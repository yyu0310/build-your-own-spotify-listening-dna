# build-your-own-spotify-listening-dna

Spotify has been recording everything you listen to for years. The problem is they use that data to recommend music based on what *other people* like, not what you actually respond to.

This pipeline takes your raw listening history, downloads the audio for every track, and runs acoustic analysis locally. The result is a personal database of 20 audio features per song, combined with your real behavioral signals: how often you listened to completion, how often you skipped, when during the day you played it. From there, six analysis scripts turn that database into charts and recommendations built entirely from your own behavior. No collaborative filtering. No "users like you."

---

## What you end up with

After a full run, you'll have:

- **`essentia_features.db`**: a SQLite database with 20 acoustic features per track across your entire listening history. Energy, valence, danceability, 6 mood axes, genre (400 Discogs classes), arousal, tempo, key.

- **`engagement.csv`**: a per-track score combining completion rate, skip rate, and recency weighting. The closest approximation of "how much do you actually like this song" that raw behavioral data can give you.

- **Five charts in `Output/`:**

  - `taste_evolution.png`: a line chart from the year you started Spotify through today, showing how your listening shifted across 5 acoustic dimensions. If your energy score trends downward over 5 years, you've been gravitating toward quieter music. If valence dips in a specific year, that period shows up.

  - `cluster_scatter.png`: a PCA scatter plot of your entire library. Each dot is a track, positioned by acoustic similarity. Dot size reflects engagement score, so your most-listened tracks sit large in the center of their cluster. You'll typically see 4–6 groups that correspond to moods: one cluster for driving/high-energy tracks, one for late-night instrumental, one for whatever genre you were deep into.

  - `av_map.png`: two axes with valence (happy vs. sad) on x and arousal (calm vs. energetic) on y. Your whole library appears as a dot cloud. Big dots are tracks you actually engaged with. If your heavy dots sit in the top-right, you gravitate toward happy and energetic; bottom-left means you tend toward reflective and mellow.

  - `context_timeofday.png`: a breakdown of your listening across four time slots (late night, morning, afternoon, evening). Shows average completion rate per slot, so you can see whether you actually enjoy what you put on at midnight versus what you play in the afternoon.

  - `context_weekday.png`: the same completion-rate analysis split by day of week.

- **A recommendation tool**: type in any song in your library and get back 10 acoustically similar tracks ranked by a combined score of audio similarity and your own past engagement. The result is songs you already own that you might have forgotten, surfaced because they match something you consistently listen to the end.

---

## How it works, layer by layer

### Layer 1: Collection

`spotify_collector.py` runs hourly via launchd (macOS) or cron (Linux). Each run calls Spotify's "recently played" endpoint and writes up to 50 tracks to SQLite.

For historical data, Spotify offers a full export of your Extended Streaming History (a GDPR data export that goes back to when you first created your account, usually 5–15 years). Once you receive those JSON files (Spotify takes 5–30 days to deliver), you seed the database in one shot and the pipeline has your entire history to work from.

### Layer 2: Download and analysis

`audio_pipeline.py` picks up every unanalyzed track in the database and finds the audio on YouTube. The search order is:

1. **spotDL**: tries Spotify's own internal YouTube Music match for the track
2. **yt-dlp with artist alias**: if spotDL fails, falls back to `"artist_name song_title"` search using `artist_aliases.json` to translate romanized Spotify names to the names YouTube actually uses (e.g., "Jay Chou" → "周杰倫", "Stefanie Sun" → "孫燕姿")

Downloads run in parallel (20 workers by default). Each downloaded file goes into `audio_analyzer.py`, which runs Essentia's TensorFlow models one at a time. Essentia already saturates all CPU cores internally, so running multiple analyzers concurrently doesn't help and only increases memory pressure. On the author's MacBook Air M3 16GB, each track takes roughly 5–10 seconds.

A 30 MB file size guard catches mismatched results; a single song in AAC or Opus should be 3–10 MB. Anything larger is probably a compilation or a wrong match.

Throughput on Apple M-series: roughly 300–600 tracks per hour.

### Layer 3: Phase 2 analysis

Once you have 1,000+ tracks analyzed, the Phase 2 scripts combine your audio features with behavioral data from the streaming history JSONs.

`phase2_engagement.py` is the foundation the other scripts depend on. For each track, it reads every play event across all your history files and computes:

- **weighted play count**: plays weighted by recency (2026 = 1.0, each prior year -0.12, floor at 0.1), so a song you rediscovered recently counts more than something you played heavily in 2018
- **completion rate**: average milliseconds played divided by track duration, pulled from a Kaggle reference dataset
- **skip rate**: fraction of plays where you hit next before the track ended

The engagement score combines all three: `log1p(weighted_play_count) × completion_rate × (1 - skip_rate)`.

The clustering and maps weight everything by this score, so the visual output reflects your actual preferences rather than raw play count.

---

## Features extracted per track

| Category | Fields |
|---|---|
| Spotify-compatible | `tempo`, `key`, `mode`, `loudness`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `valence` |
| Essentia-only | `arousal`, `tempo_confidence`, `key_strength`, `genre_discogs` (400 classes), `genre_rosamerica` |
| Mood axes | `mood_happy`, `mood_sad`, `mood_relaxed`, `mood_aggressive`, `mood_party`, `mood_electronic` |

---

## Setup

Full step-by-step setup is in [AGENTS.md](AGENTS.md), covering every step from requesting your Spotify data export to running Phase 2 analysis, with exact commands. The short version:

```bash
# 1. Request your Spotify Extended Streaming History
#    Spotify > Settings > Account > Privacy Settings > Download your data
#    (takes 5-30 days; check "Extended streaming history")

# 2. Install dependencies (two separate venvs needed)
python3 -m venv .venv
.venv/bin/pip install essentia-tensorflow numpy spotipy gspread google-auth python-dotenv requests scikit-learn pandas matplotlib

python3 -m venv .venv-spotdl
.venv-spotdl/bin/pip install spotdl yt-dlp

# 3. Configure credentials
mkdir -p ~/.config/spotify-dna
cp .env.example ~/.config/spotify-dna/.env
# Fill in SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET

# 4. Run the collector once to authenticate with Spotify
.venv/bin/python3 spotify_collector.py

# 5. Seed the database from your streaming history JSONs and start the pipeline
#    (see AGENTS.md for the exact seeding command)
caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20 > pipeline.log 2>&1 &
tail -f pipeline.log   # monitor progress

# Stop cleanly
pkill -f audio_pipeline.py
```

What each part of that command does:

- `caffeinate -i`: prevents the Mac from sleeping so the pipeline can run for hours unattended
- `--workers 20`: opens 20 parallel download threads, each searching and downloading a different track from YouTube
- `> pipeline.log 2>&1 &`: runs in the background; all output (including errors) goes to `pipeline.log` so your terminal stays free
- `tail -f pipeline.log`: stream progress in real time; prints one line per analyzed track

What the pipeline does internally for each track:

1. Search YouTube Music via spotDL (uses Spotify's own internal match, high accuracy)
2. If that fails, fall back to yt-dlp with `"artist_name song_title"`, using `artist_aliases.json` to translate romanized Spotify names
3. Download the audio (AAC or Opus, typically 3–10 MB per track)
4. Run Essentia TensorFlow models locally to extract 20 acoustic features (5–10 seconds per track on Apple Silicon)
5. Write results to `essentia_features.db`, delete the temporary audio file

The pipeline is **resumable**: stopping and restarting picks up where it left off without re-analyzing completed tracks.

Expected throughput on Apple M-series: 300–600 tracks per hour. Ten thousand tracks takes roughly 20–30 hours; you can leave it running overnight.

Credentials (Spotify API keys, Google Service Account) live in `~/.config/spotify-dna/` (outside the repo). Set `SECRETS_DIR=/your/path` to use a different location.

---

## Running Phase 2

```bash
.venv/bin/pip install scikit-learn pandas matplotlib

.venv/bin/python3 tools/phase2_engagement.py    # required first; produces engagement.csv
.venv/bin/python3 tools/phase2_evolution.py     # taste_evolution.png
.venv/bin/python3 tools/phase2_cluster.py       # cluster_scatter.png (requires engagement.csv)
.venv/bin/python3 tools/phase2_av_map.py        # av_map.png (requires clusters.csv + engagement.csv)
.venv/bin/python3 tools/phase2_context.py       # context_timeofday.png, context_weekday.png
.venv/bin/python3 tools/phase2_recommend.py "song title artist name"

.venv/bin/python3 tools/phase2_fingerprint.py  # summary printout
```

---

## Spotify artist to YouTube search mapping

`artist_aliases.json` maps Spotify display names to the search terms that actually find the right YouTube uploads. The problem: Spotify shows romanized names for many East Asian artists regardless of how those artists brand themselves on YouTube. Searching `"Jay Chou some song"` fails; `"周杰倫 some song"` finds the official upload.

**Pull requests for additional mappings are welcome**, especially for Korean, Chinese mainland, and Vietnamese artists.

| Spotify | YouTube |
|---|---|
| Jay Chou | 周杰倫 |
| Stefanie Sun | 孫燕姿 |
| Enno Cheng | 鄭宜農 Enno Cheng |
| Jonathan Lee | 李宗盛 |
| JJ Lin | 林俊傑 |
| Jolin Tsai | JOLIN 蔡依林 |
| Tanya Chua | 蔡健雅 |
| A-Mei Chang | 張惠妹 |
| Rainie Yang | 楊丞琳 |
| Princess Ai | 戴愛玲 |
| Jess Lee | 李佳薇 Jess Lee 官方音樂頻道 |
| ABAO阿爆 | ABAO阿爆_阿仍仍 |
| Joanna Wang | Joanna Wang 王若琳 |
| Wayne's so Sad | 傷心欲絕Wayneʻs So Sad |
| GBOYSWAG | 鼓鼓 呂思緯 GBOYSWAG |
| Elephant Gym | 大象體操Elephant Gym |
| Flesh Juicer | Flesh Juicer 血肉果汁機 |
| Major in Body Bear 體熊專科 | major in bodybear |
| Tizzy Bac | tizzybacvideo |
| 當代電影大師 | 當代電影大師Modern Cinema Master |
| Kairi Yagi | 八木海莉 |
| 八木海莉⚡️電音遊戯 | 八木海莉 |
| Satoko Shibata | 柴田聡子 \| Satoko Shibata |

---

## Known limitations

- **Spotify Audio Features API disabled**: Returns 403 for Developer Mode apps since November 2024. This pipeline uses local Essentia models and Kaggle/HuggingFace reference datasets as a workaround.

- **valence scores are Western-biased**: The emomusic model was trained on the emoMusic dataset (744 Western pop and rock tracks). Research shows that valence models trained on Western music have lower accuracy on Asian music. High-energy tracks from Taiwan or Japan often score low valence, even when they don't actually sound negative.

- **Single analyzer, always**: Essentia's TensorFlow runtime already uses all CPU cores through intra-op parallelism. Running two analyzers concurrently doesn't increase throughput; it halves available memory per process and causes context switching. The pipeline enforces one analyzer at a time.

- **Some tracks can't be found on YouTube**: Live recordings, regional exclusives, and Spotify-only content won't have a matching YouTube upload. These end up as `download_not_found` in the progress database. The manual workaround is finding an alternate URL yourself and adding it to `track_url_overrides.json`. Automating this would require a separate music search API with its own rate limits and matching errors.

- **New 2025-2026 releases**: Reference datasets (Kaggle, HuggingFace) have cutoff dates. Very recent tracks are often missing from both. The pipeline will still attempt to download and analyze them via YouTube, but search mismatches are more common for recent releases.
