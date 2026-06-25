# Spotify Audio Analysis Pipeline

A personal music fingerprinting pipeline that analyzes your Spotify listening history. Downloads full audio from YouTube, runs Essentia TensorFlow models locally, and builds a feature database for music recommendation.

## What It Does

1. **Collector** (`spotify_collector.py`): Hourly cron that pulls recently played tracks from Spotify API and writes to Google Sheets
2. **Pipeline** (`audio_pipeline.py`): Downloads audio via spotDL / yt-dlp, analyzes with Essentia (20 features per track), writes to SQLite
3. **Backfill** (`sheet_backfill_from_db.py`): Syncs analysis results from local DB back to Google Sheets every 30 minutes

## Features Extracted

| Category | Fields |
|---|---|
| Spotify-compatible | `tempo`, `key`, `mode`, `loudness`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `valence` |
| Essentia-only | `arousal`, `tempo_confidence`, `key_strength`, `genre_discogs` (400 classes), `genre_rosamerica` |
| Mood axes | `mood_happy`, `mood_sad`, `mood_relaxed`, `mood_aggressive`, `mood_party`, `mood_electronic` |

## Architecture

```
macOS launchd (hourly)
  → spotify_collector.py
      → Spotify API (recently played)
      → Local SQLite DBs (Kaggle / HuggingFace / Essentia)
      → Google Sheets

caffeinate (long-running)
  → audio_pipeline.py --workers 20
      → spotDL / yt-dlp (parallel download, ThreadPoolExecutor)
      → audio_analyzer.py (Essentia, single-threaded inference)
      → essentia_features.db
```

## Database Architecture

Three separate SQLite databases (split for open-source licensing clarity):

| DB | Source | Size | Notes |
|---|---|---|---|
| `kaggle_features.db` | Public Kaggle datasets | ~1M tracks | Read-only reference |
| `huggingface_features.db` | HuggingFace dataset | ~1.2M tracks | Read-only reference |
| `essentia_features.db` | Self-computed | Growing | Open-sourceable |

## Requirements

```bash
# Analysis environment
python3 -m venv .venv
.venv/bin/pip install essentia-tensorflow numpy

# Download environment
python3 -m venv .venv-spotdl
.venv-spotdl/bin/pip install spotdl yt-dlp
```

Credentials (Spotify API keys, Google Service Account) go in a separate directory outside the repo and are loaded via `python-dotenv`.

---

## Spotify Artist → YouTube Channel Mapping

`artist_aliases.json` maps Spotify artist display names to their correct YouTube channel names or search terms. This is needed because many East Asian artists use romanized names on Spotify but their own-language names on YouTube — yt-dlp's default search fails without this correction.

**We welcome community contributions to this table.**

If you listen to artists not in this list and know the correct YouTube channel or search term, open a PR to add them.

### Format

```json
{
  "Spotify Display Name": "YouTube Channel Name or Search Term"
}
```

The value can be either an exact channel name (e.g. `tizzybacvideo`) or a better search term (e.g. `蔡健雅`). Both work with yt-dlp's `ytsearch`.

### Current Mappings

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

Primarily Taiwanese and Japanese artists — contributions welcome, especially for Korean, Chinese, and other East Asian markets.

### Why This Exists

Spotify assigns romanized display names to many artists regardless of their actual branding. When yt-dlp searches `"Jay Chou some song title"` it fails; `"周杰倫 some song title"` finds the official upload immediately. This file is the bridge.

---

## Running the Pipeline

```bash
# Start pipeline (resumable, picks up where it left off)
caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20 > pipeline.log 2>&1 &

# Stop cleanly (kills all download subprocesses too)
pkill -f audio_pipeline.py

# Check progress
tail -f pipeline.log
```

Key flags: `--workers N` (parallel downloads, ~20 saturates one Essentia analyzer), `--limit N` (stop after N tracks), `--keep-audio` (don't delete after analysis).

---

## Phase 2 Analysis

After the pipeline has analyzed 1,000+ tracks, run the Phase 2 suite to visualize your music DNA.

Install additional dependencies:
```bash
.venv/bin/pip install scikit-learn pandas matplotlib
```

Run in order (each script builds on the previous):

```bash
# 1. Engagement scoring (required by steps 3, 4, 6)
#    Outputs: Output/engagement.csv
.venv/bin/python3 tools/phase2_engagement.py

# 2. Taste evolution chart (independent)
#    Outputs: Output/taste_evolution.png
.venv/bin/python3 tools/phase2_evolution.py

# 3. K-Means clustering, engagement-weighted
#    Requires: Output/engagement.csv
#    Outputs: Output/clusters.csv, cluster_scatter.png, elbow.png
.venv/bin/python3 tools/phase2_cluster.py

# 4. Arousal-Valence mood map
#    Requires: Output/clusters.csv + engagement.csv
#    Outputs: Output/av_map.png
.venv/bin/python3 tools/phase2_av_map.py

# 5. Time-of-day listening patterns (independent)
#    Outputs: Output/context_timeofday.png, context_weekday.png
.venv/bin/python3 tools/phase2_context.py

# 6. Recommendation from your own catalog
#    Requires: Output/engagement.csv
.venv/bin/python3 tools/phase2_recommend.py "Song Title Artist Name"

# Print your music DNA fingerprint summary
.venv/bin/python3 tools/phase2_fingerprint.py
```

### What Each Script Produces

| Script | Output | What It Shows |
|---|---|---|
| `phase2_engagement.py` | `engagement.csv` | Per-track score: completion rate × (1 - skip rate) × recency weight |
| `phase2_evolution.py` | `taste_evolution.png` | 5-feature trend lines across years |
| `phase2_cluster.py` | `cluster_scatter.png` | PCA scatter, point size = how much you like it |
| `phase2_av_map.py` | `av_map.png` | Emotional map: calm ↔ energetic × negative ↔ positive |
| `phase2_context.py` | `context_*.png` | Which moods you actually enjoy at different times |
| `phase2_recommend.py` | Terminal output | "You liked X, you might have forgotten Y" |

## Known Limitations

- **Spotify Audio Features API**: Returns 403 for Development Mode apps since November 2024. The pipeline works around this with local Kaggle/HuggingFace datasets and self-computed Essentia features.
- **valence bias**: The emomusic model was trained on Western music (MSD). Loud/fast/distorted tracks from Taiwan/Japan may score low valence despite being energetic and positive. Use `mood_aggressive` + `mood_party` instead for metal/rock.
- **Single analyzer throughput ceiling**: Essentia's TF intra-op parallelism already saturates the CPU; multiple analyzers add no speed — only one analyzer runs at a time.
- **30 MB file guard**: Downloads over 30 MB are assumed to be mismatched compilations/albums and are skipped (`file_too_large` in progress DB).
- **Tracks not on YouTube / Spotify-exclusive content**: Some tracks (live event recordings, commentary tracks, platform-exclusive releases) exist only on Spotify and have no matching YouTube upload. These will permanently fail with `download_not_found`. The only workaround is to manually find an alternative YouTube URL and add it to `track_url_overrides.json`. This is intentionally left as a manual step — automation would require a separate lyrics/music search API.
- **New releases not in reference datasets**: Kaggle and HuggingFace datasets have a cutoff date. Tracks released in 2025–2026 are often absent from both. The pipeline will still attempt to download and analyze them via YouTube, but if yt-dlp search fails (wrong title match, no upload yet), they end up in `download_not_found`. These will be resolved over time as official uploads appear on YouTube.
