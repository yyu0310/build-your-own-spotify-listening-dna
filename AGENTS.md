# AGENTS.md — AI Agent Context

This file provides context for AI coding assistants (Claude Code, Gemini CLI, Codex, etc.) helping users set up and run this project.

---

## What This Project Does

A personal music DNA pipeline that:
1. Collects your Spotify listening history continuously via the Spotify API
2. Downloads the full audio of each track from YouTube (yt-dlp / spotDL)
3. Runs local Essentia TensorFlow models to extract 20 acoustic features per track
4. Stores everything in SQLite and optionally syncs to Google Sheets
5. Lets you run 6 Phase 2 analysis scripts for clustering, mood mapping, taste evolution, and personalized recommendation

The key difference from streaming service recommendations: this pipeline uses **your actual listening behavior** (skip rate, completion rate, time-of-day patterns) rather than collaborative filtering.

---

## Project Structure

```
.
├── audio_pipeline.py          # Main pipeline: download + Essentia analysis
├── audio_analyzer.py          # Essentia feature extraction (single-threaded)
├── db_paths.py                # SQLite DB path constants
├── spotify_collector.py       # Hourly Spotify history collector (launchd/cron)
├── sheet_backfill_from_db.py  # Sync Essentia features → Google Sheets
│
├── artist_aliases.json        # Spotify romanized name → YouTube search term
├── skip_artist_keywords.json  # BGM/noise keywords to skip (no music content)
├── track_url_overrides.json   # Manual Spotify track_id → YouTube URL overrides
│
├── models/                    # Essentia model metadata JSON files
│   └── *.json
│
└── tools/
    ├── phase2_engagement.py   # Step 1: compute engagement scores from history
    ├── phase2_evolution.py    # Step 2: taste evolution over years (chart)
    ├── phase2_cluster.py      # Step 3: K-Means clustering (engagement-weighted)
    ├── phase2_av_map.py       # Step 4: Arousal-Valence mood map
    ├── phase2_context.py      # Step 5: time-of-day listening patterns
    ├── phase2_recommend.py    # Step 6: cosine similarity recommendation
    ├── phase2_fingerprint.py  # Print your music DNA summary
    ├── youtube_url_lookup.py  # YouTube Data API v3 URL lookup utility
    └── reset_download_not_found.py  # Pipeline recovery utility
```

**Generated at runtime (not in repo):**
```
Database/
├── essentia_features.db       # Self-computed audio features (main output)
├── essentia_progress.db       # Pipeline progress tracker
├── kaggle_features.db         # Reference features from Kaggle dataset
├── huggingface_features.db    # Reference features from HuggingFace dataset
└── Spotify Extended Streaming History/
    └── Streaming_History_Audio_*.json

Output/
├── engagement.csv
├── clusters.csv
├── cluster_scatter.png
├── av_map.png
├── taste_evolution.png
└── context_timeofday.png
```

---

## Step-by-Step Setup Guide for AI Agents

When a user asks "how do I set this up?" or "help me get started," walk through these steps in order.

### Step 1: Request Spotify Extended Streaming History

This is the most time-consuming step (Spotify takes 5–30 days to deliver the data).

Tell the user:
> Go to Spotify → Settings → Account → Privacy Settings → Download your data.
> Check "Extended streaming history" (not just "Account data").
> You'll receive a `.zip` file by email within a few days to a few weeks.
> Inside will be multiple `Streaming_History_Audio_*.json` files — keep all of them.

Place the JSON files into `Database/Spotify Extended Streaming History/`.

### Step 2: Set Up a Spotify Developer App

1. Go to https://developer.spotify.com/dashboard
2. Click "Create app"
3. Fill in any name/description
4. Set Redirect URI to: `http://127.0.0.1:8888/callback`
5. Copy the Client ID and Client Secret

### Step 3: Install Dependencies

Two separate virtual environments are needed due to spotDL/Essentia conflicts:

```bash
# Environment 1: Essentia analysis
python3 -m venv .venv
.venv/bin/pip install essentia-tensorflow numpy spotipy gspread google-auth python-dotenv requests scikit-learn pandas matplotlib

# Environment 2: Audio download (spotDL)
python3 -m venv .venv-spotdl
.venv-spotdl/bin/pip install spotdl yt-dlp
```

Also install system yt-dlp (used as fallback):
```bash
brew install yt-dlp       # macOS
# or: pip install yt-dlp  # Linux
```

### Step 4: Download Essentia Models

The `models/*.json` files in this repo are metadata only. The actual `.pb` model files must be downloaded separately.

Run this to download all required models:
```bash
mkdir -p models
.venv/bin/python3 -c "
import json, os, urllib.request
from pathlib import Path

model_dir = Path('models')
for jf in sorted(model_dir.glob('*.json')):
    meta = json.loads(jf.read_text())
    url  = meta.get('download_url') or meta.get('model_url')
    if not url:
        continue
    dest = model_dir / (jf.stem + '.pb')
    if dest.exists():
        print(f'already have {dest.name}')
        continue
    print(f'downloading {dest.name}...')
    urllib.request.urlretrieve(url, dest)
    print(f'  saved {dest.stat().st_size // 1024} KB')
print('done')
"
```

If the above fails, check each `.json` file for the `download_url` field and download manually from https://essentia.upf.edu/models/.

### Step 5: Configure Credentials

```bash
mkdir -p ~/.config/spotify-dna
cp .env.example ~/.config/spotify-dna/.env
```

Edit `~/.config/spotify-dna/.env` and fill in at minimum:
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`

The `SPOTIFY_SHEET_ID`, `LASTFM_API_KEY`, and `YOUTUBE_API_KEY` are optional.

### Step 6: Set Up Databases

Create the required directories and initialize the SQLite databases:

```bash
mkdir -p Database Output

# Create the two required databases (schema auto-created on first run)
.venv/bin/python3 -c "
import sqlite3
from pathlib import Path

Path('Database').mkdir(exist_ok=True)

# essentia_features.db
conn = sqlite3.connect('Database/essentia_features.db')
conn.execute('''CREATE TABLE IF NOT EXISTS features (
    track_id TEXT PRIMARY KEY,
    track_name TEXT, artist TEXT, album TEXT,
    energy REAL, valence REAL, danceability REAL, acousticness REAL,
    instrumentalness REAL, mood_happy REAL, mood_sad REAL,
    mood_relaxed REAL, mood_aggressive REAL, mood_party REAL,
    mood_electronic REAL, arousal REAL, tempo REAL, loudness REAL,
    key INTEGER, mode INTEGER,
    genre_discogs TEXT, genre_rosamerica TEXT,
    analyzed_at TEXT
)''')
conn.commit(); conn.close()

# essentia_progress.db
conn = sqlite3.connect('Database/essentia_progress.db')
conn.execute('''CREATE TABLE IF NOT EXISTS progress (
    track_id TEXT PRIMARY KEY,
    track_name TEXT, artist TEXT,
    status TEXT, attempts INTEGER DEFAULT 0,
    last_error TEXT, updated_at TEXT
)''')
conn.execute('''CREATE TABLE IF NOT EXISTS failed (
    track_id TEXT PRIMARY KEY,
    attempts INTEGER DEFAULT 0,
    last_error TEXT, updated_at TEXT
)''')
conn.commit(); conn.close()
print('databases initialized')
"
```

If you also have Kaggle/HuggingFace reference datasets, place them at:
- `Database/kaggle_features.db`
- `Database/huggingface_features.db`

These are optional — the pipeline falls back gracefully if they don't exist.

### Step 7: First Authentication (Spotify OAuth)

Run the collector once to trigger OAuth:

```bash
.venv/bin/python3 spotify_collector.py
```

A browser window opens. Log in and authorize. The token is cached in `~/.config/spotify-dna/.spotify_token_cache`.

To run automatically on macOS, set up a launchd job (see `launchd` section below).

### Step 8: Seed the Track List

The pipeline needs a list of track IDs to analyze. Two ways:

**Option A: Use Extended Streaming History (recommended)**

```bash
.venv/bin/python3 -c "
import json, sqlite3
from pathlib import Path

history_dir = Path('Database/Spotify Extended Streaming History')
conn = sqlite3.connect('Database/essentia_progress.db')

inserted = 0
for jf in sorted(history_dir.glob('*.json')):
    rows = json.loads(jf.read_text())
    for row in rows:
        uri = row.get('spotify_track_uri', '') or ''
        if not uri.startswith('spotify:track:'):
            continue
        track_id   = uri.split(':')[2]
        track_name = row.get('master_metadata_track_name') or ''
        artist     = row.get('master_metadata_album_artist_name') or ''
        try:
            conn.execute(
                'INSERT OR IGNORE INTO progress (track_id, track_name, artist, status, attempts) VALUES (?,?,?,?,?)',
                (track_id, track_name, artist, 'pending', 0)
            )
            inserted += 1
        except Exception:
            pass

conn.commit(); conn.close()
print(f'seeded {inserted} tracks into essentia_progress.db')
"
```

**Option B: Let the collector populate automatically**

If `spotify_collector.py` runs hourly, it will populate the database with recently played tracks over time.

### Step 9: Run the Pipeline

```bash
# Start pipeline (resumable — picks up where it left off)
caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20 > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log

# Stop cleanly
pkill -f audio_pipeline.py
```

Key flags:
- `--workers N` — parallel download threads (20 is a good starting point on Mac M-series)
- `--limit N` — stop after N tracks (useful for testing)
- `--keep-audio` — don't delete downloaded audio after analysis

Expected throughput: ~300–600 tracks/hour on Apple Silicon.

### Step 10: Phase 2 Analysis

First install analysis dependencies:
```bash
.venv/bin/pip install scikit-learn pandas matplotlib
```

Run in order (each step builds on the previous):

```bash
# Step 1: Compute engagement scores (required by steps 3, 4, 6)
.venv/bin/python3 tools/phase2_engagement.py

# Step 2: Taste evolution chart (independent)
.venv/bin/python3 tools/phase2_evolution.py

# Step 3: K-Means clustering (requires Output/engagement.csv)
.venv/bin/python3 tools/phase2_cluster.py

# Step 4: Arousal-Valence mood map (requires Output/clusters.csv + engagement.csv)
.venv/bin/python3 tools/phase2_av_map.py

# Step 5: Time-of-day context analysis (independent)
.venv/bin/python3 tools/phase2_context.py

# Step 6: Recommendation (requires Output/engagement.csv)
.venv/bin/python3 tools/phase2_recommend.py "Song Title Artist Name"

# Print your music DNA summary
.venv/bin/python3 tools/phase2_fingerprint.py
```

All charts saved to `Output/`.

---

## Running Continuously (macOS launchd)

To collect Spotify history automatically in the background:

```bash
# Create the plist
cat > ~/Library/LaunchAgents/com.user.spotify-collector.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.spotify-collector</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/project/.venv/bin/python3</string>
        <string>/path/to/project/spotify_collector.py</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

# Load it
launchctl load ~/Library/LaunchAgents/com.user.spotify-collector.plist
```

Replace `/path/to/project` with the absolute path to this directory.

---

## Troubleshooting Common Issues

### "download_not_found" for many tracks

The yt-dlp search is looking for the wrong artist name. Check `artist_aliases.json`:
- If the artist uses a different name on YouTube (common for East Asian artists), add an entry.
- Format: `"Spotify Display Name": "YouTube search term or channel name"`

If the track truly isn't on YouTube, add a manual override to `track_url_overrides.json`.

### Essentia model files missing

If you see `model not found` errors, the `.pb` files weren't downloaded. Each `models/*.json` has a download URL — download the corresponding `.pb` file to the `models/` directory.

### Pipeline stuck / stalled

```bash
# Check status
sqlite3 Database/essentia_progress.db "SELECT status, COUNT(*) FROM progress GROUP BY status;"

# Reset tracks stuck in 'downloading' state (e.g. after crash)
python3 tools/reset_download_not_found.py --reset-downloading
```

### YouTube quota exhausted (youtube_url_lookup.py)

The YouTube Data API v3 gives 100 search units/day per key (free tier). If quota is hit:
- Wait 24 hours for reset
- Or create additional API keys in Google Cloud Console

The lookup tool caches nothing — each run re-queries. Use `--min-score 0.7` to skip low-confidence results.

### Google Sheets 403 / permission error

The service account needs to be invited to the spreadsheet:
1. Open your Google Sheet
2. Share → invite the service account email (found in `service_account.json` → `client_email`)
3. Grant "Editor" access

---

## Key Design Decisions (for AI agents)

- **Single-threaded Essentia**: Essentia's TF models use all CPU cores via intra-op parallelism. Running multiple analyzers simultaneously doesn't increase speed — it only adds memory pressure and context switching overhead.
- **spotDL + yt-dlp dual strategy**: spotDL tries Spotify's internal YouTube Music matching first. If it fails, `audio_pipeline.py` falls back to yt-dlp with a `"artist title"` query (using `artist_aliases.json` to translate romanized names).
- **30 MB file guard**: Files over 30 MB are flagged as `file_too_large`. This catches compilation albums and mismatched results (a single track should be 3–10 MB in AAC/Opus format).
- **Three-database architecture**: `essentia_features.db` (self-computed, open-sourceable), `kaggle_features.db` and `huggingface_features.db` (public datasets, optional reference). Keeping them separate avoids licensing ambiguity.
- **Recency weighting in engagement**: `engagement_score = log1p(weighted_play_count) × completion_rate × (1 - skip_rate)`. Plays from 2026 have weight 1.0; each prior year is 0.12 lower (floor at 0.1). This means recent listening behavior influences clustering and recommendations more than decade-old plays.
