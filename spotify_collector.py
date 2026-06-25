#!/usr/bin/env python3
"""
Spotify Collector v1.4.0
每小時抓最近播放紀錄 + audio features + Last.fm 樂器標籤 + AcousticBrainz 特徵，去重後寫入 Google Sheet
由 macOS launchd 定時觸發
"""

import sys
import signal
import subprocess
import sqlite3
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
import gspread

SCRIPT_TIMEOUT = 300  # 5 分鐘全局 timeout


def notify_mac(title, message):
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
            timeout=5,
        )
    except Exception:
        pass


def _timeout_handler(signum, frame):
    msg = "腳本卡死超過 5 分鐘，已強制結束（可能是 gspread 無回應）"
    notify_mac("⚠️ Spotify Collector 逾時", msg)
    print(f"[ERROR] {msg}", flush=True)
    sys.exit(1)

SECRETS_DIR   = Path(os.getenv("SECRETS_DIR", str(Path.home() / ".config" / "spotify-dna")))
ENV_FILE      = SECRETS_DIR / ".env"
SPOTIFY_CACHE = SECRETS_DIR / ".spotify_token_cache"
GSHEET_SA_KEY = Path(os.getenv("GSHEET_SA_KEY_PATH", str(SECRETS_DIR / "service_account.json")))
from db_paths import KAGGLE_DB, HF_DB, ESSENTIA_DB  # DB 拆分後查三庫（2026-06-20 Phase 3）
AB_CACHE_DB    = Path(__file__).parent / "Database" / "ab_cache.db"
MB_USER_AGENT  = "SpotifyCollector/1.3.0 (your-email@example.com)"

SPOTIFY_SCOPES = " ".join([
    "user-read-recently-played",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
    "playlist-read-private",
])

TW_TZ = timezone(timedelta(hours=8))

SHEET_NAME = "聆聽紀錄"
SHEET_HEADERS = [
    # 識別
    "played_at", "track_id", "track_name", "artist", "album", "duration",
    # Essentia (20 欄，由 backfill 回填)
    "es_arousal", "es_tempo_confidence", "es_key_strength",
    "es_mood_happy", "es_mood_sad", "es_mood_relaxed", "es_mood_aggressive",
    "es_mood_party", "es_mood_electronic",
    "es_genre_rosamerica", "es_genre_discogs",
    "es_energy", "es_valence", "es_danceability", "es_instrumentalness", "es_tempo",
    "es_acousticness", "es_loudness", "es_mode", "es_key",
    # Kaggle/HF/Spotify 原始值 (12 欄)
    "sp_energy", "sp_valence", "sp_danceability", "sp_instrumentalness", "sp_tempo",
    "sp_acousticness", "sp_speechiness", "sp_liveness", "sp_loudness", "sp_mode", "sp_key", "sp_time_signature",
    # 來源標記、Last.fm、AcousticBrainz
    "af_source", "lfm_instrument_tags",
    "ab_voice_instrumental", "ab_mood_happy", "ab_mood_sad", "ab_mood_aggressive", "ab_mood_relaxed",
    "ab_danceability", "ab_genre",
]

INSTRUMENT_KEYWORDS = {
    "piano", "guitar", "strings", "violin", "cello", "bass", "drums",
    "electronic", "orchestral", "acoustic", "saxophone", "trumpet",
    "synthesizer", "synth", "ambient", "classical", "jazz", "harp",
    "flute", "clarinet", "trombone", "organ", "keyboard", "percussion",
    "brass", "woodwind", "vocal", "choir",
}


def load_spotify_client():
    load_dotenv(ENV_FILE)
    client_id    = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    if not client_id or not client_secret:
        print(f"[ERROR] 找不到 Spotify 憑證：{ENV_FILE}")
        sys.exit(1)
    cache_handler = CacheFileHandler(cache_path=str(SPOTIFY_CACHE))
    auth_manager  = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SPOTIFY_SCOPES,
        cache_handler=cache_handler,
        open_browser=True,
    )
    sp   = spotipy.Spotify(auth_manager=auth_manager, retries=0)
    user = sp.me()
    print(f"[SPOTIFY] 登入：{user['display_name']} ({user['id']})")
    return sp


def fetch_recently_played(sp):
    print("[SPOTIFY] 抓取最近播放紀錄（limit=50）...")
    results = sp.current_user_recently_played(limit=50)
    items   = results["items"]
    print(f"[SPOTIFY] 取得 {len(items)} 筆")
    return items


def format_played_at(iso_str):
    """UTC ISO → 台灣時間字串，用於去重比對（與 col_values 的顯示格式一致）。"""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.astimezone(TW_TZ).strftime("%Y-%m-%d %H:%M")


def to_sheets_timestamp(iso_str):
    """UTC ISO → Google Sheets serial number（台灣本地時間）。
    Sheets 以 1899-12-30 為起點，serial = 距起點天數（含小數）。
    """
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(TW_TZ)
    naive = dt.replace(tzinfo=None)
    return (naive - datetime(1899, 12, 30)).total_seconds() / 86400


def _num(val):
    """確保數值型欄位寫入 Google Sheets 時是 float/int，而非字串。
    空值或無法轉換時回傳空字串。
    """
    if val == "" or val is None:
        return ""
    if isinstance(val, (int, float)):
        return val
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return val


def format_duration(ms):
    return int(ms) // 1000


def format_sheet_columns(sheet):
    """設定欄位格式：A 欄 = 日期時間。只需執行一次，冪等。"""
    sheet.format("A2:A", {"numberFormat": {"type": "DATE_TIME", "pattern": "yyyy-mm-dd hh:mm"}})


def fetch_lastfm_tags(artist, track_name, api_key):
    url = "https://ws.audioscrobbler.com/2.0/"
    try:
        r = requests.get(url, params={
            "method":      "track.getTopTags",
            "artist":      artist,
            "track":       track_name,
            "api_key":     api_key,
            "format":      "json",
            "autocorrect": 1,
        }, timeout=5)
        tags = [t["name"].lower() for t in r.json().get("toptags", {}).get("tag", [])]
        matched = [t for t in tags if any(kw in t for kw in INSTRUMENT_KEYWORDS)]
        return ",".join(matched)
    except Exception:
        return ""


def ensure_sheet_headers(sheet):
    current = sheet.row_values(1)
    missing = [h for h in SHEET_HEADERS if h not in current]
    if missing:
        sheet.update(range_name="A1", values=[current + missing])
        print(f"[GSHEET] 標題列新增欄位：{missing}")


def fetch_audio_features_from_db(track_ids):
    """查三庫（DB 拆分後）。查詢順序 kaggle → hf → essentia，essentia 最後寫入故
    自算值優先覆蓋；只用非空值覆蓋，保留各庫獨有欄位（如 kaggle 的 speechiness）。"""
    if not track_ids:
        return {}
    placeholders = ",".join("?" * len(track_ids))
    features = {}
    for db in (KAGGLE_DB, HF_DB, ESSENTIA_DB):   # essentia 最後 → 自算優先
        if not os.path.exists(db):
            continue
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM features WHERE track_id IN ({placeholders})", track_ids
        ).fetchall()
        conn.close()
        for row in rows:
            d = dict(row)
            tid = d.pop("track_id")
            features.setdefault(tid, {}).update(
                {k: v for k, v in d.items() if v not in (None, "")})
    print(f"[DB] 三庫命中 {len(features)}/{len(track_ids)} 筆 audio features")
    return features


def fetch_audio_features(sp, track_ids):
    """
    Audio Features API 自 2024-11-27 起對新建 App 封鎖（Development Mode）。
    遇到 403 時回傳空 dict，聆聽紀錄照常寫入，audio feature 欄位留空。
    未來替代方案：Kaggle 預抓資料集 / 其他音樂資料庫（見工作日誌）。
    """
    print(f"[SPOTIFY] 嘗試抓取 {len(track_ids)} 首 audio features...")
    features = {}
    try:
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i : i + 100]
            results = sp.audio_features(batch)
            for f in results:
                if f:
                    features[f["id"]] = f
        print(f"[SPOTIFY] 取得 {len(features)} 筆 audio features")
    except Exception as e:
        if "403" in str(e):
            print("[SPOTIFY] Audio Features API 受限（403），欄位留空。詳見工作日誌「踩坑記錄」")
        else:
            print(f"[SPOTIFY] Audio Features 抓取失敗（{e}），欄位留空")
    return features


def fetch_isrc_map(sp, track_ids):
    """逐一查詢完整 Track Object 取得 ISRC（recently_played 精簡版不含 external_ids；
    sp.tracks() 批次端點在 Development Mode 被 403 封鎖，改用單筆 sp.track()）。
    僅對 new_items 呼叫，數量通常很少。
    """
    isrc_map = {}
    for tid in track_ids:
        try:
            t = sp.track(tid)
            isrc = t.get("external_ids", {}).get("isrc", "")
            if isrc:
                isrc_map[tid] = isrc
        except Exception as e:
            print(f"  [AB] track {tid} ISRC 查詢失敗：{e}")
    print(f"[AB] ISRC 取得 {len(isrc_map)}/{len(track_ids)} 首")
    return isrc_map


def init_ab_cache():
    conn = sqlite3.connect(AB_CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS isrc_mbid (
            isrc TEXT PRIMARY KEY,
            mbid TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ab_features (
            mbid      TEXT PRIMARY KEY,
            voice_instrumental TEXT,
            mood_happy    REAL,
            mood_sad      REAL,
            mood_aggressive REAL,
            mood_relaxed  REAL,
            danceability  TEXT,
            genre         TEXT
        )
    """)
    conn.commit()
    conn.close()


def lookup_mbid(isrc):
    """ISRC → MBID via MusicBrainz。已快取則不打 API。回傳 "" 表示查無資料，None 表示網路錯誤。"""
    conn = sqlite3.connect(AB_CACHE_DB)
    row = conn.execute("SELECT mbid FROM isrc_mbid WHERE isrc=?", (isrc,)).fetchone()
    conn.close()
    if row is not None:
        return row[0]

    try:
        time.sleep(1.1)  # MusicBrainz rate limit: 1 req/sec
        r = requests.get(
            "https://musicbrainz.org/ws/2/recording",
            params={"query": f"isrc:{isrc}", "fmt": "json"},
            headers={"User-Agent": MB_USER_AGENT},
            timeout=10,
        )
        recordings = r.json().get("recordings", [])
        mbid = recordings[0]["id"] if recordings else ""
    except Exception as e:
        print(f"  [MB] ISRC {isrc} 查詢失敗：{e}")
        return None  # 網路錯誤，不快取

    conn = sqlite3.connect(AB_CACHE_DB)
    conn.execute("INSERT OR REPLACE INTO isrc_mbid VALUES (?,?)", (isrc, mbid))
    conn.commit()
    conn.close()
    return mbid


def fetch_ab_features(mbid):
    """MBID → AcousticBrainz 高階特徵。已快取則不打 API。回傳 dict 或 {}。"""
    if not mbid:
        return {}

    conn = sqlite3.connect(AB_CACHE_DB)
    row = conn.execute("SELECT * FROM ab_features WHERE mbid=?", (mbid,)).fetchone()
    conn.close()
    if row is not None:
        cols = ["mbid", "voice_instrumental", "mood_happy", "mood_sad",
                "mood_aggressive", "mood_relaxed", "danceability", "genre"]
        d = dict(zip(cols, row))
        d.pop("mbid")
        return d

    try:
        r = requests.get(
            f"https://acousticbrainz.org/{mbid}/high-level",
            params={"format": "json"},
            timeout=10,
        )
        if r.status_code == 404:
            features = {k: "" for k in ["voice_instrumental", "mood_happy", "mood_sad",
                                         "mood_aggressive", "mood_relaxed", "danceability", "genre"]}
        else:
            hl = r.json().get("highlevel", {})
            def prob(key, cls):
                return hl.get(key, {}).get("all", {}).get(cls, "")
            features = {
                "voice_instrumental": hl.get("voice_instrumental", {}).get("value", ""),
                "mood_happy":         prob("mood_happy", "happy"),
                "mood_sad":           prob("mood_sad", "sad"),
                "mood_aggressive":    prob("mood_aggressive", "aggressive"),
                "mood_relaxed":       prob("mood_relaxed", "relaxed"),
                "danceability":       hl.get("danceability", {}).get("value", ""),
                "genre":              hl.get("genre_rosamerica", {}).get("value", ""),
            }
    except Exception as e:
        print(f"  [AB] MBID {mbid} 查詢失敗：{e}")
        return {}  # 網路錯誤，不快取

    conn = sqlite3.connect(AB_CACHE_DB)
    conn.execute(
        "INSERT OR REPLACE INTO ab_features VALUES (?,?,?,?,?,?,?,?)",
        (mbid, features["voice_instrumental"], features["mood_happy"], features["mood_sad"],
         features["mood_aggressive"], features["mood_relaxed"], features["danceability"], features["genre"]),
    )
    conn.commit()
    conn.close()
    return features


def get_gsheet(spreadsheet_id):
    if not GSHEET_SA_KEY.exists():
        print(f"[ERROR] 找不到 Google Sheets Service Account 金鑰：{GSHEET_SA_KEY}")
        print("[ERROR] 請先完成 Service Account 設定（見工作日誌 Setup 步驟）")
        sys.exit(1)
    gc = gspread.service_account(filename=str(GSHEET_SA_KEY))
    print("[GSHEET] 連線成功")
    spreadsheet = gc.open_by_key(spreadsheet_id)
    try:
        sheet = spreadsheet.worksheet(SHEET_NAME)
        print(f"[GSHEET] 找到工作表：{SHEET_NAME}")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=SHEET_NAME, rows=50000, cols=len(SHEET_HEADERS))
        sheet.append_row(SHEET_HEADERS)
        print(f"[GSHEET] 新建工作表：{SHEET_NAME}")
    return sheet


def get_existing_played_at(sheet):
    values   = sheet.col_values(1)[1:]  # 跳過標題行
    existing = set(values)
    # 將舊格式（raw UTC）也轉成新格式加入，確保不重複
    for v in list(existing):
        try:
            existing.add(format_played_at(v))
        except Exception:
            pass
    print(f"[GSHEET] 現有紀錄：{len(values)} 筆")
    return existing


def build_row(item, features_map, tags_map=None, af_source_map=None, ab_map=None):
    played_at = item["played_at"]
    track     = item["track"]
    tid       = track["id"]
    f         = features_map.get(tid, {})
    ab        = (ab_map or {}).get(tid, {})
    return [
        # 識別
        to_sheets_timestamp(played_at),
        tid,
        track["name"],
        track["artists"][0]["name"],
        track["album"]["name"],
        format_duration(track["duration_ms"]),
        # Essentia (20 欄空白佔位，backfill 回填)
        "", "", "", "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "", "",
        # Kaggle/HF/Spotify 原始值 (12 欄)
        _num(f.get("energy", "")),
        _num(f.get("valence", "")),
        _num(f.get("danceability", "")),
        _num(f.get("instrumentalness", "")),
        _num(f.get("tempo", "")),
        _num(f.get("acousticness", "")),
        _num(f.get("speechiness", "")),
        _num(f.get("liveness", "")),
        _num(f.get("loudness", "")),
        _num(f.get("mode", "")),
        _num(f.get("key", "")),
        _num(f.get("time_signature", "")),
        # 來源標記、Last.fm、AcousticBrainz
        (af_source_map or {}).get(tid, ""),
        (tags_map or {}).get(tid, ""),
        ab.get("voice_instrumental", ""),
        _num(ab.get("mood_happy", "")),
        _num(ab.get("mood_sad", "")),
        _num(ab.get("mood_aggressive", "")),
        _num(ab.get("mood_relaxed", "")),
        ab.get("danceability", ""),
        ab.get("genre", ""),
    ]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SCRIPT_TIMEOUT)

    print("=== Spotify Collector v1.4.0 START ===")

    load_dotenv(ENV_FILE)
    spreadsheet_id = os.getenv("SPOTIFY_SHEET_ID")
    if not spreadsheet_id:
        print(f"[ERROR] 找不到 SPOTIFY_SHEET_ID，請加入 {ENV_FILE}")
        sys.exit(1)

    lastfm_api_key = os.getenv("LASTFM_API_KEY")
    if lastfm_api_key:
        print("[LASTFM] API Key 已載入")
    else:
        print("[LASTFM] 未設定 LASTFM_API_KEY，跳過樂器標籤")

    sp    = load_spotify_client()
    items = fetch_recently_played(sp)

    if not items:
        print("[RESULT] 沒有播放紀錄，結束")
        return

    track_ids    = [item["track"]["id"] for item in items if item["track"]["id"]]
    features_map = fetch_audio_features_from_db(track_ids)
    db_hit_ids   = set(features_map.keys())

    missing_ids  = [tid for tid in track_ids if tid not in features_map]
    if missing_ids:
        api_features = fetch_audio_features(sp, missing_ids)
        features_map.update(api_features)
        api_hit_ids = set(api_features.keys())
    else:
        api_hit_ids = set()

    af_source_map = {
        tid: ("kaggle_db" if tid in db_hit_ids else "spotify_api" if tid in api_hit_ids else "")
        for tid in track_ids
    }

    sheet    = get_gsheet(spreadsheet_id)
    ensure_sheet_headers(sheet)
    format_sheet_columns(sheet)
    existing = get_existing_played_at(sheet)

    # 去重：用台灣時間格式比對（sheet 已統一為台灣時間）
    new_items = [item for item in items
                 if format_played_at(item["played_at"]) not in existing]

    # Last.fm 只查新歌，避免浪費 API 呼叫
    tags_map = {}
    if lastfm_api_key and new_items:
        print(f"[LASTFM] 查詢 {len(new_items)} 首樂器標籤...")
        for item in new_items:
            track = item["track"]
            tags  = fetch_lastfm_tags(track["artists"][0]["name"], track["name"], lastfm_api_key)
            tags_map[track["id"]] = tags
            if tags:
                print(f"  [LASTFM] {track['name']} → {tags}")
        hit = sum(1 for v in tags_map.values() if v)
        print(f"[LASTFM] 命中 {hit}/{len(new_items)} 首")

    # AcousticBrainz：ISRC → MBID → 高階特徵
    ab_map = {}
    if new_items:
        init_ab_cache()
        new_tids = [item["track"]["id"] for item in new_items]
        isrc_map = fetch_isrc_map(sp, new_tids)
        print(f"[AB] 查詢 {len(new_items)} 首 AcousticBrainz 特徵（rate limit: 1 req/sec）...")
        ab_hit = 0
        for item in new_items:
            track = item["track"]
            tid   = track["id"]
            isrc  = isrc_map.get(tid, "")
            if not isrc:
                ab_map[tid] = {}
                continue
            mbid = lookup_mbid(isrc)
            if mbid is None:
                ab_map[tid] = {}
                continue
            features = fetch_ab_features(mbid)
            ab_map[tid] = features
            if features.get("voice_instrumental"):
                ab_hit += 1
                print(f"  [AB] {track['name']} → {features.get('voice_instrumental')} / {features.get('genre','')}")
        print(f"[AB] 命中 {ab_hit}/{len(new_items)} 首")

    new_rows = []
    for item in new_items:
        row = build_row(item, features_map, tags_map, af_source_map, ab_map)
        new_rows.append(row)
        print(f"  [NEW] {item['played_at'][:16]} | {item['track']['artists'][0]['name']} - {item['track']['name']}")

    skipped = len(items) - len(new_rows)
    print(f"\n[RESULT] 新增 {len(new_rows)} 筆，跳過 {skipped} 筆（重複）")

    if new_rows:
        sheet.insert_rows(new_rows, row=2, value_input_option="RAW")
        print("[GSHEET] 寫入完成")

    signal.alarm(0)
    print("=== Spotify Collector v1.4.0 END ===")
    notify_mac("Spotify Collector", f"新增 {len(new_rows)} 筆，跳過 {skipped} 筆")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        notify_mac("⚠️ Spotify Collector 錯誤", str(e)[:80])
        print(f"[ERROR] 未預期錯誤：{e}", flush=True)
        sys.exit(1)
