#!/usr/bin/env python3
"""
audio_pipeline.py — Spotify 聆聽史「自算 audio features」主流程。

對「聽過但本地 DB 沒有特徵」的歌：
  下載完整音檔 → Essentia 分析 → 寫回 essentia_features.db → 刪音檔。

特性
- 可續跑：每首分析完立刻 commit；重跑會自動跳過已分析（af_source='essentia'）的歌。
  合上電腦只是暫停，回來重跑同指令即接續，零損失。
- 逐首刪音檔（除非 --keep-audio）：磁碟峰值 <10MB。
- 失敗紀錄：找不到/分析失敗的歌記進 essentia_progress.db，重試 >=MAX_ATTEMPTS 次後跳過。
- 先小批：--limit N 只跑「最常聽」的前 N 首，方便驗證品質再放大。
- 下載：預設 spotdl 主要、yt-dlp 備援（--downloader 可指定）。

用法範例
  python3 audio_pipeline.py --limit 100            # 先跑最常聽的 100 首
  python3 audio_pipeline.py                        # 全跑（約 17k 首，可中斷續跑）
  caffeinate -i python3 audio_pipeline.py          # 整夜防睡跑（lid 需開著、插電）
"""

import os
import sys
import json
import glob
import time
import signal
import sqlite3
import argparse
import subprocess
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from audio_analyzer import Analyzer

from db_paths import ESSENTIA_DB, have_feature_ids, essentia_ids

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = ESSENTIA_DB      # 寫入目標：自算特徵庫（DB 拆分後，2026-06-20 Phase 3）
PROGRESS_DB = os.path.join(HERE, "Database", "essentia_progress.db")
HIST_DIR = os.path.join(HERE, "Database", "Spotify Extended Streaming History")
TMP_DIR = os.path.join(HERE, "tmp_audio")
SPOTDL_BIN = os.path.join(HERE, ".venv-spotdl", "bin", "spotdl")

MIN_MS_PLAYED = 30000      # 過濾掉播放 <30s 的紀錄
MAX_ATTEMPTS = 3           # 同一首失敗幾次後永久跳過（50路+ 限流保護，改為 3 次）
DOWNLOAD_TIMEOUT = 180     # 單首下載逾時（秒）；15路+ 同時下載時網路競爭，需更寬裕
_ALIAS_PATH = os.path.join(HERE, "artist_aliases.json")
ARTIST_ALIASES = {k: v for k, v in json.load(open(_ALIAS_PATH, encoding="utf-8")).items()
                  if not k.startswith("_")} if os.path.exists(_ALIAS_PATH) else {}

_URL_OVERRIDE_PATH = os.path.join(HERE, "track_url_overrides.json")
TRACK_URL_OVERRIDES = {k: v for k, v in json.load(open(_URL_OVERRIDE_PATH, encoding="utf-8")).items()
                       if not k.startswith("_")} if os.path.exists(_URL_OVERRIDE_PATH) else {}

_SKIP_KW_PATH = os.path.join(HERE, "skip_artist_keywords.json")
_skip_kw_data = json.load(open(_SKIP_KW_PATH, encoding="utf-8")) if os.path.exists(_SKIP_KW_PATH) else {}
SKIP_ARTIST_KEYWORDS = [k.lower() for k in _skip_kw_data.get("keywords", [])]


def is_bgm_artist(artist: str) -> bool:
    a = artist.lower()
    return any(kw in a for kw in SKIP_ARTIST_KEYWORDS)

MAX_FILE_MB = 30           # 下載檔 >30MB（≈30分鐘，128k）判定 yt-dlp 配錯歌跳過。
                           # 錨點：情歌王(最長 KTV，7分46秒)≈7.5MB、長後搖≈15-20MB 都在門檻內；
                           # 真正炸彈是 30MB+ 的「完整專輯/數小時 mix」，解碼會吃 6-7GB RAM 觸發 kernel panic。
MAX_DURATION_SEC = 1800    # yt-dlp 源頭過濾：>30 分鐘的搜尋結果直接不下載

# 分析結果要寫進 features 表的欄位（feats dict 的 key 即欄名）
NEW_COLUMNS = {
    "af_source": "TEXT", "analyzed_at": "TEXT", "arousal": "REAL",
    "tempo_confidence": "REAL", "key_strength": "REAL",
    "mood_happy": "REAL", "mood_sad": "REAL", "mood_relaxed": "REAL",
    "mood_aggressive": "REAL", "mood_party": "REAL", "mood_electronic": "REAL",
    "genre_rosamerica": "TEXT", "genre_discogs": "TEXT",
}

_stop = False
_active_procs: set = set()
_procs_lock = threading.Lock()


def _handle_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[pipeline] 收到中斷訊號，終止所有下載子進程…", flush=True)
    with _procs_lock:
        for p in list(_active_procs):
            try:
                p.kill()
            except OSError:
                pass


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# DB 準備
# ----------------------------------------------------------------------------
def ensure_schema(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(features)")}
    added = []
    for name, typ in NEW_COLUMNS.items():
        if name not in cols:
            con.execute(f'ALTER TABLE features ADD COLUMN "{name}" {typ}')
            added.append(name)
    if added:
        con.commit()
        log(f"features 表新增欄位：{', '.join(added)}")


def ensure_progress_db():
    con = sqlite3.connect(PROGRESS_DB)
    con.execute("PRAGMA busy_timeout=30000")
    for attempt in range(10):
        try:
            con.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError:
            time.sleep(0.5)
    con.execute("""CREATE TABLE IF NOT EXISTS failed (
        track_id TEXT PRIMARY KEY, attempts INTEGER DEFAULT 0,
        last_error TEXT, updated_at TEXT)""")
    con.commit()
    return con


# ----------------------------------------------------------------------------
# 待辦清單
# ----------------------------------------------------------------------------
def build_worklist(feat_con, prog_con, limit=None, shard=None, force_essentia=False):
    """回傳 [(track_id, name, artist, plays)]，依播放次數由多到少。
    shard=(idx, count) 時，用交錯切片（todo[idx::count]）分給多進程平行跑，
    確保同一首歌只會被一個進程處理，不需要額外協調機制。
    force_essentia=True 時，只以 essentia_features.db 判斷「已分析」，
    讓 pipeline 補跑那些只有 Kaggle/HF 特徵的曲目。"""
    log("解析聆聽史 JSON…")
    plays = defaultdict(int)
    meta = {}
    for fp in glob.glob(os.path.join(HIST_DIR, "Streaming_History_Audio_*.json")):
        for r in json.load(open(fp, encoding="utf-8")):
            uri = r.get("spotify_track_uri")
            if not uri or (r.get("ms_played") or 0) < MIN_MS_PLAYED:
                continue
            tid = uri.split(":")[-1]
            plays[tid] += 1
            if tid not in meta:
                meta[tid] = (r.get("master_metadata_track_name"),
                             r.get("master_metadata_album_artist_name"))
    log(f"去重曲目（>=30s）：{len(plays)} 首")

    # force_essentia: 只跳過已在 essentia_features.db 的歌（補跑 Kaggle/HF 覆蓋的曲目）
    # 預設：三庫聯集，不論來源都跳過
    have = essentia_ids() if force_essentia else have_feature_ids()
    # 失敗達上限的永久跳過
    skip = {r[0] for r in prog_con.execute(
        "SELECT track_id FROM failed WHERE attempts >= ?", (MAX_ATTEMPTS,))}

    todo = [(tid, meta[tid][0], meta[tid][1], n)
            for tid, n in plays.items()
            if tid not in have and tid not in skip and meta[tid][0]
            and not is_bgm_artist(meta[tid][1] or "")]
    todo.sort(key=lambda x: x[3], reverse=True)

    if shard:
        idx, count = shard
        todo = todo[idx::count]
        log(f"分片 {idx}/{count}：本進程負責 {len(todo)} 首（交錯切片，跟其他分片無重疊）")

    log(f"已有特徵 {len(have)} 首；待分析 {len(todo)} 首"
        + (f"（本次只取最常聽前 {limit} 首）" if limit else ""))
    return todo[:limit] if limit else todo


# ----------------------------------------------------------------------------
# 下載
# ----------------------------------------------------------------------------
def _run(cmd, timeout):
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with _procs_lock:
            _active_procs.add(proc)
        try:
            proc.wait(timeout=timeout)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            proc.kill()
            return False
    except Exception:
        return False
    finally:
        with _procs_lock:
            _active_procs.discard(proc)


def download_spotdl(track_id, tmpdir):
    if not os.path.exists(SPOTDL_BIN):
        return None
    out_tpl = os.path.join(tmpdir, "{track-id}")
    ok = _run([SPOTDL_BIN, "download",
               f"https://open.spotify.com/track/{track_id}",
               "--output", out_tpl, "--format", "mp3", "--bitrate", "128k",
               "--threads", "1"], DOWNLOAD_TIMEOUT)
    if ok:
        for f in glob.glob(os.path.join(tmpdir, f"{track_id}.*")):
            return f
    return None


def _strip_live(name):
    """去掉常見 Live/特殊版本後綴，回傳原版曲名讓 yt-dlp 更好搜尋。"""
    import re
    return re.sub(
        r'\s*[-（(【\[]?\s*(Live|live|LIVE|現場版?|live version|live recording|concert version)\s*[-）)】\]]?\s*$',
        '', name
    ).strip()


def _strip_feat(name):
    """去掉 (feat. ...) 括號，讓 yt-dlp 用乾淨曲名搜尋。"""
    import re
    return re.sub(r'\s*[\(（][^)）]*feat[^)）]*[\)）]', '', name, flags=re.IGNORECASE).strip()


def download_ytdlp(track_id, artist, name, tmpdir):
    # 羅馬拼音藝名換成 YouTube 慣用的中文/日文名，提升搜尋命中率
    search_artist = ARTIST_ALIASES.get(artist, artist)
    clean_name = _strip_live(name)
    feat_name = _strip_feat(name)
    feat_clean_name = _strip_feat(clean_name)
    # 依序試四種 query，命中即停
    seen = set()
    queries = []
    for q in [name, clean_name, feat_name, feat_clean_name]:
        if q and q not in seen:
            queries.append(f"ytsearch1:{search_artist} {q}")
            seen.add(q)
    out_tpl = os.path.join(tmpdir, f"{track_id}.%(ext)s")
    for query in queries:
        ok = _run(["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio",
                   "--match-filter", f"duration < {MAX_DURATION_SEC}",
                   "-o", out_tpl, query], DOWNLOAD_TIMEOUT)
        if ok:
            hits = glob.glob(os.path.join(tmpdir, f"{track_id}.*"))
            if hits:
                return hits[0]
    return None


def download_url_override(track_id, tmpdir):
    """直接用指定 URL 下載，繞過搜尋（適用於上傳者是經紀公司等情況）。"""
    url = TRACK_URL_OVERRIDES.get(track_id)
    if not url:
        return None
    out_tpl = os.path.join(tmpdir, f"{track_id}.%(ext)s")
    ok = _run(["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio",
               "-o", out_tpl, url], DOWNLOAD_TIMEOUT)
    if ok:
        hits = glob.glob(os.path.join(tmpdir, f"{track_id}.*"))
        return hits[0] if hits else None
    return None


def download_track(track_id, artist, name, tmpdir, downloader):
    # URL override 最優先
    f = download_url_override(track_id, tmpdir)
    if f:
        return f, "url-override"
    if downloader in ("auto", "spotdl"):
        f = download_spotdl(track_id, tmpdir)
        if f:
            return f, "spotdl"
        if downloader == "spotdl":
            return None, None
    f = download_ytdlp(track_id, artist, name, tmpdir)
    return (f, "yt-dlp") if f else (None, None)


# ----------------------------------------------------------------------------
# 寫回 DB
# ----------------------------------------------------------------------------
def write_features(con, track_id, name, artist, feats):
    row = dict(feats)
    row["track_id"] = track_id
    row["track_name"] = name
    row["artist"] = artist
    row["af_source"] = "essentia"
    row["analyzed_at"] = datetime.now().isoformat(timespec="seconds")
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    col_sql = ",".join(f'"{c}"' for c in cols)
    updates = ",".join(f'"{c}"=excluded."{c}"' for c in cols if c != "track_id")
    con.execute(
        f'INSERT INTO features ({col_sql}) VALUES ({placeholders}) '
        f'ON CONFLICT(track_id) DO UPDATE SET {updates}',
        [row[c] for c in cols])
    con.commit()


def record_failure(prog_con, track_id, error):
    prog_con.execute(
        "INSERT INTO failed (track_id, attempts, last_error, updated_at) "
        "VALUES (?, 1, ?, ?) ON CONFLICT(track_id) DO UPDATE SET "
        "attempts = attempts + 1, last_error = excluded.last_error, "
        "updated_at = excluded.updated_at",
        (track_id, str(error)[:300], datetime.now().isoformat(timespec="seconds")))
    prog_con.commit()


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑最常聽的前 N 首（先小批驗證用）")
    ap.add_argument("--keep-audio", action="store_true",
                    help="分析後保留音檔（預設逐首刪除）")
    ap.add_argument("--downloader", choices=["auto", "spotdl", "ytdlp"],
                    default="auto", help="下載器：auto=spotdl優先yt-dlp備援")
    ap.add_argument("--workers", type=int, default=5,
                    help="並行下載執行緒數（預設 5）；TF 推論固定單執行緒，不佔額外記憶體")
    ap.add_argument("--force-essentia", action="store_true",
                    help="只跳過已在 essentia_features.db 的曲目，補跑 Kaggle/HF 覆蓋的歌")
    ap.add_argument("--shard", default=None,
                    help="平行多進程分流，格式 i/N（如 0/2、1/2）；交錯切片互不重疊。"
                         "實測 2 進程 CP 值最高、3 進程最快（GIL 限制，單進程只用 3.3/8 核）")
    args = ap.parse_args()

    shard = None
    if args.shard:
        i, n = args.shard.split("/")
        shard = (int(i), int(n))

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)
    os.makedirs(TMP_DIR, exist_ok=True)

    feat_con = sqlite3.connect(DB_PATH)
    feat_con.execute("PRAGMA busy_timeout=30000")
    feat_con.execute("PRAGMA journal_mode=WAL")
    prog_con = ensure_progress_db()
    ensure_schema(feat_con)

    todo = build_worklist(feat_con, prog_con, args.limit, shard, args.force_essentia)
    if not todo:
        log("沒有待分析的歌，結束。")
        return

    log(f"開始分析 {len(todo)} 首；下載並行={args.workers} 執行緒；TF 推論單執行緒（模型只載入一次）")
    az = Analyzer(verbose=False)

    def do_download(item):
        tid, name, artist, _ = item
        path, via = download_track(tid, artist, name, TMP_DIR, args.downloader)
        return item, path, via

    done = failed = 0
    t_start = time.time()
    total = len(todo)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(do_download, item) for item in todo]
        for i, future in enumerate(as_completed(futures), 1):
            if _stop:
                break
            try:
                (tid, name, artist, plays), path, via = future.result()
            except Exception as e:
                log(f"⚠️  下載異常（{i}/{total}）：{e}")
                failed += 1
                continue

            tag = f"({i}/{total}) {artist} - {name} [{plays}次]"
            if not path:
                log(f"❌ 找不到音源，跳過 {tag}")
                record_failure(prog_con, tid, "download_not_found")
                failed += 1
                continue

            # size guard：擋掉 yt-dlp 配錯的超長巨檔，避免 Essentia 解碼吃爆 RAM 觸發 kernel panic
            size_mb = os.path.getsize(path) / 1_000_000
            if size_mb > MAX_FILE_MB:
                log(f"🚫 巨檔 {size_mb:.0f}MB 跳過（疑似配錯歌）{tag}")
                record_failure(prog_con, tid, f"file_too_large_{size_mb:.0f}MB")
                failed += 1
                for f in glob.glob(os.path.join(TMP_DIR, f"{tid}.*")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                continue

            try:
                log(f"🎼 分析中（{via}）{tag}")
                feats = az.analyze(path)
                write_features(feat_con, tid, name, artist, feats)
                done += 1
                log(f"✅ 寫入 DB：tempo={feats['tempo']} valence={feats['valence']} "
                    f"dance={feats['danceability']} genre={feats['genre_discogs']}")
                if not args.keep_audio:
                    os.remove(path)
            except Exception as e:
                log(f"⚠️  分析失敗 {tag}：{e}")
                record_failure(prog_con, tid, e)
                failed += 1
                for f in glob.glob(os.path.join(TMP_DIR, f"{tid}.*")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

            if done and done % 10 == 0:
                rate = (time.time() - t_start) / (done + failed)
                remain = (total - i) * rate / 3600
                log(f"— 進度 {done} 成功 / {failed} 失敗；約 {rate:.0f}s/首；"
                    f"剩餘估計 {remain:.1f} 小時 —")

    log(f"本次結束：成功 {done}、失敗 {failed}、處理 {done + failed}/{total}")
    log("（重跑同指令會接續未完成的歌）")


if __name__ == "__main__":
    main()
