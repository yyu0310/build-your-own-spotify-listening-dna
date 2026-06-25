#!/usr/bin/env python3
"""
youtube_url_lookup.py — 用 YouTube Data API v3 補全 download_not_found 的 URL

用法：
  python3 tools/youtube_url_lookup.py [--dry-run] [--limit N]

  --dry-run  只印結果，不寫入 overrides.json / DB
  --limit N  只處理前 N 首（預設全部，100/天 quota 上限）

API Key setup:
  Google Cloud Console → Enable YouTube Data API v3 → Create API key
  Add YOUTUBE_API_KEY=your_key to ~/.config/spotify-dna/.env (or SECRETS_DIR)
"""

import argparse
import difflib
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENV_FILE       = Path(os.getenv("SECRETS_DIR", str(Path.home() / ".config" / "spotify-dna"))) / ".env"
PROGRESS_DB    = ROOT / "Database" / "essentia_progress.db"
OVERRIDES_FILE = ROOT / "track_url_overrides.json"
HISTORY_DIR    = ROOT / "Database" / "Spotify Extended Streaming History"
ALIASES_FILE   = ROOT / "artist_aliases.json"

YT_SEARCH_URL  = "https://www.googleapis.com/youtube/v3/search"
MIN_SIMILARITY  = 0.35  # 低於此分數視為錯誤結果，跳過


def load_api_key() -> str:
    if not ENV_FILE.exists():
        sys.exit(f"[ERROR] 找不到 {ENV_FILE}\n請先建立並加入 YOUTUBE_API_KEY=你的key")
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("YOUTUBE_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit(f"[ERROR] {ENV_FILE} 裡找不到 YOUTUBE_API_KEY=... 這行")


def get_failed_track_ids() -> list[str]:
    conn = sqlite3.connect(PROGRESS_DB)
    rows = conn.execute(
        "SELECT track_id FROM failed WHERE last_error = 'download_not_found' AND attempts >= 3"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def build_track_map(track_ids: set[str]) -> dict[str, dict]:
    result = {}
    for jf in sorted(HISTORY_DIR.glob("Streaming_History_Audio_*.json")):
        for item in json.loads(jf.read_text()):
            uri = item.get("spotify_track_uri", "")
            if not uri:
                continue
            tid = uri.split(":")[-1]
            if tid not in track_ids or tid in result:
                continue
            name   = item.get("master_metadata_track_name", "")
            artist = item.get("master_metadata_album_artist_name", "")
            if name and artist:
                result[tid] = {"artist": artist, "track": name}
    return result


def load_aliases() -> dict:
    if ALIASES_FILE.exists():
        d = json.loads(ALIASES_FILE.read_text())
        return {k: v for k, v in d.items() if not k.startswith("_")}
    return {}


def _strip_feat(name: str) -> str:
    import re
    return re.sub(r'\s*[\(（][^)）]*feat[^)）]*[\)）]', '', name, flags=re.IGNORECASE).strip()


def title_similarity(track: str, yt_title: str) -> float:
    """歌曲名稱和 YouTube 標題的相似度（0-1）。"""
    t = _strip_feat(track).lower()
    y = yt_title.lower()
    if t and t in y:  # 歌名是 YouTube 標題的 substring（含藝人名前綴的常見格式）
        return 0.9
    return difflib.SequenceMatcher(None, t, y).ratio()


def search_youtube(api_key: str, artist: str, track: str) -> list[dict]:
    """回傳最多 3 筆結果，每筆含 video_id, title, channel。"""
    query = f"{artist} {_strip_feat(track)}"
    params = urllib.parse.urlencode({
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoCategoryId": "10",  # Music
        "maxResults": "3",
        "key": api_key,
    })
    url = f"{YT_SEARCH_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:200]
        print(f"  [API ERROR] HTTP {e.code}: {err}")
        return []
    except Exception as e:
        print(f"  [API ERROR] {e}")
        return []

    results = []
    for item in body.get("items", []):
        vid = item.get("id", {}).get("videoId", "")
        snippet = item.get("snippet", {})
        if vid:
            results.append({
                "video_id": vid,
                "title":    snippet.get("title", ""),
                "channel":  snippet.get("channelTitle", ""),
                "url":      f"https://www.youtube.com/watch?v={vid}",
            })
    return results


def load_existing_overrides() -> dict:
    if OVERRIDES_FILE.exists():
        return json.loads(OVERRIDES_FILE.read_text())
    return {}


def write_overrides(overrides: dict):
    OVERRIDES_FILE.write_text(json.dumps(overrides, ensure_ascii=False, indent=2))


def reset_in_db(track_ids: list[str]):
    import datetime
    conn = sqlite3.connect(PROGRESS_DB)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for tid in track_ids:
        conn.execute(
            "UPDATE failed SET attempts=0, last_error='retry_after_url_override', updated_at=? WHERE track_id=?",
            (now, tid),
        )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=MIN_SIMILARITY)
    args = parser.parse_args()

    api_key = load_api_key()
    aliases = load_aliases()
    print("[INFO] API key 載入成功")

    failed_ids = get_failed_track_ids()
    print(f"[INFO] 待搜尋：{len(failed_ids)} 首（download_not_found / retry_dnf_2026）")

    track_map = build_track_map(set(failed_ids))
    existing_overrides = set(load_existing_overrides().keys())
    candidates = [
        (tid, track_map[tid])
        for tid in failed_ids
        if tid in track_map and tid not in existing_overrides
    ]
    if args.limit:
        candidates = candidates[:args.limit]

    print(f"[INFO] 實際查詢（扣除無歌名 / 已有 override）：{len(candidates)} 首\n")

    to_write = []  # (tid, artist, track, url)

    for i, (tid, info) in enumerate(candidates, 1):
        artist = info["artist"]
        track  = info["track"]
        search_artist = aliases.get(artist, artist)
        print(f"[{i:3}/{len(candidates)}] {artist} - {track}")
        if search_artist != artist:
            print(f"         alias → {search_artist}")
        results = search_youtube(api_key, search_artist, track)
        time.sleep(0.3)

        if not results:
            print("         ❌ 無結果")
            continue

        # 選相似度最高的結果，低於門檻則跳過
        scored = [(title_similarity(track, r["title"]), r) for r in results]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]

        for alt_score, alt in scored[1:]:
            print(f"            alt({alt_score:.2f}): {alt['url']} — {alt['title']}")

        if best_score < args.min_score:
            print(f"         ⚠️  相似度太低（{best_score:.2f}），跳過")
            print(f"            最佳候選：{best['url']} — {best['title']}")
            continue

        print(f"         ✅ ({best_score:.2f}) {best['url']}")
        print(f"            {best['title']}（{best['channel']}）")
        to_write.append((tid, artist, track, best["url"]))

    print("\n" + "=" * 60)
    print(f"  找到 {len(to_write)} / {len(candidates)} 首")
    print("=" * 60)

    if not to_write:
        print("[INFO] 沒有找到新 URL，結束。")
        return

    if args.dry_run:
        print("\n[DRY-RUN] 沒有寫入任何檔案。")
        return

    print(f"\n以上 {len(to_write)} 首將寫入 track_url_overrides.json 並 reset DB。")
    ans = input("輸入 'yes' 確認，其他鍵取消：").strip().lower()
    if ans != "yes":
        print("[取消]")
        return

    overrides = load_existing_overrides()
    overrides.setdefault("_comment_youtube_api", "YouTube Data API v3 自動查詢補全的 URL")
    for tid, _, _, url in to_write:
        overrides[tid] = url
    write_overrides(overrides)
    print(f"[DONE] track_url_overrides.json +{len(to_write)} 條")

    reset_in_db([tid for tid, *_ in to_write])
    print(f"[DONE] essentia_progress.db reset {len(to_write)} 首 → retry_after_url_override")
    print("\n接下來跑 pipeline：caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 5")


if __name__ == "__main__":
    main()
