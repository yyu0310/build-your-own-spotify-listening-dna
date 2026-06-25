#!/usr/bin/env python3
"""
phase2_engagement.py — 個人 Engagement Score 計算（地基腳本）

輸入：15 年串流歷史 JSON + kaggle_features.db（取 duration_ms）
輸出：Output/engagement.csv

欄位：track_id, artist, track_name, play_count, weighted_play_count,
       avg_ms_played, completion_rate, skip_count, skip_rate, engagement_score
"""

import json
import sqlite3
import sys
import math
import csv
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import db_paths

HISTORY_DIR = ROOT / "Database" / "Spotify Extended Streaming History"
OUTPUT_DIR  = ROOT / "Output"
MIN_MS      = 30_000   # 有效播放門檻：30 秒
CURRENT_YEAR = 2026


def recency_weight(year: int) -> float:
    return max(0.1, 1.0 - (CURRENT_YEAR - year) * 0.12)


def load_all_history() -> list[dict]:
    """讀取所有 Audio Streaming History JSON，回傳有效播放記錄。"""
    records = []
    for path in sorted(HISTORY_DIR.glob("Streaming_History_Audio_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            if not d.get("spotify_track_uri") or d.get("ms_played", 0) < MIN_MS:
                continue
            year = int(d["ts"][:4])
            records.append({
                "track_id":   d["spotify_track_uri"].split(":")[-1],
                "ms_played":  d["ms_played"],
                "year":       year,
                "rw":         recency_weight(year),
                "reason_end": d.get("reason_end", ""),
                "skipped":    bool(d.get("skipped")),
            })
    print(f"[INFO] 有效播放紀錄：{len(records):,} 筆（來自 {HISTORY_DIR}）")
    return records


def load_duration_map(track_ids: set) -> dict:
    """從 kaggle_features.db 取 duration_ms，回傳 {track_id: ms}。"""
    if not Path(db_paths.KAGGLE_DB).exists():
        print("[WARN] kaggle_features.db 不存在，無法計算完聽率")
        return {}
    ph = ",".join("?" * len(track_ids))
    ids = list(track_ids)
    conn = sqlite3.connect(db_paths.KAGGLE_DB)
    try:
        rows = conn.execute(
            f"SELECT track_id, duration_ms FROM features WHERE track_id IN ({ph})", ids
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: r[1] for r in rows if r[1]}


def load_name_map(track_ids: set) -> dict:
    """從 essentia > kaggle 取 artist + track_name。"""
    ph = ",".join("?" * len(track_ids))
    ids = list(track_ids)
    name_map = {}
    for db_path, artist_col in [
        (db_paths.KAGGLE_DB,   "artist"),
        (db_paths.ESSENTIA_DB, "artist"),
    ]:
        if not Path(db_path).exists():
            continue
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                f"SELECT track_id, {artist_col} AS artist, track_name "
                f"FROM features WHERE track_id IN ({ph})", ids
            ).fetchall()
        except Exception:
            conn.close()
            continue
        conn.close()
        for tid, artist, tname in rows:
            if tid not in name_map or (artist and tname):
                name_map[tid] = (artist or "", tname or "")
    return name_map


def compute_engagement(records: list[dict], duration_map: dict) -> list[dict]:
    """彙整每首歌的行為指標，回傳 engagement 清單（依 engagement_score 排序）。"""
    agg: dict[str, dict] = {}

    for r in records:
        tid = r["track_id"]
        if tid not in agg:
            agg[tid] = {
                "play_count": 0,
                "weighted_play_count": 0.0,
                "ms_played_list": [],
                "skip_count": 0,
                "completion_count": 0,
            }
        a = agg[tid]
        a["play_count"] += 1
        a["weighted_play_count"] += r["rw"]
        a["ms_played_list"].append(r["ms_played"])
        if r["skipped"] or r["reason_end"] in ("fwdbtn",):
            a["skip_count"] += 1
        if r["reason_end"] == "trackdone":
            a["completion_count"] += 1

    rows = []
    for tid, a in agg.items():
        avg_ms = sum(a["ms_played_list"]) / len(a["ms_played_list"])
        dur_ms = duration_map.get(tid)
        completion_rate = min(1.0, avg_ms / dur_ms) if dur_ms else 0.5  # 無時長則假設 50%
        skip_rate = a["skip_count"] / a["play_count"]
        eng = math.log1p(a["weighted_play_count"]) * completion_rate * (1 - skip_rate)
        rows.append({
            "track_id":             tid,
            "play_count":           a["play_count"],
            "weighted_play_count":  round(a["weighted_play_count"], 3),
            "avg_ms_played":        round(avg_ms),
            "completion_rate":      round(completion_rate, 4),
            "skip_count":           a["skip_count"],
            "skip_rate":            round(skip_rate, 4),
            "engagement_score":     round(eng, 6),
        })

    rows.sort(key=lambda x: x["engagement_score"], reverse=True)
    return rows


def print_top(rows: list[dict], name_map: dict):
    def name(tid):
        a, t = name_map.get(tid, ("?", "?"))
        return f"{a} — {t}"

    print("\n【加權播放 Top-10（近年偏好）】")
    sorted_w = sorted(rows, key=lambda x: x["weighted_play_count"], reverse=True)
    for i, r in enumerate(sorted_w[:10], 1):
        print(f"  {i:>2}. {name(r['track_id'])}  (加權={r['weighted_play_count']:.1f}, 共{r['play_count']}次)")

    print("\n【完聽率 Top-10（最常聽完的歌）】")
    sorted_c = sorted(
        [r for r in rows if r["play_count"] >= 3],
        key=lambda x: x["completion_rate"], reverse=True
    )
    for i, r in enumerate(sorted_c[:10], 1):
        print(f"  {i:>2}. {name(r['track_id'])}  (完聽率={r['completion_rate']:.0%}, {r['play_count']}次)")

    print("\n【跳過率 Top-10（最常跳過的歌）】")
    sorted_s = sorted(
        [r for r in rows if r["play_count"] >= 3],
        key=lambda x: x["skip_rate"], reverse=True
    )
    for i, r in enumerate(sorted_s[:10], 1):
        print(f"  {i:>2}. {name(r['track_id'])}  (跳過率={r['skip_rate']:.0%}, {r['play_count']}次)")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    records = load_all_history()
    all_ids = {r["track_id"] for r in records}
    print(f"[INFO] 唯一曲目：{len(all_ids):,} 首")

    duration_map = load_duration_map(all_ids)
    print(f"[INFO] 取得 duration_ms：{len(duration_map):,} 首")

    name_map = load_name_map(all_ids)
    rows = compute_engagement(records, duration_map)
    print(f"[INFO] 計算完成：{len(rows):,} 首")

    out_path = OUTPUT_DIR / "engagement.csv"
    fieldnames = ["track_id", "play_count", "weighted_play_count", "avg_ms_played",
                  "completion_rate", "skip_count", "skip_rate", "engagement_score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OUTPUT] {out_path}")

    print_top(rows, name_map)
    print("\n[DONE] engagement.csv 建立完成")


if __name__ == "__main__":
    main()
