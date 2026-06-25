#!/usr/bin/env python3
"""
phase2_recommend.py — 個人化混合推薦引擎

用法：
    .venv/bin/python3 tools/phase2_recommend.py "百合花 重新出發"
    .venv/bin/python3 tools/phase2_recommend.py "Shawn Mendes"    # 模糊匹配

輸入：essentia_features.db + Output/engagement.csv
輸出：終端打印 Top-10 推薦

分數 = cosine_similarity（音頻相似）× log1p（候選歌曲的 engagement_score）
→ 音頻相似 + 你以前真正喜歡過的歌，優先排前面
"""

import csv
import difflib
import math
import sqlite3
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import db_paths

OUTPUT_DIR = ROOT / "Output"
FEATURES_14 = [
    "energy", "valence", "danceability", "acousticness", "instrumentalness",
    "mood_happy", "mood_sad", "mood_relaxed", "mood_aggressive",
    "mood_party", "mood_electronic", "arousal", "tempo", "loudness",
]


def load_library() -> tuple[list, np.ndarray]:
    conn = sqlite3.connect(db_paths.ESSENTIA_DB)
    conn.row_factory = sqlite3.Row
    try:
        cols = ", ".join(FEATURES_14)
        rows = conn.execute(
            f"SELECT track_id, artist, track_name, {cols} FROM features WHERE tempo IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    meta, X = [], []
    for r in rows:
        d = dict(r)
        vals = [d.get(f) for f in FEATURES_14]
        if any(v is None for v in vals):
            continue
        meta.append({
            "track_id":   d["track_id"],
            "artist":     str(d.get("artist") or ""),
            "track_name": str(d.get("track_name") or ""),
        })
        X.append([float(v) for v in vals])

    return meta, np.array(X, dtype=float)


def load_engagement() -> dict:
    path = OUTPUT_DIR / "engagement.csv"
    if not path.exists():
        return {}
    eng = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eng[row["track_id"]] = float(row["engagement_score"])
    return eng


def fuzzy_match(query: str, meta: list) -> int | None:
    """回傳最匹配的 index，或 None。"""
    # 先嘗試 artist + track_name 拼接搜
    candidates = [f"{m['artist']} {m['track_name']}".lower() for m in meta]
    matches = difflib.get_close_matches(query.lower(), candidates, n=1, cutoff=0.4)
    if matches:
        return candidates.index(matches[0])

    # 退而求其次：track_name 單獨匹配
    names = [m["track_name"].lower() for m in meta]
    matches = difflib.get_close_matches(query.lower(), names, n=1, cutoff=0.4)
    if matches:
        return names.index(matches[0])

    return None


def main():
    if len(sys.argv) < 2:
        print("用法：python3 tools/phase2_recommend.py \"歌名\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"[INFO] 查詢：{query}")

    meta, X = load_library()
    print(f"[INFO] 曲庫：{len(meta):,} 首")
    eng = load_engagement()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    idx = fuzzy_match(query, meta)
    if idx is None:
        print(f"[ERR] 找不到相符的歌曲：{query}")
        sys.exit(1)

    seed = meta[idx]
    print(f"[INFO] 匹配到：{seed['artist']} — {seed['track_name']}\n")

    # cosine similarity
    vec = X_scaled[idx].reshape(1, -1)
    sims = cosine_similarity(vec, X_scaled)[0]

    seed_track_id = seed["track_id"]

    # final_score = cosine_sim × log1p(candidate_engagement)
    final_scores = []
    for i, (m, sim) in enumerate(zip(meta, sims)):
        if m["track_id"] == seed_track_id or sim > 0.9999:  # ponytail: 同曲異版本過濾
            continue
        eng_score = eng.get(m["track_id"], 0.0)
        final = sim * math.log1p(eng_score)
        final_scores.append((i, sim, eng_score, final))

    final_scores.sort(key=lambda x: x[3], reverse=True)

    print(f"{'排名':<4} {'藝人':<25} {'歌名':<35} {'音頻相似':<10} {'Engagement':<12} {'綜合分'}")
    print("-" * 100)
    for rank, (i, sim, eng_s, final) in enumerate(final_scores[:10], 1):
        m = meta[i]
        artist = m["artist"][:23] + ".." if len(m["artist"]) > 25 else m["artist"]
        name   = m["track_name"][:33] + ".." if len(m["track_name"]) > 35 else m["track_name"]
        print(f"{rank:<4} {artist:<25} {name:<35} {sim:.4f}    {eng_s:.4f}       {final:.4f}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
