#!/usr/bin/env python3
"""
phase2_evolution.py — 音樂口味年份進化圖

輸入：15 年串流歷史 + essentia_features.db
輸出：Output/taste_evolution.png

以 ms_played 加權計算每年 energy/valence/danceability/acousticness/arousal 均值，
畫折線圖 + ±1σ 陰影帶，直觀看出口味漂移。
"""

import json
import sqlite3
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["PingFang TC", "DejaVu Sans"]
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import db_paths

HISTORY_DIR = ROOT / "Database" / "Spotify Extended Streaming History"
OUTPUT_DIR  = ROOT / "Output"
MIN_MS      = 30_000
MIN_YEAR    = 2021

FEATURES = ["energy", "valence", "danceability", "acousticness", "arousal"]
COLORS   = ["#E63946", "#457B9D", "#2EC4B6", "#F4A261", "#8338EC"]
LABELS   = ["Energy", "Valence", "Danceability", "Acousticness", "Arousal"]


def load_history() -> list[dict]:
    records = []
    for path in sorted(HISTORY_DIR.glob("Streaming_History_Audio_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            year = int(d["ts"][:4])
            if not d.get("spotify_track_uri") or d.get("ms_played", 0) < MIN_MS or year < MIN_YEAR:
                continue
            records.append({
                "track_id": d["spotify_track_uri"].split(":")[-1],
                "ms_played": d["ms_played"],
                "year": year,
            })
    return records


def load_features(track_ids: set) -> dict:
    ph = ",".join("?" * len(track_ids))
    ids = list(track_ids)
    feat_map = {}
    conn = sqlite3.connect(db_paths.ESSENTIA_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT track_id, energy, valence, danceability, acousticness, arousal "
            f"FROM features WHERE track_id IN ({ph})", ids
        ).fetchall()
    finally:
        conn.close()
    for r in rows:
        d = dict(r)
        if all(d.get(f) is not None for f in FEATURES):
            feat_map[d["track_id"]] = d
    return feat_map


def compute_yearly(records, feat_map) -> dict:
    """回傳 {year: {feature: [weighted values, weights]}}"""
    yearly: dict[int, dict] = defaultdict(lambda: {f: ([], []) for f in FEATURES})
    for r in records:
        feat = feat_map.get(r["track_id"])
        if not feat:
            continue
        y = r["year"]
        w = r["ms_played"]
        for f in FEATURES:
            v = feat.get(f)
            if v is not None:
                yearly[y][f][0].append(v * w)
                yearly[y][f][1].append(w)
    return yearly


def weighted_mean_std(vals, weights):
    total_w = sum(weights)
    if total_w == 0:
        return 0, 0
    mean = sum(vals) / total_w
    # weighted sample std
    raw_vals = [v / w for v, w in zip(vals, weights)]
    if len(raw_vals) < 2:
        return mean, 0
    std = np.std(raw_vals)
    return mean, std


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("[INFO] 讀取串流歷史...")
    records = load_history()
    print(f"[INFO] 有效播放：{len(records):,} 筆")

    feat_map = load_features({r["track_id"] for r in records})
    print(f"[INFO] 有 Essentia 特徵：{len(feat_map):,} 首")

    yearly = compute_yearly(records, feat_map)
    years = sorted(yearly.keys())
    print(f"[INFO] 涵蓋年份：{years[0]}–{years[-1]}")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    for feat, color, label in zip(FEATURES, COLORS, LABELS):
        means, stds = [], []
        valid_years = []
        for y in years:
            vals, weights = yearly[y][feat]
            if not vals:
                continue
            m, s = weighted_mean_std(vals, weights)
            means.append(m)
            stds.append(s)
            valid_years.append(y)

        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(valid_years, means, "o-", color=color, linewidth=2,
                label=label, markersize=5)
        ax.fill_between(valid_years, means - stds, means + stds,
                        color=color, alpha=0.1)

    ax.set_xlabel("年份", fontfamily="PingFang TC", fontsize=12)
    ax.set_ylabel("特徵均值（0–1）", fontfamily="PingFang TC", fontsize=12)
    ax.set_title(f"音樂口味年份進化圖（{MIN_YEAR}–2026）", fontfamily="PingFang TC",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(years)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    out = OUTPUT_DIR / "taste_evolution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"[OUTPUT] {out}")
    print("[DONE]")


if __name__ == "__main__":
    main()
