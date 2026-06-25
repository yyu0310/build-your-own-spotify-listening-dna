#!/usr/bin/env python3
"""
phase2_av_map.py — Arousal-Valence 情緒地圖（engagement 加權）

輸入：essentia_features.db + Output/clusters.csv + Output/engagement.csv
輸出：Output/av_map.png

點大小 = engagement_score，顏色 = cluster，直觀看出你「真正喜歡」的歌落在哪個情緒象限。
"""

import csv
import sqlite3
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["PingFang TC", "DejaVu Sans"]
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import db_paths

OUTPUT_DIR = ROOT / "Output"
CLUSTER_COLORS = ["#E63946", "#457B9D", "#2EC4B6", "#F4A261", "#8338EC",
                  "#06D6A0", "#FF006E", "#3A86FF"]


def load_csv(path: Path) -> dict:
    d = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d[row["track_id"]] = row
    return d


def load_av(track_ids: set) -> dict:
    ph = ",".join("?" * len(track_ids))
    ids = list(track_ids)
    conn = sqlite3.connect(db_paths.ESSENTIA_DB)
    try:
        rows = conn.execute(
            f"SELECT track_id, arousal, valence FROM features "
            f"WHERE track_id IN ({ph}) AND arousal IS NOT NULL AND valence IS NOT NULL",
            ids
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: (r[1], r[2]) for r in rows}


def main():
    cluster_path = OUTPUT_DIR / "clusters.csv"
    eng_path     = OUTPUT_DIR / "engagement.csv"
    if not cluster_path.exists():
        print("[ERR] clusters.csv 不存在，請先跑 phase2_cluster.py")
        return

    clusters = load_csv(cluster_path)    # {track_id: {cluster_id, pca_x, pca_y}}
    eng_map  = load_csv(eng_path) if eng_path.exists() else {}
    all_ids  = set(clusters.keys())

    av_map = load_av(all_ids)
    print(f"[INFO] 有 AV 特徵：{len(av_map):,} 首")

    valences, arousals, sizes, colors, cluster_ids = [], [], [], [], []
    for tid, (arous, val) in av_map.items():
        cid = int(clusters.get(tid, {}).get("cluster_id", 0))
        eng = float(eng_map.get(tid, {}).get("engagement_score", 0.1))
        valences.append(val)
        arousals.append(arous)
        sizes.append(min(eng * 25, 150))
        colors.append(CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
        cluster_ids.append(cid)

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    ax.scatter(valences, arousals, s=sizes, c=colors, alpha=0.4, linewidths=0)

    # 四象限虛線 + 標籤
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    quad_kw = dict(fontfamily="PingFang TC", fontsize=11, alpha=0.35, fontweight="bold")
    ax.text(0.82, 0.85, "Happy\nExcited",   **quad_kw)
    ax.text(0.05, 0.85, "Angry\nTense",    **quad_kw)
    ax.text(0.78, 0.08, "Relaxed\nPeaceful", **quad_kw)
    ax.text(0.05, 0.08, "Sad\nDepressed",  **quad_kw)

    # Legend（cluster）
    n_clusters = max(cluster_ids) + 1
    patches = [mpatches.Patch(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                               label=f"Cluster {i}") for i in range(n_clusters)]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Valence（低=憂鬱 → 高=正向）", fontfamily="PingFang TC", fontsize=12)
    ax.set_ylabel("Arousal（低=平靜 → 高=激動）", fontfamily="PingFang TC", fontsize=12)
    ax.set_title("Arousal-Valence 情緒地圖（點越大 = 你越喜歡）",
                 fontfamily="PingFang TC", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    out = OUTPUT_DIR / "av_map.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"[OUTPUT] {out}")
    print("[DONE]")


if __name__ == "__main__":
    main()
