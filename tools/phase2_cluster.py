#!/usr/bin/env python3
"""
phase2_cluster.py — Engagement 加權 K-Means 聚類

輸入：essentia_features.db + Output/engagement.csv
輸出：Output/clusters.csv, Output/cluster_scatter.png, Output/elbow.png

使用 sample_weight=engagement_score 讓你真正喜歡的歌
對 cluster 中心影響更大。
"""

import csv
import sqlite3
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["PingFang TC", "DejaVu Sans"]
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
CLUSTER_COLORS = ["#E63946", "#457B9D", "#2EC4B6", "#F4A261", "#8338EC",
                  "#06D6A0", "#FF006E", "#3A86FF"]


def load_features() -> tuple[list[str], np.ndarray]:
    conn = sqlite3.connect(db_paths.ESSENTIA_DB)
    conn.row_factory = sqlite3.Row
    try:
        cols = ", ".join(FEATURES_14)
        rows = conn.execute(
            f"SELECT track_id, {cols} FROM features WHERE tempo IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    track_ids, X = [], []
    for r in rows:
        d = dict(r)
        vals = [d.get(f) for f in FEATURES_14]
        if any(v is None for v in vals):
            continue
        track_ids.append(d["track_id"])
        X.append(vals)

    print(f"[INFO] Essentia 特徵：{len(track_ids):,} 首，{len(FEATURES_14)} 維")
    return track_ids, np.array(X, dtype=float)


def load_engagement(track_ids: list[str]) -> np.ndarray:
    eng_path = OUTPUT_DIR / "engagement.csv"
    if not eng_path.exists():
        print("[WARN] engagement.csv 不存在，所有歌 weight=1")
        return np.ones(len(track_ids))

    eng_map = {}
    with open(eng_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eng_map[row["track_id"]] = float(row["engagement_score"])

    weights = np.array([max(eng_map.get(tid, 0.01), 0.01) for tid in track_ids])
    print(f"[INFO] 有 engagement score：{sum(1 for t in track_ids if t in eng_map):,} 首")
    return weights


def elbow(X_scaled: np.ndarray, weights: np.ndarray) -> int:
    """Elbow method，回傳建議 K。"""
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled, sample_weight=weights)
        inertias.append(km.inertia_)

    # 找最大曲率點（差分二次）
    d1 = np.diff(inertias)
    d2 = np.diff(d1)
    best_k = list(K_range)[np.argmin(d2) + 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_range), inertias, "o-", color="#1DB954", linewidth=2)
    ax.axvline(best_k, color="#E63946", linestyle="--", label=f"Best K={best_k}")
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title("Elbow Curve", fontsize=14)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    out = OUTPUT_DIR / "elbow.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OUTPUT] {out}（建議 K={best_k}）")
    return best_k


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    track_ids, X = load_features()
    weights = load_engagement(track_ids)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = elbow(X_scaled, weights)

    print(f"[INFO] 跑 KMeans K={best_k}...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled, sample_weight=weights)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X_scaled)
    print(f"[INFO] PCA 解釋變異量：{pca.explained_variance_ratio_.sum():.1%}")

    # 存 clusters.csv
    out_csv = OUTPUT_DIR / "clusters.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "cluster_id", "pca_x", "pca_y"])
        for tid, cid, (x, y) in zip(track_ids, labels, X2d):
            writer.writerow([tid, int(cid), round(float(x), 4), round(float(y), 4)])
    print(f"[OUTPUT] {out_csv}")

    # Scatter plot：點大小 = engagement
    max_w = np.percentile(weights, 99)
    sizes = np.clip(weights / max_w * 80, 2, 80)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")
    for cid in range(best_k):
        mask = labels == cid
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   s=sizes[mask], c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                   alpha=0.5, label=f"Cluster {cid}", linewidths=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
    ax.set_title("音樂聚類散點圖（點越大 = engagement 越高）",
                 fontfamily="PingFang TC", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", markerscale=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    out_png = OUTPUT_DIR / "cluster_scatter.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"[OUTPUT] {out_png}")

    # 打印各 cluster 代表特徵
    print("\n【各 Cluster 特徵均值 Top-3】")
    label_arr = np.array(labels)
    for cid in range(best_k):
        mask = label_arr == cid
        means = X[mask].mean(axis=0)
        top3_idx = means.argsort()[::-1][:3]
        top3 = ", ".join(f"{FEATURES_14[i]}={means[i]:.2f}" for i in top3_idx)
        high_eng = [(track_ids[i], weights[i]) for i in np.where(mask)[0]]
        high_eng.sort(key=lambda x: x[1], reverse=True)
        print(f"  Cluster {cid}（{mask.sum()} 首）：{top3}")
        print(f"    代表曲 track_id Top-3：{', '.join(t for t, _ in high_eng[:3])}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
