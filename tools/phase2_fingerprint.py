#!/usr/bin/env python3
"""
phase2_fingerprint.py — 個人音樂指紋分析
輸入：2026 Extended Streaming History + spotify_features.db
輸出：雷達圖 PNG + 文字摘要
"""

import json
import sqlite3
import statistics
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import db_paths

HISTORY_DIR = ROOT / "Database" / "Spotify Extended Streaming History"
OUTPUT_DIR  = ROOT / "Output"
TW_TZ       = timezone(timedelta(hours=8))

FEATURES = ["energy", "valence", "danceability", "acousticness",
            "instrumentalness", "speechiness", "liveness"]
LABELS_ZH = ["能量感", "正向情緒", "舞動感", "原聲比例",
              "器樂比例", "語音比例", "現場感"]


def load_data():
    with open(HISTORY_DIR / "Streaming_History_Audio_2026.json") as f:
        raw = json.load(f)
    tracks = [d for d in raw if d.get("spotify_track_uri") and d.get("ms_played", 0) >= 30000]
    unique_ids = list({d["spotify_track_uri"].split(":")[-1] for d in tracks})
    ph = ",".join("?" * len(unique_ids))

    feat_map = {}
    # essentia 優先（最後 override）；kaggle/hf 提供 speechiness/liveness（essentia 無）
    db_configs = [
        (db_paths.KAGGLE_DB,   "artist",  "speechiness, liveness"),
        (db_paths.HF_DB,       "artists", "speechiness, liveness"),
        (db_paths.ESSENTIA_DB, "artist",  "NULL AS speechiness, NULL AS liveness"),
    ]
    for db_path, artist_col, extra_cols in db_configs:
        if not Path(db_path).exists():
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"SELECT track_id, track_name, {artist_col} AS artist, "
                f"energy, valence, danceability, acousticness, instrumentalness, "
                f"{extra_cols}, tempo, loudness, mode "
                f"FROM features WHERE track_id IN ({ph})", unique_ids
            ).fetchall()
            for r in rows:
                d = dict(r)
                tid = d["track_id"]
                if tid not in feat_map:
                    feat_map[tid] = d
                else:
                    for k, v in d.items():
                        if v is not None and k != "track_id":
                            feat_map[tid][k] = v
        except Exception as e:
            print(f"[WARN] {Path(db_path).name}: {e}")
        finally:
            conn.close()

    enriched = []
    for d in tracks:
        tid = d["spotify_track_uri"].split(":")[-1]
        if tid in feat_map:
            f = feat_map[tid].copy()
            f["ts"] = d["ts"]
            f["ms_played"] = d["ms_played"]
            enriched.append(f)

    return enriched


def radar_chart(means, output_path):
    N = len(FEATURES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values = [means[f] for f in FEATURES] + [means[FEATURES[0]]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, values, "o-", linewidth=2, color="#1DB954")
    ax.fill(angles, values, alpha=0.25, color="#1DB954")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS_ZH, fontsize=13, fontfamily="PingFang TC")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_title("2026 音樂指紋", fontsize=16, fontfamily="PingFang TC",
                 fontweight="bold", pad=20, color="#191414")

    # 各項數值標註
    for angle, val, label in zip(angles[:-1], values[:-1], LABELS_ZH):
        ax.annotate(f"{val:.2f}", xy=(angle, val), xytext=(angle, val + 0.08),
                    ha="center", va="center", fontsize=9, color="#1DB954", fontweight="bold")

    fig.patch.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"[OUTPUT] 雷達圖：{output_path}")


def print_summary(data, means):
    print("\n" + "=" * 50)
    print("  2026 個人音樂指紋報告")
    print("=" * 50)
    print(f"  分析筆數：{len(data)} 筆（有 audio features 的播放紀錄）")
    print()

    print("【核心特徵】")
    desc = {
        "energy":          ("能量感",    "高=激烈/搖滾，低=舒緩"),
        "valence":         ("正向情緒",  "高=開心，低=憂鬱/中性"),
        "danceability":    ("舞動感",    "高=節拍規律適合跳舞"),
        "acousticness":    ("原聲比例",  "高=吉他/鋼琴，低=電子合成"),
        "instrumentalness":("器樂比例",  "高=純器樂，低=有人聲"),
        "speechiness":     ("語音比例",  "高=rap/口白，低=純演唱"),
        "liveness":        ("現場感",    "高=現場錄音"),
    }
    for f, (zh, hint) in desc.items():
        bar = "█" * int(means[f] * 20)
        print(f"  {zh:<6} {bar:<20} {means[f]:.3f}  ({hint})")

    print()
    print(f"  平均 tempo：{means['tempo']:.1f} BPM")
    print(f"  平均 loudness：{means['loudness']:.1f} dB")
    major = sum(1 for d in data if d.get("mode") == 1)
    print(f"  大調比例：{major/len(data)*100:.0f}%（{major}/{len(data)}）")

    print()
    print("【個性詮釋】")
    e, v, d = means["energy"], means["valence"], means["danceability"]
    a, ins = means["acousticness"], means["instrumentalness"]
    traits = []
    if e > 0.6:   traits.append("偏向高能量")
    elif e < 0.4: traits.append("偏向舒緩柔和")
    else:         traits.append("能量中性")
    if v > 0.6:   traits.append("情緒偏正向")
    elif v < 0.4: traits.append("情緒偏中性或憂鬱")
    else:         traits.append("情緒平衡")
    if d > 0.6:   traits.append("喜歡節奏感強的曲子")
    if a > 0.4:   traits.append("偏好原聲/木吉他風格")
    if ins > 0.2: traits.append("常聽器樂或輕人聲曲目")
    for t in traits:
        print(f"  · {t}")

    print()
    print("【播放最多的藝人 Top 10】")
    def clean_artist(raw):
        if not raw or raw == "None":
            return None
        s = str(raw).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip("'\"").split("',")[0].strip("'\"")
        return s or None
    artists = Counter(
        clean_artist(d.get("artist")) for d in data
        if clean_artist(d.get("artist"))
    )
    for i, (artist, cnt) in enumerate(artists.most_common(10), 1):
        print(f"  {i:>2}. {artist}  ({cnt} 次)")

    print("=" * 50)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("[INFO] 載入資料...")
    data = load_data()
    print(f"[INFO] 有效資料：{len(data)} 筆")

    means = {}
    for f in FEATURES + ["tempo", "loudness"]:
        vals = [d[f] for d in data if d.get(f) is not None and d.get(f) != ""]
        means[f] = statistics.mean(vals) if vals else 0

    print_summary(data, means)
    radar_chart(means, OUTPUT_DIR / "fingerprint_2026.png")
    print("\n[DONE] 分析完成")


if __name__ == "__main__":
    main()
