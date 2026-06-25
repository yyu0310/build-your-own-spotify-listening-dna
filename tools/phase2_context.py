#!/usr/bin/env python3
"""
phase2_context.py — 聆聽情境時間分析（含完聽率）

輸入：15 年串流歷史 + essentia_features.db + kaggle_features.db（duration_ms）
輸出：Output/context_timeofday.png, Output/context_weekday.png

不只看「你在哪個時間聽什麼風格」，
更看「你在哪個時間真正享受哪種音樂（完聽率）」。
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
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import db_paths

HISTORY_DIR = ROOT / "Database" / "Spotify Extended Streaming History"
OUTPUT_DIR  = ROOT / "Output"
TW_TZ       = timezone(timedelta(hours=8))
MIN_MS      = 30_000

TIME_SLOTS = {
    "深夜\n(22–06)": (22, 6),
    "早晨\n(06–12)": (6, 12),
    "下午\n(12–18)": (12, 18),
    "晚間\n(18–22)": (18, 22),
}
FEATURES_PLOT = ["energy", "valence", "danceability", "acousticness", "arousal"]
FEAT_COLORS   = ["#E63946", "#457B9D", "#2EC4B6", "#F4A261", "#8338EC"]
FEAT_LABELS   = ["Energy", "Valence", "Danceability", "Acousticness", "Arousal"]
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hour_to_slot(h: int) -> str:
    for label, (s, e) in TIME_SLOTS.items():
        if s > e:  # 跨午夜
            if h >= s or h < e:
                return label
        else:
            if s <= h < e:
                return label
    return "晚間\n(18–22)"


def load_history() -> list[dict]:
    records = []
    for path in sorted(HISTORY_DIR.glob("Streaming_History_Audio_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            if not d.get("spotify_track_uri") or d.get("ms_played", 0) < MIN_MS:
                continue
            dt = datetime.fromisoformat(d["ts"].replace("Z", "+00:00")).astimezone(TW_TZ)
            records.append({
                "track_id":   d["spotify_track_uri"].split(":")[-1],
                "ms_played":  d["ms_played"],
                "hour":       dt.hour,
                "weekday":    dt.weekday(),    # 0=Mon
                "reason_end": d.get("reason_end", ""),
            })
    return records


def load_audio_features(track_ids: set) -> dict:
    ph = ",".join("?" * len(track_ids))
    ids = list(track_ids)
    feat_map = {}
    conn = sqlite3.connect(db_paths.ESSENTIA_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT track_id, {', '.join(FEATURES_PLOT)} "
            f"FROM features WHERE track_id IN ({ph})", ids
        ).fetchall()
    finally:
        conn.close()
    for r in rows:
        d = dict(r)
        if all(d.get(f) is not None for f in FEATURES_PLOT):
            feat_map[d["track_id"]] = d
    return feat_map


def load_duration_map(track_ids: set) -> dict:
    if not Path(db_paths.KAGGLE_DB).exists():
        return {}
    ph = ",".join("?" * len(track_ids))
    conn = sqlite3.connect(db_paths.KAGGLE_DB)
    try:
        rows = conn.execute(
            f"SELECT track_id, duration_ms FROM features WHERE track_id IN ({ph})",
            list(track_ids)
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: r[1] for r in rows if r[1]}


def plot_timeofday(slot_data: dict):
    slots = list(TIME_SLOTS.keys())
    x = np.arange(len(slots))

    comp_rates = [
        np.mean(slot_data[s]["completion_rates"]) if slot_data[s]["completion_rates"] else 0
        for s in slots
    ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax1.set_facecolor("#F5F5F5")
    ax2 = ax1.twinx()

    # 特徵折線
    for feat, color, label in zip(FEATURES_PLOT, FEAT_COLORS, FEAT_LABELS):
        vals = [np.mean(slot_data[s]["features"][feat]) if slot_data[s]["features"][feat] else 0
                for s in slots]
        ax1.plot(x, vals, "o-", color=color, linewidth=2, markersize=6, label=label)

    # 完聽率折線（右軸，虛線）
    ax2.plot(x, comp_rates, "s--", color="#1DB954", linewidth=2.5,
             markersize=8, label="完聽率", zorder=5)
    for xi, val in zip(x, comp_rates):
        ax2.text(xi, val + 0.015, f"{val:.0%}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#1DB954")

    ax1.set_xticks(x)
    ax1.set_xticklabels(slots, fontfamily="PingFang TC", fontsize=11)
    ax1.set_ylabel("音頻特徵均值（0–1）", fontfamily="PingFang TC", fontsize=11)
    ax2.set_ylabel("完聽率", color="#1DB954", fontfamily="PingFang TC", fontsize=11)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis="y", colors="#1DB954")

    # 合併 legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title("時段 × 音頻特徵 / 完聽率", fontfamily="PingFang TC",
                  fontsize=14, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    out = OUTPUT_DIR / "context_timeofday.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"[OUTPUT] {out}")


def plot_weekday(weekday_data: dict):
    x = np.arange(7)
    energy_means = [np.mean(weekday_data[d]["energy"]) if weekday_data[d]["energy"] else 0
                    for d in range(7)]
    comp_means   = [np.mean(weekday_data[d]["completion_rates"]) if weekday_data[d]["completion_rates"] else 0
                    for d in range(7)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax2 = ax.twinx()
    ax.bar(x, energy_means, color="#E63946", alpha=0.5, label="Energy 均值")
    ax2.plot(x, comp_means, "o-", color="#1DB954", linewidth=2, label="完聽率")
    ax.set_xticks(x)
    ax.set_xticklabels(WEEKDAY_NAMES)
    ax.set_ylabel("Energy 均值", color="#E63946", fontfamily="PingFang TC")
    ax2.set_ylabel("完聽率", color="#1DB954", fontfamily="PingFang TC")
    ax.set_title("星期幾 × Energy / 完聽率", fontfamily="PingFang TC",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              prop={"family": "PingFang TC"})
    ax.spines[["top"]].set_visible(False)

    out = OUTPUT_DIR / "context_weekday.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"[OUTPUT] {out}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("[INFO] 讀取串流歷史...")
    records = load_history()
    print(f"[INFO] 有效播放：{len(records):,} 筆")

    all_ids = {r["track_id"] for r in records}
    feat_map = load_audio_features(all_ids)
    dur_map  = load_duration_map(all_ids)
    print(f"[INFO] 音頻特徵：{len(feat_map):,}，duration：{len(dur_map):,}")

    slot_data = {s: {"features": {f: [] for f in FEATURES_PLOT}, "completion_rates": []}
                 for s in TIME_SLOTS}
    weekday_data = {d: {"energy": [], "completion_rates": []} for d in range(7)}

    for r in records:
        feat = feat_map.get(r["track_id"])
        if not feat:
            continue
        dur = dur_map.get(r["track_id"])
        comp = min(1.0, r["ms_played"] / dur) if dur else None

        slot = hour_to_slot(r["hour"])
        for f in FEATURES_PLOT:
            slot_data[slot]["features"][f].append(feat[f])
        if comp is not None:
            slot_data[slot]["completion_rates"].append(comp)

        wd = r["weekday"]
        weekday_data[wd]["energy"].append(feat["energy"])
        if comp is not None:
            weekday_data[wd]["completion_rates"].append(comp)

    plot_timeofday(slot_data)
    plot_weekday(weekday_data)
    print("[DONE]")


if __name__ == "__main__":
    main()
