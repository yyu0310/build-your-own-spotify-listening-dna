#!/usr/bin/env python3
"""
db_paths.py — DB 拆分後三庫路徑集中管理（2026-06-20 Phase 3）。

純 os.path / sqlite3，無重依賴，供 audio_pipeline / analyze_local / sheet_backfill 共用，
避免各自寫死路徑（DRY），也讓 sheet_backfill 不必 import essentia 鏈。

三庫角色：
  essentia_features.db   ← 我們自算的特徵（pipeline 寫入目標，可開源）
  kaggle_features.db     ← Kaggle 三資料集合併（外部參考，唯讀）
  huggingface_features.db← HF Figueroa 1.2M（外部參考，唯讀）
"""
import os
import sqlite3

_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Database")

ESSENTIA_DB = os.path.join(_DB_DIR, "essentia_features.db")
KAGGLE_DB = os.path.join(_DB_DIR, "kaggle_features.db")
HF_DB = os.path.join(_DB_DIR, "huggingface_features.db")

REFERENCE_DBS = [KAGGLE_DB, HF_DB]       # 外部參考（唯讀）
ALL_FEATURE_DBS = [ESSENTIA_DB, KAGGLE_DB, HF_DB]


def have_feature_ids():
    """三庫聯集：所有已有特徵（tempo 非空）的 track_id。
    worklist 用它判斷哪些歌已經有特徵、該跳過，不論特徵來自哪個庫。"""
    ids = set()
    for db in ALL_FEATURE_DBS:
        if not os.path.exists(db):
            continue
        con = sqlite3.connect(db)
        try:
            ids |= {r[0] for r in con.execute(
                "SELECT track_id FROM features WHERE tempo IS NOT NULL")}
        finally:
            con.close()
    return ids


def essentia_ids():
    """只查 essentia_features.db：用於 --force-essentia 模式，
    讓 pipeline 補分析那些只有 Kaggle/HF 特徵的曲目。"""
    if not os.path.exists(ESSENTIA_DB):
        return set()
    con = sqlite3.connect(ESSENTIA_DB)
    try:
        return {r[0] for r in con.execute(
            "SELECT track_id FROM features WHERE tempo IS NOT NULL")}
    finally:
        con.close()
