#!/usr/bin/env python3
"""
sheet_backfill_from_db.py — 把三庫特徵回填到 Google Sheet「聆聽紀錄」

設計：
- es_* 欄位：從 essentia_features.db 讀取（包含獨有欄位 + 與 Kaggle 重疊欄位）
- sp_* 欄位：從 kaggle/HF DB 讀取（Kaggle/Spotify 原始值，永不被 Essentia 蓋掉）
- 完全獨立：兩組欄位各自填各自的，互不干擾
- 冪等：es_* 有值就跳過；sp_* 有值就跳過
- 2026-06-20：重構，拿掉 OVERWRITE 概念，改為完全獨立雙欄設計
"""

import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import os
import gspread

from db_paths import ESSENTIA_DB, KAGGLE_DB, HF_DB

SECRETS_DIR   = Path(os.getenv("SECRETS_DIR", str(Path.home() / ".config" / "spotify-dna")))
ENV_FILE      = SECRETS_DIR / ".env"
GSHEET_SA_KEY = Path(os.getenv("GSHEET_SA_KEY_PATH", str(SECRETS_DIR / "service_account.json")))
SHEET_NAME    = "聆聽紀錄"

# Essentia → es_* 欄位（DB 欄名 → Sheet 欄名）
ES_HEADER_MAP = {
    "arousal":          "es_arousal",
    "tempo_confidence": "es_tempo_confidence",
    "key_strength":     "es_key_strength",
    "mood_happy":       "es_mood_happy",
    "mood_sad":         "es_mood_sad",
    "mood_relaxed":     "es_mood_relaxed",
    "mood_aggressive":  "es_mood_aggressive",
    "mood_party":       "es_mood_party",
    "mood_electronic":  "es_mood_electronic",
    "genre_rosamerica": "es_genre_rosamerica",
    "genre_discogs":    "es_genre_discogs",
    # Essentia 與 Kaggle 重疊的特徵，存在 es_* 欄
    "energy":           "es_energy",
    "valence":          "es_valence",
    "danceability":     "es_danceability",
    "instrumentalness": "es_instrumentalness",
    "tempo":            "es_tempo",
    "acousticness":     "es_acousticness",
    "loudness":         "es_loudness",
    "mode":             "es_mode",
    "key":              "es_key",
}

# Kaggle/HF → sp_* 欄位（DB 欄名 → Sheet 欄名）
SP_FIELD_MAP = {
    "energy":           "sp_energy",
    "valence":          "sp_valence",
    "danceability":     "sp_danceability",
    "instrumentalness": "sp_instrumentalness",
    "tempo":            "sp_tempo",
    "acousticness":     "sp_acousticness",
    "speechiness":      "sp_speechiness",
    "liveness":         "sp_liveness",
    "loudness":         "sp_loudness",
    "mode":             "sp_mode",
    "key":              "sp_key",
    "time_signature":   "sp_time_signature",
}

ALL_NEW_HEADERS = list(ES_HEADER_MAP.values()) + list(SP_FIELD_MAP.values())


def _num(val):
    if val in ("", None):
        return ""
    if isinstance(val, (int, float)):
        return val
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return val


def load_essentia_features():
    db = Path(ESSENTIA_DB)
    if not db.exists():
        return {}
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM features WHERE af_source='essentia'").fetchall()
    conn.close()
    return {row["track_id"]: dict(row) for row in rows}


def load_sp_features():
    features = {}
    fields = list(SP_FIELD_MAP.keys())
    select = "track_id, " + ", ".join(fields)
    for db_path in [KAGGLE_DB, HF_DB]:
        db = Path(db_path)
        if not db.exists():
            continue
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {select} FROM features WHERE energy IS NOT NULL"
        ).fetchall()
        conn.close()
        for row in rows:
            d = dict(row)
            tid = d.pop("track_id")
            features.setdefault(tid, {}).update(
                {k: v for k, v in d.items() if v not in (None, "")}
            )
    return features


def ensure_headers(sheet):
    current = sheet.row_values(1)
    missing = [h for h in ALL_NEW_HEADERS if h not in current]
    if missing:
        sheet.update(range_name="A1", values=[current + missing])
        print(f"[GSHEET] 標題列新增欄位：{missing}")
        current = current + missing
    return current


def main():
    print("=== Sheet Backfill from DB START ===")
    load_dotenv(ENV_FILE)
    spreadsheet_id = os.getenv("SPOTIFY_SHEET_ID")
    if not spreadsheet_id:
        print(f"[ERROR] 找不到 SPOTIFY_SHEET_ID")
        sys.exit(1)

    essentia = load_essentia_features()
    sp_features = load_sp_features()
    print(f"[DB] essentia：{len(essentia)} 首 / sp_ 原始值：{len(sp_features)} 首")

    gc = gspread.service_account(filename=str(GSHEET_SA_KEY))
    sheet = gc.open_by_key(spreadsheet_id).worksheet(SHEET_NAME)
    print("[GSHEET] 連線成功")

    headers = ensure_headers(sheet)
    col_idx = {h: i for i, h in enumerate(headers)}
    tid_col = col_idx["track_id"]

    all_touch_cols = list(ES_HEADER_MAP.values()) + list(SP_FIELD_MAP.values())
    valid_cols = [col_idx[h] for h in all_touch_cols if h in col_idx]
    start_col = min(valid_cols)
    end_col   = max(valid_cols)

    all_rows = sheet.get_all_values()
    body = all_rows[1:]
    body = [row + [""] * max(0, end_col + 1 - len(row)) for row in body]

    updates = []
    es_count = sp_count = 0

    for i, row in enumerate(body):
        tid = row[tid_col]
        es_feat = essentia.get(tid)
        sp_feat = sp_features.get(tid)

        # es_* 需要填：
        #   - es_arousal == "" → 第一次填入（exclusive 11 欄 + overlapping 9 欄都空）
        #   - es_energy == "" → migration 後 overlapping 欄被清空需補填（exclusive 已有值）
        needs_es = bool(es_feat) and (
            row[col_idx.get("es_arousal", 0)] == ""
            or row[col_idx.get("es_energy", 0)] == ""
        )
        # sp_* 需要填：有 Kaggle/HF 資料且 sp_energy 還空著
        needs_sp = bool(sp_feat) and row[col_idx.get("sp_energy", 0)] == ""

        if not needs_es and not needs_sp:
            continue

        new_row = list(row)

        if needs_es:
            for db_field, sheet_col in ES_HEADER_MAP.items():
                if sheet_col in col_idx:
                    new_row[col_idx[sheet_col]] = _num(es_feat.get(db_field, ""))
            es_count += 1

        if needs_sp:
            for db_field, sheet_col in SP_FIELD_MAP.items():
                if sheet_col in col_idx:
                    val = _num(sp_feat.get(db_field, ""))
                    if val != "":
                        new_row[col_idx[sheet_col]] = val
            sp_count += 1

        sheet_row_num = i + 2
        a1_start = gspread.utils.rowcol_to_a1(sheet_row_num, start_col + 1)
        a1_end   = gspread.utils.rowcol_to_a1(sheet_row_num, end_col + 1)
        updates.append({
            "range":  f"{a1_start}:{a1_end}",
            "values": [new_row[start_col:end_col + 1]],
        })

    if not updates:
        print("[RESULT] 沒有需要回填的列")
        return

    print(f"[GSHEET] 回填 {len(updates)} 列（es_: {es_count}，sp_: {sp_count}）...")
    BATCH = 200
    for i in range(0, len(updates), BATCH):
        sheet.batch_update(updates[i:i + BATCH], value_input_option="USER_ENTERED")
        print(f"  已寫入 {min(i + BATCH, len(updates))}/{len(updates)}")

    print(f"[RESULT] 完成，回填 {len(updates)} 列")
    print("=== Sheet Backfill from DB END ===")


if __name__ == "__main__":
    main()
