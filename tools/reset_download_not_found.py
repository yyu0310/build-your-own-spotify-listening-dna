#!/usr/bin/env python3
"""重置所有 download_not_found (attempts<3) 讓 pipeline 重跑一次。"""
import datetime
import sqlite3
from pathlib import Path

DB = Path(__file__).parent.parent / "Database" / "essentia_progress.db"

conn = sqlite3.connect(DB)
cur = conn.execute(
    "SELECT COUNT(*) FROM failed WHERE last_error='download_not_found' AND attempts < 3"
)
count = cur.fetchone()[0]
print(f"[INFO] 待重置：{count} 首")

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
conn.execute(
    "UPDATE failed SET attempts=0, last_error='retry_dnf_2026', updated_at=? "
    "WHERE last_error='download_not_found' AND attempts < 3",
    (now,),
)
conn.commit()
conn.close()
print(f"[DONE] 已重置 {count} 首 → retry_dnf_2026")
print("接下來跑 pipeline：caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20")
