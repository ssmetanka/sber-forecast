"""Манифест входных данных с SHA-256: фиксирует версию данных и новостного корпуса.

    python tools/hash_manifest.py   →   outputs/tables/data_manifest.json
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["data/hackathonlicence/consumption.parquet", "data/hackathonlicence/market_access.parquet",
         "data/hackathonlicence/connection.parquet", "data/raw/dict/t_dict_municipal_districts.xlsx",
         "data/raw/portal/consumer_spending.parquet", "data/news/headlines.parquet", "data/news/events.parquet",
         "data/news/acts.parquet", "data/news/events_official.parquet"]
out = {}
for rel in FILES + sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "data/news/mchs").glob("*.parquet")):
    p = ROOT / rel
    if p.exists():
        out[rel] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
dest = ROOT / "outputs/tables/data_manifest.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(out), "файлов ->", dest)
