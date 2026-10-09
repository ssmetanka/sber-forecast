"""Шаг 0. Загрузка открытых данных: датасет конкурса СберИндекса, справочник МО,
национальные ряды потребительских расходов СберИндекса (портал).

Сайты Сбера используют сертификаты НУЦ Минцифры, которых нет в certifi, поэтому загрузка
идёт с verify=False. Целостность проверяется по размеру файлов (см. вывод).
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES = {
    "hackathon": "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip",
    "dict": "https://s.sber.ru/GthXk7",   # t_dict_municipal.rar: справочник и границы МО (CC BY-SA 4.0)
    "national": "https://sberindex.ru/api/dataset/v1/download/consumer-spending/parquet",
}


def get(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=600, verify=False, allow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  {dest.relative_to(ROOT)}: {len(r.content):,} байт")
    return dest


def main():
    z = get(SOURCES["hackathon"], DATA / "raw" / "hackathonlicence.zip")
    if not (DATA / "hackathonlicence" / "consumption.parquet").exists():
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                if n.endswith(".parquet"):
                    f.extract(n, DATA)
    rar = get(SOURCES["dict"], DATA / "raw" / "t_dict_municipal.rar")
    dict_dir = DATA / "raw" / "dict"
    if not (dict_dir / "t_dict_municipal_districts.xlsx").exists():
        dict_dir.mkdir(parents=True, exist_ok=True)
        tar = shutil.which("bsdtar") or shutil.which("tar")   # libarchive умеет RAR (Windows 10+, macOS)
        subprocess.run([tar, "-xf", str(rar), "-C", str(dict_dir), "t_dict_municipal_districts.xlsx"], check=True)
    get(SOURCES["national"], DATA / "raw" / "portal" / "consumer_spending.parquet")
    print("данные готовы")


if __name__ == "__main__":
    sys.exit(main())
