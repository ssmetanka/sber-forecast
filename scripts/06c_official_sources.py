"""Шаг 6c. Официальные источники о ЧС с муниципальной детализацией.

1. Акты о режиме ЧС (publication.pravo.gov.ru): даты введения/отмены регионального режима ЧС,
   все регионы, 2023–2024.
2. Новости региональных ГУ МЧС (NN.mchs.gov.ru): тексты новостей о ЧС, в которых называются
   малые МО. Регионы — все, покрытые данными СберИндекса; период — из configs/news.yaml.

Результат — data/news/events_official.parquet в формате событий шага 6b (+ колонка source).
"""
import re
import sys
import time
from pathlib import Path

import requests

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import ROOT, load_config, load_dict, load_panel  # noqa: E402
from sbx.news import acts as A  # noqa: E402
from sbx.news.mchs import EMERGENCY_RE, collect, fetch_bodies  # noqa: E402
from sbx.news.process import Lemmatizer, build_gazetteer, geocode  # noqa: E402

ncfg = load_config("configs/news.yaml")
ocfg = ncfg["official"]
cfg = load_config()
out = ROOT / "data" / "news"
dct = load_dict(cfg)
dct = dct[dct.year_to >= 2024]
covered_regions = sorted(set(dct.loc[dct.index.isin(load_panel(cfg).territory), "region_code"].astype(int)))

# --- 1. акты о режиме ЧС -----------------------------------------------------------------
f_acts = out / "acts.parquet"
if not f_acts.exists() or "--rebuild" in sys.argv:
    try:
        raw = A.fetch(ocfg["acts_start"], ocfg["acts_end"])
        A.classify(raw).to_parquet(f_acts)
    except Exception as e:
        print(f"портал актов недоступен ({type(e).__name__})")
if not f_acts.exists():
    sys.exit("нет актов и нет доступа к порталу")
acts = pd.read_parquet(f_acts)
reg_acts = acts[acts.is_regime & (acts.action == "введение")]
print(f"актов о ЧС: {len(acts)}, из них о введении режима: {len(reg_acts)}; "
      f"в регионах с данными СберИндекса: {reg_acts.region_code.isin(covered_regions).sum()}")

# --- 2. новости ГУ МЧС ---------------------------------------------------------------------
mdir = out / "mchs"
mdir.mkdir(parents=True, exist_ok=True)
regions = covered_regions if ocfg["mchs_regions"] == "covered" else ocfg["mchs_regions"]
max_min = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--max-minutes=")), None)
no_collect = "--no-collect" in sys.argv
if not no_collect:
    try:                                    # сайты МЧС могут быть недоступны из-за рубежа
        requests.get("https://56.mchs.gov.ru/", timeout=15, verify=False, headers={"User-Agent": "Mozilla/5.0"}).raise_for_status()
    except Exception as e:
        print(f"сайты ГУ МЧС недоступны ({type(e).__name__}) — используем только уже собранное")
        no_collect = True
t_start = time.time()
frames = []
for reg in regions:
    f = mdir / f"{reg:02d}_{ocfg['mchs_start']}_{ocfg['mchs_end']}.parquet"
    if not f.exists():
        if no_collect or (max_min and time.time() - t_start > 60 * max_min):
            continue                       # только уже собранное (без сетевых загрузок)
        try:
            df = collect(reg, ocfg["mchs_start"], ocfg["mchs_end"], section="novosti")
            b = fetch_bodies(df) if len(df) else df.assign(body="")
            b.to_parquet(f)
        except Exception as e:  # сайт региона недоступен — пропускаем, фиксируем
            print(f"  регион {reg:02d}: ошибка {type(e).__name__}")
            continue
    frames.append(pd.read_parquet(f))
# плюс все ранее собранные периоды (например, полный 2024 г. из отдельного сбора в Colab)
seen = {str(x) for x in mdir.glob("*.parquet")}
for extra in sorted(mdir.glob("*.parquet")):
    if not any(str(extra) == str(mdir / f"{r:02d}_{ocfg['mchs_start']}_{ocfg['mchs_end']}.parquet") for r in regions):
        frames.append(pd.read_parquet(extra))
mchs = pd.concat(frames, ignore_index=True).drop_duplicates("url") if frames else pd.DataFrame()
print(f"новостей ГУ МЧС о ЧС с текстом: {len(mchs)} из {len(frames)} регионов")

# --- геопривязка по предложениям о ЧС ------------------------------------------------------
lem = Lemmatizer()
stop = set(ncfg["geocode"]["stop_names"])
cities, districts, regs = build_gazetteer(dct, lem, stop)
SENT = re.compile(r"(?<=[.!?;])\s+")
rows = []
for r in mchs.itertuples():
    text = f"{r.title}. {r.body}"
    mos = set()
    for s in SENT.split(text):
        if re.search(EMERGENCY_RE, s, re.I):
            m, _ = geocode(lem.tokens(s), cities, districts, {k: v for k, v in regs.items() if v == r.region_code}, stop)
            mos.update(t for t in m if int(dct.loc[t, "region_code"]) == r.region_code)
    rows.append({"date": str(r.date), "time": "00:00", "url": r.url, "title": r.title, "types": ["emergency"],
                 "territory_ids": sorted(mos), "region_codes": [] if mos else [int(r.region_code)], "source": "mchs"})
for a in reg_acts.itertuples():
    rows.append({"date": a.publish_date, "time": "00:00", "url": f"http://publication.pravo.gov.ru/document/{a.eo_number}",
                 "title": a.title, "types": ["emergency"], "territory_ids": [], "region_codes": [int(a.region_code)],
                 "source": "acts"})
ev = pd.DataFrame(rows)
ev.to_parquet(out / "events_official.parquet")
n_mo = ev[ev.source == "mchs"].territory_ids.explode().dropna().nunique()
print(f"официальных событий: {len(ev)}; МО, названных ГУ МЧС: {n_mo}")
