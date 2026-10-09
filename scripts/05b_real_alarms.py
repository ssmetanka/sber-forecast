"""Шаг 5b. Тревоги на реальных данных 2024 г. и их интерпретация.

1. Рабочий детектор (BOCPD с новостным hazard, FAR 3 %) на фактических рядах.
2. Системные месяцы: если в категории за месяц тревог больше, чем 2 × FAR, это сдвиг общего
   характера (методика, платёжный канал, общероссийское событие), а не локальный шок.
3. Локальные тревоги вне системных месяцев: МО, регион, величина отклонения и новости о
   событиях в этом МО/регионе за ±1 месяц (если новости собраны).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import CATEGORIES, ROOT, load_config, load_dict, path  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.features import FeatureBuilder  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
dcfg = load_config("configs/detection.yaml")
P = dcfg["params"]
p = ctx.panel
mi = p.month_idx
tables = path(ctx.cfg, "tables")
fb = FeatureBuilder(ctx, ctx.cfg["data"]["artifact_months"])
artifact = tuple(mi(m) for m in ctx.cfg["data"]["artifact_months"])
x = D.levels(ctx.d, mi(dcfg["calib_end"]), artifact + tuple(a + 1 for a in artifact))
c = D.deviation(x, season_alpha=dcfg["deviation"]["season_alpha"], ref_start=max(artifact))
months = np.arange(mi(dcfg["monitor"][0]), mi(dcfg["monitor"][1]) + 1)


news_file = ROOT / "data" / "news" / "news_monthly.parquet"
mult = None
if news_file.exists():
    from sbx.news.process import hazard_multiplier
    mult = hazard_multiplier(pd.read_parquet(news_file), p.territory, p.months, load_config("configs/news.yaml"))
score = D.bocpd(x, hazard_mult=mult, **P["bocpd"])
far = 0.03
th = np.quantile(score[:, months], 1 - far)
alarm = score > th

# системные месяцы по категориям
share = pd.DataFrame({cat: alarm[ctx.cat_idx == k][:, months].mean(0) for k, cat in enumerate(CATEGORIES)},
                     index=[str(p.months[t]) for t in months])
share.to_csv(tables / "alarm_share_by_month_category.csv")
systemic = share > 2 * far
print("Доля рядов с тревогой (FAR 3 %), по месяцам и категориям:")
print((100 * share).round(1))

# локальные тревоги вне системных месяцев
dct = load_dict(ctx.cfg)
tid = p.territory
dev_pct = 100 * (np.exp(ctx.d - np.nanmedian(ctx.d[:, max(artifact):mi("2023-12") + 1], 1, keepdims=True)) - 1)
rows = []
for t in months:
    m = str(p.months[t])
    for s in np.where(alarm[:, t])[0]:
        cat = CATEGORIES[ctx.cat_idx[s]]
        if systemic.loc[m, cat]:
            continue
        rows.append({"month": m, "territory_id": int(tid[s]), "МО": dct.loc[tid[s], "municipal_district_name"],
                     "регион": dct.loc[tid[s], "region_name"], "category": cat, "score": round(float(score[s, t]), 2),
                     "отклонение_от_уровня_2023_%": round(dev_pct[s, t], 1),
                     "первая_тревога": not alarm[s, t - 1]})
loc = pd.DataFrame(rows)

# новости рядом с тревогой
ev_file = ROOT / "data" / "news" / "events.parquet"
if ev_file.exists() and len(loc):
    ev = pd.read_parquet(ev_file)
    ev["month"] = pd.PeriodIndex(ev.date.str[:7], freq="M")
    by_mo, by_reg = {}, {}
    for e in ev.itertuples():
        for t_ in e.territory_ids:
            by_mo.setdefault(int(t_), []).append((e.month, e.date, e.title, ",".join(e.types)))
        for r_ in e.region_codes:
            by_reg.setdefault(int(r_), []).append((e.month, e.date, e.title, ",".join(e.types)))

    def near(row):
        m = pd.Period(row.month, "M")
        own = [x for x in by_mo.get(row.territory_id, []) if abs((x[0] - m).n) <= 1]
        reg = [x for x in by_reg.get(int(dct.loc[row.territory_id, "region_code"]), []) if abs((x[0] - m).n) <= 1
               and ("emergency" in x[3] or "utility" in x[3])]
        pick = own[:2] + reg[:1]
        return " || ".join(f"{d} [{ty}] {ti}" for _, d, ti, ty in pick)

    loc["новости_±1_мес"] = loc.apply(near, axis=1)
loc = loc.sort_values(["первая_тревога", "score"], ascending=False)
loc.to_csv(tables / "local_alarms_2024.csv", index=False)
print(f"\nЛокальных тревог вне системных месяцев: {len(loc)}, из них первых (начало эпизода): "
      f"{int(loc['первая_тревога'].sum()) if len(loc) else 0}")
if len(loc):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    cols = ["month", "МО", "category", "отклонение_от_уровня_2023_%"] + (["новости_±1_мес"] if "новости_±1_мес" in loc else [])
    print(loc[loc["первая_тревога"]].head(25)[cols].to_string(index=False))
