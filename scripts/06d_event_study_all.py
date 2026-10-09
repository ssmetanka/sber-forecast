"""Шаг 6d. Сводный event study по всем ЧС 2024 г. с привязкой к МО (не только паводок).

Отбор по известному событию (паводок) снимается: берём все сообщения о ЧС (ГУ МЧС, Lenta.ru),
где назван конкретный МО, за весь 2024 г. «Событие» — МО × месяц с сообщением о ЧС (дата
публикации). Контроль — МО того же региона без сообщений в этом месяце.

Исходы:
  1) ошибка прогноза ансамбля на шаг вперёд в единицах шума ряда (z): событие против контроля;
  2) доля рядов с тревогой рабочего детектора (BOCPD + новости, FAR 3 %) в месяц события и
     следующий: событие против контроля.
Значимость — кластерный бутстреп по регионам (события одного региона зависимы).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import CATEGORIES, ROOT, load_config, load_dict, path  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.news.process import hazard_multiplier  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
g, p = ctx.grid, ctx.panel
dcfg, ncfg = load_config("configs/detection.yaml"), load_config("configs/news.yaml")
tid = p.territory
dct = load_dict(ctx.cfg)
region = pd.Series(tid).map(dct.region_code).to_numpy()
col = {str(m): j for j, m in enumerate(p.months)}
news = ROOT / "data" / "news"

ev = pd.read_parquet(news / "events.parquet").assign(source="lenta")
if (news / "events_official.parquet").exists():
    ev = pd.concat([ev, pd.read_parquet(news / "events_official.parquet")], ignore_index=True)
ev["types"] = ev["types"].apply(list)
ev["month"] = ev.date.astype(str).str[:7]
ev = ev[ev.month.str.startswith("2024") & ev.types.apply(lambda t: "emergency" in t)]
pairs = sorted({(int(t), m) for ids, m in zip(ev.territory_ids, ev.month) for t in ids if int(t) in set(tid)})
print(f"событий МО × месяц (ЧС, 2024): {len(pairs)}; МО: {len({t for t, _ in pairs})}; "
      f"регионов: {len({int(dct.loc[t, 'region_code']) for t, _ in pairs})}")

# ошибка прогноза ансамбля на шаг вперёд, в единицах шума ряда
ens = ctx.load("ensemble")
err = np.full(ctx.Y.shape, np.nan)
for j, o in enumerate(g.origins):
    if o + 1 < g.T:
        err[:, o + 1] = np.log(ctx.Y[:, o + 1]) - ens[:, j, 0]
months24 = [col[f"2024-{m:02d}"] for m in range(1, 13)]
sig = np.nanstd(err[:, months24], axis=1)
z = err / np.maximum(sig, 1e-9)[:, None]

# тревоги рабочего детектора
mi = p.month_idx
artifact = tuple(mi(m) for m in ctx.cfg["data"]["artifact_months"])
x = D.levels(ctx.d, mi(dcfg["calib_end"]), artifact + tuple(a + 1 for a in artifact))
mult = hazard_multiplier(pd.read_parquet(news / "news_monthly.parquet"), tid, p.months, ncfg)
score = D.bocpd(x, hazard_mult=mult, **dcfg["params"]["bocpd"])
mon = np.arange(mi("2024-01"), mi("2024-12") + 1)
alarm = score > np.nanquantile(score[:, mon], 0.97)

treated_mo_month = {(t, m) for t, m in pairs}
rows = []
for t_mo, m in pairs:
    t = col[m]
    rg = int(dct.loc[t_mo, "region_code"])
    same = region == rg
    ctrl_mo = [mo for mo in set(tid[same]) if (mo, m) not in treated_mo_month and mo != t_mo]
    if not ctrl_mo:
        continue
    for c, cat in enumerate(CATEGORIES):
        rt = np.where((tid == t_mo) & (ctx.cat_idx == c))[0]
        rc = np.where(np.isin(tid, ctrl_mo) & (ctx.cat_idx == c))[0]
        ts = [t] + ([t + 1] if t + 1 < g.T else [])
        rows.append({"region": rg, "territory_id": t_mo, "month": m, "category": cat,
                     "dz": z[rt, t].mean() - np.nanmean(z[rc, t]),
                     "alarm_t": alarm[np.ix_(rt, ts)].any(1).mean(), "alarm_c": alarm[np.ix_(rc, ts)].any(1).mean()})
es = pd.DataFrame(rows)
es.to_csv(path(ctx.cfg, "tables") / "event_study_all_events_raw.csv", index=False)


def cluster_boot(df, col_, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    regs = df.region.unique()
    by = {r: df.loc[df.region == r, col_].to_numpy() for r in regs}
    est = df[col_].mean()
    bs = [np.concatenate([by[r] for r in rng.choice(regs, len(regs))]).mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_ = 2 * min((np.array(bs) <= 0).mean(), (np.array(bs) >= 0).mean())
    return est, lo, hi, p_


out = []
for cat in ["Все категории"] + [c for c in CATEGORIES if c != "Все категории"]:
    d_ = es[es.category == cat].dropna(subset=["dz"])
    if len(d_) < 5:
        continue
    est, lo, hi, p_ = cluster_boot(d_, "dz")
    d_ = d_.assign(dal=d_.alarm_t - d_.alarm_c)
    a_est, a_lo, a_hi, a_p = cluster_boot(d_, "dal")
    out.append({"категория": cat, "событий": len(d_), "регионов": d_.region.nunique(),
                "сдвиг ошибки, σ": est, "ДИ низ": lo, "ДИ верх": hi, "p (кластерный бутстреп)": p_,
                "тревоги в МО с ЧС, %": 100 * d_.alarm_t.mean(), "тревоги в контроле, %": 100 * d_.alarm_c.mean(),
                "разница тревог, п.п.": 100 * a_est, "p тревог": a_p})
res = pd.DataFrame(out)
res.to_csv(path(ctx.cfg, "tables") / "event_study_all_events.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
