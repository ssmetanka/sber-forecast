"""Шаг 6b. Новости → события → экспозиция МО; четыре роли новостей.

  (a) прогноз: поправочный слой log ŷ += β·экспозиция (β на validation), абляция на test;
  (b) детекция: BOCPD с hazard, повышенным в МО с событиями, против обычного BOCPD при равном FAR;
  (c) watch-list: МО с событиями ЧС → precision@K относительно фактических провалов;
  (d) event study паводка апреля 2024: МО, названные в новостях, против МО тех же регионов и страны.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import CATEGORIES, ROOT, load_config, load_dict, path  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.news.process import extract_events, hazard_multiplier, monthly_exposure  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ncfg = load_config("configs/news.yaml")
dcfg = load_config("configs/detection.yaml")
ctx = setup()
p, g = ctx.panel, ctx.grid
tables = path(ctx.cfg, "tables")
news_dir = ROOT / "data" / "news"
dct = load_dict(ctx.cfg)
dct_cur = dct[dct.year_to >= 2024]

# --- события --------------------------------------------------------------------------------
ev_file = news_dir / "events.parquet"
if ev_file.exists() and "--rebuild" not in sys.argv:
    events = pd.read_parquet(ev_file)
else:
    heads = pd.read_parquet(news_dir / "headlines.parquet")
    events = extract_events(heads, dct_cur, ncfg)
    events.to_parquet(ev_file)
    print(f"заголовков {len(heads):,} → событий с геопривязкой {len(events):,}")
events["source"] = "lenta"
off_file = news_dir / "events_official.parquet"
if off_file.exists():                      # ГУ МЧС и акты о режиме ЧС (шаг 6c)
    events = pd.concat([events, pd.read_parquet(off_file)], ignore_index=True)
print("событий по источникам:", events.source.value_counts().to_dict())
events["types"] = events["types"].apply(list)
events["territory_ids"] = events["territory_ids"].apply(list)
events["region_codes"] = events["region_codes"].apply(list)
covered = set(p.territory)
in_data_regions = set(dct.loc[dct.index.isin(covered), "region_code"])
share_cov = events.apply(lambda e: any(t in covered for t in e.territory_ids) or any(r in in_data_regions for r in e.region_codes), axis=1)
print(f"событий с территорией, покрытой данными СберИндекса: {share_cov.mean():.1%}")
print("событий по типам:", events.explode("types").types.value_counts().to_dict())
print("с привязкой к МО:", (events.territory_ids.str.len() > 0).sum(), "только к региону:",
      ((events.territory_ids.str.len() == 0) & (events.region_codes.str.len() > 0)).sum())
# выборка для ручной проверки качества геопривязки и типов
events.sample(min(60, len(events)), random_state=0).assign(
    mo_names=lambda e: e.territory_ids.apply(lambda ids: "; ".join(dct.loc[i, "municipal_district_name"] for i in ids)),
    regions=lambda e: e.region_codes.apply(lambda rs: "; ".join(dct.groupby("region_code").region_name.first()[r] for r in rs)),
)[["date", "title", "types", "mo_names", "regions"]].to_csv(tables / "news_review_sample.csv", index=False)

expo = monthly_exposure(events, dct_cur, p.months, ncfg)
expo.to_parquet(news_dir / "news_monthly.parquet")

# --- матрица экспозиции ЧС+коммунальных аварий S × T ------------------------------------------
tid = p.territory
col = {str(m): j for j, m in enumerate(p.months)}
E = np.zeros((len(tid), len(p.months)))
e_sel = expo[expo.type.isin(ncfg["hazard"]["types"])].groupby(["territory_id", "month"]).exposure.sum()
pos = pd.Series(range(len(tid)), index=tid)
for (t, m), v in e_sel.items():
    if t in pos.index and m in col:
        E[pos.loc[[t]].to_numpy(), col[m]] = v

# прямые упоминания МО в событиях ЧС по месяцам
direct = events[events.types.apply(lambda t: "emergency" in t)].explode("territory_ids").dropna(subset=["territory_ids"])
direct["month"] = direct.date.str[:7]
direct_cnt = direct.groupby(["territory_ids", "month"]).size()

# --- (d) event study: паводок апреля 2024 и Курская область, август 2024 -----------------------
ens = ctx.load("ensemble")                 # лог-прогноз уровней
err = np.full(ctx.Y.shape, np.nan)         # ошибка прогноза на шаг вперёд, лог: факт − прогноз
for j, o in enumerate(g.origins):
    if o + 1 < g.T:
        err[:, o + 1] = np.log(ctx.Y[:, o + 1]) - ens[:, j, 0]
rc_by_name = dct.groupby("region_name").region_code.first()
region = pd.Series(tid).map(dct.region_code).to_numpy()
ev_month = events.assign(month=events.date.str[:7])


def event_study(case, region_names, months_, types):
    regs = [rc_by_name[n] for n in region_names if n in rc_by_name]
    sel = ev_month[ev_month.month.isin(months_) & ev_month.types.apply(lambda t: bool(set(t) & set(types)))]
    named = {int(t) for ids in sel.territory_ids for t in ids if dct.loc[int(t), "region_code"] in regs}
    is_named = np.isin(tid, list(named))
    in_region = np.isin(region, regs)
    rows = []
    for c, cat in enumerate(CATEGORIES):
        m = ctx.cat_idx == c
        for month in months_:
            t = col[month]
            a, b, r = err[m & is_named, t], err[m & in_region & ~is_named, t], err[m & ~in_region, t]
            ok_a, ok_b = np.isfinite(a).sum() > 2, np.isfinite(b).sum() > 2
            rows.append({"case": case, "category": cat, "month": month, "n_named": int(np.isfinite(a).sum()),
                         "named_%": 100 * np.nanmedian(a) if ok_a else np.nan,
                         "same_region_%": 100 * np.nanmedian(b) if ok_b else np.nan,
                         "rest_%": 100 * np.nanmedian(r),
                         "p_named_vs_region": stats.mannwhitneyu(a[np.isfinite(a)], b[np.isfinite(b)]).pvalue if ok_a and ok_b else np.nan,
                         "p_named_vs_rest": stats.mannwhitneyu(a[np.isfinite(a)], r[np.isfinite(r)]).pvalue if ok_a else np.nan})
    names = sorted(dct.loc[list(named), "municipal_district_name_short"].astype(str)) if named else []
    print(f"\n{case}: МО, названных в новостях: {len(named)} — {', '.join(names)}")
    return pd.DataFrame(rows), is_named, in_region


es_flood, is_named, in_region = event_study("Паводок 2024", ["Оренбургская область", "Курганская область", "Тюменская область"],
                                            ["2024-04", "2024-05"], ["emergency"])
# Приграничные регионы (Белгородская, Брянская, Воронежская, Курская, Ростовская обл., Краснодарский
# край, Крым, Севастополь) в муниципальных данных СберИндекса отсутствуют целиком, поэтому кейсы
# атак/вторжения 2024 г. на этих данных проверить нельзя.
es = es_flood
es[es.case == "Паводок 2024"].drop(columns="case").to_csv(tables / "event_study_flood2024.csv", index=False)
es.to_csv(tables / "event_studies.csv", index=False)
print(es.round(3).to_string(index=False))

# --- (b) BOCPD с реальным новостным hazard -----------------------------------------------------
mi = p.month_idx
artifact = tuple(mi(m) for m in ctx.cfg["data"]["artifact_months"])
x = D.levels(ctx.d, mi(dcfg["calib_end"]), artifact + tuple(a + 1 for a in artifact))
months = np.arange(mi(dcfg["monitor"][0]), mi(dcfg["monitor"][1]) + 1)
mult = hazard_multiplier(expo, tid, p.months, ncfg)
bp = dcfg["params"]["bocpd"]
s_plain = D.bocpd(x, **bp)
s_news = D.bocpd(x, hazard_mult=mult, **bp)
rows = []
for name, s in [("BOCPD", s_plain), ("BOCPD + новости", s_news)]:
    th = np.nanquantile(s[:, months], 0.97)
    al = s > th
    t4, t5 = col["2024-04"], col["2024-05"]
    for cat_name, mc in [("все категории рядов", np.ones(len(tid), bool)), ("Маркетплейсы", ctx.cat_idx == 2)]:
        rows.append({"detector": name, "ряды": cat_name,
                     "тревоги в названных МО (апр–май), %": 100 * al[np.ix_(is_named & mc, [t4, t5])].any(1).mean(),
                     "тревоги в остальных МО регионов, %": 100 * al[np.ix_(in_region & ~is_named & mc, [t4, t5])].any(1).mean(),
                     "тревоги по стране, %": 100 * al[np.ix_(~in_region & mc, [t4, t5])].any(1).mean()})
det = pd.DataFrame(rows)
det.to_csv(tables / "news_bocpd_flood.csv", index=False)
print("\n=== BOCPD с реальными новостями (FAR 3 %), паводок 2024 ===")
print(det.round(1).to_string(index=False))

# --- (a) поправочный слой прогноза по новостям -------------------------------------------------
# признак — экспозиция на origin (as-of), цель — ошибка прогноза ансамбля; β по validation
def pairs(split):
    lo, hi = (g.val if split == "val" else g.test)
    X, Yv = [], []
    for j, o in enumerate(g.origins):
        for h in g.horizons:
            t = o + h
            if t < g.T and lo <= t <= hi:
                X.append(E[:, o])
                Yv.append(np.log(ctx.Y[:, t]) - ens[:, j, h - 1])
    return np.concatenate(X), np.concatenate(Yv)

try:
    Xv, Yv = pairs("val")
except ValueError:                         # нет целей в validation (урезанный бэктест)
    Xv, Yv = np.zeros(1), np.zeros(1)
beta = np.sum(Xv * Yv) / max(np.sum(Xv ** 2), 1e-9)
try:
    Xt, Yt = pairs("test")
except ValueError:
    Xt, Yt = np.zeros(1), np.zeros(1)
lvl_parts = [ctx.Y[:, o + h] for j, o in enumerate(g.origins) for h in g.horizons
             if o + h < g.T and g.test[0] <= o + h <= g.test[1]]
lvl = np.concatenate(lvl_parts) if lvl_parts else np.ones(1)
pred = lvl / np.exp(Yt)
mae0 = np.abs(lvl - pred).mean()
mae1 = np.abs(lvl - pred * np.exp(beta * Xt)).mean()
corr = stats.spearmanr(Xt[Xt > 0], Yt[Xt > 0]) if (Xt > 0).sum() > 10 else None
pd.DataFrame([{"beta_val": beta, "MAE_test_без_новостей": mae0, "MAE_test_с_новостями": mae1,
               "доля_пар_с_экспозицией_%": 100 * (Xt > 0).mean(),
               "spearman_экспозиция_ошибка": corr.statistic if corr else np.nan,
               "p": corr.pvalue if corr else np.nan}]).to_csv(tables / "news_forecast_ablation.csv", index=False)
print(f"\n(a) поправка прогноза по новостям: β={beta:.4f}; MAE test {mae0:.1f} → {mae1:.1f}")

# --- (c) watch-list: МО с событиями ЧС в месяце t против провалов в t+1 ----------------------
sig = np.nanstd(err[:, months], axis=1)
hits = []
for t in months[:-1]:
    exposed = E[:, t] > 0.5
    if exposed.sum() == 0:
        continue
    zt = err[:, t + 1] / sig                                     # ошибка в единицах шума своего ряда
    drop = zt < np.nanpercentile(zt, 10)                         # худшие 10 % (провалы)
    hits.append({"month": str(p.months[t]), "n_watch": int(exposed.sum()),
                 "precision": drop[exposed].mean(), "base_rate": drop.mean()})
wl = pd.DataFrame(hits)
wl.to_csv(tables / "news_watchlist.csv", index=False)
if len(wl):
    print(f"(c) watch-list: precision {wl.precision.mean():.3f} при базовой доле {wl.base_rate.mean():.3f}")
