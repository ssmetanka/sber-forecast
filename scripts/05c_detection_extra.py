"""Шаг 5c. Дополнительные проверки детекторов.

1. Ориентир случайного детектора и средняя полнота по кривой FAR (1/3/5 %).
2. Чувствительность «BOCPD + новости» к качеству новостей: полнота новостей 0 (новости — чистый
   шум), 0.3, 0.6, 0.9 при 3 % ложных «новостей». При шумовых новостях выигрыша быть не должно —
   это проверка, что новостной prior не работает как «оракул».
3. Реальные события, вне синтетики: (a) паводок апреля 2024 (МО из новостей ГУ МЧС/Lenta),
   (b) все региональные режимы ЧС из официальных актов 2024 г. в регионах с данными (кроме
   паводковых) — проверка вне выборки; плацебо — те же регионы в случайные месяцы.
4. Детектор «выход за интервал Chronos-2» (прогноз на шаг вперёд) и покрытие 80 %-интервалов FM.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import CATEGORIES, ROOT, load_config, load_dict, path  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.detection.synthetic import run_grid  # noqa: E402
from sbx.features import FeatureBuilder  # noqa: E402
from sbx.news.process import hazard_multiplier  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
dcfg = load_config("configs/detection.yaml")
ncfg = load_config("configs/news.yaml")
P = dcfg["params"]
p, g, d = ctx.panel, ctx.grid, ctx.d
mi = p.month_idx
T = path(ctx.cfg, "tables")
fb = FeatureBuilder(ctx, ctx.cfg["data"]["artifact_months"])
tid = p.territory
mo_list, first = np.unique(tid, return_index=True)
mo_rows = fb.mo_rows[first]
mo_of_row = np.searchsorted(mo_list, tid)
nb_mo = np.where(fb.nb_rows[first] >= 0, mo_of_row[fb.nb_rows[first].clip(0)], -1)
artifact = tuple(mi(m) for m in ctx.cfg["data"]["artifact_months"])
exclude = artifact + tuple(a + 1 for a in artifact)
calib_end = mi(dcfg["calib_end"])
months = np.arange(mi(dcfg["monitor"][0]), mi(dcfg["monitor"][1]) + 1)
onset_range = (mi(dcfg["onset"][0]), mi(dcfg["onset"][1]))
ref_start = max(artifact)


def streams(x):
    lv = D.levels(x, calib_end, exclude)
    return {"z": D.innovations(x, calib_end, exclude=exclude), "x": lv,
            "c": D.deviation(lv, season_alpha=dcfg["deviation"]["season_alpha"], ref_start=ref_start)}


# --- 1. случайный детектор и площадь под кривой FAR ---------------------------------------
raw = pd.read_csv(T / "detection_synthetic_raw.csv")
tol = 2
rnd = {f: 1 - (1 - f) ** (tol + 1) for f in raw.far.unique()}
auc = raw.groupby(["detector", "far"]).recall.mean().unstack()
auc["средняя полнота (FAR 1–5 %)"] = auc.mean(1)
auc.loc["Случайные тревоги (ориентир)"] = [rnd[f] for f in auc.columns[:-1]] + [np.mean(list(rnd.values()))]
auc = auc.sort_values("средняя полнота (FAR 1–5 %)", ascending=False)
auc.to_csv(T / "detection_far_curve_auc.csv")
print("=== Полнота по кривой FAR и ориентир случайного детектора ===")
print(auc.round(3))

# --- 2. чувствительность к качеству новостей ----------------------------------------------
rows = []
for recall in (0.0, 0.3, 0.6, 0.9):
    def news_sim(shocked, onset, rng, recall=recall):
        S, Tn = shocked.shape[0], d.shape[1]
        m = np.ones((S, Tn))
        idx = np.where(shocked & (rng.random(S) < recall))[0]
        m[idx, onset[idx]] = dcfg["news_sim"]["mult"]
        m[rng.random((S, Tn)) < dcfg["news_sim"]["false_rate"]] = dcfg["news_sim"]["mult"]
        return m
    dets = {"BOCPD": lambda z, nw: D.bocpd(z["x"], **P["bocpd"]),
            "BOCPD + новости": lambda z, nw: D.bocpd(z["x"], hazard_mult=nw, **P["bocpd"])}
    r = run_grid(d, dets, mo_rows, nb_mo, streams, months, fars=(0.03,), magnitudes=(0.05, 0.10),
                 shapes=("step", "dip"), scopes=("series", "mo"), onset_range=onset_range, frac=0.1,
                 seeds=(0, 1), news_sim=news_sim, log=lambda *_: None)
    m = r.groupby("detector").recall.mean()
    rows.append({"полнота новостей": recall, "BOCPD": m["BOCPD"], "BOCPD + новости": m["BOCPD + новости"],
                 "прирост, п.п.": 100 * (m["BOCPD + новости"] - m["BOCPD"])})
sens = pd.DataFrame(rows)
sens.to_csv(T / "news_sim_sensitivity.csv", index=False)
print("\n=== Чувствительность к качеству новостей (FAR 3 %, ложные новости 3 %) ===")
print(sens.round(3).to_string(index=False))

# --- 3. реальные события ------------------------------------------------------------------
st = streams(d)
news_file = ROOT / "data" / "news" / "news_monthly.parquet"
mult = hazard_multiplier(pd.read_parquet(news_file), tid, p.months, ncfg) if news_file.exists() else None


def rank(s):
    ref = np.sort(s[:, months][np.isfinite(s[:, months])])
    return np.searchsorted(ref, s, side="right") / len(ref)


real_det = {
    "z-score (инновация)": D.zscore(st["z"]),
    "EWMA": D.ewma(st["z"], **P["ewma"]),
    "CUSUM оконный": D.cusum_w(st["c"], **P["cusum_w"]),
    "GLR (онлайн-разрыв)": D.glr(st["x"], **P["glr"]),
    "BOCPD": D.bocpd(st["x"], **P["bocpd"]),
    "BOCPD + новости": D.bocpd(st["x"], hazard_mult=mult, **P["bocpd"]),
    "Stouffer-МО": D.stouffer_mo(st["c"], fb.mo_rows, **P["cusum_w"]),
    "Пространственный": D.spatial(st["c"], fb.nb_rows, w_self=P["spatial_w_self"], **P["cusum_w"]),
}

# FM-интервал: факт против прогноза Chronos-2 на шаг вперёд (квантили 0.1/0.5/0.9 по d)
qf = path(ctx.cfg, "cache") / "fm_quantiles" / "chronos2_joint.npy"
coverage = []
if qf.exists():
    Q = np.load(qf).astype(float)                     # S × O × H × 3 (на сетке текущего бэктеста)
    if Q.shape[1] == len(g.origins):
        zfm = np.full(d.shape, np.nan)
        for j, o in enumerate(g.origins):
            t = o + 1
            if t < d.shape[1]:
                q10, q50, q90 = Q[:, j, 0, 0], Q[:, j, 0, 1], Q[:, j, 0, 2]
                zfm[:, t] = np.abs(d[:, t] - q50) / np.maximum((q90 - q10) / 2.563, 1e-6)
        real_det["Выход за интервал Chronos-2"] = zfm
        for name in ["chronos2_joint", "chronos_bolt", "chronos_bolt_small", "chronos2", "timesfm"]:
            f = path(ctx.cfg, "cache") / "fm_quantiles" / f"{name}.npy"
            if not f.exists():
                continue
            Qm = np.load(f).astype(float)
            if Qm.shape[1] != len(g.origins):
                continue
            tidx = g.target_idx()
            for split, (lo, hi) in [("val", g.val), ("test", g.test)]:
                for h in (1, 3, 6, 12):
                    js = [j for j in range(len(g.origins)) if tidx[j, h - 1] >= 0 and lo <= tidx[j, h - 1] <= hi]
                    if not js:
                        continue
                    y = np.concatenate([d[:, tidx[j, h - 1]] for j in js])
                    lo_q = np.concatenate([Qm[:, j, h - 1, 0] for j in js])
                    hi_q = np.concatenate([Qm[:, j, h - 1, 2] for j in js])
                    ok = np.isfinite(lo_q) & np.isfinite(hi_q)
                    coverage.append({"модель": name, "split": split, "h": h,
                                     "покрытие 80 %-интервала": ((y >= lo_q) & (y <= hi_q))[ok].mean()})
# конформная калибровка: множитель ширины интервала вокруг медианы, при котором покрытие на
# validation = 80 %; на test проверяем покрытие откалиброванного интервала
conf_rows = []
for name in ["chronos2_joint", "chronos_bolt", "chronos_bolt_small", "chronos2", "timesfm"]:
    f = path(ctx.cfg, "cache") / "fm_quantiles" / f"{name}.npy"
    if not f.exists():
        continue
    Qm = np.load(f).astype(float)
    if Qm.shape[1] != len(g.origins):
        continue
    tidx = g.target_idx()
    for h in (1, 3, 6, 12):
        def pairs(lo, hi):
            js = [j for j in range(len(g.origins)) if tidx[j, h - 1] >= 0 and lo <= tidx[j, h - 1] <= hi]
            y = np.concatenate([d[:, tidx[j, h - 1]] for j in js])
            q = np.concatenate([Qm[:, j, h - 1, :] for j in js])
            ok = np.isfinite(q).all(1)
            return y[ok], q[ok]
        yv, qv = pairs(*g.val)
        yt, qt = pairs(*g.test)
        # нормированное отклонение от медианы в единицах полуширины интервала
        sv = np.abs(yv - qv[:, 1]) / np.maximum((qv[:, 2] - qv[:, 0]) / 2, 1e-9)
        k = np.quantile(sv, 0.80)
        st_ = np.abs(yt - qt[:, 1]) / np.maximum((qt[:, 2] - qt[:, 0]) / 2, 1e-9)
        conf_rows.append({"модель": name, "h": h, "множитель ширины (по validation)": k,
                          "покрытие test до калибровки": (st_ <= 1).mean(), "покрытие test после калибровки": (st_ <= k).mean()})
if conf_rows:
    cf = pd.DataFrame(conf_rows)
    cf.to_csv(T / "fm_interval_conformal.csv", index=False)
    print("\n=== Конформная калибровка 80 %-интервалов FM (множитель по validation, проверка на test) ===")
    print(cf.round(3).to_string(index=False))
if coverage:
    cov = pd.DataFrame(coverage)
    cov.to_csv(T / "fm_interval_coverage.csv", index=False)
    print("\n=== Покрытие 80 %-интервалов FM (номинал 0.80) ===")
    print(cov.pivot_table(index=["модель", "split"], columns="h", values="покрытие 80 %-интервала").round(3))

# события: паводок (МО из новостей) и официальные режимы ЧС (регионы) вне паводка
dct = load_dict(ctx.cfg)
region = pd.Series(tid).map(dct.region_code).to_numpy()
ev = pd.read_parquet(ROOT / "data" / "news" / "events.parquet")
off = ROOT / "data" / "news" / "events_official.parquet"
if off.exists():
    ev = pd.concat([ev, pd.read_parquet(off)], ignore_index=True)
ev["month"] = ev.date.astype(str).str[:7]
flood_regs = [dct.groupby("region_name").region_code.first()[n] for n in
              ["Оренбургская область", "Курганская область", "Тюменская область"]]
sel = ev[ev.month.isin(["2024-04", "2024-05"]) & ev.types.apply(lambda t: "emergency" in list(t))]
named = {int(t) for ids in sel.territory_ids for t in ids if int(dct.loc[int(t), "region_code"]) in flood_regs}
is_named = np.isin(tid, list(named))
in_flood_region = np.isin(region, flood_regs)
acts = pd.read_parquet(ROOT / "data" / "news" / "acts.parquet")
acts = acts[acts.is_regime & (acts.action == "введение") & acts.publish_date.str.startswith("2024")]
acts = acts[~acts.region_code.isin(flood_regs) & acts.region_code.isin(set(region))]
act_events = sorted({(int(r.region_code), r.publish_date[:7]) for r in acts.itertuples()})
col = {str(m): j for j, m in enumerate(p.months)}
rng = np.random.default_rng(0)
mon_names = [str(p.months[t]) for t in months]

from scipy import stats as sst
rows = []
for name, s in real_det.items():
    th = np.nanquantile(s[:, months], 0.97)
    al = s > th
    t4, t5 = col["2024-04"], col["2024-05"]
    hit_named = al[np.ix_(is_named, [t4, t5])].any(1)
    hit_base = al[np.ix_(~in_flood_region, [t4, t5])].any(1)
    a_named, a_base = hit_named.mean(), hit_base.mean()
    p_flood = sst.fisher_exact([[hit_named.sum(), (~hit_named).sum()], [hit_base.sum(), (~hit_base).sum()]],
                               alternative="greater")[1]
    # акты: доля рядов региона с тревогой в месяц акта или следующий
    hit, plc, pairs_ = [], [], []
    for rc, m in act_events:
        rows_r = region == rc
        if m not in col:
            continue
        t = col[m]
        ts = [t] + ([t + 1] if t + 1 < d.shape[1] else [])
        h_ev = al[np.ix_(rows_r, ts)].any(1).mean()
        hit.append(h_ev)
        p_ev = []
        for _ in range(5):                            # плацебо: тот же регион, случайный месяц 2024
            tp = col[rng.choice(mon_names)]
            tps = [tp] + ([tp + 1] if tp + 1 < d.shape[1] else [])
            p_ev.append(al[np.ix_(rows_r, tps)].any(1).mean())
        plc += p_ev
        pairs_.append((h_ev, np.mean(p_ev)))
    rows.append({"детектор": name, "паводок: МО из новостей, %": 100 * a_named, "паводок: страна, %": 100 * a_base,
                 "lift паводка": a_named / max(a_base, 1e-9),
                 "режимы ЧС (акты, вне паводка), %": 100 * np.mean(hit) if hit else np.nan,
                 "плацебо (случайный месяц), %": 100 * np.mean(plc) if plc else np.nan,
                 "lift актов": (np.mean(hit) / max(np.mean(plc), 1e-9)) if hit else np.nan,
                 "p паводок (Фишер)": p_flood,
                 "p акты (Уилкоксон)": sst.wilcoxon([a - b for a, b in pairs_], alternative="greater").pvalue
                 if pairs_ and any(abs(a - b) > 0 for a, b in pairs_) else np.nan})
real = pd.DataFrame(rows).sort_values("lift паводка", ascending=False)

# --- проверка без круга: группу МО задают только новости ГУ МЧС, а новостной hazard детектора
# строится без них (Lenta.ru + акты). Тогда эффект не может возникнуть из-за того, что hazard
# и группа питаются одними и теми же сообщениями.
from sbx.news.process import monthly_exposure  # noqa: E402
ev_lenta = pd.read_parquet(ROOT / "data" / "news" / "events.parquet").assign(source="lenta")
ev_all = pd.concat([ev_lenta, pd.read_parquet(off)], ignore_index=True) if off.exists() else ev_lenta
for c_ in ("types", "territory_ids", "region_codes"):
    ev_all[c_] = ev_all[c_].apply(list)
dct_cur = dct[dct.year_to >= 2024] if "year_to" in dct else dct
mult_wo = hazard_multiplier(monthly_exposure(ev_all[ev_all.source != "mchs"], dct_cur, p.months, ncfg), tid, p.months, ncfg)
ev_all["month"] = ev_all.date.astype(str).str[:7]
sel_m = ev_all[(ev_all.source == "mchs") & ev_all.month.isin(["2024-04", "2024-05"])
               & ev_all.types.apply(lambda t: "emergency" in t)]
named_m = {int(t) for ids in sel_m.territory_ids for t in ids if int(dct.loc[int(t), "region_code"]) in flood_regs}
is_named_m = np.isin(tid, list(named_m))
nc_rows = []
for name, s in {"BOCPD (без новостей)": real_det["BOCPD"],
                "BOCPD + новости (hazard без ГУ МЧС)": D.bocpd(st["x"], hazard_mult=mult_wo, **P["bocpd"]),
                "BOCPD + новости (все источники, круговая)": real_det["BOCPD + новости"],
                "Выход за интервал Chronos-2": real_det.get("Выход за интервал Chronos-2")}.items():
    if s is None:
        continue
    al = s > np.nanquantile(s[:, months], 0.97)
    t4, t5 = col["2024-04"], col["2024-05"]
    hn = al[np.ix_(is_named_m, [t4, t5])].any(1)
    hb = al[np.ix_(~in_flood_region, [t4, t5])].any(1)
    nc_rows.append({"детектор": name, "МО из сообщений ГУ МЧС": int(is_named_m.sum() // 6),
                    "тревоги в этих МО, %": 100 * hn.mean(), "тревоги по стране, %": 100 * hb.mean(),
                    "lift": hn.mean() / max(hb.mean(), 1e-9),
                    "p (Фишер)": sst.fisher_exact([[hn.sum(), (~hn).sum()], [hb.sum(), (~hb).sum()]], alternative="greater")[1]})
nc = pd.DataFrame(nc_rows)
nc.to_csv(T / "detection_flood_noncircular.csv", index=False)
print("\n=== Паводок без круга: группа — ГУ МЧС, hazard — Lenta.ru + акты ===")
print(nc.round(3).to_string(index=False))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


allp = np.concatenate([real["p паводок (Фишер)"].to_numpy(), real["p акты (Уилкоксон)"].fillna(1).to_numpy()])
adj = holm(allp)
real["p паводок, Холм"] = adj[: len(real)]
real["p акты, Холм"] = adj[len(real):]
real.to_csv(T / "detection_real_events.csv", index=False)
print(f"\n=== Реальные события (FAR 3 %): МО паводка из новостей — {is_named.sum() // 6}; "
      f"региональных режимов ЧС из актов вне паводка — {len(act_events)} ===")
print(real.round(2).to_string(index=False))
