"""Шаг 5. Сравнение детекторов структурных сдвигов.

1) Онлайн-детекторы на полусинтетике (форма × величина × охват × seed) при равном FAR.
2) Офлайн-методы ruptures (PELT, BinSeg, BottomUp, Window, KernelCPD) — ретроспективная разметка.
3) Реальные данные 2024: тревоги лучшего детектора и кейс паводка апреля 2024.

    python scripts/05_detection.py [--quick]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import load_config, path  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.detection.synthetic import calibrate, run_grid  # noqa: E402
from sbx.detection.offline import offline_grid  # noqa: E402
from sbx.features import FeatureBuilder  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

quick = "--quick" in sys.argv
ctx = setup()
dcfg = load_config("configs/detection.yaml")
fb = FeatureBuilder(ctx, ctx.cfg["data"]["artifact_months"])
p, d = ctx.panel, ctx.d
tables = path(ctx.cfg, "tables")
mi = p.month_idx

# структура МО: строки рядов каждого МО и соседние МО
tid = p.territory
mo_list, first = np.unique(tid, return_index=True)
mo_rows = fb.mo_rows[first]                                   # n_MO × 6
mo_of_row = np.searchsorted(mo_list, tid)
nb_mo = np.where(fb.nb_rows[first] >= 0, mo_of_row[fb.nb_rows[first].clip(0)], -1)

calib_end = mi(dcfg["calib_end"])
artifact = tuple(mi(m) for m in ctx.cfg["data"]["artifact_months"])
exclude = artifact + tuple(a + 1 for a in artifact)
ref_start = max(artifact) if artifact else 0
months = np.arange(mi(dcfg["monitor"][0]), mi(dcfg["monitor"][1]) + 1)
onset_range = (mi(dcfg["onset"][0]), mi(dcfg["onset"][1]))


def z_fn(x):
    """Потоки для детекторов: z — инновации на шаг вперёд, x — уровни, c — отклонение от эталона."""
    lv = D.levels(x, calib_end, exclude)
    return {"z": D.innovations(x, calib_end, exclude=exclude), "x": lv,
            "c": D.deviation(lv, season_alpha=dcfg["deviation"]["season_alpha"], ref_start=ref_start)}


def norm97(s):
    """Ранговая нормировка по месяцам мониторинга (эмпирическая функция распределения) —
    делает разные детекторы сопоставимыми для ИЛИ-комбинации."""
    q = s[:, months]
    ref = np.sort(q[np.isfinite(q)])
    out = np.searchsorted(ref, s, side="right") / len(ref)
    return np.where(np.isfinite(s), out, np.nan)


P = dcfg["params"]
detectors = {
    "z-score (инновация)": lambda z, nw: D.zscore(z["z"]),
    "EWMA": lambda z, nw: D.ewma(z["z"], **P["ewma"]),
    "CUSUM классический": lambda z, nw: D.cusum(z["z"], **P["cusum"]),
    "Page-Hinkley": lambda z, nw: D.page_hinkley(z["z"], **P["page_hinkley"]),
    "CUSUM оконный": lambda z, nw: D.cusum_w(z["c"], **P["cusum_w"]),
    "GLR (онлайн-разрыв)": lambda z, nw: D.glr(z["x"], **P["glr"]),
    "BOCPD": lambda z, nw: D.bocpd(z["x"], **P["bocpd"]),
    "BOCPD + новости": lambda z, nw: D.bocpd(z["x"], hazard_mult=nw, **P["bocpd"]),
    "Stouffer-МО": lambda z, nw: D.stouffer_mo(z["c"], fb.mo_rows, **P["cusum_w"]),
    "Пространственный": lambda z, nw: D.spatial(z["c"], fb.nb_rows, w_self=P["spatial_w_self"], **P["cusum_w"]),
    "Комбо: BOCPD ∨ Stouffer-МО": lambda z, nw: np.maximum(
        norm97(D.bocpd(z["x"], **P["bocpd"])), norm97(D.stouffer_mo(z["c"], fb.mo_rows, **P["cusum_w"]))),
    "Комбо: оконный ∨ Stouffer ∨ пространств.": lambda z, nw: np.maximum.reduce([
        norm97(D.cusum_w(z["c"], **P["cusum_w"])),
        norm97(D.stouffer_mo(z["c"], fb.mo_rows, **P["cusum_w"])),
        norm97(D.spatial(z["c"], fb.nb_rows, w_self=P["spatial_w_self"], **P["cusum_w"]))]),
}


def news_sim(shocked, onset, rng, recall=dcfg["news_sim"]["recall"], false_rate=dcfg["news_sim"]["false_rate"],
             mult=dcfg["news_sim"]["mult"]):
    """Симуляция новостного потока: о доле `recall` шоков есть новость в месяц начала,
    плюс ложные новости с частотой false_rate на ряд-месяц. Возвращает множитель hazard."""
    S, T = shocked.shape[0], d.shape[1]
    m = np.ones((S, T))
    idx = np.where(shocked & (rng.random(S) < recall))[0]
    m[idx, onset[idx]] = mult
    m[rng.random((S, T)) < false_rate] = mult
    return m


# --- 1. онлайн-детекторы на полусинтетике ------------------------------------------------
t0 = time.time()
g = dcfg["grid"]
res = run_grid(d, detectors, mo_rows, nb_mo, z_fn, months,
               fars=tuple(g["fars"]), magnitudes=tuple(g["magnitudes"]) if not quick else (0.10,),
               shapes=tuple(g["shapes"]), scopes=tuple(g["scopes"]), onset_range=onset_range,
               frac=g["frac"], seeds=tuple(range(g["seeds"])) if not quick else (0,), news_sim=news_sim)
res.to_csv(tables / "detection_synthetic_raw.csv", index=False)
print(f"полусинтетика: {time.time() - t0:.0f} с")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
at3 = res[res.far == 0.03]
summary = at3.groupby("detector")[["recall", "recall_at_onset", "delay", "precision", "F1", "far_per100"]].mean()
summary = summary.sort_values("recall", ascending=False)
summary.to_csv(tables / "detection_summary_far3.csv")
print("\n=== Средние по сетке, FAR = 3 ложных тревоги на 100 ряд-месяцев ===")
print(summary.round(3))
by_scope = at3.pivot_table(index="detector", columns="scope", values="recall").round(3)
by_scope.to_csv(tables / "detection_recall_by_scope.csv")
print("\n=== Полнота по охвату шока ===")
print(by_scope)
by_mag = at3.pivot_table(index="detector", columns="magnitude", values="recall").round(3)
by_mag.to_csv(tables / "detection_recall_by_magnitude.csv")
print("\n=== Полнота по величине шока ===")
print(by_mag)
by_far = res.pivot_table(index="detector", columns="far", values="recall").round(3)
by_far.to_csv(tables / "detection_recall_by_far.csv")
print("\n=== Полнота при разных FAR ===")
print(by_far)

# --- 2. офлайн-методы ruptures ------------------------------------------------------------
if not quick:
    t0 = time.time()
    off = offline_grid(d, mo_rows, nb_mo, onset_range, dcfg["offline"], log=print)
    off.to_csv(tables / "detection_offline.csv", index=False)
    print(f"\n=== Офлайн (ретроспективно, весь ряд), {time.time() - t0:.0f} с ===")
    print(off.groupby("method")[["recall", "precision", "F1", "false_cp_per100"]].mean().round(3))

# --- 3. реальные данные -------------------------------------------------------------------
z = z_fn(d)
best_name = dcfg["production_detector"]
score = detectors[best_name](z, None)
th = np.nanquantile(score[:, months], 1 - 0.03)
alarm = score > th
dct = fb.ctx.panel.y.index.to_frame(index=False)
dct["region"] = fb.region
alarms = []
for t in months:
    rows = np.where(alarm[:, t])[0]
    for r in rows:
        alarms.append({"month": str(p.months[t]), "territory_id": tid[r], "category": p.category[r],
                       "score": score[r, t], "dev": z["c"][r, t]})
pd.DataFrame(alarms).to_csv(tables / "alarms_2024.csv", index=False)
print(f"\nТревоги 2024 ({best_name}, FAR 3 %):",
      pd.DataFrame(alarms).groupby("month").size().to_dict() if alarms else 0)
np.save(path(ctx.cfg, "cache") / "z_real.npy", z["c"])
