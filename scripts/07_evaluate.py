"""Шаг 7. Сводная оценка прогнозов: все модели, ансамбль, сравнение с Prophet.

Таблицы → outputs/tables, графики → outputs/figures. Модели, прогнозы которых ещё не
посчитаны, пропускаются.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import ensemble  # noqa: E402
from sbx.backtest import evaluate, save_forecast  # noqa: E402
from sbx.data import CATEGORIES, load_config, path  # noqa: E402
from sbx.pipeline import pivot, setup  # noqa: E402
from sbx.stats import compare  # noqa: E402

ctx = setup()
g, out = ctx.grid, ctx.out
tables, figs = path(ctx.cfg, "tables"), path(ctx.cfg, "figures")
nb = ctx.load("national_best")

# --- кандидаты: лог-прогнозы уровней ------------------------------------------------------
LOCAL = {"d__last": "Панель: последнее отклонение", "d__ses05": "Панель: SES(0.5) на d",
         "d__level_season03": "Панель: уровень + сезонность d", "d__lgbm": "LightGBM на d",
         "d__lgbm_v2": "LightGBM v2 на d (без региона)",
         "d__fm_chronos2_joint": "Chronos-2 (6 категорий совместно) на d",
         "d__fm_chronos_bolt": "Chronos-Bolt (base) на d", "d__fm_chronos_bolt_small": "Chronos-Bolt (small) на d",
         "d__fm_chronos2": "Chronos-2 (одномерный) на d", "d__fm_timesfm": "TimesFM-2.5 на d",
         "d__fm_chronos2_joint_lora_c": "Chronos-2 совместный + LoRA (дообучение до 12.2023) на d",
         "prophet__local": "Prophet на d (локальная компонента)",
         "d__fm_tirex": "TiRex на d"}
RAW = {"raw__naive": "Наивная (последнее значение)", "raw__snaive": "Сезонная наивная",
       "raw__snaive_growth": "Сезонная наивная × рост", "raw__fm_chronos_bolt": "Chronos-Bolt (base) на сыром ряду",
       "raw__fm_chronos_bolt_small": "Chronos-Bolt (small) на сыром ряду", "raw__fm_chronos2": "Chronos-2 на сыром ряду", "raw__fm_tirex": "TiRex на сыром ряду",
       "raw__fm_timesfm": "TimesFM-2.5 на сыром ряду",
       "prophet__default": "Prophet (по умолчанию)", "prophet__default_full": "Prophet (по умолчанию) — вся панель", "prophet__yearly": "Prophet (log, годовая сезонность, праздники)"}

logF, labels = {}, {}


def guard_prophet(F, name):
    """Страховка от расходимости Prophet (в его пользу): прогноз дальше чем в 3 раза за пределами
    истории ряда до origin заменяется последним значением. На ранних origin (7–12 точек) годовая
    сезонность в лог-шкале иначе экстраполируется экспонентой."""
    ly = ctx.panel.ly
    out = F.copy()
    n_bad = 0
    for j, o in enumerate(g.origins):
        hist = ly[:, : o + 1]
        lo, hi = np.nanmin(hist, 1) - np.log(3), np.nanmax(hist, 1) + np.log(3)
        bad = np.isfinite(F[:, j]) & ((F[:, j] < lo[:, None]) | (F[:, j] > hi[:, None]))
        n_bad += bad.sum()
        out[:, j] = np.where(bad, ly[:, o][:, None], F[:, j])
    tot = np.isfinite(F).sum()
    GUARD[name] = {"заменено ячеек": int(n_bad), "доля, %": round(100 * n_bad / max(tot, 1), 3)}
    return out


GUARD = {}
for k, lab in LOCAL.items():
    if (out / f"{k}.npy").exists():
        logF[k] = nb[ctx.cat_idx] + ctx.load(k)
        labels[k] = lab
for k, lab in RAW.items():
    if (out / f"{k}.npy").exists():
        logF[k] = ctx.load(k)
        labels[k] = lab
for k in [k for k in logF if k.startswith("prophet__")]:
    logF[k] = guard_prophet(logF[k], k)
json.dump(GUARD, open(tables / "prophet_guard.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("страховка Prophet:", GUARD)
print("модели:", list(labels.values()))

# --- ансамбль (веса на validation, только модели на всей панели) ---------------------------
pool = [k for k in logF if k in LOCAL and not k.startswith("prophet")]   # Prophet — только выборка МО
ecfg = load_config("configs/models.yaml")["ensemble"]
W = ensemble.caruana({k: logF[k] for k in pool}, ctx.Y, g, rounds=ecfg["rounds"])
ens = ensemble.apply({k: logF[k] for k in pool}, W, g)
logF["ensemble"] = np.log(ens)
labels["ensemble"] = "Ансамбль (Caruana по группам h)"
save_forecast(np.log(ens), out, "ensemble")
# абляция: тот же отбор без foundation models — вклад FM в ансамбль
pool_nofm = [k for k in pool if "__fm_" not in k]
if len(pool_nofm) < len(pool):
    W0 = ensemble.caruana({k: logF[k] for k in pool_nofm}, ctx.Y, g, rounds=ecfg["rounds"])
    logF["ensemble_nofm"] = np.log(ensemble.apply({k: logF[k] for k in pool_nofm}, W0, g))
    labels["ensemble_nofm"] = "Ансамбль без foundation models (абляция)"
json.dump({str(g.groups[i]): {labels.get(m, m): round(w, 3) for m, w in ws.items()} for i, ws in W.items()},
          open(tables / "ensemble_weights.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("веса ансамбля:", json.dumps({str(g.groups[i]): ws for i, ws in W.items()}, ensure_ascii=False))

# --- метрики на всей панели -----------------------------------------------------------------
full_mask = np.ones(len(ctx.Y), bool)
mask_file = out / "prophet_sample_mask.npy"
sample = np.load(mask_file) if mask_file.exists() else full_mask

# «лучший Prophet задним числом»: на каждом h — конфигурация с наименьшей MAE на test
# (заведомо в пользу Prophet; используется только как самый строгий ориентир)
pk = [k for k in logF if k.startswith("prophet__")]
if len(pk) > 1:
    best = np.full_like(logF[pk[0]], np.nan)
    choice = {}
    for h in range(1, g.H + 1):
        maes = {}
        for k in pk:
            r = evaluate(g, ctx.Y[sample], np.exp(logF[k][sample]), splits=("test",), horizons=[h])
            if len(r):
                maes[k] = r.MAE.iloc[0]
        if maes:
            kb = min(maes, key=maes.get)
            choice[h] = labels[kb]
            best[:, :, h - 1] = logF[kb][:, :, h - 1]
    logF["prophet_best_test"] = best
    labels["prophet_best_test"] = "Prophet: лучшая конфигурация по test (в пользу Prophet)"
    json.dump(choice, open(tables / "prophet_best_by_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def eval_on(mask, keys):
    rows = []
    for k in keys:
        r = evaluate(g, ctx.Y[mask], np.exp(logF[k][mask]))
        r.insert(0, "model", labels[k])
        rows.append(r)
    return pd.concat(rows)


# модели на всей панели — с прогнозами и вне выборки Prophet
panel_keys = [k for k in logF if sample.all() or np.isfinite(logF[k][~sample]).any()]
res_full = eval_on(full_mask, panel_keys)
res_full.to_csv(tables / "forecast_metrics_full.csv", index=False)
res_sample = eval_on(sample, list(logF))
res_sample.to_csv(tables / "forecast_metrics_sample.csv", index=False)

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)
for name, res in [("вся панель, 12 096 рядов", res_full), (f"выборка Prophet, {sample.sum()} рядов", res_sample)]:
    for split in ("val", "test"):
        print(f"\n=== MAE, {split} — {name} ===")
        print(pivot(res, "MAE", split).round(0))
print("\n=== R² г/г прироста, test — выборка ===")
print(pivot(res_sample, "R2_yoy", "test").round(3))
print("\n=== R² уровней, test — выборка ===")
print(pivot(res_sample, "R2", "test").round(4))

# --- значимость: ансамбль против лучшей конфигурации Prophet --------------------------------
prophets = [k for k in logF if k.startswith("prophet")]   # все конфигурации + лучшая по test
if prophets:
    tid = ctx.panel.territory[sample]
    rows = []
    for pk in prophets:
        for h in g.horizons:
            for split in ("test", "all"):
                r = compare(g, ctx.Y[sample], np.exp(logF["ensemble"][sample]), np.exp(logF[pk][sample]),
                            tid, h, split=None if split == "all" else split)
                rows.append({"prophet": labels[pk], "split": split, **r})
    sig = pd.DataFrame(rows)
    sig.to_csv(tables / "ensemble_vs_prophet.csv", index=False)
    print("\n=== Ансамбль против Prophet (кластерный бутстреп по МО, DM по месяцам) ===")
    print(sig.round(3).to_string(index=False))

# --- Prophet по умолчанию на всей панели (12 096 рядов) ---------------------------------------
if "prophet__default_full" in logF:
    rows = []
    for h in g.horizons:
        for split in ("test", "all"):
            r = compare(g, ctx.Y, np.exp(logF["ensemble"]), np.exp(logF["prophet__default_full"]),
                        ctx.panel.territory, h, split=None if split == "all" else split)
            rows.append({"prophet": labels["prophet__default_full"], "split": split, **r})
    sigf = pd.DataFrame(rows)
    sigf.to_csv(tables / "ensemble_vs_prophet_full_panel.csv", index=False)
    print("\n=== Ансамбль против Prophet по умолчанию на всей панели ===")
    print(sigf[sigf.split == "test"].round(3).to_string(index=False))

# --- по категориям -------------------------------------------------------------------------
rows = []
for c, cat in enumerate(CATEGORIES):
    m = ctx.cat_idx == c
    for k in ["ensemble"] + prophets + ["d__last"]:
        if k in logF:
            mm = m & sample if k.startswith("prophet") else m & sample
            r = evaluate(g, ctx.Y[mm], np.exp(logF[k][mm]), splits=("test",))
            r.insert(0, "model", labels[k])
            r.insert(0, "category", cat)
            rows.append(r)
bycat = pd.concat(rows)
bycat.to_csv(tables / "forecast_by_category.csv", index=False)
print("\n=== MAE по категориям, test, выборка ===")
print(bycat.pivot_table(index=["category", "model"], columns="h", values="MAE").round(0))

# --- МО с неполной историей: панельный фолбэк -------------------------------------------------
from sbx.backtest import Grid, run_local  # noqa: E402
from sbx.data import load_panel  # noqa: E402
from sbx.models import baselines as B  # noqa: E402
from sbx.national import decompose  # noqa: E402

allp = load_panel(ctx.cfg)
_, d_all, cat_all = decompose(allp)            # n — по полным МО, как в основной панели
part = ~allp.full_mask
F = run_local(B.naive, d_all[part], g)         # последнее наблюдённое отклонение
Yp = allp.y.to_numpy(float)[part]
res_part = evaluate(g, Yp, np.exp(nb[cat_all[part]] + F))
res_part.insert(0, "model", "Панельный фолбэк (неполные ряды)")
res_part.to_csv(tables / "forecast_metrics_partial.csv", index=False)
print(f"\n=== Неполные ряды: {part.sum()} рядов, {len(np.unique(allp.territory[part]))} МО ===")
print(res_part[res_part.split != "val"].round(3).to_string(index=False))
