"""Шаг 7f. Приложение к отчёту: все модели с местами на validation и test и дополнительные метрики.

1. `all_models_ranked.csv` — каждая модель × горизонт: MAE на validation и test и место среди моделей
   (вся панель; модели, посчитанные только на выборке МО, — отдельным блоком).
2. `extra_metrics.csv` — для ключевых моделей на test: MAE, MdAPE, смещение (знаковая ошибка
   в % от факта), WAPE, R² уровней, R² годового прироста. Считается по сохранённым прогнозам.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import path  # noqa: E402
from sbx.pipeline import guard_prophet, setup  # noqa: E402

ctx = setup()
g, Y = ctx.grid, ctx.Y
T = path(ctx.cfg, "tables")
HS = [1, 3, 6, 12]

# --- 1. все модели с местами ----------------------------------------------------------------
out = []
for fname, scope in [("forecast_metrics_full.csv", "вся панель"), ("forecast_metrics_sample.csv", "выборка 500 МО")]:
    m = pd.read_csv(T / fname)
    if scope != "вся панель":
        full_models = set(pd.read_csv(T / "forecast_metrics_full.csv").model)
        m = m[~m.model.isin(full_models)]
    for split in ("val", "test"):
        pv = m[m.split == split].pivot_table(index="model", columns="h", values="MAE")
        for h in HS:
            if h not in pv.columns:
                continue
            col = pv[h].dropna()
            for model, v in col.items():
                out.append({"область": scope, "модель": model, "split": split, "h": h, "MAE": v,
                            "место": int(col.rank(method="min")[model]), "моделей": len(col)})
allm = pd.DataFrame(out)
wide = allm.pivot_table(index=["область", "модель"], columns=["split", "h"], values=["MAE", "место"])
wide.columns = [f"{a} {s} h={h}" for a, s, h in wide.columns]
wide = wide.reset_index().sort_values(["область", "MAE test h=1"])
wide.to_csv(T / "all_models_ranked.csv", index=False)
print(f"моделей: {wide.shape[0]}")

# --- 2. дополнительные метрики ключевых моделей -------------------------------------------
N = np.load(path(ctx.cfg, "forecasts") / "national_best.npy")[ctx.cat_idx]       # лог национальной части
logF = {
    "Ансамбль": ctx.load("ensemble"),
    "TimesFM-2.5 на d": N + ctx.load("d__fm_timesfm"),
    "LightGBM на d": N + ctx.load("d__lgbm"),
    "Панель: уровень + сезонность d": N + ctx.load("d__level_season03"),
    "Prophet по умолчанию (вся панель)": guard_prophet(ctx, ctx.load("prophet__default_full")),
}
tidx = g.target_idx()
rows = []
for name, F in logF.items():
    for h in HS:
        j = np.where((tidx[:, h - 1] >= g.test[0]) & (tidx[:, h - 1] <= g.test[1]))[0]
        t = tidx[j, h - 1]
        y = Y[:, t]
        yh = np.exp(F[:, j, h - 1])
        y12 = Y[:, t - 12] if (t - 12 >= 0).all() else np.full_like(y, np.nan)
        ok = np.isfinite(y) & np.isfinite(yh) & (y > 0)
        e = yh[ok] - y[ok]
        r2 = 1 - np.sum(e ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
        ok2 = ok & np.isfinite(y12) & (y12 > 0)
        gy, gh = np.log(y[ok2] / y12[ok2]), np.log(yh[ok2] / y12[ok2])
        r2y = 1 - np.sum((gh - gy) ** 2) / np.sum((gy - gy.mean()) ** 2)
        rows.append({"модель": name, "h": h, "MAE": np.abs(e).mean(),
                     "MdAPE, %": 100 * np.median(np.abs(e) / y[ok]),
                     "смещение, %": 100 * e.sum() / y[ok].sum(),
                     "WAPE, %": 100 * np.abs(e).sum() / y[ok].sum(),
                     "R² уровней": r2, "R² г/г": r2y, "целевых месяцев": len(np.unique(t))})
ex = pd.DataFrame(rows)
ex.to_csv(T / "extra_metrics.csv", index=False)
pd.set_option("display.width", 220)
print(ex.round(3).to_string(index=False))
