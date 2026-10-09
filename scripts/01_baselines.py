"""Шаг 1. Национальная компонента и простые модели.

1) Сравнивает модели национальной компоненты n при истинном d (изолирует ошибку n).
2) Выбирает лучшую на validation и сохраняет её прогноз `national_best`.
3) Прогоняет простые модели на сыром log y и на локальной компоненте d.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import run_local, save_forecast  # noqa: E402
from sbx.data import load_national, path  # noqa: E402
from sbx.models import baselines as B  # noqa: E402
from sbx.national import forecast_national, national_log  # noqa: E402
from sbx.pipeline import pivot, setup  # noqa: E402

ctx = setup()
g, cfg = ctx.grid, ctx.cfg
L, off = national_log(load_national(cfg), ctx.panel.months, cfg["national"]["mapping"])

# --- 1. национальная компонента при истинном d -----------------------------------------
true_d = np.full((ctx.d.shape[0], len(g.origins), g.H), np.nan)
tidx = g.target_idx()
for j in range(len(g.origins)):
    for h in range(g.H):
        if tidx[j, h] >= 0:
            true_d[:, j, h] = ctx.d[:, tidx[j, h]]

nat_variants = {
    "snaive_own": dict(method="snaive_own"),
    "snaive_own_d05": dict(method="snaive_own", damping=0.5),
    "snaive_nat": dict(method="snaive_nat"),
    "nat_profile_y1": dict(method="nat_profile", years=1),
    "nat_profile_y3": dict(method="nat_profile", years=3),
    "nat_profile_g_y3": dict(method="nat_profile_g", years=3),
    "nat_profile_g05_y3": dict(method="nat_profile_g", years=3, damping=0.5),
}
N = {k: forecast_national(ctx.n, g, L=L, offset=off, **v) for k, v in nat_variants.items()}
res_n = pd.concat([ctx.evaluate(ctx.compose(F, true_d), f"N:{k}") for k, F in N.items()])

# Выбор по категориям на validation нестабилен: на h=6 там один origin, на h=12 — ни одного.
# Поэтому берём равновзвешенное среднее трёх разных по природе моделей (прогнозная комбинация):
# собственная сезонность + свой рост, своя сезонность + рост национального ряда,
# сезонный профиль национального ряда за 3 года. На validation оно лучшее на h=3 и h=6.
# основной вариант — консервативный as-of: национальный ряд портала с лагом публикации nat_lag
lag = cfg["national"].get("nat_lag", 1)
N["avg3"] = (N["snaive_own"] + forecast_national(ctx.n, g, "snaive_nat", L=L, offset=off, nat_lag=lag)
             + forecast_national(ctx.n, g, "nat_profile", L=L, offset=off, years=3, nat_lag=lag)) / 3
res_n = pd.concat([res_n, ctx.evaluate(ctx.compose(N["avg3"], true_d), "N:avg3")])
best = N["avg3"]
save_forecast(best, ctx.out, "national_best")
oracle = ctx.compose(ctx.n[:, tidx.clip(0)], true_d)
res_n = pd.concat([res_n, ctx.evaluate(oracle, "N:oracle")])

# --- 2. простые модели ---------------------------------------------------------------
res = []
ly = ctx.panel.ly
for name, fn in {"raw:naive": B.naive, "raw:snaive": B.seasonal_naive,
                 "raw:snaive_growth": B.snaive_growth}.items():
    F = run_local(fn, ly, g)
    save_forecast(F, ctx.out, name.replace(":", "__"))
    res.append(ctx.evaluate(np.exp(F), name))

local = {
    "d:last": (B.naive, {}),
    "d:mean3": (B.mean_k, {"k": 3}),
    "d:ses05": (B.ses, {"alpha": 0.5}),
    "d:level_season03": (B.level_season, {"alpha_season": 0.3}),
    "d:level_season05": (B.level_season, {"alpha_season": 0.5}),
}
for name, (fn, kw) in local.items():
    D = run_local(fn, ctx.d, g, **kw)
    save_forecast(D, ctx.out, name.replace(":", "__"))
    res.append(ctx.evaluate(ctx.compose(best, D), name))
res = pd.concat(res)

allres = pd.concat([res_n, res])
allres.to_csv(path(cfg, "tables") / "baselines.csv", index=False)
pd.set_option("display.width", 200)
for split in ("val", "test"):
    print(f"\n=== MAE, {split} ===")
    print(pivot(allres, "MAE", split).round(0))
print("\n=== R2_yoy, test ===")
print(pivot(allres, "R2_yoy", "test").round(3))
