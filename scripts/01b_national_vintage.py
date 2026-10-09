"""Шаг 1b. Чувствительность национальной компоненты к запаздыванию и пересмотрам национального ряда.

Национальный ряд портала скачан в 2026 г. Чтобы оценить риск «знания будущего» через пересмотры
и задержку публикации, повторяем прогноз n со сдвигом национального ряда на 1 и 2 месяца назад
(на origin используются только данные, опубликованные минимум 1–2 месяца назад) и сравниваем MAE
при истинной локальной компоненте.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import load_national, path  # noqa: E402
from sbx.national import forecast_national, national_log  # noqa: E402
from sbx.pipeline import pivot, setup  # noqa: E402

ctx = setup()
g, cfg = ctx.grid, ctx.cfg
L, off = national_log(load_national(cfg), ctx.panel.months, cfg["national"]["mapping"])
tidx = g.target_idx()
true_d = np.where(tidx[None] >= 0, ctx.d[:, tidx.clip(0)], np.nan)
rows = []
for lag in (0, 1, 2):
    F = (forecast_national(ctx.n, g, "snaive_own") + forecast_national(ctx.n, g, "snaive_nat", L=L, offset=off, nat_lag=lag)
         + forecast_national(ctx.n, g, "nat_profile", L=L, offset=off, years=3, nat_lag=lag)) / 3
    rows.append(ctx.evaluate(ctx.compose(F, true_d), f"среднее трёх, лаг национального ряда {lag} мес."))
F_own = forecast_national(ctx.n, g, "snaive_own")
rows.append(ctx.evaluate(ctx.compose(F_own, true_d), "без национального ряда (только собственный рост)"))
res = pd.concat(rows)
res.to_csv(path(cfg, "tables") / "national_vintage_sensitivity.csv", index=False)
pd.set_option("display.width", 200)
for s in ("val", "test"):
    print(s); print(pivot(res, "MAE", s).round(0))
