"""Шаг 7b. Откуда ошибка: национальная компонента или локальная.

Подставляем в прогноз ансамбля истинную национальную компоненту n вместо прогнозной:
MAE(ŷ) − MAE(ŷ | истинное n) — вклад ошибки прогноза n (общероссийской динамики).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import evaluate  # noqa: E402
from sbx.data import path  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
g = ctx.grid
nb = ctx.load("national_best")
ens = ctx.load("ensemble")
tidx = g.target_idx()
n_true = np.where(tidx[None] >= 0, ctx.n[:, tidx.clip(0)], np.nan)
oracle = ens - nb[ctx.cat_idx] + n_true[ctx.cat_idx]
rows = []
for name, F in [("ансамбль", ens), ("ансамбль с истинной n", oracle)]:
    r = evaluate(g, ctx.Y, np.exp(F), splits=("test", "all"))
    r.insert(0, "model", name)
    rows.append(r)
res = pd.concat(rows)
piv = res.pivot_table(index=["split", "model"], columns="h", values="MAE")
share = {}
for split in ("test", "all"):
    a, b = piv.loc[(split, "ансамбль")], piv.loc[(split, "ансамбль с истинной n")]
    share[split] = (1 - b / a).round(3)
out = piv.round(0)
print(out)
print("доля ошибки ансамбля из-за прогноза национальной компоненты:")
print(pd.DataFrame(share).T)
out.to_csv(path(ctx.cfg, "tables") / "error_decomposition.csv")
pd.DataFrame(share).T.to_csv(path(ctx.cfg, "tables") / "error_share_national.csv")
