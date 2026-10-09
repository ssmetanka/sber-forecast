"""Шаг 7e. Устойчивость к пересмотрам национального ряда портала: тот же ансамбль, но национальная
компонента прогнозируется только по собственной панели (сезонная наивная с ростом), без рядов портала.

Это не альтернативный основной вариант: на validation он заметно хуже (выбор остаётся за средним
трёх моделей), а здесь показывает, сколько преимущества над Prophet остаётся без портала совсем.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import ensemble  # noqa: E402
from sbx.data import path  # noqa: E402
from sbx.national import forecast_national  # noqa: E402
from sbx.pipeline import guard_prophet, pivot, setup  # noqa: E402
from sbx.stats import compare  # noqa: E402

ctx = setup()
g = ctx.grid
POOL = ["d__last", "d__ses05", "d__level_season03", "d__lgbm", "d__lgbm_v2", "d__fm_chronos2_joint",
        "d__fm_chronos_bolt", "d__fm_chronos_bolt_small", "d__fm_chronos2", "d__fm_timesfm", "d__fm_chronos2_joint_lora_c"]
own = forecast_national(ctx.n, g, "snaive_own")
logF = {k: own[ctx.cat_idx] + ctx.load(k) for k in POOL}
E = ensemble.apply(logF, ensemble.caruana(logF, ctx.Y, g, rounds=30), g)
res = ctx.evaluate(E, "ансамбль без рядов портала")
P = guard_prophet(ctx, ctx.load("prophet__default_full"))
rows = []
for h in g.horizons:
    r = compare(g, ctx.Y, E, np.exp(P), ctx.panel.territory, h)          # apply → уровни, Prophet в логах
    rows.append({"h": h, "MAE без портала": r["MAE_a"], "MAE Prophet": r["MAE_b"], "разница, %": 100 * r["rel"],
                 "ДИ низ": r["ci_low"], "ДИ верх": r["ci_high"], "месяцев лучше, %": 100 * r["share_months_better"],
                 "p (знак)": r["sign_p"]})
out = pd.DataFrame(rows)
val = pivot(res, "MAE", "val")
out["MAE validation"] = [val.iloc[0].get(h, np.nan) for h in g.horizons]
out.to_csv(path(ctx.cfg, "tables") / "ensemble_no_portal.csv", index=False)
print(out.round(3).to_string(index=False))
