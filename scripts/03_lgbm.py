"""Шаг 3. Глобальный LightGBM на локальной компоненте d.

    python scripts/03_lgbm.py            # все origin
    python scripts/03_lgbm.py --quick    # последний origin, быстрая проверка
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import save_forecast  # noqa: E402
from sbx.data import load_config, path  # noqa: E402
from sbx.features import FeatureBuilder  # noqa: E402
from sbx.models.lgbm import lgbm_forecast  # noqa: E402
from sbx.pipeline import pivot, setup  # noqa: E402

ctx = setup()
variant = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--variant=")), "lgbm")
mcfg = load_config("configs/models.yaml")[variant]
fb = FeatureBuilder(ctx, ctx.cfg["data"]["artifact_months"])
grid = ctx.grid
if "--quick" in sys.argv:
    grid.origins = grid.origins[-1:]
t0 = time.time()
F, imp = lgbm_forecast(fb, grid, params=mcfg["params"], num_rounds=mcfg["num_rounds"],
                       max_rows=mcfg["max_rows"], seed=ctx.cfg["seed"], drop=mcfg.get("drop", []))
print(f"готово за {time.time() - t0:.0f} с")
if "--quick" in sys.argv:
    sys.exit()
save_forecast(F, ctx.out, f"d__{variant}")
imp.to_csv(path(ctx.cfg, "tables") / f"{variant}_importance.csv")
print(imp.head(15).round(0))
nb = ctx.load("national_best")
res = pd.concat([ctx.evaluate(ctx.compose(nb, F), f"d:{variant}"),
                 ctx.evaluate(ctx.compose(nb, ctx.load("d__last")), "d:last")])
print(pivot(res, "MAE", "val").round(0))
print(pivot(res, "MAE", "test").round(0))
