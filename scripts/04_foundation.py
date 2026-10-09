"""Шаг 4. Foundation models zero-shot.

На локальной компоненте d — все ряды и все origin. Абляция на сыром log y — на выборке МО
из шага 2 (та же, что у Prophet), чтобы показать, что выигрыш даёт подготовка входа.

    python scripts/04_foundation.py chronos2_joint chronos_bolt timesfm
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import save_forecast  # noqa: E402
from sbx.data import load_config, path  # noqa: E402
from sbx.models.foundation import build  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

torch.set_num_threads(int(load_config("configs/models.yaml")["foundation"].get("threads", 8)))
ctx = setup()
g = ctx.grid
fcfg = load_config("configs/models.yaml")["foundation"]
mask_file = ctx.out / "prophet_sample_mask.npy"
mask = np.load(mask_file) if mask_file.exists() else np.ones(len(ctx.d), bool)
qdir = path(ctx.cfg, "cache") / "fm_quantiles"
qdir.mkdir(parents=True, exist_ok=True)

for name in sys.argv[1:] or ["chronos2_joint", "chronos_bolt", "timesfm"]:
    key = name.replace("_joint", "")              # chronos2_joint берёт настройки chronos2
    kw = {k: v for k, v in fcfg.get(key, {}).items() if k in ("model_id", "batch_size")}
    model = build(name, **kw)
    for target, X, rows in [("d", ctx.d, np.ones(len(ctx.d), bool)), ("raw", ctx.panel.ly, mask)]:
        if target == "raw" and (not fcfg["raw_ablation"] or name == "chronos2_joint"):
            continue
        # чекпоинт по каждому origin: при перезапуске готовые origin пропускаются
        ck = qdir / f"ckpt_{target}_{name}.npz"
        if ck.exists():
            z = np.load(ck)
            F, Q, done = z["F"], z["Q"], list(z["done"])
        else:
            F = g.empty(len(X))
            Q = np.full(F.shape + (3,), np.nan, dtype=np.float32)
            done = []
        t0 = time.time()
        for j, o in enumerate(g.origins):
            if j in done:
                continue
            H = min(g.H, g.T - 1 - o)
            med, q = model.forecast(X[rows, : o + 1], H)
            F[rows, j, :H] = med[:, :H]
            Q[rows, j, :H] = q[:, :H]
            done.append(j)
            np.savez(ck, F=F, Q=Q, done=np.array(done))
            print(f"[{name}/{target}] origin {ctx.panel.months[o]}: {time.time() - t0:.0f} с", flush=True)
        save_forecast(F, ctx.out, f"{target}__fm_{name}")
        if target == "d":
            np.save(qdir / f"{name}.npy", Q)
