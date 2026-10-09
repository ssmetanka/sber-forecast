"""Переносит прогнозы, посчитанные на старой сетке (origins 2023-12…2024-11), в контрольные точки
новой расширенной сетки (origins 2023-06…2024-11), чтобы тяжёлые модели досчитывали только новые origin.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import path  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
g = ctx.grid
old_dir = ctx.out.parent / "forecasts_v1"
qdir = path(ctx.cfg, "cache") / "fm_quantiles"
old_first = ctx.panel.month_idx("2023-12")
shift = int(np.where(g.origins == old_first)[0][0])
done = list(range(shift, len(g.origins)))
print("сдвиг индексов origin:", shift)


def embed(old):
    new = np.full((old.shape[0], len(g.origins)) + old.shape[2:], np.nan, dtype=np.float32)
    new[:, shift:] = old
    return new


for name in ["chronos2_joint", "chronos_bolt"]:
    for target in ["d", "raw"]:
        f = old_dir / f"{target}__fm_{name}.npy"
        if not f.exists():
            continue
        F = embed(np.load(f))
        qf = qdir / f"{name}.npy"
        Q = embed(np.load(qf)) if (target == "d" and qf.exists()) else np.full(F.shape + (3,), np.nan, np.float32)
        np.savez(qdir / f"ckpt_{target}_{name}.npz", F=F, Q=Q, done=np.array(done))
        print("FM", name, target, "→ контрольная точка")
for cfg in ["default"]:
    f = old_dir / f"prophet__{cfg}.npy"
    if f.exists():
        np.savez(ctx.out / f"ckpt_prophet_{cfg}.npz", F=embed(np.load(f)), done=np.array(done))
        print("Prophet", cfg, "→ контрольная точка")
