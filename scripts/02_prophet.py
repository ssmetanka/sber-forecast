"""Шаг 2. Prophet (базовая модель) по всем origin.

Prophet стоит ~0,3 с CPU на подгонку, полная панель — 12 096 рядов × 12 origin × конфигурации,
это часы. Поэтому считаем на случайной выборке МО (все 6 категорий каждого МО, seed фиксирован),
а сравнение с нашими моделями делаем парно на той же выборке.

    python scripts/02_prophet.py [--sample 500] default yearly
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import save_forecast  # noqa: E402
from sbx.data import load_config  # noqa: E402
from sbx.models.prophet_m import prophet_forecast  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

args = sys.argv[1:]
mcfg = load_config("configs/models.yaml")["prophet"]
n_mo = mcfg["sample_mo"]
if "--sample" in args:
    i = args.index("--sample")
    n_mo = int(args[i + 1])
    del args[i: i + 2]
sub_mo = None
if "--subsample" in args:            # вложенная подвыборка МО для дорогих конфигураций
    i = args.index("--subsample")
    sub_mo = int(args[i + 1])
    del args[i: i + 2]
configs = [a for a in args if not a.startswith("--")] or mcfg["configs"]
args = [a for a in args]

ctx = setup()
g = ctx.grid
tid = ctx.panel.territory
rng = np.random.default_rng(ctx.cfg["seed"])
mos = np.unique(tid)
chosen = rng.choice(mos, min(n_mo, len(mos)), replace=False) if n_mo else mos
mask = np.isin(tid, chosen)
np.save(ctx.out / "prophet_sample_mask.npy", mask)
print(f"Prophet: {mask.sum()} рядов ({len(chosen)} МО)", flush=True)
if "--mask-only" in args:       # только выборка (нужна FM-абляции до запуска Prophet)
    sys.exit()

start = ctx.panel.months[0].to_timestamp().strftime("%Y-%m-%d")
base_mask = mask
full = "--full" in args             # вся панель: досчитываем ряды вне выборки, выборку берём из готового
for config in configs:
    mask = base_mask
    suffix = ""
    prefill = None
    if full:
        suffix = "_full"
        mask = ~base_mask
        prev = ctx.out / f"prophet__{config}.npy"
        if prev.exists():
            prefill = np.load(prev)
        print(f"[{config}] вся панель: досчитываем {mask.sum()} рядов вне выборки", flush=True)
    if sub_mo and config in mcfg.get("subsample_configs", []):
        mask = base_mask & np.isin(tid, chosen[:sub_mo])
        print(f"[{config}] подвыборка: {mask.sum()} рядов ({sub_mo} МО)", flush=True)
    X = (ctx.d if config == "local" else ctx.panel.ly)[mask]
    ck = ctx.out / f"ckpt_prophet_{config}{suffix}.npz"  # чекпоинт по origin для перезапуска
    if ck.exists():
        z = np.load(ck)
        F, done = z["F"], list(z["done"])
    else:
        F, done = g.empty(len(tid)), []
    t0 = time.time()
    for j, o in enumerate(g.origins):
        if j in done:
            continue
        H = min(g.H, g.T - 1 - o)
        F[mask, j, :H] = prophet_forecast(X[:, : o + 1], H, config, start, n_jobs=mcfg["n_jobs"], chunk=50)
        done.append(j)
        np.savez(ck, F=F, done=np.array(done))
        print(f"[{config}] origin {ctx.panel.months[o]} готов, {time.time() - t0:.0f} с", flush=True)
    if prefill is not None:
        F[base_mask] = prefill[base_mask]
    save_forecast(F, ctx.out, f"prophet__{config}{suffix}")
