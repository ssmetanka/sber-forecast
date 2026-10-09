"""Шаг 7d. Связка «детектор → прогноз».

Если на origin рабочий детектор (BOCPD + новости) сигнализирует о недавнем разрыве, сглаживающие
модели медленно догоняют новый уровень. Правило: в рядах с тревогой на origin (или месяцем раньше)
локальная часть прогноза смещается к последнему наблюдённому отклонению:
    d̂' = (1 − w) · d̂_ансамбля + w · d[origin],
вес w выбирается по MAE на validation из {0, 0.25, 0.5, 0.75, 1}; на test — только отчёт.
Детектор использует только данные ≤ origin (онлайн), поэтому утечки нет.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import evaluate  # noqa: E402
from sbx.data import ROOT, load_config, path  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.news.process import hazard_multiplier  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
g, p = ctx.grid, ctx.panel
dcfg, ncfg = load_config("configs/detection.yaml"), load_config("configs/news.yaml")
mi = p.month_idx
artifact = tuple(mi(m) for m in ctx.cfg["data"]["artifact_months"])
x = D.levels(ctx.d, mi(dcfg["calib_end"]), artifact + tuple(a + 1 for a in artifact))
nf = ROOT / "data" / "news" / "news_monthly.parquet"
mult = hazard_multiplier(pd.read_parquet(nf), p.territory, p.months, ncfg) if nf.exists() else None
score = D.bocpd(x, hazard_mult=mult, **dcfg["params"]["bocpd"])
months = np.arange(mi(dcfg["monitor"][0]), mi(dcfg["monitor"][1]) + 1)
th = np.nanquantile(score[:, months], 0.97)
alarm = score > th

nb = ctx.load("national_best")
ens = ctx.load("ensemble")                               # лог-прогноз уровней ансамбля
d_ens = ens - nb[ctx.cat_idx]
F_last = np.stack([np.repeat(ctx.d[:, o][:, None], g.H, 1) for o in g.origins], axis=1)
flag = np.stack([alarm[:, o] | alarm[:, max(o - 1, 0)] for o in g.origins], axis=1)   # S × O
print(f"доля пар ряд×origin с тревогой: {flag.mean():.3%}")

rows, best = [], None
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    D_new = np.where(flag[:, :, None], (1 - w) * d_ens + w * F_last, d_ens)
    Y = ctx.compose(nb, D_new)
    r = evaluate(g, ctx.Y, Y, splits=("val", "test"))
    r.insert(0, "w", w)
    rows.append(r)
    v = r[r.split == "val"].MAE.mean()
    if best is None or v < best[0]:
        best = (v, w)
res = pd.concat(rows)
res.to_csv(path(ctx.cfg, "tables") / "detector_forecast_link.csv", index=False)
pd.set_option("display.width", 200)
print(res.pivot_table(index=["split", "w"], columns="h", values="MAE").round(1))
print(f"вес по validation: w = {best[1]}")
# MAE только на рядах с тревогой — там правило и должно работать
sub = []
for w in (0.0, best[1]):
    D_new = np.where(flag[:, :, None], (1 - w) * d_ens + w * F_last, d_ens)
    Y = np.exp(nb[ctx.cat_idx] + D_new)
    Yf = np.where(flag[:, :, None], Y, np.nan)
    r = evaluate(g, ctx.Y, Yf, splits=("test",))
    r.insert(0, "w", w)
    sub.append(r)
print("\nтолько ряды с тревогой на origin (test):")
print(pd.concat(sub).pivot_table(index="w", columns="h", values="MAE").round(1))
