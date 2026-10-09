"""Быстрый пересчёт главных чисел из сохранённых прогнозов (~1–2 мин, без обучения моделей).

Берёт outputs/forecasts/ensemble.npy и prophet__default_full.npy, заново считает MAE на test,
выигрыш против Prophet, двусторонний бутстреп и знаковый тест и сверяет с outputs/tables.

    python tools/quick_check.py          # код выхода 0 — пересчёт совпал с таблицами
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbx.pipeline import pivot, setup  # noqa: E402
from sbx.stats import compare  # noqa: E402

ctx = setup()
g = ctx.grid


def guard(F):
    """Та же страховка Prophet, что в 07_evaluate.py: прогноз дальше ×3 за пределами истории ряда
    до origin заменяется последним значением (в пользу Prophet)."""
    ly, out = ctx.panel.ly, F.copy()
    for j, o in enumerate(g.origins):
        hist = ly[:, : o + 1]
        lo, hi = np.nanmin(hist, 1) - np.log(3), np.nanmax(hist, 1) + np.log(3)
        bad = np.isfinite(F[:, j]) & ((F[:, j] < lo[:, None]) | (F[:, j] > hi[:, None]))
        out[:, j] = np.where(bad, ly[:, o][:, None], F[:, j])
    return out


E, P = np.exp(ctx.load("ensemble")), np.exp(guard(ctx.load("prophet__default_full")))
mae = pivot(ctx.evaluate(E, "ансамбль"), "MAE", "test").iloc[0]
tab = pd.read_csv(ROOT / "outputs/tables/ensemble_vs_prophet_full_panel.csv")
tab = tab[tab.split == "test"].set_index("h")
bad = 0
print(f"{'h':>3} {'MAE':>8} {'Prophet':>8} {'разница':>8} {'95% ДИ, ₽':>18} {'мес. лучше':>10} {'p':>6}  сверка")
for h in (1, 3, 6, 12):
    r = compare(g, ctx.Y, E, P, ctx.panel.territory, h)
    ok = abs(r["MAE_a"] - tab.loc[h, "MAE_a"]) < 0.5 and abs(r["rel"] - tab.loc[h, "rel"]) < 1e-3 \
        and abs(mae[h] - r["MAE_a"]) < 0.5
    bad += not ok
    print(f"{h:>3} {r['MAE_a']:>8.1f} {r['MAE_b']:>8.1f} {100 * r['rel']:>7.1f}% "
          f"{r['ci_low']:>8.0f} … {r['ci_high']:<6.0f} {100 * r['share_months_better']:>9.0f}% {r['sign_p']:>6.3f}  "
          f"{'OK' if ok else 'РАСХОЖДЕНИЕ'}")
print("всё совпало" if not bad else f"расхождений: {bad}")
sys.exit(1 if bad else 0)
