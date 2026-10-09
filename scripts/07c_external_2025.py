"""Шаг 7c. Внешняя проверка на данных, которых не было при разработке: 2025 год.

Муниципальных данных за 2025 г. нет, но национальные ряды портала СберИндекса доступны до 2026 г.
Прогнозируем национальную компоненту из декабря 2024 г. (последний месяц данных) на 12 месяцев
тем же методом (среднее трёх моделей), используя только данные ≤ декабря 2024, и сравниваем
прогноз годового роста 2025/2024 с фактом национального ряда портала для сопоставимых категорий.

Оговорка: n — медиана по МО расходов на жителя, а портал — совокупные расходы страны; разница
включает динамику населения (≈ −0,3 % в год) и структурные сдвиги между МО.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import CATEGORIES, load_national, path  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
cfg = ctx.cfg
nat = load_national(cfg)
mapping = cfg["national"]["mapping"]
n = ctx.n
months = ctx.panel.months
o = len(months) - 1                                       # декабрь 2024
ext = pd.period_range(nat.index.min(), "2025-12", freq="M")
off = ext.get_loc(months[0])
L = np.stack([np.log(nat[mapping[c]].reindex(ext).to_numpy()) for c in CATEGORIES])

lag = cfg["national"].get("nat_lag", 1)                   # тот же as-of, что в основном расчёте
fc = np.zeros((len(CATEGORIES), 12))
for h in range(1, 13):
    t = o + h
    own = n[:, t - 12] + (n[:, o] - n[:, o - 12])
    natg = n[:, t - 12] + (L[:, off + o - lag] - L[:, off + o - lag - 12])
    ks = [k for k in range(1, 5) if t - 12 * k <= o - lag][:3]
    prof = n[:, o] + np.mean([L[:, off + t - 12 * k] - L[:, off + o - 12 * k] for k in ks], axis=0)
    fc[:, h - 1] = (own + natg + prof) / 3

rows = []
for c, cat in enumerate(CATEGORIES):
    pred_g = fc[c].mean() - n[c, 12:24].mean()            # прогноз: средний лог-прирост 2025 к 2024
    fact = L[c, off + 24: off + 36]
    fact_g = np.nanmean(fact) - np.nanmean(L[c, off + 12: off + 24]) if np.isfinite(fact).all() else np.nan
    rows.append({"категория МО": cat, "национальный ряд портала": mapping[cat],
                 "прогноз роста 2025/2024, %": 100 * (np.exp(pred_g) - 1),
                 "факт портала 2025/2024, %": 100 * (np.exp(fact_g) - 1),
                 "корреляция помесячного профиля 2025": np.corrcoef(fc[c] - fc[c].mean(), fact - fact.mean())[0, 1]
                 if np.isfinite(fact).all() else np.nan})
res = pd.DataFrame(rows)
res.to_csv(path(cfg, "tables") / "external_check_2025.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
