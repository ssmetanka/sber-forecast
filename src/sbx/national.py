"""Декомпозиция log y = n + d и модели национальной компоненты n.

n[c, t] — медиана log-расходов по полным МО категории c в месяце t: общая сезонность и инфляция.
d[s, t] — локальное отклонение ряда s от национальной компоненты своей категории.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Grid
from .data import CATEGORIES, Panel


def decompose(panel: Panel, full_mask: np.ndarray | None = None):
    """Возвращает (n: C × T, d: S × T, cat_idx: S). n считается только по полным МО."""
    ly = panel.ly
    cat_idx = np.array([CATEGORIES.index(c) for c in panel.category])
    mask = panel.full_mask if full_mask is None else full_mask
    n = np.stack([np.nanmedian(ly[(cat_idx == c) & mask], axis=0) for c in range(len(CATEGORIES))])
    d = ly - n[cat_idx]
    return n, d, cat_idx


def national_log(national: pd.DataFrame, months: pd.PeriodIndex, mapping: dict) -> tuple[np.ndarray, int]:
    """Лог национальных рядов портала для каждой категории на шкале от 2018-12 до конца панели.

    Возвращает (L: C × T_ext, offset), где столбец offset + t соответствует месяцу months[t].
    """
    ext = pd.period_range(national.index.min(), months[-1], freq="M")
    L = np.stack([np.log(national[mapping[c]].reindex(ext).to_numpy()) for c in CATEGORIES])
    offset = ext.get_loc(months[0])
    return L, offset


def _own_growth(n: np.ndarray, o: int) -> np.ndarray:
    """Годовой лог-прирост n на origin; если года истории нет — аннуализированный тренд H2 к H1."""
    if o >= 12:
        return n[:, o] - n[:, o - 12]
    k = (o + 1) // 2
    return 12 / k * (n[:, o + 1 - k: o + 1].mean(1) - n[:, :k].mean(1))


def forecast_national(n: np.ndarray, grid: Grid, method: str, L: np.ndarray | None = None,
                      offset: int = 0, years: int = 3, damping: float = 1.0, nat_lag: int = 0) -> np.ndarray:
    """Прогноз n: массив C × O × H (лог).

    method:
      snaive_own   — n[t-12] + собственный годовой прирост на origin (тренд для первого origin);
      snaive_nat   — n[t-12] + годовой прирост национального ряда портала на origin;
      nat_profile  — n[o] + средняя за `years` прошлых лет динамика национального ряда от
                     месяца origin до целевого месяца (сезонность + типичный рост);
      nat_profile_g— то же, но типичный рост заменён текущим годовым ростом национального ряда.
    Все варианты используют только данные ≤ origin.
    """
    C = n.shape[0]
    F = np.full((C, len(grid.origins), grid.H), np.nan)
    for j, o in enumerate(grid.origins):
        for h in range(1, min(grid.H, grid.T - 1 - o) + 1):
            t = o + h
            if t < grid.first_target:          # цель вне окна оценки (ранний origin)
                continue
            if method == "snaive_own":
                F[:, j, h - 1] = n[:, t - 12] + damping * _own_growth(n, o)
            elif method == "snaive_nat":
                a = o - nat_lag                    # последний опубликованный месяц на origin (as-of)
                g = L[:, offset + a] - L[:, offset + a - 12]
                F[:, j, h - 1] = n[:, t - 12] + damping * g
            elif method in ("nat_profile", "nat_profile_g"):
                deltas, growths = [], []
                for k in range(1, years + 1 + (1 if nat_lag else 0)):
                    a, b = offset + o - 12 * k, offset + t - 12 * k
                    if t - 12 * k > o - nat_lag:   # месяц ещё не опубликован на origin — берём год раньше
                        continue
                    if a - 12 < 0 or len(deltas) == years:
                        break
                    deltas.append(L[:, b] - L[:, a])
                    growths.append(L[:, a] - L[:, a - 12])
                delta = np.mean(deltas, axis=0)
                if method == "nat_profile_g":
                    g_now = L[:, offset + o - nat_lag] - L[:, offset + o - nat_lag - 12]
                    delta = delta + h / 12 * damping * (g_now - np.mean(growths, axis=0))
                F[:, j, h - 1] = n[:, o] + delta
            else:
                raise ValueError(method)
    return F
