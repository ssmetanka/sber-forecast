"""Простые модели. Все функции: (история S × t, H) -> прогноз S × H; NaN в истории допустимы."""
from __future__ import annotations

import numpy as np


def _last_valid(X: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(X), np.arange(X.shape[1])[None, :], -1).max(1)
    out = np.full(X.shape[0], np.nan)
    ok = idx >= 0
    out[ok] = X[np.where(ok)[0], idx[ok]]
    return out


def naive(X, H):
    return np.repeat(_last_valid(X)[:, None], H, axis=1)


def mean_k(X, H, k=3):
    m = np.nanmean(X[:, -k:], axis=1)
    m = np.where(np.isfinite(m), m, _last_valid(X))
    return np.repeat(m[:, None], H, axis=1)


def seasonal_naive(X, H):
    """x[t-12]; при h ≤ 12 сезонный аналог всегда есть в истории."""
    t = X.shape[1]
    return np.stack([X[:, t - 1 + h - 12] for h in range(1, H + 1)], axis=1)


def snaive_growth(X, H):
    """Для log y: x[t-12] + годовой прирост на origin (тренд H2/H1 при коротком ряде)."""
    t = X.shape[1]
    if t >= 13:
        g = X[:, -1] - X[:, -13]
    else:
        k = t // 2
        g = 12 / k * (np.nanmean(X[:, -k:], 1) - np.nanmean(X[:, :k], 1))
    return seasonal_naive(X, H) + g[:, None]


def ses(X, H, alpha=0.5):
    """Простое экспоненциальное сглаживание уровня (векторно по всем рядам)."""
    level = None
    for j in range(X.shape[1]):
        x = X[:, j]
        if level is None:
            level = x.copy()
            continue
        level = np.where(np.isfinite(x), np.where(np.isfinite(level), alpha * x + (1 - alpha) * level, x), level)
    return np.repeat(level[:, None], H, axis=1)


def level_season(X, H, alpha_season=0.3, k=3):
    """Текущий уровень (среднее за k мес) + сжатая собственная сезонность прошлого года.

    Для отклонения d: d̂[t] = mean(d[o-k+1..o]) + α · (d[t-12] − mean последних 12 мес).
    """
    lvl = mean_k(X, 1, k)[:, 0]
    base = np.nanmean(X[:, -12:], axis=1)
    sn = seasonal_naive(X, H)
    season = np.nan_to_num(sn - base[:, None])
    return lvl[:, None] + alpha_season * season
