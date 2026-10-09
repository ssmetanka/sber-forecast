"""Rolling-origin бэктест и метрики.

Прогнозы всех моделей хранятся как массив F[S, O, H] (ряд × origin × шаг), NaN там, где цель
выходит за конец данных. Модель получает только историю до origin включительно, поэтому
утечка будущего исключена конструктивно.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import Panel


class Grid:
    """Сетка origin × h для панели."""

    def __init__(self, panel: Panel, cfg: dict):
        b = cfg["backtest"]
        self.months = panel.months
        self.T = len(self.months)
        o0, o1 = (panel.month_idx(m) for m in b["origins"])
        self.origins = np.arange(o0, o1 + 1)
        self.H = b["max_h"]
        self.horizons = b["horizons"]
        self.val = tuple(panel.month_idx(m) for m in b["val_targets"])
        self.test = tuple(panel.month_idx(m) for m in b["test_targets"])
        self.groups = b["horizon_groups"]
        # оцениваются только цели внутри validation ∪ test; ранние origin (короткая история)
        # дают прогнозы лишь на те h, где цель попадает в это окно (так h = 12 получает 6 целей)
        self.first_target, self.last_target = self.val[0], self.test[1]

    def empty(self, S: int) -> np.ndarray:
        return np.full((S, len(self.origins), self.H), np.nan)

    def target_idx(self) -> np.ndarray:
        """Индекс целевого месяца для каждой пары (origin, h); -1, если за пределами данных."""
        t = self.origins[:, None] + np.arange(1, self.H + 1)[None, :]
        return np.where((t < self.T) & (t >= self.first_target), t, -1)

    def mask(self, F: np.ndarray) -> np.ndarray:
        """NaN в ячейках (origin, h), цель которых вне окна оценки."""
        F = F.copy()
        F[:, self.target_idx() < 0] = np.nan
        return F


def run_local(fn: Callable[[np.ndarray, int], np.ndarray], X: np.ndarray, grid: Grid, **kw) -> np.ndarray:
    """Прогоняет модель `fn(history, H) -> (S, H)` по всем origin.

    `X` — матрица S × T (например, локальная компонента d или log y); в fn уходит X[:, :o+1].
    """
    F = grid.empty(X.shape[0])
    for j, o in enumerate(grid.origins):
        H = min(grid.H, grid.T - 1 - o)
        F[:, j, :H] = fn(X[:, : o + 1], H, **kw)[:, :H]
    return grid.mask(F)


def save_forecast(F: np.ndarray, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.npy", F.astype(np.float32))


def load_forecast(out: Path, name: str) -> np.ndarray:
    return np.load(out / f"{name}.npy").astype(float)


# ----------------------------------------------------------------------------- метрики

def _pairs(grid: Grid, Y: np.ndarray, Yhat: np.ndarray, h: int, split: str | None):
    """Собирает пары (факт, прогноз, прошлогодний факт) для горизонта h и выборки split."""
    j_h = h - 1
    t = grid.origins + h
    ok = (t < grid.T) & (t >= grid.first_target) & (t <= grid.last_target)
    if split == "val":
        ok &= (t >= grid.val[0]) & (t <= grid.val[1])
    elif split == "test":
        ok &= (t >= grid.test[0]) & (t <= grid.test[1])
    js = np.where(ok)[0]
    yt = np.concatenate([Y[:, t[j]] for j in js]) if len(js) else np.array([])
    yp = np.concatenate([Yhat[:, j, j_h] for j in js]) if len(js) else np.array([])
    yl = np.concatenate([Y[:, t[j] - 12] for j in js]) if len(js) else np.array([])
    return yt, yp, yl


def metrics(yt: np.ndarray, yp: np.ndarray, ylag: np.ndarray | None = None) -> dict:
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    err = yt - yp
    res = {
        "MAE": np.abs(err).mean(),
        "R2": 1 - (err ** 2).sum() / ((yt - yt.mean()) ** 2).sum(),
        "WAPE": np.abs(err).sum() / np.abs(yt).sum(),
        "sMAPE": np.mean(2 * np.abs(err) / (np.abs(yt) + np.abs(yp))),
        "n": int(m.sum()),
    }
    if ylag is not None:
        yl = ylag[m]
        ok = yl > 0
        g, gp = np.log(yt[ok] / yl[ok]), np.log(yp[ok] / yl[ok])
        res["R2_yoy"] = 1 - ((g - gp) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    return res


def evaluate(grid: Grid, Y: np.ndarray, Yhat: np.ndarray, splits=("val", "test", "all"),
             horizons=None) -> pd.DataFrame:
    rows = []
    for split in splits:
        for h in horizons or grid.horizons:
            yt, yp, yl = _pairs(grid, Y, Yhat, h, None if split == "all" else split)
            if len(yt) == 0:
                continue
            rows.append({"split": split, "h": h, **metrics(yt, yp, yl)})
    return pd.DataFrame(rows)


def abs_errors(grid: Grid, Y: np.ndarray, Yhat: np.ndarray, h: int, split: str | None = None) -> np.ndarray:
    """Матрица абсолютных ошибок S × (число origin) для теста Дибольда–Мариано и бутстрепа."""
    t = grid.origins + h
    ok = (t < grid.T) & (t >= grid.first_target) & (t <= grid.last_target)
    if split == "val":
        ok &= (t >= grid.val[0]) & (t <= grid.val[1])
    elif split == "test":
        ok &= (t >= grid.test[0]) & (t <= grid.test[1])
    js = np.where(ok)[0]
    if len(js) == 0:
        return None
    return np.stack([np.abs(Y[:, t[j]] - Yhat[:, j, h - 1]) for j in js], axis=1)
