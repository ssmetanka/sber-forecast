"""Общий контекст запуска: конфиг, панель, сетка, декомпозиция, сборка прогноза."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Grid, evaluate, load_forecast
from .data import Panel, load_config, load_panel, path
from .national import decompose


@dataclass
class Context:
    cfg: dict
    panel: Panel          # только полные ряды (основной оценочный набор)
    grid: Grid
    n: np.ndarray         # C × T
    d: np.ndarray         # S × T
    cat_idx: np.ndarray   # S
    Y: np.ndarray         # уровни S × T

    @property
    def out(self):
        return path(self.cfg, "forecasts")

    def compose(self, Nhat: np.ndarray, Dhat: np.ndarray) -> np.ndarray:
        """Уровень ŷ = exp(n̂ + d̂): Nhat C × O × H, Dhat S × O × H."""
        return np.exp(Nhat[self.cat_idx] + Dhat)

    def evaluate(self, Yhat: np.ndarray, name: str, **kw) -> pd.DataFrame:
        r = evaluate(self.grid, self.Y, Yhat, **kw)
        r.insert(0, "model", name)
        return r

    def load(self, name: str) -> np.ndarray:
        F = load_forecast(self.out, name)
        return self.grid.mask(F) if F.ndim == 3 and F.shape[1] == len(self.grid.origins) else F


def setup(config: str = "configs/default.yaml") -> Context:
    cfg = load_config(config)
    np.random.seed(cfg["seed"])
    panel = load_panel(cfg).full()
    grid = Grid(panel, cfg)
    n, d, cat_idx = decompose(panel)
    return Context(cfg, panel, grid, n, d, cat_idx, panel.y.to_numpy(float))


def pivot(res: pd.DataFrame, metric: str = "MAE", split: str = "test") -> pd.DataFrame:
    r = res[res.split == split]
    if r.empty or metric not in r:
        return pd.DataFrame()
    t = r.pivot_table(index="model", columns="h", values=metric)
    return t.sort_values(t.columns[0])


def guard_prophet(ctx, F: np.ndarray) -> np.ndarray:
    """Страховка Prophet (в его пользу), как в 07_evaluate.py: лог-прогноз дальше чем в 3 раза за
    пределами истории ряда до origin заменяется последним значением."""
    ly, out = ctx.panel.ly, F.copy()
    for j, o in enumerate(ctx.grid.origins):
        hist = ly[:, : o + 1]
        lo, hi = np.nanmin(hist, 1) - np.log(3), np.nanmax(hist, 1) + np.log(3)
        bad = np.isfinite(F[:, j]) & ((F[:, j] < lo[:, None]) | (F[:, j] > hi[:, None]))
        out[:, j] = np.where(bad, ly[:, o][:, None], F[:, j])
    return out
