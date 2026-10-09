"""Ансамбль Caruana: жадный отбор моделей с возвращением по MAE на validation, отдельно по группам h.

Комбинируем в лог-пространстве: log ŷ = Σ w_m · log ŷ_m, веса неотрицательные и в сумме 1.
"""
from __future__ import annotations

import numpy as np


def _val_cells(grid, h_list):
    """Список (j_origin, j_h, t) для validation-целей в группе горизонтов."""
    cells = []
    for j, o in enumerate(grid.origins):
        for h in h_list:
            t = o + h
            if t < grid.T and grid.val[0] <= t <= grid.val[1]:
                cells.append((j, h - 1, t))
    return cells


def caruana(logF: dict[str, np.ndarray], Y: np.ndarray, grid, rounds: int = 30) -> dict:
    """Возвращает {group_index: {model: weight}}."""
    names = list(logF)
    weights = {}
    for gi, hs in enumerate(grid.groups):
        cells = _val_cells(grid, hs)
        if not cells:
            weights[gi] = {names[0]: 1.0}
            continue
        yt = np.concatenate([Y[:, t] for _, _, t in cells])
        P = np.stack([np.concatenate([logF[m][:, j, k] for j, k, _ in cells]) for m in names])
        ok = np.isfinite(P).all(0) & np.isfinite(yt)
        P, yt = P[:, ok], yt[ok]
        counts = np.zeros(len(names))
        cur = np.zeros(P.shape[1])
        for r in range(rounds):
            scores = [np.abs(yt - np.exp((cur * r + P[i]) / (r + 1))).mean() for i in range(len(names))]
            i = int(np.argmin(scores))
            counts[i] += 1
            cur = (cur * r + P[i]) / (r + 1)
        w = counts / counts.sum()
        # усадка к равным весам: λ выбирается по MAE на validation (защита от переобучения весов
        # там, где validation-ячеек мало, — длинные горизонты)
        best = None
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            wl = (1 - lam) * w + lam / len(names)
            mae = np.abs(yt - np.exp(wl @ P)).mean()
            if best is None or mae < best[0] - 1e-9:
                best = (mae, lam, wl)
        _, lam, w = best
        weights[gi] = {names[i]: float(w[i]) for i in range(len(names)) if w[i] > 1e-6}
        weights[gi]["_shrink_lambda"] = lam
    return weights


def apply(logF: dict[str, np.ndarray], weights: dict, grid) -> np.ndarray:
    out = np.full_like(next(iter(logF.values())), np.nan)
    for gi, hs in enumerate(grid.groups):
        k = np.array(hs) - 1
        acc = sum(w * logF[m][:, :, k] for m, w in weights[gi].items() if not m.startswith("_"))
        out[:, :, k] = acc
    return np.exp(out)
