"""Офлайн-методы ruptures: ретроспективный поиск точек разрыва по всему ряду.

Штраф подбирается так, чтобы на рядах без шока было `target_false_per100` ложных точек на
100 ряд-месяцев 2024 г. — так методы сравниваются честно, как и онлайн-детекторы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt

from .synthetic import inject

PENALTIES = [0.5, 1, 2, 3, 5, 8, 12, 20, 35]


def _algo(method, x):
    x = x.reshape(-1, 1)
    if method == "pelt":
        return rpt.Pelt(model="l2", min_size=2, jump=1).fit(x)
    if method == "binseg":
        return rpt.Binseg(model="l2", min_size=2, jump=1).fit(x)
    if method == "bottomup":
        return rpt.BottomUp(model="l2", min_size=2, jump=1).fit(x)
    if method == "window":
        return rpt.Window(width=6, model="l2", min_size=2, jump=1).fit(x)
    if method == "kernel":
        return rpt.KernelCPD(kernel="rbf", min_size=2).fit(x)
    raise ValueError(method)


def _changepoints(method, X, pens):
    """{pen: список массивов точек разрыва} для каждого ряда X (N × T), ряды стандартизуются."""
    out = {p: [] for p in pens}
    for x in X:
        xs = (x - np.median(x)) / (np.median(np.abs(np.diff(x))) * 1.4826 / np.sqrt(2) + 1e-6)
        a = _algo(method, xs)
        for p in pens:
            out[p].append(np.array(a.predict(pen=p)[:-1]))
    return out


def offline_grid(d, mo_rows, nb_mo, onset_range, cfg, seed=0, log=print) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = d.shape[1]
    mon0 = onset_range[0] - 1                                  # начало 2024 г.
    ds, shocked, onset = inject(d, mo_rows, nb_mo, "series", "step", cfg["magnitude"], onset_range,
                                0.5, rng)
    n = cfg["n_series"]
    pos = rng.choice(np.where(shocked)[0], n // 2, replace=False)
    neg = rng.choice(np.where(~shocked)[0], n // 2, replace=False)
    rows = []
    for method in cfg["methods"]:
        cp_neg = _changepoints(method, ds[neg], PENALTIES)
        # штраф по ложным точкам в 2024 г. на чистых рядах
        rates = {p: 100 * sum(((c >= mon0) & (c < T)).sum() for c in cp_neg[p]) / (len(neg) * (T - mon0))
                 for p in PENALTIES}
        pen = min(PENALTIES, key=lambda p: abs(rates[p] - cfg["target_false_per100"]))
        cp_pos = _changepoints(method, ds[pos], [pen])[pen]
        hits = [np.any(np.abs(c - onset[r]) <= cfg["tol"]) for c, r in zip(cp_pos, pos)]
        n_cp_pos = sum(((c >= mon0)).sum() for c in cp_pos)
        n_tp = sum(int(h) for h in hits)
        n_fp_neg = sum(((c >= mon0)).sum() for c in cp_neg[pen])
        recall = np.mean(hits)
        precision = n_tp / max(n_cp_pos + n_fp_neg, 1)
        rows.append({"method": method, "penalty": pen, "recall": recall, "precision": precision,
                     "F1": 2 * recall * precision / max(recall + precision, 1e-9),
                     "false_cp_per100": rates[pen]})
        log(f"  offline {method}: recall={recall:.3f}, pen={pen}")
    return pd.DataFrame(rows)
