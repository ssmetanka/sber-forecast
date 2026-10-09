"""Значимость разницы с Prophet при зависимых наблюдениях.

Ряды одного месяца связаны общим шоком, поэтому бутстреп только по МО даёт ложно узкие
интервалы. Основной интервал — двусторонний бутстреп: МО и целевые месяцы ресэмплируются
независимо. Дополнительно — тест Дибольда–Мариано по ряду средних по панели дифференциалов
потерь за целевые месяцы с HAC-дисперсией и поправкой Харви–Лейборна–Ньюболда (HLN)
на малую выборку и перекрытие горизонтов.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .backtest import abs_errors


def _mo_month_matrix(diff: np.ndarray, territory: np.ndarray):
    """Средний дифференциал потерь по (МО, месяц): матрица n_MO × n_месяцев."""
    mos, inv = np.unique(territory, return_inverse=True)
    M = np.zeros((len(mos), diff.shape[1]))
    np.add.at(M, inv, diff)
    return M / np.bincount(inv)[:, None]


def dm_hln(d_t: np.ndarray, h: int) -> tuple[float, float]:
    """DM-тест с HAC (Ньюи–Уэст, лаг h−1, обрезан до T−2) и поправкой HLN; t-распределение T−1."""
    T = len(d_t)
    if T < 3:
        return np.nan, np.nan
    lag = max(0, min(h - 1, T - 2))
    e = d_t - d_t.mean()
    gam = [np.mean(e[k:] * e[: T - k]) for k in range(lag + 1)]
    var = (gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))) / T
    if var <= 0:
        return np.nan, np.nan
    dm = d_t.mean() / np.sqrt(var)
    hh = lag + 1
    corr = np.sqrt(max((T + 1 - 2 * hh + hh * (hh - 1) / T) / T, 1e-9))
    dm *= corr
    return dm, 2 * stats.t.sf(abs(dm), df=T - 1)


def compare(grid, Y, Yhat_a, Yhat_b, territory, h, split="test", n_boot=2000, seed=0) -> dict:
    """Сравнение модели a с b (обычно b — Prophet) на горизонте h; diff < 0 — a лучше."""
    ea, eb = abs_errors(grid, Y, Yhat_a, h, split), abs_errors(grid, Y, Yhat_b, h, split)
    if ea is None or eb is None:
        return {"h": h}
    ok = np.isfinite(ea).all(1) & np.isfinite(eb).all(1)
    ea, eb, territory = ea[ok], eb[ok], territory[ok]
    diff = ea - eb                                   # S × месяцы
    res = {"h": h, "MAE_a": ea.mean(), "MAE_b": eb.mean(), "diff": diff.mean(),
           "rel": ea.mean() / eb.mean() - 1, "n_months": diff.shape[1]}
    M = _mo_month_matrix(diff, territory)
    rng = np.random.default_rng(seed)
    n_mo, n_m = M.shape
    two_way, mo_only = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n_mo, n_mo)
        j = rng.integers(0, n_m, n_m)
        two_way.append(M[np.ix_(i, j)].mean())
        mo_only.append(M[i].mean())
    res["ci_low"], res["ci_high"] = np.percentile(two_way, [2.5, 97.5])          # МО × месяцы
    res["ci_mo_low"], res["ci_mo_high"] = np.percentile(mo_only, [2.5, 97.5])    # только МО (для сравнения)
    res["share_mo_better"] = (M.mean(1) < 0).mean()
    res["share_months_better"] = (M.mean(0) < 0).mean()
    res["dm_stat"], res["dm_p"] = dm_hln(M.mean(0), h)
    # знаковый тест по целевым месяцам: устойчив к зависимости рядов внутри месяца
    k = int((M.mean(0) < 0).sum())
    res["sign_p"] = stats.binomtest(k, n_m, 0.5).pvalue if n_m else np.nan
    return res
