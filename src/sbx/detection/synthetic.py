"""Полусинтетическая проверка детекторов: в реальные ряды d вставляем шоки известной формы,
величины и охвата, калибруем пороги на равную частоту ложных тревог (FAR) по рядам без шоков
и меряем полноту, задержку, точность и F1 с допуском.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHAPES = ("step", "dip", "ramp")
SCOPES = ("series", "mo", "cluster")


def shock_profile(shape: str, T: int, tau: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Матрица N × T добавок к d (лог-шкала) для N шоков с началом tau и величиной delta."""
    t = np.arange(T)[None, :]
    tau = tau[:, None]
    if shape == "step":
        w = (t >= tau).astype(float)
    elif shape == "dip":           # провал на 2 месяца с возвратом
        w = ((t >= tau) & (t < tau + 2)).astype(float)
    elif shape == "ramp":          # линейно за 3 месяца до полной величины
        w = np.clip((t - tau + 1) / 3, 0, 1)
    else:
        raise ValueError(shape)
    return w * delta[:, None]


def inject(d, mo_rows, nb_mo, scope, shape, magnitude, onset_range, frac, rng):
    """Возвращает (d_shocked, shocked_mask S, onset S (−1 — без шока)).

    scope: series — одна случайная категория МО; mo — все категории МО; cluster — МО и его
    соседи по автодороге (все категории). Знак шока случайный.
    """
    S, T = d.shape
    n_mo = mo_rows.shape[0]
    centers = rng.choice(n_mo, int(frac * n_mo), replace=False)
    onset = np.full(S, -1)
    delta = np.zeros(S)
    for c in centers:
        tau = rng.integers(onset_range[0], onset_range[1] + 1)
        dl = np.log1p(magnitude) * rng.choice([-1, 1])
        if scope == "series":
            rows = [mo_rows[c, rng.integers(1, mo_rows.shape[1])]]
        elif scope == "mo":
            rows = list(mo_rows[c])
        else:
            mos = [c] + [m for m in nb_mo[c] if m >= 0]
            rows = list(mo_rows[mos].ravel())
        for r in rows:
            if onset[r] < 0:          # пересечения кластеров: первый шок побеждает
                onset[r], delta[r] = tau, dl
    m = onset >= 0
    ds = d.copy()
    ds[m] += shock_profile(shape, T, onset[m], delta[m])
    return ds, m, onset


def calibrate(score, clean_mask, months, far):
    """Порог, при котором доля тревог на рядах без шока в месяцах мониторинга равна far."""
    vals = score[np.ix_(clean_mask, months)].ravel()
    vals = vals[np.isfinite(vals)]
    return np.quantile(vals, 1 - far)


def evaluate_alarms(alarm, shocked, onset, clean_mask, months, tol=2):
    """alarm: bool S × T. Обнаружен — есть тревога в [τ, τ+tol]."""
    S, T = alarm.shape
    mon = np.zeros(T, bool)
    mon[months] = True
    idx = np.where(shocked)[0]
    t = np.arange(T)[None, :]
    win = (t >= onset[idx, None]) & (t <= onset[idx, None] + tol) & mon[None, :]
    hit = alarm[idx] & win
    detected = hit.any(1)
    first = np.where(detected, np.argmax(hit, 1) - onset[idx], np.nan)
    tp_alarms = hit.sum()
    all_alarms = alarm[:, mon].sum()
    false_alarms = alarm[np.ix_(clean_mask, months)].sum()
    recall = detected.mean()
    precision = tp_alarms / max(all_alarms, 1)
    return {
        "recall": recall,
        "recall_at_onset": (first == 0).mean() if len(idx) else np.nan,
        "delay": np.nanmean(first),
        "precision": precision,
        "F1": 2 * precision * recall / max(precision + recall, 1e-9),
        "far_per100": 100 * false_alarms / max(clean_mask.sum() * len(months), 1),
    }


def run_grid(d, detectors: dict, mo_rows, nb_mo, z_fn, months, fars=(0.01, 0.03, 0.05),
             magnitudes=(0.03, 0.05, 0.10, 0.20), shapes=SHAPES, scopes=SCOPES, onset_range=None,
             frac=0.1, seeds=(0, 1, 2), news_sim=None, log=print) -> pd.DataFrame:
    """detectors: {name: fn(z, d_shocked, news_mult) -> score S × T}."""
    rows = []
    for scope in scopes:
        for shape in shapes:
            for mag in magnitudes:
                for seed in seeds:
                    rng = np.random.default_rng(seed + int(mag * 1000) + 7 * SHAPES.index(shape))
                    ds, shocked, onset = inject(d, mo_rows, nb_mo, scope, shape, mag, onset_range, frac, rng)
                    z = z_fn(ds)
                    news = news_sim(shocked, onset, rng) if news_sim else None
                    clean = ~shocked
                    for name, fn in detectors.items():
                        sc = fn(z, news)
                        for far in fars:
                            th = calibrate(sc, clean, months, far)
                            r = evaluate_alarms(sc > th, shocked, onset, clean, months)
                            rows.append({"scope": scope, "shape": shape, "magnitude": mag, "seed": seed,
                                         "detector": name, "far": far, **r})
                log(f"  {scope}/{shape}: готово")
    return pd.DataFrame(rows)
