"""Глобальный LightGBM на локальной компоненте d (прямая стратегия, h — признак).

На каждом origin o модель переобучается на парах (o', h') с o' + h' ≤ o — только прошлое.
Цель — d[o'+h'] − d[o'], лосс L1, веса — уровень расходов на o' (≈ оптимизация MAE в рублях).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import FeatureBuilder

DEFAULT_PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                      feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                      verbose=-1, num_threads=8)


_CACHE: dict = {}


def _feats(fb: FeatureBuilder, o: int, h: int) -> pd.DataFrame:
    """Признаки пары (o, h) не зависят от того, на каком origin обучаем модель, — кэшируем."""
    key = (id(fb), o, h)
    if key not in _CACHE:
        _CACHE[key] = fb.build(o, h).astype("float32")
    return _CACHE[key]


def _train_rows(fb: FeatureBuilder, o: int, H: int, min_origin: int, max_rows: int, rng):
    X, y, w = [], [], []
    for oo in range(min_origin, o):
        for h in range(1, H + 1):
            if oo + h > o:
                break
            f = _feats(fb, oo, h)
            X.append(f)
            y.append(fb.d[:, oo + h] - fb.d[:, oo])
            w.append(np.exp(fb.ly[:, oo]))
    X = pd.concat(X, ignore_index=True)
    y, w = np.concatenate(y), np.concatenate(w)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    return X.iloc[idx], y[idx], w[idx]


def lgbm_forecast(fb: FeatureBuilder, grid, params=None, num_rounds=400, min_origin=2,
                  max_rows=1_500_000, seed=42, drop=(), log=print):
    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    rng = np.random.default_rng(seed)
    F = grid.empty(fb.d.shape[0])
    importances = []
    for j, o in enumerate(grid.origins):
        H = min(grid.H, grid.T - 1 - o)
        X, y, w = _train_rows(fb, o, grid.H, min_origin, max_rows, rng)
        X = X.drop(columns=list(drop))
        cats = [c for c in FeatureBuilder.CATEGORICAL if c in X.columns]
        ds = lgb.Dataset(X, y, weight=w, categorical_feature=cats, free_raw_data=True)
        model = lgb.train(params, ds, num_boost_round=num_rounds)
        for h in range(1, H + 1):
            F[:, j, h - 1] = fb.d[:, o] + model.predict(_feats(fb, o, h).drop(columns=list(drop)))
        importances.append(pd.Series(model.feature_importance("gain"), index=X.columns))
        log(f"[lgbm] origin {fb.months[o]}: {len(y):,} строк обучения")
    imp = pd.concat(importances, axis=1).mean(1).sort_values(ascending=False)
    return F, imp
