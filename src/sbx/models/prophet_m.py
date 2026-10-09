"""Prophet — базовая модель организаторов. Несколько конфигураций, чтобы сравнение было честным.

  default  — Prophet() как «из коробки» на уровнях. На ряде короче двух лет авто-режим сам
             отключает годовую сезонность, остаётся кусочно-линейный тренд;
  yearly   — log y, принудительная годовая сезонность (Фурье порядка 3), праздники РФ;
  local    — Prophet на локальной компоненте d (сезонность и инфляция уже сняты национальной).
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

CONFIGS = {
    "default": dict(transform="level", prophet={}),
    "yearly": dict(transform="log", prophet=dict(yearly_seasonality=3, weekly_seasonality=False,
                                                  daily_seasonality=False, changepoint_prior_scale=0.05)),
    "local": dict(transform="none", prophet=dict(yearly_seasonality=False, weekly_seasonality=False,
                                                  daily_seasonality=False, changepoint_prior_scale=0.01)),
}


def _quiet():
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    warnings.filterwarnings("ignore")


FIT_TIMEOUT_S = 30


def _fit_with_timeout(model, df, future, timeout):
    """Подгонка и прогноз в потоке-демоне со сторожем: на отдельных рядах Stan зависает
    (найдено на практике), ограничение итераций это не лечит. Зависший поток не блокирует выход."""
    import threading
    box = {}

    def work():
        try:
            model.fit(df, algorithm="LBFGS", iter=3000)
            box["p"] = model.predict(future)["yhat"].to_numpy()
        except Exception:
            box["p"] = None

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(timeout)
    return box.get("p")


def _fit_chunk(X: np.ndarray, start: str, H: int, cfg: dict) -> np.ndarray:
    _quiet()
    from prophet import Prophet

    ds = pd.date_range(start, periods=X.shape[1], freq="MS")
    future = pd.DataFrame({"ds": pd.date_range(ds[-1], periods=H + 1, freq="MS")[1:]})
    out = np.full((X.shape[0], H), np.nan)
    for i, x in enumerate(X):
        ok = np.isfinite(x)
        if ok.sum() < 3:
            continue
        y = np.exp(x[ok]) if cfg["transform"] == "level" else x[ok]
        m = Prophet(uncertainty_samples=0, **cfg["prophet"])   # нужна только медиана: без симуляций интервалов
        if cfg["transform"] == "log":
            m.add_country_holidays(country_name="RU")
        p = _fit_with_timeout(m, pd.DataFrame({"ds": ds[ok], "y": y}), future, FIT_TIMEOUT_S)
        if p is None:  # сбой или зависание оптимизатора (редкие ряды) — фолбэк на последнее значение
            p = np.repeat(y[-1], H)
        out[i] = np.log(np.maximum(p, 1.0)) if cfg["transform"] == "level" else p
    return out


def prophet_forecast(X: np.ndarray, H: int, config: str, start: str, n_jobs: int = -1,
                     chunk: int = 256) -> np.ndarray:
    """X — история S × t (лог-уровни для default/yearly, d для local). Возвращает S × H на той же шкале."""
    cfg = CONFIGS[config]
    # Подгонка Prophet — в основном ожидание внешнего процесса Stan, поэтому потоки эффективны
    # и избавляют от зависаний пула процессов на Windows (backend="threading")
    parts = Parallel(n_jobs=n_jobs or os.cpu_count(), backend=os.environ.get("PROPHET_BACKEND", "threading"))(
        delayed(_fit_chunk)(X[i:i + chunk], start, H, cfg) for i in range(0, X.shape[0], chunk))
    return np.vstack(parts)
