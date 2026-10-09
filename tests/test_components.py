"""Модульные тесты компонентов на игрушечных данных (без загрузки панели):
as-of экспозиция новостей, hazard-множитель, калибровка FAR, оценка тревог, онлайн-свойство
BOCPD, ансамбль Caruana и тест Дибольда–Мариано.

    python -m pytest tests -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import ensemble  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.detection.synthetic import calibrate, evaluate_alarms  # noqa: E402
from sbx.news.process import hazard_multiplier, monthly_exposure  # noqa: E402
from sbx.stats import dm_hln  # noqa: E402

MONTHS = pd.period_range("2024-01", "2024-06", freq="M")
CFG = {"align": {"half_life_months": 1, "max_lag_months": 3, "publication_lag_days": 0},
       "geocode": {"region_weight": 0.3},
       "hazard": {"types": ["emergency"], "beta": 4, "cap": 2}}
DCT = pd.DataFrame({"region_code": [1, 1, 2]}, index=pd.Index([10, 11, 20], name="territory_id"))


def _events(date, mos=(10,), regions=(), types=("emergency",)):
    return pd.DataFrame({"date": [pd.Timestamp(date)], "territory_ids": [list(mos)],
                         "region_codes": [list(regions)], "types": [list(types)]})


def _expo(ev):
    e = monthly_exposure(ev, DCT, MONTHS, CFG)
    return e.set_index(["territory_id", "month"]).exposure


def test_exposure_as_of_no_lookback():
    """Новость 20 марта: в феврале экспозиции нет, в марте — доля месяца после публикации."""
    e = _expo(_events("2024-03-20"))
    assert (10, "2024-02") not in e.index
    assert e[(10, "2024-03")] == pytest.approx(12 / 31)
    assert e[(10, "2024-04")] == pytest.approx(1.0)
    assert e[(10, "2024-05")] == pytest.approx(0.5)          # полураспад 1 месяц


def test_exposure_region_weight_and_other_regions():
    """Упоминание региона даёт его МО вес region_weight; МО другого региона не затрагиваются."""
    e = _expo(_events("2024-02-01", mos=(), regions=(1,)))
    assert e[(10, "2024-02")] == pytest.approx(0.3)
    assert e[(11, "2024-02")] == pytest.approx(0.3)
    assert not any(t == 20 for t, _ in e.index)


def test_hazard_multiplier_cap_and_type_filter():
    ev = pd.concat([_events("2024-03-01"), _events("2024-03-01"), _events("2024-03-01"),
                    _events("2024-03-01", mos=(11,), types=("economy",))], ignore_index=True)
    expo = monthly_exposure(ev, DCT, MONTHS, CFG)
    M = hazard_multiplier(expo, np.array([10, 11, 20]), MONTHS, CFG)
    assert M[0, 2] == pytest.approx(1 + 4 * 2)                 # экспозиция 3 обрезана cap = 2
    assert M[0, 1] == 1.0                                     # до события множитель 1
    assert (M[1] == 1.0).all() and (M[2] == 1.0).all()        # не тот тип / нет событий


def test_calibrate_reaches_target_far():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(4000, 12))
    months = np.arange(6, 12)
    clean = np.ones(4000, bool)
    th = calibrate(s, clean, months, 0.03)
    assert (s[:, months] > th).mean() == pytest.approx(0.03, abs=0.003)


def test_evaluate_alarms_toy():
    alarm = np.zeros((3, 10), bool)
    alarm[0, 6] = True                                        # шок с t=5, тревога через 1 мес.
    alarm[2, 8] = True                                        # ложная тревога на чистом ряду
    shocked = np.array([True, True, False])
    onset = np.array([5, 5, 0])
    r = evaluate_alarms(alarm, shocked, onset, ~shocked, np.arange(4, 10))
    assert r["recall"] == 0.5 and r["delay"] == 1.0 and r["recall_at_onset"] == 0.0
    assert r["precision"] == 0.5


def test_bocpd_online_and_reacts_to_break():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 24))
    x[:100, 14:] += 3.0                                       # ступенька у половины рядов
    s = D.bocpd(x, hazard=0.02, kappa0=0.05, recent=1)
    s_cut = D.bocpd(x[:, :16], hazard=0.02, kappa0=0.05, recent=1)
    np.testing.assert_allclose(s[:, :16], s_cut, rtol=1e-9, atol=1e-9)   # прошлое не зависит от будущего
    assert np.nanmean(s[:100, 14:17]) > np.nanmean(s[100:, 14:17]) + 1


def test_caruana_prefers_better_model_and_weights_sum_to_one():
    rng = np.random.default_rng(2)
    S, T, H = 300, 24, 12
    Y = np.exp(rng.normal(8, 0.3, size=(S, T)))
    origins = np.arange(6, 23)
    grid = SimpleNamespace(origins=origins, T=T, val=(12, 17), groups=[[1, 2], [3, 4, 5], list(range(6, 13))])
    t = origins[:, None] + np.arange(1, H + 1)
    truth = np.where(t < T, np.log(Y[:, np.minimum(t, T - 1)]), np.nan)
    good = truth + rng.normal(0, 0.02, truth.shape)
    bad = truth + rng.normal(0, 0.3, truth.shape)
    W = ensemble.caruana({"good": good, "bad": bad}, Y, grid)
    for w in W.values():
        ws = {k: v for k, v in w.items() if not k.startswith("_")}
        assert sum(ws.values()) == pytest.approx(1.0)
        assert ws.get("good", 0) > ws.get("bad", 0)


def test_dm_hln_null_and_alternative():
    rng = np.random.default_rng(3)
    _, p0 = dm_hln(rng.normal(size=60), h=1)
    _, p1 = dm_hln(rng.normal(-1.0, 1, size=60), h=1)
    assert p0 > 0.01 and p1 < 0.001
