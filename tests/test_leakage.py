"""Тесты на заглядывание в будущее: прогноз и признаки на origin o не должны меняться,
если все данные после o заменить шумом.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import run_local  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.models import baselines as B  # noqa: E402
from sbx.pipeline import setup  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return setup()


def _scramble_after(X, o, seed=0):
    rng = np.random.default_rng(seed)
    Y = X.copy()
    Y[:, o + 1:] = rng.normal(size=Y[:, o + 1:].shape)
    return Y


@pytest.mark.parametrize("fn", [B.naive, B.mean_k, B.ses, B.level_season, B.snaive_growth])
def test_baselines_no_future(ctx, fn):
    g = ctx.grid
    F = run_local(fn, ctx.d, g)
    j = min(3, len(g.origins) - 1)
    o = g.origins[j]
    F2 = run_local(fn, _scramble_after(ctx.d, o), g)
    assert np.allclose(F[:, j], F2[:, j], equal_nan=True)


def test_features_no_future(ctx):
    from sbx.features import FeatureBuilder
    fb = FeatureBuilder(ctx)
    o, h = 15, 6
    f1 = fb.build(o, h)
    fb.d = _scramble_after(ctx.d, o)
    fb.ly = _scramble_after(ctx.panel.ly, o)
    f2 = fb.build(o, h)
    assert np.allclose(f1.to_numpy(float), f2.to_numpy(float), equal_nan=True)


@pytest.mark.parametrize("det", [D.cusum_w, D.zscore, D.ewma, D.bocpd, D.glr])
def test_detectors_online(ctx, det):
    """Онлайн-детектор: score в месяц t зависит только от данных ≤ t."""
    x = D.levels(ctx.d[:500], 11)
    t = 16
    s1 = det(x)[:, t]
    s2 = det(_scramble_after(x, t))[:, t]
    assert np.allclose(s1, s2, equal_nan=True)


def test_national_median_is_cross_sectional(ctx):
    """n[t] — медиана по МО в том же месяце t: не использует другие месяцы."""
    from sbx.national import decompose
    n1, _, _ = decompose(ctx.panel)
    p2 = ctx.panel
    y = p2.y.copy()
    y.iloc[:, 20:] = 1.0
    from sbx.data import Panel
    n2, _, _ = decompose(Panel(y, p2.full_mask))
    assert np.allclose(n1[:, :20], n2[:, :20])
