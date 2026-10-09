"""Foundation models временных рядов в режиме zero-shot (без дообучения).

Каждый класс: forecast(X: S × t, H) -> (медиана S × H, квантили S × H × 3 для уровней 0.1/0.5/0.9).
Подаём локальную компоненту d (сезонность и инфляция уже сняты) или, для абляции, сырой log y.
"""
from __future__ import annotations

import numpy as np
import torch

QUANTILES = [0.1, 0.5, 0.9]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # TimesFM выбирает устройство сам


class ChronosBolt:
    def __init__(self, model_id="amazon/chronos-bolt-base", batch_size=512):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=DEVICE, torch_dtype=torch.float32)
        self.bs = batch_size

    @torch.no_grad()
    def forecast(self, X, H):
        qs = []
        for i in range(0, len(X), self.bs):
            q, _ = self.pipe.predict_quantiles(torch.tensor(X[i:i + self.bs], dtype=torch.float32),
                                               prediction_length=H, quantile_levels=QUANTILES)
            qs.append(q.cpu().numpy())
        q = np.concatenate(qs)
        return q[:, :, 1], q


class Chronos2:
    """Chronos-2. joint=True — 6 категорий одного МО подаются как один многомерный ряд."""

    def __init__(self, model_id="amazon/chronos-2", batch_size=256, joint=False, n_var=6):
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=DEVICE, torch_dtype=torch.float32)
        self.bs, self.joint, self.n_var = batch_size, joint, n_var

    @torch.no_grad()
    def forecast(self, X, H):
        S, t = X.shape
        inp = X.reshape(S // self.n_var, self.n_var, t) if self.joint else X[:, None, :]
        q, _ = self.pipe.predict_quantiles(inp.astype(np.float32), prediction_length=H,
                                           quantile_levels=QUANTILES, batch_size=self.bs)
        q = np.stack([x.cpu().numpy() for x in q])          # (N, n_var, H, 3)
        q = q.reshape(S, H, len(QUANTILES))
        return q[:, :, 1], q


class TimesFM25:
    def __init__(self, model_id="google/timesfm-2.5-200m-pytorch", batch_size=512):
        import timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        self.model.compile(timesfm.ForecastConfig(
            max_context=64, max_horizon=16, normalize_inputs=True, per_core_batch_size=batch_size,
            use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=False,
            fix_quantile_crossing=True))

    def forecast(self, X, H):
        point, quant = self.model.forecast(horizon=H, inputs=[x.astype(np.float32) for x in X])
        # quant: (S, H, 10) — среднее + квантили 0.1…0.9
        q = np.stack([quant[:, :H, 1], quant[:, :H, 5], quant[:, :H, 9]], axis=-1)
        return point[:, :H], q


class TiRex:
    def __init__(self, model_id="NX-AI/TiRex", batch_size=512):
        from tirex import load_model
        self.model = load_model(model_id, device=DEVICE, backend="torch")
        self.bs = batch_size

    @torch.no_grad()
    def forecast(self, X, H):
        qs = []
        for i in range(0, len(X), self.bs):
            q, _ = self.model.forecast(context=torch.tensor(X[i:i + self.bs], dtype=torch.float32),
                                       prediction_length=H)
            qs.append(q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q))
        q = np.concatenate(qs)                         # (S, H, 9) — квантили 0.1…0.9
        q = np.stack([q[:, :, 0], q[:, :, 4], q[:, :, 8]], axis=-1)
        return q[:, :, 1], q


REGISTRY = {"chronos_bolt": ChronosBolt, "chronos2": Chronos2, "timesfm": TimesFM25, "tirex": TiRex}


def build(name: str, **kw):
    if name == "chronos2_joint":
        return Chronos2(joint=True, **kw)
    if name == "chronos_bolt_small":
        return ChronosBolt(**{"model_id": "amazon/chronos-bolt-small", **kw})
    return REGISTRY[name](**kw)
