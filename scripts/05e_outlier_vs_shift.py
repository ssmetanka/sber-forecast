"""Шаг 5e. Выброс против сдвига: отличает ли детектор разовый всплеск от смены уровня.

В 10 % рядов (по одной категории МО) вносится либо разовый выброс на один месяц, либо ступенька
той же величины (10 % и 20 %). Порог каждого детектора — 3 % ложных тревог на рядах без вставки.
Хороший детектор сдвигов:
  * на ступеньке тревожит в окне [τ, τ+2] (полнота);
  * на выбросе после того, как он прошёл (месяцы τ+1, τ+2), молчит — иначе это ложный «сдвиг».
Отношение «полнота на ступеньке / тревоги после выброса» показывает, насколько детектор отличает
одно от другого.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import load_config, path  # noqa: E402
from sbx.detection import detectors as D  # noqa: E402
from sbx.features import FeatureBuilder  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

ctx = setup()
dcfg = load_config("configs/detection.yaml")
P = dcfg["params"]
p, d = ctx.panel, ctx.d
mi = p.month_idx
fb = FeatureBuilder(ctx, ctx.cfg["data"]["artifact_months"])
artifact = tuple(mi(m) for m in ctx.cfg["data"]["artifact_months"])
exclude = artifact + tuple(a + 1 for a in artifact)
calib_end = mi(dcfg["calib_end"])
months = np.arange(mi(dcfg["monitor"][0]), mi(dcfg["monitor"][1]) + 1)
lo, hi = mi(dcfg["onset"][0]), mi(dcfg["onset"][1])
ref_start = max(artifact)


def detect(x):
    lv = D.levels(x, calib_end, exclude)
    z = D.innovations(x, calib_end, exclude=exclude)
    c = D.deviation(lv, season_alpha=dcfg["deviation"]["season_alpha"], ref_start=ref_start)
    return {"z-score (инновация)": D.zscore(z), "EWMA": D.ewma(z, **P["ewma"]),
            "CUSUM оконный": D.cusum_w(c, **P["cusum_w"]), "GLR (онлайн-разрыв)": D.glr(lv, **P["glr"]),
            "BOCPD": D.bocpd(lv, **P["bocpd"])}


rows = []
S, Tn = d.shape
for mag in (0.10, 0.20):
    for seed in (0, 1):
        rng = np.random.default_rng(100 + seed + int(mag * 100))
        sel = rng.random(S) < 0.10
        tau = rng.integers(lo, hi + 1, size=S)
        sign = rng.choice([-1, 1], size=S)
        t = np.arange(Tn)[None, :]
        delta = (np.log1p(mag) * sign)[:, None]
        res = {}
        for kind in ("выброс", "ступенька"):
            w = (t == tau[:, None]) if kind == "выброс" else (t >= tau[:, None])
            x = d + np.where(sel[:, None], w * delta, 0.0)
            res[kind] = detect(x)
        for det in res["выброс"]:
            out = {}
            for kind in ("выброс", "ступенька"):
                s = res[kind][det]
                th = np.nanquantile(s[np.ix_(~sel, months)], 0.97)
                al = s > th
                idx = np.where(sel)[0]
                if kind == "ступенька":
                    win = (t >= tau[idx, None]) & (t <= tau[idx, None] + 2)
                    out["полнота на ступеньке"] = (al[idx] & win).any(1).mean()
                else:
                    at = (t == tau[idx, None])
                    after = (t >= tau[idx, None] + 1) & (t <= tau[idx, None] + 2)
                    out["тревога в месяц выброса"] = (al[idx] & at).any(1).mean()
                    out["тревога после выброса (ложный сдвиг)"] = (al[idx] & after).any(1).mean()
            rows.append({"детектор": det, "величина": mag, "seed": seed, **out})
r = pd.DataFrame(rows).groupby(["детектор", "величина"]).mean(numeric_only=True).drop(columns="seed").reset_index()
r["различение (ступенька / ложный сдвиг)"] = r["полнота на ступеньке"] / r["тревога после выброса (ложный сдвиг)"].clip(lower=1e-3)
r.to_csv(path(ctx.cfg, "tables") / "detection_outlier_vs_shift.csv", index=False)
pd.set_option("display.width", 220)
print(r.round(3).sort_values(["величина", "различение (ступенька / ложный сдвиг)"], ascending=[True, False]).to_string(index=False))
