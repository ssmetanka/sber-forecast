"""Признаки для глобальной модели локальной компоненты d.

Все признаки строятся по истории ≤ origin o; цель — d[o+h] − d[o].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORIES, load_dict, load_market_access, load_neighbors


class FeatureBuilder:
    def __init__(self, ctx, artifact_months=("2023-04",)):
        self.ctx = ctx
        p = ctx.panel
        self.d, self.ly = ctx.d, p.ly
        self.T = self.d.shape[1]
        tid = p.territory
        self.cat = ctx.cat_idx
        self.months = p.months
        self.artifact = np.array([p.month_idx(m) for m in artifact_months])

        # строка ряда «Все категории» того же МО и все строки МО
        row = {(t, c): i for i, (t, c) in enumerate(zip(tid, self.cat))}
        self.row_all = np.array([row[(t, 0)] for t in tid])
        self.mo_rows = np.array([[row[(t, c)] for c in range(len(CATEGORIES))] for t in tid])

        # k ближайших соседей по автодороге (только присутствующие в панели), та же категория
        nb = load_neighbors(ctx.cfg)
        present = set(tid)
        nb = nb[nb.dst.isin(present)]
        k = ctx.cfg["spatial"]["k_neighbors"]
        nbr = {s: g.dst.to_numpy()[:k] for s, g in nb.groupby("src")}
        self.nb_rows = np.full((len(tid), k), -1)
        self.nb_dist = np.full(len(tid), np.nan)
        for i, (t, c) in enumerate(zip(tid, self.cat)):
            ds = nbr.get(t, [])
            for j, dt in enumerate(ds):
                self.nb_rows[i, j] = row[(dt, c)]
        d1 = nb[nb["rank"] == 1].set_index("src").distance
        self.nb_dist = pd.Series(tid).map(d1).to_numpy(float)

        # статика: справочник и доступность рынков
        dct = load_dict(ctx.cfg)
        ma = load_market_access(ctx.cfg)
        self.region = pd.Series(tid).map(dct.region_code).fillna(-1).to_numpy(int)
        types = {t: i for i, t in enumerate(sorted(dct.municipal_district_type.dropna().unique()))}
        self.mo_type = pd.Series(tid).map(dct.municipal_district_type).map(types).fillna(-1).to_numpy(int)
        self.lat = pd.Series(tid).map(dct.municipal_district_center_lat).to_numpy(float)
        self.lon = pd.Series(tid).map(dct.municipal_district_center_lon).to_numpy(float)
        self.market_access = pd.Series(tid).map(ma).to_numpy(float)
        # группы для региональных средних (регион × категория)
        self.reg_cat = self.region * 10 + self.cat

    def _col(self, X, j):
        return X[:, j] if 0 <= j < self.T else np.full(X.shape[0], np.nan)

    def _group_mean(self, v, key):
        s = pd.Series(v).groupby(key).transform("mean").to_numpy()
        return s

    def build(self, o: int, h: int) -> pd.DataFrame:
        """Признаки для пары (origin o, горизонт h) по всем рядам."""
        d, c = self.d, self._col
        t = o + h
        lags = {f"d_l{k}": c(d, o - k) for k in range(6)}
        hist = d[:, max(0, o - 5): o + 1]
        last12 = d[:, max(0, o - 11): o + 1]
        f = pd.DataFrame(lags)
        f["d_mean3"] = np.nanmean(d[:, max(0, o - 2): o + 1], 1)
        f["d_mean6"] = np.nanmean(hist, 1)
        f["d_std6"] = np.nanstd(hist, 1)
        f["d_slope6"] = (f["d_mean3"] - np.nanmean(d[:, max(0, o - 5): max(1, o - 2)], 1)) / 3
        f["d_seas"] = c(d, t - 12) - np.nanmean(last12, 1)          # собственная сезонность цели
        f["d_t12"] = c(d, t - 12)
        f["d_yoy"] = c(d, o) - c(d, o - 12)
        f["d_all0"] = d[self.row_all, o]                              # «Все категории» того же МО
        f["d_mo_mean0"] = d[self.mo_rows, o].mean(1)
        nbv = np.where(self.nb_rows >= 0, d[self.nb_rows.clip(0), o], np.nan)
        f["nb_mean0"] = np.nanmean(nbv, 1)
        f["nb_gap0"] = f["d_l0"] - f["nb_mean0"]
        nbv3 = np.where(self.nb_rows >= 0, d[self.nb_rows.clip(0), max(0, o - 3)], np.nan)
        f["nb_chg3"] = f["nb_mean0"] - np.nanmean(nbv3, 1)
        f["reg_mean0"] = self._group_mean(d[:, o], self.reg_cat)
        f["reg_chg3"] = f["reg_mean0"] - self._group_mean(c(d, o - 3), self.reg_cat)
        f["ly0"] = self.ly[:, o]
        f["market_access"] = self.market_access
        f["nb_dist1"] = self.nb_dist
        f["lat"], f["lon"] = self.lat, self.lon
        f["mo_type"] = self.mo_type
        f["region"] = self.region
        f["category"] = self.cat
        f["h"] = h
        f["target_month"] = self.months[t].month if t < self.T else (self.months[o].month + h - 1) % 12 + 1
        f["origin_month"] = self.months[o].month
        f["artifact_in_window"] = np.isin(self.artifact, np.arange(o - 5, o + 1)).any().astype(int)
        f["artifact_seas"] = int(np.isin(t - 12, self.artifact).any())
        return f

    CATEGORICAL = ["mo_type", "region", "category"]
