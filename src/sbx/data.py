"""Загрузка данных СберИндекса и справочников в удобные для бэктеста структуры."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ["Все категории", "Продовольствие", "Маркетплейсы", "Здоровье",
              "Общественное питание", "Транспорт"]


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    if key in {"cache", "forecasts", "tables", "figures"}:
        p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Panel:
    """Широкая панель: строки — ряды (МО × категория), столбцы — месяцы.

    `full` — ряды с полной историей (основной оценочный набор, 2 016 МО × 6 категорий);
    `y` — уровни (₽ на жителя в месяц), `ly` — логарифм.
    """
    y: pd.DataFrame
    full_mask: np.ndarray

    @property
    def months(self) -> pd.PeriodIndex:
        return self.y.columns

    @property
    def index(self) -> pd.MultiIndex:
        return self.y.index

    @property
    def territory(self) -> np.ndarray:
        return self.y.index.get_level_values("territory_id").to_numpy()

    @property
    def category(self) -> np.ndarray:
        return self.y.index.get_level_values("category").to_numpy()

    @cached_property
    def ly(self) -> np.ndarray:
        return np.log(self.y.to_numpy(dtype=float))

    def month_idx(self, m: str | pd.Period) -> int:
        return self.months.get_loc(pd.Period(m, "M"))

    def full(self) -> "Panel":
        return Panel(self.y[self.full_mask], np.ones(self.full_mask.sum(), bool))


def load_panel(cfg: dict) -> Panel:
    c = pd.read_parquet(path(cfg, "consumption"))
    c["date"] = pd.PeriodIndex(c["date"], freq="M")
    y = c.pivot_table(index=["territory_id", "category"], columns="date", values="value", aggfunc="first")
    months = pd.period_range(cfg["data"]["start"], cfg["data"]["end"], freq="M")
    y = y.reindex(columns=months)
    cat_order = {c: i for i, c in enumerate(CATEGORIES)}
    order = sorted(range(len(y)), key=lambda i: (y.index[i][0], cat_order[y.index[i][1]]))
    y = y.iloc[order]
    full_mask = y.notna().all(axis=1).to_numpy()
    # МО считаем полным, только если полны все 6 его категорий
    tid = y.index.get_level_values(0)
    mo_full = pd.Series(full_mask, index=tid).groupby(level=0).transform("all").to_numpy()
    return Panel(y, mo_full)


def load_dict(cfg: dict) -> pd.DataFrame:
    """Справочник МО: id, название, регион, ОКТМО, координаты центра (актуальные записи)."""
    d = pd.read_excel(path(cfg, "municipal_dict"))
    d = d.sort_values("year_to").drop_duplicates("territory_id", keep="last")
    return d.set_index("territory_id")


def load_national(cfg: dict) -> pd.DataFrame:
    """Национальные месячные ряды СберИндекса (млрд ₽), столбцы — типы трат."""
    n = pd.read_parquet(path(cfg, "national"))
    n["period"] = pd.PeriodIndex(pd.to_datetime(n["period"]), freq="M")
    return n.pivot_table(index="period", columns="type", values="value")


def load_market_access(cfg: dict) -> pd.Series:
    return pd.read_parquet(path(cfg, "market_access")).set_index("territory_id")["market_access"]


def load_neighbors(cfg: dict, k: int | None = None) -> pd.DataFrame:
    """k ближайших соседей каждого МО по автодорожному расстоянию (кэшируется)."""
    k = k or cfg["spatial"]["k_neighbors"]
    cache = path(cfg, "cache") / f"neighbors_k{k}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    e = pd.read_parquet(path(cfg, "connection"), filters=[("type", "==", "highway")],
                        columns=["territory_id_x", "territory_id_y", "distance"])
    both = pd.concat([
        e.rename(columns={"territory_id_x": "src", "territory_id_y": "dst"}),
        e.rename(columns={"territory_id_y": "src", "territory_id_x": "dst"}),
    ], ignore_index=True)
    both = both[both.src != both.dst]
    nb = both.sort_values(["src", "distance"]).groupby("src").head(k).reset_index(drop=True)
    nb["rank"] = nb.groupby("src").cumcount() + 1
    nb.to_parquet(cache)
    return nb
