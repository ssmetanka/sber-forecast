"""Шаг 8. Графики для отчёта и презентации (outputs/figures/*.png)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import load_dict, path  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

# палитра (validated default из гайда dataviz): синий — наше решение, оранжевый — Prophet,
# бирюзовый — третья серия; прочие модели — серые
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2ab", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "axes.titleweight": "bold", "axes.titlesize": 11, "figure.facecolor": "white",
})

ctx = setup()
tables, figs = path(ctx.cfg, "tables"), path(ctx.cfg, "figures")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(figs / name, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


# 1. MAE по горизонтам ---------------------------------------------------------------------
f = tables / "forecast_metrics_sample.csv"
if f.exists():
    r = pd.read_csv(f)
    r = r[r.split == "test"]
    piv = r.pivot_table(index="model", columns="h", values="MAE")
    fig, ax = plt.subplots(figsize=(7, 4))
    hs = piv.columns.to_numpy()
    main = {"Ансамбль (Caruana по группам h)": BLUE, "Панель: последнее отклонение": AQUA}
    prophets = [m for m in piv.index if m.startswith("Prophet")]
    best_p = min(prophets, key=lambda m: piv.loc[m].mean()) if prophets else None
    for m in piv.index:
        if m in main or m == best_p or piv.loc[m].max() > 2000:
            continue
        ax.plot(hs, piv.loc[m], color=GRAY, lw=1, alpha=0.8)
    ax.plot([], [], color=GRAY, lw=1, label="прочие модели")
    if best_p:
        ax.plot(hs, piv.loc[best_p], color=ORANGE, marker="o", ms=5, label=f"{best_p} — лучший Prophet")
    for m, c in main.items():
        if m in piv.index:
            ax.plot(hs, piv.loc[m], color=c, marker="o", ms=5, label=m)
    ax.set_xticks(hs)
    ax.set_xlabel("горизонт прогноза, мес.")
    ax.set_ylabel("MAE, ₽ на жителя в месяц")
    ax.set_title("Ошибка прогноза на отложенном полугодии (июль–декабрь 2024)", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "mae_by_horizon.png")

    # 2. абляция FM: сырой ряд против подготовленного входа
    pairs = [("Chronos-Bolt base", "Chronos-Bolt (base) на сыром ряду", "Chronos-Bolt (base) на d"),
             ("Chronos-Bolt small", "Chronos-Bolt (small) на сыром ряду", "Chronos-Bolt (small) на d"),
             ("Chronos-2", "Chronos-2 на сыром ряду", "Chronos-2 (одномерный) на d"),
             ("TimesFM-2.5", "TimesFM-2.5 на сыром ряду", "TimesFM-2.5 на d"),
             ("TiRex", "TiRex на сыром ряду", "TiRex на d")]
    pairs = [p for p in pairs if p[1] in piv.index and p[2] in piv.index]
    if pairs:
        fig, ax = plt.subplots(figsize=(1.9 * len(pairs) + 2.5, 3.6))
        x = np.arange(len(pairs))
        w = 0.36
        raw = [piv.loc[p[1], 1] for p in pairs]
        prep = [piv.loc[p[2], 1] for p in pairs]
        b1 = ax.bar(x - w / 2 - 0.01, raw, w, color=ORANGE, label="сырой log y")
        b2 = ax.bar(x + w / 2 + 0.01, prep, w, color=BLUE, label="локальная компонента d (наш вход)")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.0f}",
                        ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, [p[0] for p in pairs])
        ax.set_ylabel("MAE h=1, ₽")
        ax.set_title("Foundation models: выигрыш даёт подготовка входа", loc="left")
        ax.legend(fontsize=8.5)
        save(fig, "fm_ablation.png")

# 3. примеры рядов --------------------------------------------------------------------------
fe = ctx.out / "ensemble.npy"
if fe.exists():
    ens = np.exp(ctx.load("ensemble"))
    dct = load_dict(ctx.cfg)
    tid = ctx.panel.territory
    hit_o = np.where(ctx.grid.origins == ctx.panel.month_idx("2024-06"))[0]
    j0 = int(hit_o[0]) if len(hit_o) else 0
    names = {"Орск": None, "Новотроицк": None, "Курган": None, "Ишим": None}
    for nm in list(names):
        hit = dct[(dct.municipal_district_name_short == nm) & dct.index.isin(tid)]
        names[nm] = int(hit.index[0]) if len(hit) else None
    names = {k: v for k, v in names.items() if v is not None}
    pf = ctx.out / "prophet__yearly.npy"
    prop = np.exp(ctx.load("prophet__yearly")) if pf.exists() else None
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), sharey=False)
    axes = np.atleast_1d(axes)
    months = ctx.panel.months.to_timestamp()
    for ax, (nm, t) in zip(axes, names.items()):
        s = np.where((tid == t) & (ctx.cat_idx == 2))[0][0]          # Маркетплейсы
        ax.plot(months, ctx.Y[s], color=INK, lw=1.6, label="факт")
        o = ctx.grid.origins[j0]
        fut = months[o + 1: o + 7]
        ax.plot(fut, ens[s, j0, :len(fut)], color=BLUE, marker="o", ms=4, label="ансамбль")
        if prop is not None and np.isfinite(prop[s, j0, 0]):
            ax.plot(fut, prop[s, j0, :len(fut)], color=ORANGE, marker="o", ms=4, label="Prophet")
        ax.axvline(months[o], color=GRAY, lw=1, ls="--")
        ax.axvspan(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01"), color="#f0efec", zorder=0)
        ax.set_title(f"{nm}: маркетплейсы", loc="left", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].set_ylabel("₽ на жителя в месяц")
    axes[0].legend(fontsize=8)
    fig.suptitle("Прогноз из июня 2024 на 1–6 мес.; серая полоса — паводок апреля 2024", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    save(fig, "examples_flood_cities.png")

# 4–5. детекция ---------------------------------------------------------------------------
f = tables / "detection_synthetic_raw.csv"
if f.exists():
    r = pd.read_csv(f)
    order = r[r.far == 0.03].groupby("detector").recall.mean().sort_values(ascending=False)
    top = list(order.index[:3])
    colors = dict(zip(top, [BLUE, ORANGE, AQUA]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    by_far = r.groupby(["detector", "far"]).recall.mean().unstack()
    for det in by_far.index:
        c = colors.get(det, GRAY)
        axes[0].plot(100 * by_far.columns, by_far.loc[det], color=c, lw=2 if det in colors else 1,
                     marker="o" if det in colors else None, ms=5, label=det if det in colors else None)
    rnd = [1 - (1 - f) ** 3 for f in by_far.columns]
    axes[0].plot(100 * by_far.columns, rnd, color=INK2, lw=1.2, ls="--", label="случайные тревоги")
    axes[0].plot([], [], color=GRAY, lw=1, label="прочие детекторы")
    axes[0].set_xlabel("ложных тревог на 100 ряд-месяцев")
    axes[0].set_ylabel("полнота (доля найденных шоков)")
    axes[0].set_title("Полнота при равной частоте ложных тревог", loc="left")
    axes[0].legend(fontsize=8)
    by_mag = r[r.far == 0.03].groupby(["detector", "magnitude"]).recall.mean().unstack()
    for det in by_mag.index:
        c = colors.get(det, GRAY)
        axes[1].plot(100 * by_mag.columns, by_mag.loc[det], color=c, lw=2 if det in colors else 1,
                     marker="o" if det in colors else None, ms=5)
    axes[1].set_xlabel("величина шока, %")
    axes[1].set_title("Полнота по величине шока (FAR 3 %)", loc="left")
    for ax in axes:
        ax.set_ylim(0, 1)
    save(fig, "detection_recall.png")

f = tables / "alarms_2024.csv"
if f.exists():
    a = pd.read_csv(f)
    cnt = a.groupby("month").size()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(cnt.index, cnt.values, color=BLUE, width=0.8)
    ax.axhline(0.03 * len(ctx.Y), color=INK2, lw=1, ls="--")
    ax.text(len(cnt) - 0.5, 0.03 * len(ctx.Y), "  ожидаемо при 3 % ложных", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_ylabel("тревог (ряды)")
    ax.set_title("Тревоги детектора на реальных данных 2024 г.", loc="left")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    save(fig, "alarms_2024.png")

# 6. паводок ------------------------------------------------------------------------------
f = tables / "event_study_flood2024.csv"
if f.exists():
    es = pd.read_csv(f)
    es = es[es.month == "2024-04"].set_index("category")
    cats = list(es.index)
    x = np.arange(len(cats))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (col, lab, c) in enumerate([("named_%", "МО из новостей о ЧС", BLUE),
                                        ("same_region_%", "другие МО тех же регионов", AQUA),
                                        ("rest_%", "остальная страна", GRAY)]):
        ax.bar(x + (i - 1) * (w + 0.01), es[col], w, color=c, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, cats, rotation=15, fontsize=9)
    ax.set_ylabel("медианная ошибка прогноза, %")
    ax.set_title("Апрель 2024: отклонение факта от прогноза на 1 мес. (паводок)", loc="left")
    ax.legend(fontsize=8.5)
    save(fig, "flood_event_study.png")

# 7. кейсы тревог с новостями ------------------------------------------------------------
f = tables / "local_alarms_2024.csv"
if f.exists():
    la = pd.read_csv(f)
    dct = load_dict(ctx.cfg)
    tid = ctx.panel.territory
    cases = [("Торопецкий", "Маркетплейсы", "атака и эвакуация жителей 18.09.2024"),
             ("Саракташский", "Маркетплейсы", "паводок в Оренбургской обл., апрель 2024"),
             ("Орск", "Маркетплейсы", "прорыв дамбы 05.04.2024")]
    from sbx.data import CATEGORIES
    fig, axes = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.2))
    months = ctx.panel.months.to_timestamp()
    for ax, (nm, cat, note) in zip(axes, cases):
        hit = dct[dct.municipal_district_name_short.astype(str).str.startswith(nm) & dct.index.isin(tid)]
        if not len(hit):
            ax.set_visible(False)
            continue
        t = int(hit.index[0])
        s = np.where((tid == t) & (ctx.cat_idx == CATEGORIES.index(cat)))[0][0]
        ax.plot(months, ctx.Y[s], color=INK, lw=1.6)
        al = la[(la.territory_id == t) & (la.category == cat)]
        for m in al.month:
            ax.axvline(pd.Timestamp(m + "-01"), color=ORANGE, lw=2, alpha=0.8)
        ax.plot([], [], color=ORANGE, lw=2, label="тревога BOCPD")
        ax.set_title(f"{hit.municipal_district_name_short.iloc[0]}: {cat.lower()}", loc="left", fontsize=10)
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.5, color=INK2)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].set_ylabel("₽ на жителя в месяц")
    axes[0].legend(fontsize=8, loc="upper left")
    save(fig, "alarm_cases.png")
