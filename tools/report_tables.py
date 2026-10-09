"""Таблица §7.1 отчёта из outputs/tables (чтобы числа отчёта не расходились с расчётом).

    python tools/report_tables.py            # печать таблицы в Markdown
    python tools/report_tables.py --write    # замена таблицы в reports/methodology.md
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
HS = [1, 3, 6, 12]
ROWS = [  # (подпись в отчёте, имя модели в таблицах, сноска)
    ("**Ансамбль (наше решение)**", "Ансамбль (Caruana по группам h)", ""),
    ("TimesFM-2.5 на d", "TimesFM-2.5 на d", ""),
    ("Chronos-2 совместный (6 категорий) на d", "Chronos-2 (6 категорий совместно) на d", ""),
    ("Chronos-2 совместный + LoRA (дообучение на данных до 12.2023)", "Chronos-2 совместный + LoRA (дообучение до 12.2023) на d", "³"),
    ("Ансамбль без foundation models (абляция)", "Ансамбль без foundation models (абляция)", ""),
    ("LightGBM на d", "LightGBM на d", ""),
    ("Chronos-2 одномерный на d", "Chronos-2 (одномерный) на d", ""),
    ("Chronos-Bolt base на d", "Chronos-Bolt (base) на d", ""),
    ("Chronos-Bolt small на d", "Chronos-Bolt (small) на d", ""),
    ("LightGBM v2 на d (без географии)", "LightGBM v2 на d (без региона)", "⁴"),
    ("Панель: последнее отклонение", "Панель: последнее отклонение", ""),
    ("Панель: уровень + сезонность d", "Панель: уровень + сезонность d", ""),
    ("Панель: SES(0,5) на d", "Панель: SES(0.5) на d", ""),
    ("Prophet на d (локальная компонента)", "Prophet на d (локальная компонента)", "¹"),
    ("Prophet по умолчанию (вся панель / выборка)", ("Prophet (по умолчанию) — вся панель", "Prophet (по умолчанию)"), ""),
    ("Prophet: log + годовая сезонность + праздники", "Prophet (log, годовая сезонность, праздники)", "¹ ²"),
    ("TimesFM-2.5 на сыром ряду", "TimesFM-2.5 на сыром ряду", ""),
    ("Chronos-2 на сыром ряду", "Chronos-2 на сыром ряду", ""),
    ("Chronos-Bolt base на сыром ряду", "Chronos-Bolt (base) на сыром ряду", ""),
    ("Наивная (последнее значение)", "Наивная (последнее значение)", ""),
]


def ru(x, d=0):
    if pd.isna(x):
        return "—"
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−")


def table() -> str:
    full = pd.read_csv(T / "forecast_metrics_full.csv")
    smp = pd.read_csv(T / "forecast_metrics_sample.csv")
    f = full[full.split == "test"].pivot_table(index="model", columns="h", values=["MAE", "R2_yoy"])
    s = smp[smp.split == "test"].pivot_table(index="model", columns="h", values=["MAE", "R2_yoy"])
    out = ["| Модель | MAE h=1 | h=3 | h=6 | h=12 | R² г/г: h=1 / 3 / 6 / 12 | MAE на выборке: h=1 / 3 / 6 / 12 |",
           "|---|---|---|---|---|---|---|"]
    for label, name, note in ROWS:
        nf, ns = (name if isinstance(name, tuple) else (name, name))
        b = label.startswith("**")
        fm = [ru(f.loc[nf, ("MAE", h)]) if nf in f.index else "—" for h in HS]
        src = f if nf in f.index else s
        key = nf if nf in f.index else ns
        r2 = [" / ".join(ru(src.loc[key, ("R2_yoy", h)], 2) for h in HS) if key in src.index else "—"]
        sm = " / ".join(ru(s.loc[ns, ("MAE", h)]) for h in HS) if ns in s.index else "—"
        if "LoRA" in label:
            r2 = ["—"]
        cells = fm + r2 + [sm]
        if b:
            cells = [f"**{c}**" for c in cells]
        out.append(f"| {label}{note} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    tb = table()
    print(tb)
    if "--write" in sys.argv:
        p = ROOT / "reports" / "methodology.md"
        txt = p.read_text(encoding="utf-8")
        start = txt.index("| Модель | MAE h=1 | h=3 | h=6 | h=12 | R² г/г")
        end = txt.index("\n\n", start)
        p.write_text(txt[:start] + tb + txt[end:], encoding="utf-8")
        print("таблица §7.1 обновлена")
