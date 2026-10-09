"""Сверка чисел отчёта и README с таблицами расчёта (outputs/tables).

Проверяет, что ключевые числа, процитированные в reports/methodology.md и README.md, совпадают с
тем, что лежит в outputs/tables после прогона, и что таблица §7.1 собрана из текущих результатов.

    python tools/verify_report.py        # код выхода 0 — всё сходится
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
sys.path.insert(0, str(ROOT / "tools"))
from report_tables import table  # noqa: E402

REPORT = (ROOT / "reports" / "methodology.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
HS = [1, 3, 6, 12]


def ru(x, d=0):
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−")


def pct(x):
    return f"{ru(100 * x)} %"


checks = []


def need(text, where, what):
    ok = text in (REPORT if where == "отчёт" else README)
    checks.append((ok, where, what, text))


full = pd.read_csv(T / "forecast_metrics_full.csv")
t = full[full.split == "test"].pivot_table(index="model", columns="h", values="MAE")
ens = [ru(t.loc["Ансамбль (Caruana по группам h)", h]) for h in HS]
need(" / ".join(ens), "отчёт", "MAE ансамбля на всей панели")
need(f"| Ансамбль, вся панель (12 096 рядов) | **{ens[0]}** | **{ens[1]}** | **{ens[2]}** | **{ens[3]}** |", "README", "MAE ансамбля")

fp = pd.read_csv(T / "ensemble_vs_prophet_full_panel.csv")
fp = fp[fp.split == "test"].set_index("h")
rel_full = [ru(100 * fp.loc[h, "rel"]) for h in HS]
need(" / ".join(rel_full) + " %", "отчёт", "выигрыш против Prophet, вся панель")
need("| " + " | ".join(f"{r} %" for r in rel_full) + " |", "README", "выигрыш против Prophet, вся панель")
sp = pd.read_csv(T / "ensemble_vs_prophet.csv")
best = sp[(sp.split == "test") & sp.prophet.str.contains("лучшая")].set_index("h")
rel_best = [ru(100 * best.loc[h, "rel"]) for h in HS]
need(" / ".join(rel_best) + " %", "отчёт", "выигрыш против лучшего Prophet")
need(f"p = {ru(fp.sign_p.max(), 3)}", "отчёт", "знаковый тест")

ex = pd.read_csv(T / "external_check_2025.csv").iloc[0]
need(f"+{ru(ex['прогноз роста 2025/2024, %'], 1)} %", "отчёт", "прогноз роста 2025")
need(f"+{ru(ex['факт портала 2025/2024, %'], 1)} %", "отчёт", "факт роста 2025")

au = pd.read_csv(T / "news_manual_audit_summary.csv").iloc[0]
need(f"{ru(au['полностью верно, %'])} %", "отчёт", "ручной аудит новостей")

det = pd.read_csv(T / "detection_summary_far3.csv").set_index("detector")
need(ru(det.loc["BOCPD + новости", "recall"], 3).replace(",", ","), "отчёт", "полнота BOCPD + новости")
real = pd.read_csv(T / "detection_real_events.csv").set_index("детектор")
need(f"p = {ru(real.loc['BOCPD + новости', 'p паводок (Фишер)'], 3)}", "отчёт", "p паводка (Фишер)")

w = json.load(open(T / "ensemble_weights.json", encoding="utf-8"))
need(f"TimesFM-2.5 {ru(w['[1, 2]']['TimesFM-2.5 на d'], 2)}", "отчёт", "вес TimesFM в ансамбле h = 1–2")

ok_table = table() in REPORT
checks.append((ok_table, "отчёт", "таблица §7.1 совпадает с outputs/tables", "tools/report_tables.py --write"))

bad = [c for c in checks if not c[0]]
for ok, where, what, text in checks:
    print(("OK   " if ok else "FAIL ") + f"[{where}] {what}: {text}")
print(f"\n{len(checks) - len(bad)} из {len(checks)} проверок сошлись")
sys.exit(1 if bad else 0)
