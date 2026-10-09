"""Детекторы в постановках, которые встречаются в открытых решениях конкурса.

Наша сетка шире обычной: величины 3–20 %, формы ступенька/провал/рампа, охват ряд/МО/кластер.
Средняя полнота по такой сетке (0,27 при FAR 3 %) несопоставима с числами, посчитанными
только на крупных ступеньках. Скрипт ничего не пересчитывает — агрегирует готовый
outputs/tables/detection_synthetic_raw.csv по подмножествам сценариев (секунды).

    python tools/detection_comparable.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "outputs/tables/detection_synthetic_raw.csv")

SETUPS = [
    ("Вся наша сетка (3–20 %, 3 формы, 3 охвата)", 0.03, lambda d: d),
    ("Шоки 5–20 %", 0.03, lambda d: d[d["magnitude"] >= 0.05]),
    ("Шоки 10–20 %", 0.03, lambda d: d[d["magnitude"] >= 0.10]),
    ("Ступеньки 5–20 %", 0.03, lambda d: d[(d["magnitude"] >= 0.05) & (d["shape"] == "step")]),
    ("Шоки 5–20 %, охват МО (все категории)", 0.03, lambda d: d[(d["magnitude"] >= 0.05) & (d["scope"] == "mo")]),
    ("Шоки 5–20 %", 0.05, lambda d: d[d["magnitude"] >= 0.05]),
]
DETECTORS = ["BOCPD + новости", "BOCPD", "Stouffer-МО", "z-score (инновация)"]

rows = []
for name, far, pick in SETUPS:
    sub = pick(raw[raw["far"] == far])
    g = sub.groupby("detector")[["recall", "recall_at_onset", "precision", "F1", "delay"]].mean()
    for det in DETECTORS:
        r = g.loc[det]
        rows.append({"постановка": name, "ложных тревог на 100 ряд-мес": int(far * 100), "детектор": det,
                     "сценариев": int(len(sub) / sub["detector"].nunique()),
                     "полнота": round(r["recall"], 3), "в месяц шока": round(r["recall_at_onset"], 3),
                     "точность": round(r["precision"], 3), "F1": round(r["F1"], 3), "задержка, мес": round(r["delay"], 2)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs/tables/detection_comparable.csv", index=False)

best = out[out["детектор"] == "BOCPD + новости"]
print("| Постановка | Ложных на 100 | Полнота | В месяц шока | Точность | F1 |")
print("|---|---|---|---|---|---|")
for _, r in best.iterrows():
    print(f"| {r['постановка']} | {r['ложных тревог на 100 ряд-мес']} | {r['полнота']:.2f} | "
          f"{r['в месяц шока']:.2f} | {r['точность']:.2f} | {r['F1']:.2f} |".replace(".", ","))
