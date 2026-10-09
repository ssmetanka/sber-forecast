"""Шаг 6a. Сбор заголовков новостей (Lenta.ru, рубрики «Россия» и «Экономика», 2023–2024)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import ROOT, load_config  # noqa: E402
from sbx.news.collect import collect  # noqa: E402

c = load_config("configs/news.yaml")["collect"]
out = ROOT / "data" / "news"
df = collect(c["start"], c["end"], c["rubrics"], out / "cache", workers=c["workers"], delay=c["delay"])
df.to_parquet(out / "headlines.parquet")
print(f"заголовков: {len(df):,}; по рубрикам: {df.rubric.value_counts().to_dict()}")
