"""Ноутбук Colab для сбора новостей региональных ГУ МЧС за 2024 г. (сайты МЧС недоступны с
некоторых российских IP при массовой выгрузке, а из Colab доступны).

    python tools/make_news_notebook.py   →   notebooks/sberindex_mchs_news_colab.ipynb
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "configs", "data/hackathonlicence/consumption.parquet",
           "data/raw/dict/t_dict_municipal_districts.xlsx"]


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in INCLUDE:
            tar.add(ROOT / rel, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


INTRO = """
# Сбор новостей ГУ МЧС за 2024 г. (для задачи 2 СберИндекса)

**Запуск:** `Среда выполнения → Выполнить все` (GPU не нужен). Один раз Colab спросит доступ к Google Диску —
нажмите «Разрешить»: каждый собранный регион сразу сохраняется в `Мой диск/sberindex_mchs`, поэтому при обрыве
сессии ничего не теряется, а повторный запуск продолжит с места остановки. Без Диска всё тоже работает (локально).

Своего лимита времени у ноутбука нет. Ограничения самого Colab: бесплатная сессия живёт до ~12 ч и может
отключиться, если вкладку закрыть или долго не трогать, — держите вкладку открытой. Сайты МЧС ограничивают
частоту запросов, поэтому регионы собираются аккуратно (3 параллельно), а отказавшие повторяются позже.
В конце браузер скачает `mchs_news_2024.zip` (архив обновляется после каждого региона — его можно скачать
из панели файлов в любой момент).
"""

SETUP = """
#@title Настройки, распаковка, Google Диск
START, END = "2024-01-01", "2024-12-31"
PARALLEL_REGIONS = 3        # больше — сайты МЧС начинают отказывать
PAGE_WORKERS = 2
RETRY_ROUNDS = 3            # повторные проходы по регионам, где сайт отказал
RETRY_PAUSE_MIN = 10
import base64, io, tarfile, pathlib, os, sys, time
WORK = pathlib.Path("/content/sbx"); WORK.mkdir(exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{bundle}")), mode="r:gz").extractall(WORK)
os.chdir(WORK); sys.path.insert(0, str(WORK / "src"))
!pip install -q pymorphy3==2.0.6 openpyxl beautifulsoup4 pyyaml 2>&1 | tail -1
out = WORK / "data/news/mchs"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    out = pathlib.Path("/content/drive/MyDrive/sberindex_mchs")
    print("сохраняю на Google Диск:", out)
except Exception as e:
    print("Google Диск недоступен — сохраняю локально:", e)
out.mkdir(parents=True, exist_ok=True)
print("уже собрано регионов:", len([f for f in out.glob("*.parquet")]))
"""

RUN = """
#@title Сбор (без лимита времени; отказавшие регионы — повторно)
import pandas as pd, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from sbx.data import load_config, load_dict, load_panel
from sbx.news.mchs import collect, fetch_bodies
cfg = load_config()
dct = load_dict(cfg); dct = dct[dct.year_to >= 2024]
regions = sorted(set(dct.loc[dct.index.isin(load_panel(cfg).territory), "region_code"].astype(int)))
Z = pathlib.Path("/content/mchs_news_2024.zip")
T0 = time.time()

# --- живой прогресс-бар: собрано регионов, скорость текущего запуска, оценка остатка
import threading
from IPython.display import display, HTML
DONE0 = len([f for f in out.glob(f"*_{START}_{END}.parquet")])   # собранные в прошлых запусках
STATE = {"phase": "сбор", "new": 0}

def bar_html():
    done = len([f for f in out.glob(f"*_{START}_{END}.parquet")])
    total = len(regions)
    pct = done / total
    el = (time.time() - T0) / 60
    new = done - DONE0
    left = f"≈ {(total - done) * el / new:.0f} мин" if new else "оценю после первого региона"   # по средней скорости этого запуска
    return (f"<div style='font:13px/1.5 monospace;margin:4px 0'>"
            f"<div style='width:460px;height:14px;background:#e3e8e5;border-radius:7px;overflow:hidden'>"
            f"<div style='width:{100*pct:.1f}%;height:14px;background:#21a038'></div></div>"
            f"<b>{done} из {total} регионов ({100*pct:.0f} %)</b> · {STATE['phase']} · прошло {el:.0f} мин · "
            f"<b>осталось {left}</b></div>")

BAR = display(HTML(bar_html()), display_id=True)
STOP = threading.Event()
def _tick():
    while not STOP.wait(15):
        BAR.update(HTML(bar_html()))
threading.Thread(target=_tick, daemon=True).start()

def pack():
    with zipfile.ZipFile(Z, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob("*.parquet")):
            z.write(f, f"mchs/{f.name}")

def one(reg):
    f = out / f"{reg:02d}_{START}_{END}.parquet"
    if f.exists():
        return reg, True, "уже есть"
    try:
        df = collect(reg, START, END, section="novosti", workers=PAGE_WORKERS, delay=0.5, log=lambda *_: None)
        b = fetch_bodies(df, workers=PAGE_WORKERS, delay=0.5) if len(df) else df.assign(body="")
        b.to_parquet(f)
        return reg, True, f"{len(df)} новостей, {len(b)} о ЧС"
    except Exception as e:                       # сайт отказал — не сохраняем пустой файл, повторим
        return reg, False, f"отказ сайта ({type(e).__name__}) — повторю позже"

todo = list(regions)
for rnd in range(RETRY_ROUNDS + 1):
    if rnd:
        STATE["phase"] = f"пауза перед повтором {rnd}/{RETRY_ROUNDS}"
        print(f"\\n── повтор {rnd}/{RETRY_ROUNDS}: {len(todo)} регионов, пауза {RETRY_PAUSE_MIN} мин"); time.sleep(60 * RETRY_PAUSE_MIN)
        STATE["phase"] = f"повтор {rnd}/{RETRY_ROUNDS} ({len(todo)} регионов)"
    failed = []
    with ThreadPoolExecutor(PARALLEL_REGIONS) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r, ok, msg = fu.result()
            if not ok:
                failed.append(r)
            else:
                pack()
            print(f"[{i}/{len(todo)}] регион {r:02d}: {msg}  ({(time.time()-T0)/60:.0f} мин)", flush=True)
            BAR.update(HTML(bar_html()))
    todo = failed
    if not todo:
        break
STATE["phase"] = "готово"; STOP.set(); BAR.update(HTML(bar_html()))
print(f"\\nготово: {len(regions) - len(todo)} из {len(regions)} регионов" + (f"; не удалось: {todo}" if todo else ""))
"""

FINISH = """
#@title Архив (скачается автоматически)
pack()
print(Z, f"{Z.stat().st_size/1e6:.1f} МБ,", len(list(out.glob('*.parquet'))), "регионов")
try:
    from google.colab import files; files.download(str(Z))
except Exception as e:
    print("скачайте вручную из панели файлов:", e)
"""


def main():
    cells = [cell("markdown", INTRO), cell("code", SETUP.replace("{bundle}", bundle())), cell("code", RUN),
             cell("code", FINISH)]
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / "notebooks" / "sberindex_mchs_news_colab.ipynb"
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(p, f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
