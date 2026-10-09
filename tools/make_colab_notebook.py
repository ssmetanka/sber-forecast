"""Собирает самодостаточный ноутбук для Google Colab (T4): код + данные внутри, запуск «Run all».

    python tools/make_colab_notebook.py   →   notebooks/sberindex_task2_colab.ipynb
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "src", "scripts", "configs", "tests", "tools", "run_all.py", "requirements.txt", "README.md",
    "reports/methodology.md",
    "data/hackathonlicence/consumption.parquet", "data/hackathonlicence/market_access.parquet",
    "data/raw/dict/t_dict_municipal_districts.xlsx", "data/raw/portal/consumer_spending.parquet",
    "outputs/cache/neighbors_k10.parquet",
    "data/news/events.parquet", "data/news/acts.parquet", "data/news/mchs",
]


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in INCLUDE:
            p = ROOT / rel
            tar.add(p, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# СберИндекс 2026 · Задача 2 — полный прогон в Colab

**Как запустить:** `Среда выполнения → Сменить среду выполнения → T4 GPU`, затем `Среда выполнения → Выполнить все`.
Больше ничего делать не нужно: код и данные встроены в ноутбук, модели скачиваются с Hugging Face.

Время: **~3–3,5 ч** (foundation models на GPU в фоне; Prophet и LightGBM на 2 ядрах CPU). Вкладку держите открытой.
По окончании браузер сам скачает `sberindex_results.zip` (таблицы, графики, лендинг, логи, `summary.md`);
архив также лежит в `/content/sberindex_results.zip`. Ошибка отдельного шага не останавливает прогон — она попадает
в `summary.md` и логи.
"""

SETTINGS = """
#@title Настройки (можно не трогать)
PROPHET_SAMPLE_MO = 300     # МО в выборке для Prophet (все 6 категорий каждого МО); на 2 ядрах ≈ 1,5 ч
LGBM_MAX_ROWS = 400_000     # подвыборка обучающих пар LightGBM на каждом origin
LGBM_ROUNDS = 250
MCHS_MAX_MINUTES = 100      # лимит на досбор новостей ГУ МЧС (идёт в фоне параллельно с моделями)
TIREX_MAX_MINUTES = 60      # лимит на TiRex (GPU)
AUTO_DOWNLOAD = True        # скачать архив с результатами в конце
import time, os, sys, subprocess, json
T0 = time.time()
STATUS = {}
"""

GPU = """
#@title Проверка GPU
import torch
GPU_OK = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if GPU_OK else "нет — foundation models пойдут на CPU (медленно)")
print("torch", torch.__version__, "| python", sys.version.split()[0])
!nproc; free -g | head -2
"""

UNPACK = """
#@title Распаковка кода и данных (встроены в ноутбук)
import base64, io, tarfile, pathlib
WORK = pathlib.Path("/content/sberhakmax")
WORK.mkdir(parents=True, exist_ok=True)
BUNDLE = "{bundle}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(WORK)
del BUNDLE
os.chdir(WORK)
(WORK / "logs").mkdir(exist_ok=True)
print("распаковано:", sum(1 for _ in WORK.rglob("*") if _.is_file()), "файлов")
"""

INSTALL = """
#@title Установка зависимостей (torch из Colab не трогаем)
# pandas/numpy оставляем из Colab: пайплайн проверен на pandas 2.2 и 3.0
!pip install -q lightgbm==4.7.0 prophet==1.5.0 ruptures==1.1.10 chronos-forecasting==2.3.2 \\
    pymorphy3==2.0.6 openpyxl xlrd pyyaml joblib beautifulsoup4 tabulate tirex-ts==1.4.2 peft 2>&1 | tail -3
!pip install -q --no-deps "timesfm @ git+https://github.com/google-research/timesfm.git" 2>&1 | tail -2
!pip install -q huggingface_hub safetensors 2>&1 | tail -1
import importlib
for m in ["pandas", "lightgbm", "prophet", "ruptures", "chronos", "timesfm", "pymorphy3"]:
    try:
        mod = importlib.import_module(m); print(f"{m:10s} ok", getattr(mod, "__version__", ""))
    except Exception as e:
        print(f"{m:10s} ОШИБКА: {e}")
"""

PROFILE = """
#@title Профиль Colab: 2 ядра CPU + T4
import yaml
def edit(path, fn):
    p = WORK / path
    d = yaml.safe_load(open(p, encoding="utf-8")); fn(d)
    yaml.safe_dump(d, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
def models(d):
    d["prophet"]["sample_mo"] = PROPHET_SAMPLE_MO
    d["prophet"]["n_jobs"] = 2
    for k in ("lgbm", "lgbm_v2"):
        d[k]["max_rows"] = LGBM_MAX_ROWS
        d[k]["num_rounds"] = LGBM_ROUNDS
        d[k]["params"]["num_threads"] = 2
    d["foundation"]["threads"] = 1          # FM считает на GPU, CPU оставляем Prophet/LightGBM
    d["foundation"]["chronos_bolt"]["batch_size"] = 2048
    d["foundation"]["timesfm"]["batch_size"] = 1024
edit("configs/models.yaml", models)
print(open(WORK / "configs/models.yaml", encoding="utf-8").read()[:900])
"""

RUNNER = """
#@title Запускатель шагов и прогресс-бар (лог каждого шага — logs/*.log)
import threading, statistics
from IPython.display import display, HTML
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
NOISE = ("Warning", "warn(", "Loading weights", "plotly", "torch_dtype", "FutureWarning", "it/s]")
# ожидаемая длительность шагов на T4, мин (по прошлому прогону); оценка подстраивается под реальную скорость
PLAN = {"01_baselines": 0.5, "02_mask": 0.3, "03_lgbm": 12, "03_lgbm_v2": 8, "02_prophet": 55,
        "05_detection": 6, "05c_detection_extra": 8, "05e_outlier_vs_shift": 3, "04_foundation": 3,
        "04b_fm_finetune": 20, "04_tirex": 30, "06c_official": 5, "07_evaluate": 1,
        "07b_error_decomposition": 0.3, "07c_external_2025": 0.2, "07e_no_portal": 1, "07f_appendix": 0.5,
        "06b_news": 0.5, "06d_event_study_all": 1, "05b_real_alarms": 0.5, "08_figures": 0.3, "09_site": 0.2,
        "tests": 0.4, "quick_check": 0.2, "verify_report": 0.1}
if not GPU_OK:
    PLAN.pop("04_tirex", None)
TOTAL = sum(PLAN.values())
DONE = {"w": 0.0, "n": 0}
SPEED = []

def _bar_html(name, step_sec):
    exp = PLAN.get(name, 1)
    done = DONE["w"] + min(step_sec / 60, exp * 0.95)
    pct = min(done / TOTAL, 1)
    k = statistics.median(SPEED) if SPEED else 1.0
    left = max(TOTAL - done, 0) * k
    return (f"<div style='font:13px/1.5 monospace;margin:4px 0'>"
            f"<div style='width:460px;height:14px;background:#e3e8e5;border-radius:7px;overflow:hidden'>"
            f"<div style='width:{100*pct:.1f}%;height:14px;background:#21a038'></div></div>"
            f"<b>{100*pct:.0f} %</b> · шаг {DONE['n']+1} из {len(PLAN)}: <b>{name}</b> "
            f"({step_sec/60:.0f} из ~{exp*k:.0f} мин) · прошло {(time.time()-T0)/60:.0f} мин · "
            f"<b>осталось ≈ {left:.0f} мин</b></div>")

def _live(name, t, stop):
    h = display(HTML(_bar_html(name, 0)), display_id=True)
    while not stop.wait(15):
        h.update(HTML(_bar_html(name, time.time() - t)))
    h.update(HTML(_bar_html(name, time.time() - t)))

def _finish(name, t):
    if name in PLAN:
        el = (time.time() - t) / 60
        if PLAN[name] >= 3:
            SPEED.append(max(el / PLAN[name], 0.2))
        DONE["w"] += PLAN[name]; DONE["n"] += 1

def run(name, args, show=True):
    \"\"\"Шаг на переднем плане: живой прогресс-бар, вывод в ячейку (без шума) и в logs/<name>.log. Не падает.\"\"\"
    t = time.time()
    stop = threading.Event()
    th = threading.Thread(target=_live, args=(name, t, stop), daemon=True); th.start()
    with open(WORK / "logs" / f"{name}.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, *args], cwd=WORK, env=ENV, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            log.write(line)
            if show and not any(n in line for n in NOISE) and line.strip():
                print(line.rstrip()[:220])
        p.wait()
    _finish(name, t); stop.set(); th.join()
    ok = p.returncode == 0
    STATUS[name] = ("ok" if ok else f"ОШИБКА (код {p.returncode}), см. logs/{name}.log") + f", {time.time()-t:.0f} с"
    print(f"── {name}: {STATUS[name]}  [всего {(time.time()-T0)/60:.0f} мин]")
    return ok

BG = {}
def start_bg(name, args):
    \"\"\"Шаг в фоне (GPU или сеть), лог — logs/<name>.log.\"\"\"
    log = open(WORK / "logs" / f"{name}.log", "w", encoding="utf-8")
    BG[name] = (subprocess.Popen([sys.executable, *args], cwd=WORK, env=ENV, stdout=log, stderr=subprocess.STDOUT), time.time(), log)
    print(f"── {name}: запущен в фоне")

def wait_bg(name, max_min=None):
    p, t0, log = BG[name]
    t = time.time()
    stop = threading.Event()
    th = threading.Thread(target=_live, args=(name, t, stop), daemon=True); th.start()
    while p.poll() is None:
        if max_min and time.time() - t0 > 60 * max_min:
            p.terminate(); print(f"── {name}: превышен лимит {max_min} мин — остановлен, берём готовое"); break
        time.sleep(20)
    _finish(name, t); stop.set(); th.join()
    log.close()
    STATUS[name] = ("ok" if p.returncode == 0 else f"ОШИБКА (код {p.returncode}), см. logs/{name}.log") + f", {time.time()-t0:.0f} с"
    tail = open(WORK / "logs" / f"{name}.log", encoding="utf-8", errors="replace").read().splitlines()
    print("\\n".join(l[:200] for l in tail if l.strip() and not any(n in l for n in NOISE))[-1500:])
    print(f"── {name}: {STATUS[name]}")
"""

STEP1 = """
#@title 1. Национальная компонента, простые модели, выборка для Prophet
run("01_baselines", ["scripts/01_baselines.py"])
run("02_mask", ["scripts/02_prophet.py", "--mask-only"])
"""

STEP2 = """
#@title 2. В фоне: foundation models на GPU и досбор новостей ГУ МЧС
start_bg("04_foundation", ["scripts/04_foundation.py", "chronos2_joint", "chronos_bolt", "chronos_bolt_small", "chronos2", "timesfm"])
start_bg("06c_official", ["scripts/06c_official_sources.py", f"--max-minutes={MCHS_MAX_MINUTES}"])
"""

STEP3 = """
#@title 3. LightGBM (две версии) — CPU
run("03_lgbm", ["scripts/03_lgbm.py"])
run("03_lgbm_v2", ["scripts/03_lgbm.py", "--variant=lgbm_v2"])
"""

STEP4 = """
#@title 4. Prophet (две конфигурации) — CPU, самый долгий шаг
run("02_prophet", ["scripts/02_prophet.py", "default", "yearly", "local", "--subsample", "200"])
"""

STEP5 = """
#@title 5. Обнаружение структурных сдвигов (полусинтетика, 108 сценариев)
run("05_detection", ["scripts/05_detection.py"])
run("05c_detection_extra", ["scripts/05c_detection_extra.py"])
run("05e_outlier_vs_shift", ["scripts/05e_outlier_vs_shift.py"])
"""

STEP6 = """
#@title 6. Ждём фоновые шаги, затем оценка, новости, тревоги, графики, лендинг
wait_bg("04_foundation")
run("04b_fm_finetune", ["scripts/04b_fm_finetune.py"])   # каузальное LoRA-дообучение Chronos-2
# TiRex — только на GPU (на CPU непрактично медленный); с лимитом времени
if GPU_OK:
    start_bg("04_tirex", ["scripts/04_foundation.py", "tirex"]); wait_bg("04_tirex", max_min=TIREX_MAX_MINUTES)
wait_bg("06c_official", max_min=MCHS_MAX_MINUTES + 10)
run("07_evaluate", ["scripts/07_evaluate.py"])
run("07b_error_decomposition", ["scripts/07b_error_decomposition.py"])
run("07c_external_2025", ["scripts/07c_external_2025.py"])
run("07e_no_portal", ["scripts/07e_no_portal.py"])
run("07f_appendix", ["scripts/07f_appendix.py"])
run("06b_news", ["scripts/06b_news.py"])
run("06d_event_study_all", ["scripts/06d_event_study_all.py"])
run("05b_real_alarms", ["scripts/05b_real_alarms.py"], show=False)
run("08_figures", ["scripts/08_figures.py"])
run("09_site", ["scripts/09_site.py"])
run("tests", ["-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])
run("quick_check", ["tools/quick_check.py"])      # пересчёт главных чисел из прогнозов
run("verify_report", ["tools/verify_report.py"])  # сверка с отчётом (с TiRex числа могут сдвинуться)
"""

FINISH = """
#@title 7. Итог: summary.md + архив результатов (скачается автоматически)
import pandas as pd, shutil, zipfile, platform
T = WORK / "outputs" / "tables"
lines = ["# Итог прогона в Colab", "", f"Время: {(time.time()-T0)/60:.0f} мин; GPU: {torch.cuda.get_device_name(0) if GPU_OK else 'нет'}; "
         f"python {platform.python_version()}; pandas {pd.__version__}; torch {torch.__version__}", "", "## Статус шагов", ""]
lines += [f"- `{k}`: {v}" for k, v in STATUS.items()]
def add(title, f, fn=lambda d: d):
    p = T / f
    if p.exists():
        try:
            d = fn(pd.read_csv(p))
            try:
                txt = d.to_markdown(index=False)
            except ImportError:
                txt = "```" + chr(10) + d.to_string(index=False) + chr(10) + "```"
            lines.extend(["", f"## {title}", "", txt])
        except Exception as e:
            lines.extend(["", f"## {title}", f"(не удалось прочитать: {e})"])
pv = lambda m: (lambda d: d[d.split == "test"].pivot_table(index="model", columns="h", values=m).round(3 if m != "MAE" else 0).reset_index())
add("MAE, test — выборка Prophet", "forecast_metrics_sample.csv", pv("MAE"))
add("MAE, test — вся панель", "forecast_metrics_full.csv", pv("MAE"))
add("R² г/г, test — выборка", "forecast_metrics_sample.csv", pv("R2_yoy"))
add("Ансамбль против Prophet", "ensemble_vs_prophet.csv", lambda d: d.round(3))
add("Разложение ошибки (истинная n)", "error_decomposition.csv")
add("Детекторы, FAR 3 %", "detection_summary_far3.csv", lambda d: d.round(3))
add("Детекторы по охвату", "detection_recall_by_scope.csv")
add("Офлайн-детекторы", "detection_offline.csv", lambda d: d.round(3))
add("Детекторы на реальных событиях", "detection_real_events.csv", lambda d: d.round(2))
add("Чувствительность к качеству новостей", "news_sim_sensitivity.csv", lambda d: d.round(3))
add("Покрытие интервалов FM", "fm_interval_coverage.csv", lambda d: d.round(3))
add("Внешняя проверка 2025", "external_check_2025.csv", lambda d: d.round(3))
add("Event study паводка", "event_study_flood2024.csv", lambda d: d.round(3))
add("BOCPD с новостями", "news_bocpd_flood.csv", lambda d: d.round(1))
add("Новости в прогнозе", "news_forecast_ablation.csv", lambda d: d.round(3))
w = T / "ensemble_weights.json"
if w.exists():
    lines.extend(["", "## Веса ансамбля", "", "```json", w.read_text(encoding="utf-8"), "```"])
(WORK / "summary.md").write_text("\\n".join(lines), encoding="utf-8")
print("\\n".join(lines[:60]))

ZIP = pathlib.Path("/content/sberindex_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in ["summary.md", "outputs/tables", "outputs/figures", "reports/site", "logs", "configs",
                "outputs/forecasts/d__fm_tirex.npy", "outputs/forecasts/raw__fm_tirex.npy",
                "outputs/forecasts/d__fm_timesfm.npy", "outputs/forecasts/prophet__yearly.npy",
                "outputs/forecasts/prophet_sample_mask.npy", "outputs/forecasts/ensemble.npy",
                "data/news/mchs", "data/news/events_official.parquet"]:
        p = WORK / rel
        if not p.exists():
            continue
        for f in ([p] if p.is_file() else p.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(WORK))
print(f"\\nархив: {ZIP} ({ZIP.stat().st_size/1e6:.1f} МБ)")
if AUTO_DOWNLOAD:
    try:
        from google.colab import files
        files.download(str(ZIP))
    except Exception as e:
        print("автоскачивание не сработало — скачайте вручную из панели файлов:", e)
"""


def main():
    cells = [md(INTRO), code(SETTINGS), code(GPU), code(UNPACK.replace("{bundle}", bundle())), code(INSTALL),
             code(PROFILE), code(RUNNER), code(STEP1), code(STEP2), code(STEP3), code(STEP4), code(STEP5),
             code(STEP6), code(FINISH)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "notebooks" / "sberindex_task2_colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
