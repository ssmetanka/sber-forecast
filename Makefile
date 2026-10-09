PY ?= .venv/bin/python

setup:
	python -m venv .venv && $(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/00_get_data.py

forecast:
	$(PY) scripts/01_baselines.py
	$(PY) scripts/02_prophet.py
	$(PY) scripts/03_lgbm.py
	$(PY) scripts/04_foundation.py chronos2_joint chronos_bolt timesfm
	$(PY) scripts/07_evaluate.py

detection:
	$(PY) scripts/05_detection.py

news:
	$(PY) scripts/06a_collect_news.py
	$(PY) scripts/06b_news.py

figures:
	$(PY) scripts/08_figures.py

test:
	$(PY) -m pytest tests -q

all: data forecast detection news figures

.PHONY: setup data forecast detection news figures test all

verify:
	$(PY) tools/verify_report.py

quick:
	$(PY) tools/quick_check.py
	$(PY) tools/verify_report.py
