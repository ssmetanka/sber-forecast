"""Полный прогон пайплайна одной командой (кроссплатформенно, вместо make):

    python run_all.py            # всё
    python run_all.py --from 5   # начиная с шага 5
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("00_get_data.py", []),
    ("01_baselines.py", []),
    ("02_prophet.py", ["default", "yearly", "local", "--subsample", "200"]),
    ("03_lgbm.py", []),
    ("03_lgbm.py", ["--variant=lgbm_v2"]),
    ("04_foundation.py", ["chronos2_joint", "chronos_bolt", "chronos_bolt_small", "chronos2", "timesfm"]),
    ("04b_fm_finetune.py", []),
    ("05_detection.py", []),
    ("05c_detection_extra.py", []),
    ("05e_outlier_vs_shift.py", []),
    ("06a_collect_news.py", []),
    ("06c_official_sources.py", []),
    ("07_evaluate.py", []),
    ("07b_error_decomposition.py", []),
    ("07c_external_2025.py", []),
    ("07e_no_portal.py", []),
    ("07f_appendix.py", []),
    ("06b_news.py", []),
    ("06d_event_study_all.py", []),
    ("05b_real_alarms.py", []),
    ("08_figures.py", []),
    ("09_site.py", ["--pdf"]),
]

start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
for i, (script, args) in enumerate(STEPS):
    if int(script[:2]) < start:
        continue
    print(f"\n===== {script} =====", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True, cwd=ROOT)
