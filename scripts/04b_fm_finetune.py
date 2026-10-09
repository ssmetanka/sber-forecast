"""Шаг 4b. Каузальное дообучение Chronos-2 (LoRA) против zero-shot.

Дообучаем один раз на локальной компоненте d, известной к декабрю 2023 г. (12 месяцев, 6 категорий
МО как многомерный ряд), и сравниваем с zero-shot на origin с декабря 2023 г. и позже — для них
обучение не использует будущего. Вопрос: помогает ли дообучение на 12 точках контекста?
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import evaluate, save_forecast  # noqa: E402
from sbx.data import load_config, path  # noqa: E402
from sbx.models.foundation import QUANTILES, Chronos2  # noqa: E402
from sbx.pipeline import setup  # noqa: E402

cfg_ft = load_config("configs/models.yaml")["finetune"]
torch.set_num_threads(cfg_ft.get("threads", 8))
ctx = setup()
g, d = ctx.grid, ctx.d
cut = ctx.panel.month_idx(cfg_ft["train_until"])                 # последний месяц обучающих данных
S, T = d.shape
train = d[:, : cut + 1].reshape(S // 6, 6, cut + 1).astype(np.float32)

base = Chronos2(joint=True)
t0 = time.time()
ft_pipe = base.pipe.fit(
    list(train), prediction_length=cfg_ft["prediction_length"], finetune_mode="lora",
    learning_rate=cfg_ft["learning_rate"], num_steps=cfg_ft["num_steps"], batch_size=cfg_ft["batch_size"],
    min_past=cfg_ft["min_past"], output_dir=str(path(ctx.cfg, "cache") / "chronos2_lora"),
    remove_printer_callback=True)
print(f"дообучение: {time.time() - t0:.0f} с", flush=True)

ft = Chronos2(joint=True)
ft.pipe = ft_pipe
F = g.empty(S)
valid = [j for j, o in enumerate(g.origins) if o >= cut]           # каузально допустимые origin
for j in valid:
    o = g.origins[j]
    H = min(g.H, g.T - 1 - o)
    med, _ = ft.forecast(d[:, : o + 1], H)
    F[:, j, :H] = med[:, :H]
    print(f"[ft] origin {ctx.panel.months[o]}", flush=True)
F = g.mask(F)
save_forecast(F, ctx.out, "d__fm_chronos2_joint_lora")
# для ансамбля: LoRA там, где она каузально допустима, иначе zero-shot той же модели
zs_all = ctx.load("d__fm_chronos2_joint")
save_forecast(np.where(np.isfinite(F), F, zs_all), ctx.out, "d__fm_chronos2_joint_lora_c")

nb = ctx.load("national_best")
zs = ctx.load("d__fm_chronos2_joint").copy()
keep = np.zeros(len(g.origins), bool)
keep[valid] = True
zs[:, ~keep] = np.nan                                              # те же origin, что у дообученной
res = pd.concat([ctx.evaluate(ctx.compose(nb, zs), "Chronos-2 совместный, zero-shot"),
                 ctx.evaluate(ctx.compose(nb, F), "Chronos-2 совместный, LoRA (обучение до 12.2023)")])
res.to_csv(path(ctx.cfg, "tables") / "fm_finetune_lora.csv", index=False)
pd.set_option("display.width", 200)
print(res[res.split == "test"].pivot_table(index="model", columns="h", values="MAE").round(1))
print(res[res.split == "val"].pivot_table(index="model", columns="h", values="MAE").round(1))
