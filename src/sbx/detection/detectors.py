"""Онлайн-детекторы структурных сдвигов. Работают на локальной компоненте d (национальная
сезонность и инфляция уже сняты), векторно по всем рядам.

Каждый детектор: score(d, ctx) -> массив S × T «силы тревоги» в месяц t по данным ≤ t.
Тревога — score > порог; порог подбирается так, чтобы частота ложных тревог была одинаковой.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


# ----------------------------------------------------------------------------- инновации

def innovations(d: np.ndarray, calib_end: int, k: int = 3, alpha_season: float = 0.3,
                exclude: tuple = ()) -> np.ndarray:
    """Стандартизованная ошибка прогноза на шаг вперёд z[s, t] по данным ≤ t−1.

    Прогноз d̂[t] = mean(d[t−k..t−1]) + α·(d[t−12] − mean последних 12) (если есть сезонный лаг).
    Масштаб — робастное σ ряда по ошибкам до calib_end (MAD), месяцы `exclude` не используются.
    """
    S, T = d.shape
    e = np.full((S, T), np.nan)
    for t in range(k, T):
        lvl = d[:, t - k:t].mean(1)
        if t >= 12:
            base = d[:, t - 12:t].mean(1)
            lvl = lvl + alpha_season * (d[:, t - 12] - base)
        e[:, t] = d[:, t] - lvl
    cal = [t for t in range(k, calib_end + 1) if t not in exclude]
    med = np.nanmedian(e[:, cal], 1, keepdims=True)
    mad = np.nanmedian(np.abs(e[:, cal] - med), 1, keepdims=True) * 1.4826
    # нижняя граница σ — 25-й перцентиль по рядам категории не нужен: берём глобальный минимум
    mad = np.maximum(mad, np.nanpercentile(mad, 5))
    return (e - med) / mad


def levels(d: np.ndarray, calib_end: int, exclude: tuple = ()) -> np.ndarray:
    """Ряд уровней в единицах шума: x = (d − медиана) / σ, σ — робастный шум месячных приращений."""
    cal = [t for t in range(calib_end + 1) if t not in exclude]
    med = np.nanmedian(d[:, cal], 1, keepdims=True)
    dif = np.diff(d[:, : calib_end + 1], axis=1)
    keep = [t for t in range(dif.shape[1]) if t not in exclude and t + 1 not in exclude]
    sig = np.nanmedian(np.abs(dif[:, keep] - np.nanmedian(dif[:, keep], 1, keepdims=True)), 1, keepdims=True)
    sig = sig * 1.4826 / np.sqrt(2)
    sig = np.maximum(sig, np.nanpercentile(sig, 5))
    return (d - med) / sig


def deviation(x: np.ndarray, ref=(12, 4), season_alpha: float = 0.5, ref_start: int = 0) -> np.ndarray:
    """Отклонение от лагированного эталона с поправкой на собственную сезонность:
    c[t] = x[t] − медиана x[t−12 … t−4] − α·(x[t−12] − медиана x за тот год).

    Эталон отстаёт на 4 месяца, поэтому свежий сдвиг не «съедается» эталоном, а медленный
    локальный дрейф — съедается (нет накопления, как у классического CUSUM). Сезонный член
    убирает местные особенности декабря/января, которые не сняла национальная компонента.
    ref_start — эталон не заходит раньше известного системного разрыва (артефакт 2023-04)."""
    S, T = x.shape
    c = np.full((S, T), np.nan)
    for t in range(ref[1], T):
        lo = max(0, t - ref[0])
        if t - ref[1] >= ref_start:
            lo = max(lo, ref_start)
        c[:, t] = x[:, t] - np.nanmedian(x[:, lo: t - ref[1] + 1], 1)
        if t >= 12 and season_alpha:
            yr = x[:, t - 12 - min(t - 12, 6): t - 12 + 7]
            c[:, t] -= season_alpha * (x[:, t - 12] - np.nanmedian(yr, 1))
    return c


# ----------------------------------------------------------------------------- детекторы

def cusum_w(c, k=0.5, window=3, **_):
    """Оконный двусторонний CUSUM: max по началу τ в последних `window` месяцах сумм (c − k).
    Стационарен во времени (в отличие от классического), ловит устойчивый сдвиг за 1–3 мес."""
    S, T = c.shape
    out = np.full((S, T), np.nan)
    cc = np.nan_to_num(c)
    for t in range(T):
        best = np.zeros(S)
        sp = np.zeros(S)
        sm = np.zeros(S)
        for u in range(t, max(-1, t - window), -1):
            sp = sp + cc[:, u] - k
            sm = sm - cc[:, u] - k
            best = np.maximum(best, np.maximum(sp, sm))
        out[:, t] = best
    return out

def zscore(z, **_):
    return np.abs(z)


def cusum(z, k=0.5, **_):
    """Двусторонний CUSUM Пейджа на z; score = max(S+, S−)."""
    S, T = z.shape
    sp = np.zeros(S)
    sm = np.zeros(S)
    out = np.full((S, T), np.nan)
    for t in range(T):
        x = np.nan_to_num(z[:, t])
        sp = np.maximum(0, sp + x - k)
        sm = np.maximum(0, sm - x - k)
        out[:, t] = np.maximum(sp, sm)
    return out


def ewma(z, lam=0.4, **_):
    S, T = z.shape
    m = np.zeros(S)
    out = np.full((S, T), np.nan)
    norm = np.sqrt(lam / (2 - lam))
    for t in range(T):
        m = lam * np.nan_to_num(z[:, t]) + (1 - lam) * m
        out[:, t] = np.abs(m) / norm
    return out


def page_hinkley(z, delta=0.25, **_):
    """Page–Hinkley: накопленное отклонение от текущего среднего ряда (двусторонний)."""
    S, T = z.shape
    mean = np.zeros(S)
    cum_up = np.zeros(S)
    cum_dn = np.zeros(S)
    min_up = np.zeros(S)
    max_dn = np.zeros(S)
    out = np.full((S, T), np.nan)
    for t in range(T):
        x = np.nan_to_num(z[:, t])
        mean = mean + (x - mean) / (t + 1)
        cum_up += x - mean - delta
        cum_dn += x - mean + delta
        min_up = np.minimum(min_up, cum_up)
        max_dn = np.maximum(max_dn, cum_dn)
        out[:, t] = np.maximum(cum_up - min_up, max_dn - cum_dn)
    return out


def glr(z, window=8, **_):
    """GLR сдвига среднего внутри окна: max по точке разрыва τ последних `window` месяцев
    статистики (n1·n2/n)·(mean_после − mean_до)² — аналог онлайн-PELT для одного разрыва."""
    S, T = z.shape
    out = np.full((S, T), np.nan)
    for t in range(T):
        w = np.nan_to_num(z[:, max(0, t - window + 1): t + 1])
        n = w.shape[1]
        best = np.zeros(S)
        for tau in range(1, n):
            a, b = w[:, :tau], w[:, tau:]
            stat = tau * (n - tau) / n * (b.mean(1) - a.mean(1)) ** 2
            # интересует только недавний разрыв: τ в последних 3 месяцах окна
            if n - tau <= 3:
                best = np.maximum(best, stat)
        out[:, t] = best
    return out


def bocpd(z, hazard=1 / 24, mu0=0.0, kappa0=1.0, sigma=1.0, recent=1, hazard_mult=None, **_):
    """Байесовское онлайн-обнаружение разрывов (Adams & MacKay, 2007) для сдвига среднего:
    гауссово правдоподобие с известной дисперсией, сопряжённое нормальное априорное для среднего.

    Ветка «разрыв в t» оценивает x_t по априорному предсказательному (новый режим начинается
    с x_t), поэтому вероятность разрыва в месяц t зависит от самого x_t. score — апостериорная
    лог-шансы того, что текущий режим начался не раньше `recent` месяцев назад:
    log P(r_t ≤ recent) − log P(r_t > recent) (вероятность насыщается до 1, лог-шансы — нет).
    hazard_mult (S × T) — множитель hazard из новостей: событие в МО повышает априорную
    вероятность разрыва в этом месяце (§7 SOLUTION.md)."""
    S, T = z.shape
    logR = np.zeros((S, 1))                              # log P(r_{t-1} | x_1..t-1)
    mus = np.full((S, 1), mu0)
    kap = np.full((S, 1), kappa0)
    prior_var = sigma ** 2 * (1 + 1 / kappa0)
    out = np.full((S, T), np.nan)
    for t in range(T):
        x = np.nan_to_num(z[:, t])[:, None]
        pred_var = sigma ** 2 * (1 + 1 / kap)
        logpred = -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (x - mus) ** 2 / pred_var
        logprior = -0.5 * np.log(2 * np.pi * prior_var) - 0.5 * (x - mu0) ** 2 / prior_var
        h = np.full((S, 1), hazard)
        if hazard_mult is not None:
            h = np.clip(h * hazard_mult[:, t:t + 1], 1e-4, 0.9)
        growth = logR + logpred + np.log(1 - h)
        cp = logsumexp(logR, axis=1, keepdims=True) + np.log(h) + logprior
        logR = np.concatenate([cp, growth], axis=1)
        logR -= logsumexp(logR, axis=1, keepdims=True)
        # обновление апостериорных средних: новая ветка уже видела x_t
        mus = np.concatenate([(kappa0 * mu0 + x) / (kappa0 + 1), (kap * mus + x) / (kap + 1)], axis=1)
        kap = np.concatenate([np.full((S, 1), kappa0 + 1), kap + 1], axis=1)
        short = logsumexp(logR[:, : recent + 1], axis=1)
        long = logsumexp(logR[:, recent + 1:], axis=1) if logR.shape[1] > recent + 1 else np.full(S, -np.inf)
        out[:, t] = short - long
    return out


# ----------------------------------------------------------------------------- панельные

def stouffer_mo(z, mo_rows, base=None, **kw):
    """Сумма Стауффера по 5 непересекающимся категориям МО («Все категории» исключены), затем
    базовый детектор на агрегате. Тревога МО раздаётся всем его рядам."""
    base = base or cusum_w
    sub = mo_rows[:, 1:]                                   # 5 категорий без «Все категории»
    zs = np.nansum(z[sub], axis=1) / np.sqrt(sub.shape[1])
    return base(zs, **kw)


def spatial(z, nb_rows, base=None, w_self=0.5, **kw):
    """Пространственный детектор: смесь собственного z и среднего z соседей по автодороге
    (та же категория), нормированная к единичной дисперсии. Ловит региональные шоки (паводки)."""
    nbz = np.where(nb_rows[:, :, None] >= 0, z[nb_rows.clip(0)], np.nan)
    m = np.nanmean(nbz, axis=1)
    k = np.maximum((nb_rows >= 0).sum(1, keepdims=True), 1)
    base = base or cusum_w
    comb = w_self * z + (1 - w_self) * m
    scale = np.sqrt(w_self ** 2 + (1 - w_self) ** 2 / k)    # при независимых z
    return base(comb / scale, **kw)


def combo_max(scores: list[np.ndarray], thresholds: list[float]) -> np.ndarray:
    """ИЛИ-комбинация: max нормированных порогами score (порог итога снова калибруется)."""
    return np.nanmax(np.stack([s / th for s, th in zip(scores, thresholds)]), axis=0)
