"""Шаг 9. Лендинг (reports/site/index.html) и презентация-слайды (reports/site/slides*.html → PDF).

Цифры берутся из outputs/tables и встраиваются в страницу как JSON; графики рисует Chart.js и
перекрашивает при смене темы (светлая / тёмная). Слайды 16:9 печатаются в PDF:
`python scripts/09_site.py --pdf` (нужен Microsoft Edge или Chrome).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbx.data import load_config, load_dict  # noqa: E402

T = ROOT / "outputs" / "tables"
OUT = ROOT / "reports" / "site"
OUT.mkdir(parents=True, exist_ok=True)
HS = [1, 3, 6, 12]


def csv(name):
    f = T / name
    return pd.read_csv(f) if f.exists() else None


def num(x, d=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if np.isfinite(x) else None


def sig_rows(df, label):
    return [{"vs": label, "h": int(r.h), "a": num(r.MAE_a), "b": num(r.MAE_b), "rel": num(100 * r.rel),
             "lo": num(r.ci_low), "hi": num(r.ci_high), "months": num(100 * r.share_months_better),
             "mo": num(100 * r.share_mo_better, 1), "p": num(r.sign_p, 3), "dm": num(r.dm_p, 3)} for _, r in df.iterrows()]


def build_data() -> dict:
    D = {}
    # --- прогноз, вся панель
    full = csv("forecast_metrics_full.csv")
    t = full[full.split == "test"]
    mae = t.pivot_table(index="model", columns="h", values="MAE")
    r2 = t.pivot_table(index="model", columns="h", values="R2_yoy")
    D["models"] = [{"name": m, "mae": [num(mae.loc[m, h]) for h in HS], "r2": [num(r2.loc[m, h], 2) for h in HS]}
                   for m in mae.sort_values(1).index]
    pick = {"Ансамбль (Caruana по группам h)": "Ансамбль", "TimesFM-2.5 на d": "TimesFM-2.5",
            "Chronos-2 (6 категорий совместно) на d": "Chronos-2", "LightGBM на d": "LightGBM",
            "Панель: уровень + сезонность d": "Простая панельная", "Prophet (по умолчанию) — вся панель": "Prophet"}
    D["mae_chart"] = [{"name": s, "values": [num(mae.loc[m, h]) for h in HS]} for m, s in pick.items() if m in mae.index]
    # --- значимость
    sig = []
    sf = csv("ensemble_vs_prophet_full_panel.csv")
    if sf is not None:
        sig += sig_rows(sf[sf.split == "test"], "Prophet по умолчанию, вся панель")
    ss = csv("ensemble_vs_prophet.csv")
    if ss is not None:
        sig += sig_rows(ss[(ss.split == "test") & ss.prophet.str.contains("лучшая")], "Лучший Prophet задним числом")
    D["sig"] = sig
    # --- FM: сырой ряд против d (h = 1), LoRA
    smp = csv("forecast_metrics_sample.csv")
    m1 = smp[(smp.split == "test") & (smp.h == 1)].set_index("model").MAE
    pairs = [("Chronos-Bolt base", "Chronos-Bolt (base) на сыром ряду", "Chronos-Bolt (base) на d"),
             ("Chronos-Bolt small", "Chronos-Bolt (small) на сыром ряду", "Chronos-Bolt (small) на d"),
             ("Chronos-2", "Chronos-2 на сыром ряду", "Chronos-2 (одномерный) на d"),
             ("TimesFM-2.5", "TimesFM-2.5 на сыром ряду", "TimesFM-2.5 на d")]
    D["fm_ablation"] = [{"name": n, "raw": num(m1[r]), "d": num(m1[dd])} for n, r, dd in pairs if r in m1 and dd in m1]
    lora = csv("fm_finetune_lora.csv")
    if lora is not None:
        lt = lora[lora.split == "test"].pivot_table(index="model", columns="h", values="MAE")
        D["lora"] = [{"name": m, "mae": [num(lt.loc[m, h]) if h in lt.columns else None for h in HS]} for m in lt.index]
    # --- детекция
    far = csv("detection_far_curve_auc.csv")
    D["far_curve"] = [{"name": r.detector, "values": [num(100 * r["0.01"], 1), num(100 * r["0.03"], 1), num(100 * r["0.05"], 1)]}
                      for _, r in far.iterrows()]
    mag = csv("detection_recall_by_magnitude.csv").set_index("detector")
    D["magnitude"] = [{"name": n, "values": [num(100 * v, 1) for v in mag.loc[n].values]}
                      for n in ["BOCPD + новости", "BOCPD", "z-score (инновация)", "Stouffer-МО"] if n in mag.index]
    summ = csv("detection_summary_far3.csv")
    D["detectors"] = [{"name": r.detector, "recall": num(r.recall, 3), "onset": num(r.recall_at_onset, 3), "delay": num(r.delay, 2),
                       "precision": num(r.precision, 3), "f1": num(r.F1, 3)} for _, r in summ.iterrows()]
    cmp = csv("detection_comparable.csv")          # tools/detection_comparable.py
    if cmp is not None:
        c = cmp[cmp["детектор"] == "BOCPD + новости"]
        D["comparable"] = [{"setup": r["постановка"], "far": int(r["ложных тревог на 100 ряд-мес"]),
                            "recall": num(r["полнота"], 2), "onset": num(r["в месяц шока"], 2),
                            "precision": num(r["точность"], 2), "f1": num(r["F1"], 2)} for _, r in c.iterrows()]
        big = c[c["постановка"] == "Шоки 10–20 %"]
        D["recall_big"] = num(100 * big["полнота"].iloc[0]) if len(big) else None
    real = csv("detection_real_events.csv")
    D["real"] = [{"name": r["детектор"], "named": num(r["паводок: МО из новостей, %"], 1), "base": num(r["паводок: страна, %"], 1),
                  "lift": num(r["lift паводка"], 2), "p": num(r["p паводок (Фишер)"], 3),
                  "acts": num(r["режимы ЧС (акты, вне паводка), %"], 1), "placebo": num(r["плацебо (случайный месяц), %"], 1),
                  "lift_acts": num(r["lift актов"], 2)} for _, r in real.iterrows()]
    nc = csv("detection_flood_noncircular.csv")
    if nc is not None:
        r = nc[nc["детектор"].str.contains("без ГУ МЧС")].iloc[0]
        D["noncirc"] = {"lift": num(r["lift"], 2), "p": num(r["p (Фишер)"], 3), "n": int(r["МО из сообщений ГУ МЧС"])}
    # --- новости
    es = csv("event_study_flood2024.csv")
    D["flood"] = [{"label": [r.category, "апрель" if str(r.month).endswith("04") else "май"], "named": num(r["named_%"], 1),
                   "region": num(r["same_region_%"], 1), "rest": num(r["rest_%"], 1)}
                  for _, r in es.iterrows() if r.category in ("Маркетплейсы", "Все категории", "Транспорт")]
    au = csv("news_manual_audit_summary.csv")
    D["audit"] = {k: num(v, 1) for k, v in au.iloc[0].items()} if au is not None else {}
    sens = csv("news_sim_sensitivity.csv")
    D["sens_max"] = num(sens["прирост, п.п."].max(), 1) if sens is not None else None
    gpu = ROOT / "outputs" / "colab_t4" / "tables"
    ea = pd.read_csv(gpu / "event_study_all_events.csv") if (gpu / "event_study_all_events.csv").exists() \
        else csv("event_study_all_events.csv")
    regions_gpu = pd.read_csv(gpu / "mchs_regions.csv") if (gpu / "mchs_regions.csv").exists() else None
    off = ROOT / "data/news/events_official.parquet"
    if off.exists():
        eo = pd.read_parquet(off)
        m = eo[eo.source == "mchs"]
        D["mchs"] = {"n": int(len(m)), "mo": int(m.territory_ids.explode().dropna().nunique())}
    D["all_events"] = None
    if ea is not None and len(ea):
        alpha = 0.05 / (len(ea) - 1)                # Бонферрони по категориям
        es_sig = ea[(ea["категория"] != "Все категории") & (ea["p (кластерный бутстреп)"] < alpha)]
        D["all_events"] = {"n": int(ea["событий"].iloc[0]), "regions": int(ea["регионов"].iloc[0]),
                           "mchs_regions": int(len(regions_gpu)) if regions_gpu is not None else None,
                           "sig": [{"cat": r["категория"], "shift": num(r["сдвиг ошибки, σ"], 2),
                                    "p": num(r["p (кластерный бутстреп)"], 3)} for _, r in es_sig.iterrows()]}
        if "ДИ низ" in ea.columns:
            D["es_cats"] = [{"cat": r["категория"], "shift": num(r["сдвиг ошибки, σ"], 3), "lo": num(r["ДИ низ"], 3),
                             "hi": num(r["ДИ верх"], 3), "p": num(r["p (кластерный бутстреп)"], 4)} for _, r in ea.iterrows()]
    # --- реальные эпизоды
    cons = pd.read_parquet(ROOT / "data/hackathonlicence/consumption.parquet")
    dct = load_dict(load_config())
    la = csv("local_alarms_2024.csv")
    cases = []
    for nm, note in [("Торопецкий", "атака и эвакуация жителей, сентябрь 2024"),
                     ("Орск", "прорыв дамбы, 5 апреля 2024"), ("Саракташский", "паводок, апрель 2024")]:
        hit = dct.index[dct.municipal_district_name_short.astype(str) == nm]
        if not len(hit):
            continue
        tid = int(hit[0])
        s = cons[(cons.territory_id == tid) & (cons.category == "Маркетплейсы")].sort_values("date")
        al = la[(la.territory_id == tid) & (la.category == "Маркетплейсы")].month.tolist() if la is not None else []
        cases.append({"name": str(dct.loc[tid, "municipal_district_name"]), "note": note,
                      "dates": s.date.astype(str).tolist(), "values": s.value.round().astype(int).tolist(), "alarms": al})
    D["cases"] = cases
    # --- проверки
    ex = csv("external_check_2025.csv")
    if ex is not None:
        r = ex.iloc[0]
        D["external"] = {"pred": num(r["прогноз роста 2025/2024, %"], 1), "fact": num(r["факт портала 2025/2024, %"], 1),
                         "corr": num(r["корреляция помесячного профиля 2025"], 2)}
    dec = csv("error_decomposition.csv")
    if dec is not None:
        dt = dec[dec.split == "test"].set_index("model")
        D["decomp"] = {"ens": [num(dt.loc["ансамбль", str(h)]) for h in HS],
                       "oracle": [num(dt.loc["ансамбль с истинной n", str(h)]) for h in HS]}
    rel = {s["h"]: s["rel"] for s in sig if s["vs"].startswith("Prophet по умолчанию")}
    D["kpi"] = {"rel": [rel.get(h) for h in HS], "recall": num(100 * summ.recall.max()),
                "random": num(100 * far.set_index("detector").loc["Случайные тревоги (ориентир)", "0.03"]),
                "lift": D["real"][0]["lift"] if D["real"] else None, "recall_big": D.get("recall_big")}
    return D


# ------------------------------------------------------------------ общий стиль и графики
CSS = r"""
:root{--bg:#ffffff;--bg2:#f3f7f4;--card:#ffffff;--ink:#0b0d0c;--ink2:#46504c;--ink3:#7a8580;--line:#e1e8e3;
--green:#21a038;--green2:#107f3c;--glow:rgba(33,160,56,.14);--grid:#e8eee9;
--s1:#21a038;--s2:#0f7f8a;--s3:#8fd19e;--s4:#2b3a33;--s5:#c3ccc7;--neg:#8b9490;--shadow:0 12px 32px rgba(10,40,20,.08)}
[data-theme=dark]{--bg:#090b0a;--bg2:#101412;--card:#131816;--ink:#f2f5f3;--ink2:#a7b2ad;--ink3:#78837e;--line:#222a26;
--green:#2bc24a;--green2:#62e27d;--glow:rgba(43,194,74,.16);--grid:#1c2320;
--s1:#2bc24a;--s2:#38bcc7;--s3:#a5e6b4;--s4:#e6ece8;--s5:#3b4641;--neg:#87928d;--shadow:0 12px 32px rgba(0,0,0,.5)}
*{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--ink);font:16px/1.6 Manrope,system-ui,-apple-system,"Segoe UI",sans-serif;transition:background .4s,color .4s;overflow-x:hidden}
a{color:var(--green);text-decoration:none}code{font-family:ui-monospace,Consolas,monospace;font-size:.9em;background:var(--bg2);padding:2px 6px;border-radius:6px}
"""

CHART_JS = r"""
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const FONT="Manrope, system-ui, -apple-system, 'Segoe UI', Arial, sans-serif";
Chart.defaults.font.family=FONT;
const charts=[];
function fmt(v,d=0){if(v==null)return '—';const s=Math.abs(v).toLocaleString('ru-RU',{minimumFractionDigits:d,maximumFractionDigits:d});return (v<0?'−':'')+s}
function O(extra){const ink=css('--ink2'),grid=css('--grid'),f={family:FONT,size:12};
 const o={responsive:true,maintainAspectRatio:false,animation:ANIM?{duration:1100,easing:'easeOutQuart'}:false,
  plugins:{legend:{labels:{color:ink,font:f,boxWidth:10,boxHeight:10,usePointStyle:true,padding:14}},
   tooltip:{backgroundColor:css('--card'),titleColor:css('--ink'),bodyColor:css('--ink2'),borderColor:css('--line'),borderWidth:1,padding:10,titleFont:{family:FONT,weight:'700'},bodyFont:f}},
  scales:{x:{ticks:{color:ink,font:f},grid:{color:grid},border:{color:grid}},y:{ticks:{color:ink,font:f},grid:{color:grid},border:{color:grid},beginAtZero:true}}};
 if(extra.xTitle)o.scales.x.title={display:true,text:extra.xTitle,color:css('--ink3'),font:f};
 if(extra.yTitle)o.scales.y.title={display:true,text:extra.yTitle,color:css('--ink3'),font:f};
 if(extra.yZero===false)o.scales.y.beginAtZero=false;
 if(extra.noLegend)o.plugins.legend.display=false;
 if(extra.xMax)o.scales.x.ticks.maxTicksLimit=extra.xMax;
 if(extra.unit)o.plugins.tooltip.callbacks={label:c=>' '+c.dataset.label+': '+fmt(c.parsed.y,extra.dec||0)+extra.unit};
 return o}
const PAL=()=>[css('--s1'),css('--s2'),css('--s4'),css('--s3'),css('--s5'),css('--neg')];
function mk(id,cfg){const el=document.getElementById(id);if(!el)return;charts.push({el,cfg,c:new Chart(el,cfg())})}
function rebuild(){charts.forEach(x=>{x.c.destroy();x.c=new Chart(x.el,x.cfg())})}
function line(s,i,dashed){const P=PAL(),c=dashed?css('--neg'):P[i%P.length];return{label:s.name,data:s.values,borderColor:c,backgroundColor:c,
 borderWidth:i===0?4:2,borderDash:dashed?[6,5]:[],pointRadius:i===0?5:3,pointHoverRadius:7,tension:.25}}
function drawAll(){
 const H=['1 мес.','3 мес.','6 мес.','12 мес.'];
 mk('chMae',()=>({type:'line',data:{labels:H,datasets:DATA.mae_chart.map((s,i)=>line(s,i,s.name==='Prophet'))},options:O({yTitle:'MAE, ₽ на жителя в месяц',unit:' ₽'})}));
 mk('chFm',()=>({type:'bar',data:{labels:DATA.fm_ablation.map(x=>x.name),datasets:[
  {label:'сырой ряд',data:DATA.fm_ablation.map(x=>x.raw),backgroundColor:css('--s5'),borderRadius:8},
  {label:'отклонение МО (наш вход)',data:DATA.fm_ablation.map(x=>x.d),backgroundColor:css('--s1'),borderRadius:8}]},options:O({yTitle:'MAE на 1 мес., ₽',unit:' ₽'})}));
 const keep=['BOCPD + новости','BOCPD','z-score (инновация)','Stouffer-МО','Случайные тревоги (ориентир)'];
 const far=keep.map(n=>DATA.far_curve.find(x=>x.name===n)).filter(Boolean);
 mk('chFar',()=>({type:'line',data:{labels:['1 %','3 %','5 %'],datasets:far.map((s,i)=>line(s,i,s.name.startsWith('Случайные')))},
  options:O({xTitle:'ложных тревог на 100 ряд-месяцев',yTitle:'полнота, %',unit:' %',dec:1})}));
 mk('chMag',()=>({type:'bar',data:{labels:['3 %','5 %','10 %','20 %'],datasets:DATA.magnitude.map((s,i)=>({label:s.name,data:s.values,backgroundColor:PAL()[i],borderRadius:6}))},
  options:O({xTitle:'величина шока',yTitle:'полнота при 3 % ложных, %',unit:' %',dec:1})}));
 mk('chFlood',()=>({type:'bar',data:{labels:DATA.flood.map(x=>x.label),datasets:[
  {label:'МО из сообщений о ЧС',data:DATA.flood.map(x=>x.named),backgroundColor:css('--s1'),borderRadius:6},
  {label:'другие МО тех же регионов',data:DATA.flood.map(x=>x.region),backgroundColor:css('--s2'),borderRadius:6},
  {label:'остальная страна',data:DATA.flood.map(x=>x.rest),backgroundColor:css('--s5'),borderRadius:6}]},
  options:O({yTitle:'ошибка прогноза на 1 мес., %',yZero:false,unit:' %',dec:1})}));
 if(DATA.es_cats){const es=DATA.es_cats.filter(x=>x.cat!=='Все категории').concat(DATA.es_cats.filter(x=>x.cat==='Все категории'));
  const thr=0.05/Math.max(1,DATA.es_cats.length-1),sg=x=>x.p<thr&&x.cat!=='Все категории';
  mk('chEs',()=>{const o=O({yTitle:'сдвиг ошибки прогноза, σ',yZero:false});
   o.plugins.tooltip.callbacks={label:c=>c.datasetIndex===0?' 95 % ДИ: '+fmt(c.raw[0],2)+' … '+fmt(c.raw[1],2)+'σ':' сдвиг '+fmt(c.parsed.y,2)+'σ, p '+(es[c.dataIndex].p<0.001?'< 0,001':'= '+fmt(es[c.dataIndex].p,3))};
   return{type:'bar',data:{labels:es.map(x=>x.cat),datasets:[
    {label:'95 % ДИ',data:es.map(x=>[x.lo,x.hi]),backgroundColor:es.map(x=>sg(x)?css('--glow'):'rgba(128,128,128,.14)'),borderColor:es.map(x=>sg(x)?css('--green'):css('--s5')),borderWidth:1.5,borderRadius:8,borderSkipped:false,barPercentage:.45},
    {type:'line',label:'сдвиг, σ',data:es.map(x=>x.shift),showLine:false,pointRadius:8,pointHoverRadius:10,pointBackgroundColor:es.map(x=>sg(x)?css('--green'):css('--ink3')),pointBorderColor:css('--card'),pointBorderWidth:2}]},options:o}});}
 DATA.cases.forEach((c,k)=>mk('chCase'+k,()=>({type:'line',data:{labels:c.dates,datasets:[
  {label:'маркетплейсы, ₽ на жителя',data:c.values,borderColor:css('--ink2'),backgroundColor:css('--ink2'),borderWidth:2,pointRadius:0,tension:.25},
  {label:'тревога детектора',data:c.dates.map((d,i)=>c.alarms.includes(d)?c.values[i]:null),showLine:false,pointRadius:7,pointHoverRadius:9,
   pointBackgroundColor:css('--green'),pointBorderColor:css('--card'),pointBorderWidth:2}]},
  options:O({yZero:false,noLegend:k>0,xMax:6,unit:' ₽'})})));
}
function tbl(head,rows,hl){return '<div class="tw"><table><thead><tr>'+head.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+
 rows.map((r,i)=>'<tr'+(hl&&hl(i)?' class="hl"':'')+'>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
function li(xs){return xs.filter(Boolean).map(x=>'<li>'+x+'</li>').join('')}
const LORA=()=>{const l=DATA.lora||[];return{L:l.find(x=>x.name.includes('LoRA')),Z:l.find(x=>x.name.includes('zero'))}};
"""

SITE = r"""<!doctype html><html lang="ru" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Сейсмограф трат — прогноз расходов МО и раннее обнаружение шоков</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>(function(){let t=null;try{t=localStorage.getItem('theme')}catch(e){}if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';document.documentElement.dataset.theme=t})()</script>
<style>__CSS__
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
.progress{position:fixed;top:0;left:0;height:3px;background:linear-gradient(90deg,var(--green),var(--s2));width:0;z-index:60}
header{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--bg) 84%,transparent);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
.bar{display:flex;align-items:center;gap:18px;height:64px}
.mark{width:28px;height:28px;border-radius:50%;background:conic-gradient(from 210deg,var(--green),var(--s2),var(--green));box-shadow:0 0 0 4px var(--glow);flex:none;animation:spin 12s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.brand{font-weight:800;letter-spacing:-.01em;white-space:nowrap}
nav{display:flex;gap:2px;margin-left:auto;overflow-x:auto;scrollbar-width:none;min-width:0}nav::-webkit-scrollbar{display:none}
nav a{color:var(--ink2);font-weight:600;font-size:14px;padding:8px 12px;border-radius:999px;white-space:nowrap;transition:.25s}
nav a:hover{color:var(--ink);background:var(--bg2)}nav a.on{color:var(--green);background:var(--glow)}
.tgl{flex:none;width:42px;height:42px;border-radius:50%;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;display:grid;place-items:center;transition:.3s}
.tgl:hover{border-color:var(--green);color:var(--green);transform:rotate(-20deg)}.tgl svg{width:19px;height:19px}
[data-theme=light] .moon,[data-theme=dark] .sun{display:none}
.hero{position:relative;padding:92px 0 76px;overflow:hidden}
.blob{position:absolute;border-radius:50%;filter:blur(72px);pointer-events:none;animation:float 16s ease-in-out infinite}
.b1{width:540px;height:540px;background:var(--glow);top:-170px;right:-140px}
.b2{width:400px;height:400px;background:color-mix(in srgb,var(--s2) 18%,transparent);bottom:-200px;left:-100px;animation-delay:-7s}
@keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-46px,34px) scale(1.1)}}
.rings{position:absolute;right:-90px;top:-40px;width:460px;height:460px;pointer-events:none;opacity:.8}
.rings circle{fill:none;stroke:var(--green);stroke-width:1.2;opacity:.25;transform-origin:center;animation:pulse 6s ease-out infinite}
.rings circle:nth-child(2){animation-delay:2s}.rings circle:nth-child(3){animation-delay:4s}
@keyframes pulse{0%{transform:scale(.3);opacity:.5}100%{transform:scale(1);opacity:0}}
.eyebrow{display:inline-flex;gap:8px;align-items:center;font-weight:700;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--green);background:var(--glow);padding:7px 14px;border-radius:999px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:blink 1.6s ease-in-out infinite}@keyframes blink{50%{opacity:.25}}
h1{font-size:clamp(34px,5.4vw,64px);line-height:1.05;letter-spacing:-.035em;font-weight:800;margin:22px 0 18px;max-width:900px}
h1 em{font-style:normal;background:linear-gradient(90deg,var(--green),var(--s2),var(--green));background-size:200% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 6s linear infinite}
@keyframes shine{to{background-position:200% center}}
.lead{font-size:clamp(17px,1.6vw,20px);color:var(--ink2);max-width:740px}
.author{margin-top:18px;color:var(--ink2);font-size:16px}.author b{color:var(--ink)}
.cta{display:flex;gap:12px;margin-top:30px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;padding:13px 22px;border-radius:999px;font-weight:700;transition:.25s;border:1px solid var(--line);color:var(--ink);background:var(--card)}
.btn.pri{background:var(--green);color:#fff;border-color:var(--green)}.btn:hover{transform:translateY(-2px);box-shadow:var(--shadow)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:56px;position:relative}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:22px;box-shadow:var(--shadow);transition:transform .3s,border-color .3s}
.kpi:hover{transform:translateY(-4px);border-color:var(--green)}
.kpi .v{font-size:clamp(30px,3.4vw,44px);font-weight:800;letter-spacing:-.03em;color:var(--green);line-height:1.1;font-variant-numeric:tabular-nums}
.kpi .l{color:var(--ink2);font-size:14px;margin-top:6px}
section{padding:88px 0;border-top:1px solid var(--line)}section.alt{background:var(--bg2)}
.sh{display:flex;align-items:baseline;gap:14px;margin-bottom:12px}.sh .n{font-weight:800;color:var(--green);font-size:15px}
h2{font-size:clamp(28px,3.6vw,42px);letter-spacing:-.03em;line-height:1.1;font-weight:800}
.sub{color:var(--ink2);max-width:780px;margin-bottom:34px;font-size:17px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:20px}.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.mt{margin-top:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:26px;transition:border-color .3s,box-shadow .3s,transform .3s;min-width:0}
.card:hover{border-color:color-mix(in srgb,var(--green) 50%,var(--line));box-shadow:var(--shadow)}
.card h3{font-size:19px;letter-spacing:-.01em;margin-bottom:6px}.card>p{color:var(--ink2);font-size:15px}
.step:hover{transform:translateY(-4px)}
.step .num{width:40px;height:40px;border-radius:12px;background:var(--glow);color:var(--green);font-weight:800;display:grid;place-items:center;margin-bottom:14px;transition:.3s}
.step:hover .num{background:var(--green);color:#fff}
.chart{position:relative;height:340px;margin-top:14px}.chart.sm{height:230px}
.big{font-size:44px;font-weight:800;letter-spacing:-.03em;color:var(--green);line-height:1.05;margin:6px 0 8px}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:16px;background:var(--card);margin-top:14px}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:10px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--ink3);font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.04em;background:var(--bg2)}
tr:last-child td{border-bottom:none}tbody tr{transition:background .2s}tbody tr:hover{background:var(--bg2)}
tr.hl td{font-weight:800}tr.hl td:first-child{color:var(--green)}.good{color:var(--green);font-weight:700}
details{margin-top:16px}summary{cursor:pointer;color:var(--green);font-weight:700;list-style:none;padding:6px 0}summary::-webkit-details-marker{display:none}
summary::before{content:'+';display:inline-block;width:18px;transition:.3s}details[open] summary::before{transform:rotate(45deg)}
.pill{display:inline-block;padding:4px 11px;border-radius:999px;background:var(--glow);color:var(--green);font-weight:700;font-size:12px}
.flow{width:100%;height:auto;display:block}.flow .node rect{fill:var(--card);stroke:var(--line);stroke-width:1.5}.flow .node.acc rect{stroke:var(--green);stroke-width:2.5}
.flow text{fill:var(--ink);font:700 15px Manrope,sans-serif}.flow text.sm{fill:var(--ink3);font-size:12px;font-weight:500}
.flow .edge{stroke:var(--green);stroke-width:2;fill:none;stroke-dasharray:7 7;animation:dash 1.4s linear infinite}
@keyframes dash{to{stroke-dashoffset:-28}}
.flow .node{opacity:0;transform:translateY(10px);transition:opacity .6s,transform .6s}.reveal.in .flow .node{opacity:1;transform:none}
.lvl{display:grid;grid-template-columns:220px 1fr;gap:18px;padding:16px 0;border-bottom:1px solid var(--line)}.lvl:last-child{border:none}.lvl b{color:var(--green)}.lvl div{color:var(--ink2)}
ul.clean{list-style:none;margin-top:8px}ul.clean li{padding:8px 0 8px 26px;position:relative;color:var(--ink2);font-size:15px}
ul.clean li::before{content:'';position:absolute;left:3px;top:16px;width:9px;height:9px;border-radius:3px;background:var(--green)}
ul.clean b{color:var(--ink)}
footer{padding:44px 0;color:var(--ink3);font-size:14px;border-top:1px solid var(--line)}
.reveal{opacity:0;transform:translateY(26px);transition:opacity .8s cubic-bezier(.2,.7,.2,1),transform .8s cubic-bezier(.2,.7,.2,1)}
.reveal.in{opacity:1;transform:none}
.g4>.reveal:nth-child(2),.g3>.reveal:nth-child(2),.g2>.reveal:nth-child(2),.kpis>.reveal:nth-child(2){transition-delay:.1s}
.g4>.reveal:nth-child(3),.g3>.reveal:nth-child(3),.kpis>.reveal:nth-child(3){transition-delay:.2s}
.g4>.reveal:nth-child(4),.kpis>.reveal:nth-child(4){transition-delay:.3s}
@media (max-width:900px){.kpis,.g4{grid-template-columns:repeat(2,1fr)}.g2,.g3{grid-template-columns:1fr}.lvl{grid-template-columns:1fr;gap:4px}.brand,.rings{display:none}.hero{padding:64px 0 56px}section{padding:64px 0}}
@media (max-width:520px){.kpis,.g4{grid-template-columns:1fr}.card{padding:20px}}
.mark{background:none!important;box-shadow:none!important;animation:none!important;display:grid;place-items:center}
.mark svg{width:30px;height:30px;overflow:visible}
.mark path{stroke:var(--green);stroke-width:2.6;fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:64;animation:ekg 2.4s linear infinite}
@keyframes ekg{0%{stroke-dashoffset:64}55%{stroke-dashoffset:0}100%{stroke-dashoffset:-64}}
.seis{position:relative;margin-top:40px;height:170px;border:1px solid var(--line);border-radius:22px;background:color-mix(in srgb,var(--card) 72%,transparent);overflow:hidden;box-shadow:var(--shadow)}
.seis canvas{width:100%;height:100%;display:block}
.seis .lab{position:absolute;top:12px;left:18px;font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3)}
.seis .prob{position:absolute;top:9px;right:14px;font-size:13px;font-weight:800;color:var(--ink2);font-variant-numeric:tabular-nums;padding:5px 12px;border-radius:999px;background:var(--bg2);transition:background .3s,color .3s,box-shadow .3s}
.seis .prob.hot{background:var(--green);color:#fff;box-shadow:0 0 0 6px var(--glow)}
.ticker{overflow:hidden;border-bottom:1px solid var(--line);background:var(--bg2);padding:16px 0;-webkit-mask-image:linear-gradient(90deg,transparent,#000 7%,#000 93%,transparent);mask-image:linear-gradient(90deg,transparent,#000 7%,#000 93%,transparent)}
.ticker .tr{display:flex;gap:52px;width:max-content;animation:tick 46s linear infinite}
.ticker:hover .tr{animation-play-state:paused}
.ticker span{font-weight:700;font-size:17px;white-space:nowrap;color:var(--ink2)}.ticker span i{font-style:normal;font-weight:800;color:var(--green);margin-right:4px}
@keyframes tick{to{transform:translateX(-50%)}}
.card{position:relative;overflow:hidden}
.card::after{content:'';position:absolute;inset:0;pointer-events:none;opacity:0;transition:opacity .35s;background:radial-gradient(440px circle at var(--mx,50%) var(--my,50%),var(--glow),transparent 62%)}
.card:hover::after{opacity:1}
.replay{display:grid;grid-template-columns:1.6fr 1fr;gap:20px}
.status{display:flex;align-items:center;gap:12px;margin:8px 0 4px}
.month{font-size:32px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;min-width:150px}
.chip{padding:7px 14px;border-radius:999px;font-weight:800;font-size:13px;background:var(--bg2);color:var(--ink2);transition:background .3s,color .3s,box-shadow .3s}
.chip.hot{background:var(--green);color:#fff;box-shadow:0 0 0 7px var(--glow);animation:beat .9s ease-in-out 2}
@keyframes beat{50%{transform:scale(1.1)}}
.feed{display:flex;flex-direction:column;gap:10px;margin-top:12px;min-height:250px}
.ev{padding:11px 14px;border-radius:14px;border:1px solid var(--line);background:var(--bg2);font-size:14px;color:var(--ink2);line-height:1.45;animation:pop .5s cubic-bezier(.2,.8,.2,1) both}
.ev b{color:var(--ink)}.ev.alarm{border-color:var(--green);background:var(--glow)}.ev.alarm b{color:var(--green)}
.ev.news{border-color:color-mix(in srgb,var(--s2) 55%,var(--line))}.ev.news b{color:var(--s2)}
@keyframes pop{from{opacity:0;transform:translateY(10px) scale(.97)}to{opacity:1;transform:none}}
.crit{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}
.crit .card{display:flex;gap:18px;align-items:flex-start}
.ring{flex:none;width:68px;height:68px}
.ring circle{fill:none;stroke-width:7}.ring .bg{stroke:var(--bg2)}
.ring .fg{stroke:var(--green);stroke-linecap:round;transform:rotate(-90deg);transform-origin:34px 34px;stroke-dasharray:176;stroke-dashoffset:176;transition:stroke-dashoffset 1.5s cubic-bezier(.2,.7,.2,1) .2s}
.reveal.in .ring .fg{stroke-dashoffset:0}
.ring text{fill:var(--ink);font:800 15px Manrope,sans-serif;text-anchor:middle;dominant-baseline:central}
.crit h3{font-size:17px}.crit p{color:var(--ink2);font-size:14px;margin:4px 0 8px}.crit a{font-weight:700;font-size:13px}
.gloss{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.gloss details{margin:0;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:12px 18px;transition:border-color .3s,box-shadow .3s}
.gloss details:hover{box-shadow:var(--shadow)}.gloss details[open]{border-color:var(--green)}
.gloss summary{color:var(--ink);font-weight:700}.gloss summary b{color:var(--green);margin-right:6px}
.gloss p{color:var(--ink2);font-size:15px;margin:6px 0 4px}
@media (max-width:900px){.replay,.gloss{grid-template-columns:1fr}.seis{height:130px}.crit{grid-template-columns:1fr}}
.static .ticker .tr{animation:none}.static .ring .fg{stroke-dashoffset:0;transition:none}
.static .reveal,.static .flow .node{opacity:1;transform:none;transition:none}
@media (prefers-reduced-motion:reduce){*,*::before{animation:none!important;transition:none!important}.reveal,.flow .node{opacity:1;transform:none}}
</style></head><body>
<div class="progress" id="prog"></div>
<header><div class="wrap bar"><div class="mark"><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M1 17h7l3-9 4 16 3-12 2 5h11"/></svg></div><div class="brand">Сейсмограф трат</div>
<nav id="nav"><a href="#idea">Идея</a><a href="#live">В действии</a><a href="#arch">Архитектура</a><a href="#fc">Прогноз</a><a href="#fm">Foundation models</a><a href="#det">Детекция</a><a href="#news">Новости</a><a href="#cases">Эпизоды</a><a href="#crit">Критерии</a><a href="#gloss">Глоссарий</a><a href="#check">Проверки</a></nav>
<button class="tgl" id="tgl" aria-label="Сменить тему" title="Светлая / тёмная тема"><svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg><svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg></button>
</div></header>

<div class="hero"><div class="blob b1"></div><div class="blob b2"></div>
<svg class="rings" viewBox="0 0 200 200"><circle cx="100" cy="100" r="98"/><circle cx="100" cy="100" r="98"/><circle cx="100" cy="100" r="98"/></svg>
<div class="wrap" style="position:relative">
<span class="eyebrow reveal"><span class="dot"></span>Сейсмограф трат · СберИндекс 2026 · Задача № 2</span>
<h1 class="reveal">Сейсмограф трат: <em>ранние толчки</em> в расходах муниципалитетов</h1>
<p class="lead reveal">Прогноз расходов на 1, 3, 6 и 12 месяцев и раннее обнаружение шоков. 2 016 муниципалитетов × 6 категорий безналичных трат. Ансамбль foundation models на отклонении МО от общей динамики страны, детектор разрывов с новостным априором и новости ГУ МЧС с привязкой к муниципалитетам.</p>
<p class="author reveal">Автор — <b>Максим Караханов</b></p>
<div class="cta reveal"><a class="btn pri" href="#fc">Смотреть результаты</a><a class="btn" href="../presentation.pdf">Презентация PDF</a><a class="btn" href="../methodology.md">Отчёт</a></div>
<div class="seis reveal"><canvas id="seis"></canvas><span class="lab">Сейсмограф · отклонение МО · иллюстрация работы детектора</span><span class="prob" id="seisP">P(разрыв) 0,02</span></div>
<div class="kpis" id="kpis"></div></div></div>
<div class="ticker" aria-hidden="true"><div class="tr" id="tick"></div></div>

<section id="idea"><div class="wrap"><div class="sh reveal"><span class="n">01</span><h2>Идея за минуту</h2></div>
<p class="sub reveal">Около 70 % колебаний расходов — общая динамика страны. Её берём из панели и длинных национальных рядов СберИндекса, а модели учим только на том, что отличает муниципалитет.</p>
<div class="g4">
<div class="card step reveal"><div class="num">1</div><h3>Две части</h3><p>log y = n + d: общая для страны часть n (сезонность, инфляция, онлайн-покупки) и отклонение МО d.</p></div>
<div class="card step reveal"><div class="num">2</div><h3>Ансамбль на d</h3><p>TimesFM-2.5, Chronos-2, LightGBM и простые модели; веса по горизонтам на январе–июне 2024.</p></div>
<div class="card step reveal"><div class="num">3</div><h3>Детектор шоков</h3><p>BOCPD на отклонении МО; порог — 3 ложные тревоги на 100 ряд-месяцев.</p></div>
<div class="card step reveal"><div class="num">4</div><h3>Новости — где искать</h3><p>Сообщения о ЧС по месту и дате публикации повышают априорную вероятность разрыва.</p></div>
</div></div></section>

<section id="live" class="alt"><div class="wrap"><div class="sh reveal"><span class="n">02</span><h2>Сейсмограф в действии</h2></div>
<p class="sub reveal">Реальные данные СберИндекса: маркетплейсы в Орске, 2023–2024. Месяц за месяцем — что видит детектор и когда приходят новости о ЧС.</p>
<div class="replay"><div class="card reveal"><h3 id="rpName"></h3><p>₽ на жителя в месяц · зелёные точки — тревоги BOCPD с новостным hazard</p><div class="chart"><canvas id="chReplay"></canvas></div></div>
<div class="card reveal"><h3>Пульт детектора</h3><div class="status"><span class="month" id="rpMonth">—</span><span class="chip" id="rpChip">норма</span></div>
<div class="feed" id="rpFeed"></div><button class="btn" id="rpBtn" style="margin-top:14px">▶ Запустить заново</button></div></div></div></section>

<section id="arch"><div class="wrap"><div class="sh reveal"><span class="n">03</span><h2>Архитектура</h2></div>
<p class="sub reveal">На каждом шаге только прошлое: модели видят историю до точки прогноза, новости — опубликованные к этому моменту, национальный ряд — с лагом 1 месяц.</p>
<div class="card reveal" style="overflow-x:auto"><svg class="flow" viewBox="0 0 1100 340" style="min-width:760px" role="img" aria-label="Схема решения">
<path class="edge" d="M220 77 C260 77 260 42 300 42"/><path class="edge" d="M220 77 C260 77 260 128 300 128"/>
<path class="edge" d="M510 42 C550 42 550 85 590 85"/><path class="edge" d="M510 128 C550 128 550 85 590 85"/>
<path class="edge" d="M820 85 L880 85"/><path class="edge" d="M510 128 C700 128 700 240 880 240"/>
<path class="edge" d="M220 283 L590 271"/><path class="edge" d="M820 271 L880 271"/>
<g class="node" style="transition-delay:.05s"><rect x="20" y="40" width="200" height="74" rx="16"/><text x="40" y="72">Расходы МО</text><text class="sm" x="40" y="95">СберИндекс, 2023–2024</text></g>
<g class="node" style="transition-delay:.1s"><rect x="20" y="246" width="200" height="74" rx="16"/><text x="40" y="278">Новости и акты</text><text class="sm" x="40" y="301">Lenta, ГУ МЧС, pravo.gov.ru</text></g>
<g class="node" style="transition-delay:.2s"><rect x="300" y="10" width="210" height="64" rx="16"/><text x="320" y="40">n — вся страна</text><text class="sm" x="320" y="60">медиана МО + портал с 2018 г.</text></g>
<g class="node" style="transition-delay:.25s"><rect x="300" y="96" width="210" height="64" rx="16"/><text x="320" y="126">d — отклонение МО</text><text class="sm" x="320" y="146">почти без сезонности</text></g>
<g class="node acc" style="transition-delay:.35s"><rect x="590" y="40" width="230" height="90" rx="16"/><text x="610" y="74">Ансамбль</text><text class="sm" x="610" y="96">TimesFM · Chronos · LightGBM</text><text class="sm" x="610" y="114">веса по горизонтам</text></g>
<g class="node" style="transition-delay:.4s"><rect x="590" y="226" width="230" height="90" rx="16"/><text x="610" y="260">Экспозиция МО × месяц</text><text class="sm" x="610" y="282">геопривязка, as-of</text></g>
<g class="node acc" style="transition-delay:.5s"><rect x="880" y="40" width="200" height="90" rx="16"/><text x="900" y="74">Прогноз</text><text class="sm" x="900" y="96">ŷ = exp(n̂ + d̂)</text><text class="sm" x="900" y="114">1 / 3 / 6 / 12 мес.</text></g>
<g class="node acc" style="transition-delay:.55s"><rect x="880" y="196" width="200" height="124" rx="16"/><text x="900" y="230">Детектор</text><text class="sm" x="900" y="252">BOCPD + новостной hazard</text><text class="sm" x="900" y="272">Stouffer по категориям</text><text class="sm" x="900" y="292">3 % ложных тревог</text></g>
</svg></div></div></section>

<section id="fc" class="alt"><div class="wrap"><div class="sh reveal"><span class="n">04</span><h2>Точнее Prophet на всех горизонтах</h2></div>
<p class="sub reveal">Rolling-origin: 18 точек прогноза, test — июль–декабрь 2024, на каждом горизонте 6 целевых месяцев. Вся панель — 12 096 рядов.</p>
<div class="g2"><div class="card reveal"><h3>MAE по горизонтам</h3><p>₽ на жителя в месяц, меньше — лучше; пунктир — Prophet</p><div class="chart"><canvas id="chMae"></canvas></div></div>
<div class="card reveal"><h3>Значимость разницы с Prophet</h3><p>95 % ДИ — двусторонний бутстреп по МО и месяцам; p — знаковый тест по 6 месяцам</p><div id="sigTable"></div></div></div>
<div class="card reveal mt"><h3>Все модели</h3><p>MAE и R² годового прироста на test, вся панель</p><div id="modelsTable"></div></div>
</div></section>

<section id="fm"><div class="wrap"><div class="sh reveal"><span class="n">05</span><h2>Foundation models: решает подготовка входа</h2></div>
<p class="sub reveal">На сыром ряду из 7–23 точек модели не видят годового цикла. На отклонении МО те же модели в 1,7–2 раза точнее; TimesFM-2.5 — лучшая одиночная модель.</p>
<div class="g2"><div class="card reveal"><h3>Сырой ряд против отклонения МО</h3><p>MAE на горизонте 1 месяц, выборка МО</p><div class="chart"><canvas id="chFm"></canvas></div></div>
<div class="card reveal"><h3>Что ещё проверили</h3><ul class="clean" id="fmFacts"></ul></div></div></div></section>

<section id="det" class="alt"><div class="wrap"><div class="sh reveal"><span class="n">06</span><h2>Детекция: 12 методов при равных ложных тревогах</h2></div>
<p class="sub reveal">108 полусинтетических сценариев (форма × величина 3–20 % × охват) и реальные события. Пороги всех детекторов подобраны на одинаковую частоту ложных тревог.</p>
<div class="g2"><div class="card reveal"><h3>Полнота против ложных тревог</h3><p>пунктир — случайные тревоги той же частоты</p><div class="chart"><canvas id="chFar"></canvas></div></div>
<div class="card reveal"><h3>Полнота по величине шока</h3><p>3 ложные тревоги на 100 ряд-месяцев</p><div class="chart"><canvas id="chMag"></canvas></div></div></div>
<div class="card reveal mt"><h3>Реальные события</h3><p>Паводок апреля 2024 (19 МО из сообщений о ЧС) и 32 региональных режима ЧС из официальных актов против плацебо</p><div id="realTable"></div><p id="ncNote" style="margin-top:12px;color:var(--ink2);font-size:15px"></p>
<details open><summary>Сопоставимые постановки: BOCPD + новости</summary><p style="margin:8px 0;color:var(--ink3)">Наша сетка шире обычной: шоки от 3 %, провалы и рампы. На подмножествах, как в других решениях, полнота выше.</p><div id="detCmp"></div></details>
<details><summary>Все детекторы на полусинтетике</summary><div id="detTable"></div></details></div>
<div class="card reveal mt"><h3>Рабочая схема</h3>
<div class="lvl"><b>Ряд МО × категория</b><div>BOCPD с новостным hazard — локальный разрыв: паводок, эвакуация, закрытие предприятия → проверить события в МО за ±1 месяц.</div></div>
<div class="lvl"><b>МО целиком / регион</b><div>Сумма Стауффера по 5 категориям и соседи по дорогам — одновременный сдвиг нескольких категорий или соседних МО.</div></div>
<div class="lvl"><b>Категория по стране</b><div>Доля тревог за месяц больше 2 × FAR — системный сдвиг (методика, платёжный канал): не считать локальным шоком.</div></div></div>
</div></section>

<section id="news"><div class="wrap"><div class="sh reveal"><span class="n">07</span><h2>Новости и данные СберИндекса</h2></div>
<p class="sub reveal">Новость — событие с датой публикации и местом в тексте; данные — месячные оценки по МО. Согласование: геопривязка к territory_id, доля месяца после события, только опубликованное к моменту прогноза.</p>
<div class="g4" id="newsSteps"></div>
<div class="g2 mt"><div class="card reveal"><h3>Паводок апреля 2024</h3><p>ошибка прогноза на 1 месяц в МО из сообщений о ЧС, у соседей по региону и по стране</p><div class="chart"><canvas id="chFlood"></canvas></div></div>
<div class="card reveal"><h3>Качество и честность</h3><ul class="clean" id="newsFacts"></ul></div></div>
<div class="card reveal mt"><h3>Все ЧС 2024 года: как сдвигается ошибка прогноза</h3><p>361 событие в 56 регионах, новости ГУ МЧС по 52 регионам · точка — сдвиг в σ, полоса — 95 % ДИ (кластерный бутстреп) · зелёным — значимо с поправкой Бонферрони</p><div class="chart"><canvas id="chEs"></canvas></div></div>
</div></section>

<section id="cases" class="alt"><div class="wrap"><div class="sh reveal"><span class="n">08</span><h2>Реальные эпизоды 2024 года</h2></div>
<p class="sub reveal">Тревоги рабочего детектора на фактических данных и события рядом с ними. Зелёные точки — месяцы с тревогой.</p>
<div class="g3" id="caseCards"></div></div></section>

<section id="crit" class="alt"><div class="wrap"><div class="sh reveal"><span class="n">09</span><h2>Как закрыты критерии оценки</h2></div>
<p class="sub reveal">Вес критерия — в кольце; по ссылке — раздел с доказательствами.</p>
<div class="crit"><div class="card reveal"><svg class="ring" viewBox="0 0 68 68"><circle class="bg" cx="34" cy="34" r="28"/><circle class="fg" cx="34" cy="34" r="28"/><text x="34" y="35">10%</text></svg><div><h3>Понятная методология</h3><p>log y = n + d: общая динамика страны плюс отклонение МО; схема, глоссарий, отчёт.</p><a href="#idea">Смотреть →</a></div></div><div class="card reveal"><svg class="ring" viewBox="0 0 68 68"><circle class="bg" cx="34" cy="34" r="28"/><circle class="fg" cx="34" cy="34" r="28"/><text x="34" y="35">20%</text></svg><div><h3>Прогноз лучше Prophet на 1/3/6/12 мес.</h3><p>−49 / −41 / −30 / −50 % MAE против Prophet на всей панели; бутстреп и знаковый тест.</p><a href="#fc">Смотреть →</a></div></div><div class="card reveal"><svg class="ring" viewBox="0 0 68 68"><circle class="bg" cx="34" cy="34" r="28"/><circle class="fg" cx="34" cy="34" r="28"/><text x="34" y="35">20%</text></svg><div><h3>Детекторы сдвигов и выбор лучшего</h3><p>12 онлайн- и 5 офлайн-методов при равной частоте ложных тревог, 108 сценариев, реальный паводок.</p><a href="#det">Смотреть →</a></div></div><div class="card reveal"><svg class="ring" viewBox="0 0 68 68"><circle class="bg" cx="34" cy="34" r="28"/><circle class="fg" cx="34" cy="34" r="28"/><text x="34" y="35">15%</text></svg><div><h3>Foundation models</h3><p>Chronos-2, Chronos-Bolt, TimesFM-2.5, TiRex и LoRA-дообучение — на отклонении МО в 1,7–2 раза точнее.</p><a href="#fm">Смотреть →</a></div></div><div class="card reveal"><svg class="ring" viewBox="0 0 68 68"><circle class="bg" cx="34" cy="34" r="28"/><circle class="fg" cx="34" cy="34" r="28"/><text x="34" y="35">15%</text></svg><div><h3>Новости и согласование с СберИндексом</h3><p>Lenta, ГУ МЧС по 52 регионам, акты о ЧС; геопривязка к МО, as-of, hazard детектора, event study.</p><a href="#news">Смотреть →</a></div></div><div class="card reveal"><svg class="ring" viewBox="0 0 68 68"><circle class="bg" cx="34" cy="34" r="28"/><circle class="fg" cx="34" cy="34" r="28"/><text x="34" y="35">10%</text></svg><div><h3>MAE и R²</h3><p>MAE на всех горизонтах; R² уровней 0,994–0,998 и R² годового прироста.</p><a href="#fc">Смотреть →</a></div></div><div class="card reveal"><svg class="ring" viewBox="0 0 68 68"><circle class="bg" cx="34" cy="34" r="28"/><circle class="fg" cx="34" cy="34" r="28"/><text x="34" y="35">10%</text></svg><div><h3>Интерпретация и воспроизводимость</h3><p>Colab одной кнопкой, 20 тестов, сверка чисел отчёта, проверка на 2025 г.</p><a href="#check">Смотреть →</a></div></div></div></div></section>

<section id="gloss"><div class="wrap"><div class="sh reveal"><span class="n">10</span><h2>Глоссарий</h2></div>
<p class="sub reveal">Термины, которые встречаются на странице, — в двух строках.</p>
<div class="gloss"><details class="reveal"><summary><b>n</b>национальная компонента</summary><p>Медиана логарифма расходов по всем МО за месяц: сезонность, инфляция, рост онлайн-покупок — общая для страны.</p></details><details class="reveal"><summary><b>d</b>отклонение МО</summary><p>log y − n: то, что отличает муниципалитет от страны. Почти без сезонности — его прогнозируют модели и на нём ищут разрывы.</p></details><details class="reveal"><summary><b>Rolling-origin</b>честный бэктест</summary><p>18 точек прогноза подряд; в каждой модель видит только прошлое и прогнозирует на 1, 3, 6 и 12 месяцев.</p></details><details class="reveal"><summary><b>MAE</b>средняя абсолютная ошибка</summary><p>В рублях на жителя в месяц — понятно бизнесу и не зависит от масштаба категорий.</p></details><details class="reveal"><summary><b>R²</b>уровней и прироста</summary><p>R² уровней почти всегда ≈ 0,99 из-за разницы между МО; R² годового прироста показывает, понимает ли модель динамику.</p></details><details class="reveal"><summary><b>Foundation model</b>предобученная модель рядов</summary><p>Chronos-2, TimesFM-2.5, TiRex обучены на миллионах рядов и прогнозируют без дообучения (zero-shot).</p></details><details class="reveal"><summary><b>BOCPD</b>байесовское онлайн-обнаружение разрывов</summary><p>Каждый месяц оценивает вероятность, что недавно начался новый режим. Тревога — когда она выше порога.</p></details><details class="reveal"><summary><b>Hazard</b>априорная вероятность разрыва</summary><p>Новость о ЧС в МО повышает её в месяц публикации: детектор становится чувствительнее там, где что-то случилось.</p></details><details class="reveal"><summary><b>FAR</b>частота ложных тревог</summary><p>Ложных тревог на 100 ряд-месяцев. Все детекторы сравниваются при одинаковом FAR — иначе сравнение нечестное.</p></details><details class="reveal"><summary><b>Stouffer</b>сумма z-оценок</summary><p>Объединяет сигналы шести категорий МО — ловит шок, который затронул муниципалитет целиком.</p></details><details class="reveal"><summary><b>As-of</b>только опубликованное</summary><p>В момент прогноза используем только данные и новости, доступные на эту дату, — без заглядывания в будущее.</p></details><details class="reveal"><summary><b>Event study</b>до и после события</summary><p>Сравниваем ошибку прогноза в МО с ЧС и в контрольных МО вокруг даты события.</p></details></div></div></section>

<section id="check" class="alt"><div class="wrap"><div class="sh reveal"><span class="n">11</span><h2>Проверки и воспроизводимость</h2></div>
<div class="g3" id="checkCards"></div>
<div class="card reveal mt"><h3>Ограничения — честно</h3><ul class="clean">
<li>24 месяца истории: на каждый горизонт 6 целевых месяцев; тест Дибольда–Мариано маломощен, основная оценка — бутстреп и знаковый тест.</li>
<li>Шоки в 3–5 % в одном МО статистически почти не видны; надёжно видны шоки от 10 % и групповые.</li>
<li>Новости не улучшают месячный прогноз — они полезны детектору и как объяснение тревог.</li>
<li>8 приграничных регионов отсутствуют в муниципальных данных СберИндекса.</li></ul></div>
</div></section>
<footer><div class="wrap"><b style="color:var(--ink)">Автор — Максим Караханов.</b> Данные СберИндекса — CC BY-SA 4.0. Код, конфигурации и отчёт — в репозитории проекта. Запуск: <code>python run_all.py</code> или ноутбук Colab одной кнопкой.</div></footer>

<script>const DATA=__DATA__;const STATIC=location.hash==='#static';const ANIM=!STATIC&&!matchMedia('(prefers-reduced-motion: reduce)').matches;if(STATIC)document.documentElement.classList.add('static');</script>
<script>__CHARTJS__
function fill(){
 const K=DATA.kpi;
 const items=[[K.rel[0],'%','MAE против Prophet, горизонт 1 месяц'],[K.rel[3],'%','MAE против Prophet, горизонт 12 месяцев'],
  [K.recall,'%','полнота при 3 % ложных тревог (случайно — '+fmt(K.random)+' %)'+(K.recall_big?'; шоки 10–20 % — '+fmt(K.recall_big)+' %':'')],[K.lift,'×','тревоги в МО паводка чаще фона']];
 document.getElementById('kpis').innerHTML=items.map(x=>'<div class="kpi reveal"><div class="v" data-to="'+x[0]+'" data-suf="'+x[1]+'">0</div><div class="l">'+x[2]+'</div></div>').join('');
 document.getElementById('sigTable').innerHTML=tbl(['сравнение','ансамбль','Prophet','разница','95 % ДИ, ₽','p'],
  DATA.sig.map(s=>[(s.vs.startsWith('Лучший')?'лучший Prophet':'Prophet, вся панель')+' · '+s.h+' мес.',fmt(s.a),fmt(s.b),'<span class="good">'+fmt(s.rel)+' %</span>',fmt(s.lo)+' … '+fmt(s.hi),fmt(s.p,3)]));
 const ms=DATA.models;
 document.getElementById('modelsTable').innerHTML=tbl(['модель','1 мес.','3 мес.','6 мес.','12 мес.','R² г/г, 1 мес.','R² г/г, 3 мес.'],
  ms.map(m=>[m.name,...m.mae.map(v=>fmt(v)),fmt(m.r2[0],2),fmt(m.r2[1],2)]),i=>ms[i].name.startsWith('Ансамбль ('));
 const {L,Z}=LORA();
 document.getElementById('fmFacts').innerHTML=li([
  '<b>Шесть моделей:</b> Chronos-2 (совместно по 6 категориям и одномерно), Chronos-Bolt base и small, TimesFM-2.5, TiRex на GPU.',
  L&&Z?'<b>LoRA-дообучение</b> Chronos-2 на данных до 12.2023: MAE '+fmt(L.mae[0])+' / '+fmt(L.mae[1])+' / '+fmt(L.mae[2])+' против '+fmt(Z.mae[0])+' / '+fmt(Z.mae[1])+' / '+fmt(Z.mae[2])+' у zero-shot (1 / 3 / 6 мес.).':'',
  '<b>Вклад в ансамбль</b> (абляция без FM): −2 % MAE на 1 месяц, −2…−3 % на 3 месяца, около нуля дальше.',
  '<b>Интервалы:</b> покрытие 80 %-интервала у Chronos 0,82–0,91, у TimesFM 0,74–0,82; конформная калибровка исправляет TimesFM.',
  '<b>Выход факта за интервал</b> Chronos-2 — сам по себе хороший детектор: на паводке тревоги в 2,29 раза чаще фона.']);
 document.getElementById('realTable').innerHTML=tbl(['детектор','паводок: МО с ЧС','страна','эффект','p','акты ЧС','плацебо','эффект'],
  DATA.real.map(r=>[r.name,fmt(r.named,1)+' %',fmt(r.base,1)+' %','<b>'+fmt(r.lift,2)+'×</b>',fmt(r.p,3),fmt(r.acts,1)+' %',fmt(r.placebo,1)+' %',fmt(r.lift_acts,2)+'×']),i=>i===0);
 if(DATA.noncirc)document.getElementById('ncNote').innerHTML='<b style="color:var(--green)">Проверка без круга:</b> группу МО задают только новости ГУ МЧС, а hazard детектора строится без них (Lenta.ru и акты) — эффект сохраняется: <b>'+fmt(DATA.noncirc.lift,2)+'×, p = '+fmt(DATA.noncirc.p,3)+'</b>.';
 if(DATA.comparable)document.getElementById('detCmp').innerHTML=tbl(['постановка','ложных на 100 ряд-мес.','полнота','в месяц шока','точность','F1'],
  DATA.comparable.map(c=>[c.setup,c.far,fmt(c.recall,2),fmt(c.onset,2),fmt(c.precision,2),fmt(c.f1,2)]),i=>DATA.comparable[i].setup==='Шоки 10–20 %');
 document.getElementById('detTable').innerHTML=tbl(['детектор','полнота','в месяц шока','задержка, мес.','точность','F1'],
  DATA.detectors.map(d=>[d.name,fmt(d.recall,3),fmt(d.onset,3),fmt(d.delay,2),fmt(d.precision,3),fmt(d.f1,3)]),i=>i===0);
 const steps=[['Сбор','Lenta.ru (98 тыс. заголовков), '+(DATA.mchs?fmt(DATA.mchs.n)+' новостей ГУ МЧС о ЧС':'тексты ГУ МЧС')+', акты о режиме ЧС'],['Леммы и тип','pymorphy3 и правила: ЧС, атаки, аварии, экономика'],
  ['Геопривязка','справочник СберИндекса → territory_id, омонимы — по региону'],['Экспозиция as-of','доля месяца после публикации, затухание, только прошлое']];
 document.getElementById('newsSteps').innerHTML=steps.map((s,i)=>'<div class="card step reveal"><div class="num">'+(i+1)+'</div><h3>'+s[0]+'</h3><p>'+s[1]+'</p></div>').join('');
 const A=DATA.audit,AE=DATA.all_events;
 document.getElementById('newsFacts').innerHTML=li([
  '<b>Ручной аудит</b> 60 событий: '+fmt(A['полностью верно, %'])+' % размечены полностью верно, территория — '+fmt(A['территория верна, %'])+' %.',
  '<b>Не оракул:</b> при шумовых новостях выигрыша нет, он растёт с их качеством (до +'+fmt(DATA.sens_max,1)+' п.п. полноты).',
  '<b>Паводок:</b> в апреле маркетплейсы в МО из сообщений о ЧС на 5–6 п.п. ниже соседей по региону, в мае — восстановительный отскок.',
  AE?'<b>Без отбора по паводку:</b> по всем '+AE.n+' событиям ЧС 2024 г. с названным МО в '+AE.regions+' регионах'+(AE.mchs_regions?' (новости ГУ МЧС по '+AE.mchs_regions+' регионам, прогон на GPU)':'')+
   (AE.sig&&AE.sig.length?': «Все категории» не сдвигаются, но реагируют '+AE.sig.map(s=>s.cat.toLowerCase()+' ('+(s.shift>0?'+':'')+fmt(s.shift,2)+'σ, p '+(s.p<0.001?'< 0,001':'= '+fmt(s.p,3))+')').join(' и ')+' — с поправкой Бонферрони на 6 категорий.':' значимого сдвига нет: большинство ЧС расходы не меняют, в отличие от крупного паводка.'):'',
  'Новости не улучшают месячный прогноз — их роль в детекции и объяснении тревог.']);
 document.getElementById('caseCards').innerHTML=DATA.cases.map((c,i)=>'<div class="card reveal"><span class="pill">'+c.note+'</span><h3 style="margin-top:12px">'+c.name+'</h3><div class="chart sm"><canvas id="chCase'+i+'"></canvas></div></div>').join('');
 const E=DATA.external||{},Dc=DATA.decomp||{ens:[0],oracle:[0]};
 document.getElementById('checkCards').innerHTML=[
  ['Внешняя проверка 2025','<div class="big">+'+fmt(E.pred,1)+' %</div><p>прогноз роста «Всех категорий» на 2025 г. против факта портала <b>+'+fmt(E.fact,1)+' %</b>; профиль по месяцам — корреляция '+fmt(E.corr,2)+'.</p>'],
  ['Откуда ошибка','<div class="big">'+Math.round(100*(1-Dc.oracle[0]/Dc.ens[0]))+' %</div><p>ошибки на 1 месяц — прогноз общероссийской динамики: с истинной национальной частью MAE было бы '+fmt(Dc.oracle[0])+' вместо '+fmt(Dc.ens[0])+'.</p>'],
  ['Воспроизводимость','<div class="big">1 кнопка</div><p>ноутбук Colab с кодом и данными внутри; независимый прогон на T4 повторил результаты; 20 тестов: утечка будущего и компоненты; SHA-256 всех данных.</p>']
 ].map(x=>'<div class="card reveal"><h3>'+x[0]+'</h3>'+x[1]+'</div>').join('');
}
function countUp(el){const to=parseFloat(el.dataset.to),suf=el.dataset.suf,dec=suf==='×'?2:0,t0=performance.now(),dur=1400;
 (function f(t){const p=Math.min(1,(t-t0)/dur),e=1-Math.pow(1-p,3);el.textContent=fmt(to*e,dec)+(suf==='%'?' %':suf);if(p<1)requestAnimationFrame(f)})(t0)}
function ticker(){const AE=DATA.all_events||{};const it=[['12 096','рядов в бэктесте'],['2 016','муниципалитетов'],['6','категорий трат'],['18','точек прогноза'],
 [String(DATA.models.length),'моделей в сравнении'],[String(DATA.detectors.length),'детекторов'],[AE.n?String(AE.n):'','ЧС в event study'],[AE.mchs_regions?String(AE.mchs_regions):'','регионов с новостями МЧС'],
 ['98 тыс.','заголовков Lenta.ru'],[fmt(DATA.kpi.rel[0])+' %','MAE к Prophet на 1 месяц'],[fmt(DATA.kpi.rel[3])+' %','MAE к Prophet на 12 месяцев']].filter(x=>x[0]);
 const h=it.map(x=>'<span><i>'+x[0]+'</i>'+x[1]+'</span>').join('');document.getElementById('tick').innerHTML=h+h}
function seismo(){const cv=document.getElementById('seis');if(!cv)return;const ctx=cv.getContext('2d'),P=document.getElementById('seisP');
 const STEP=2.4;let W=0,H=0,pts=[],marks=[],t=0,shock=null,next=90,prob=.02,live=true;const dpr=Math.min(2,devicePixelRatio||1);
 const base=()=>{t++;return Math.sin(t/31)*.16+Math.sin(t/9.7)*.07+(Math.random()-.5)*.16};
 function sample(){let v=base();if(shock){const k=shock.age++;
   v+=shock.kind==='spike'?shock.a*Math.exp(-k/10)*Math.sin(k*.9):shock.a*(1-Math.exp(-k/5))*Math.exp(-Math.max(0,k-55)/14);
   if(k===shock.det){marks.push({x:W});prob=.84+Math.random()*.12}if(k>shock.len)shock=null}
  else if(--next<=0){shock={kind:Math.random()<.5?'spike':'step',a:(Math.random()<.5?-1:1)*(.85+Math.random()*.45),age:0,det:4+Math.floor(Math.random()*5),len:110};next=170+Math.random()*130}
  prob=Math.max(.02,prob*.988);return v}
 function size(){W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  const n=Math.ceil(W/STEP)+2;while(pts.length<n)pts.push(sample());pts=pts.slice(-n)}
 function draw(){const g=css('--green'),gl=css('--glow'),mid=H*.6,amp=H*.24;ctx.clearRect(0,0,W,H);
  ctx.strokeStyle=css('--grid');ctx.lineWidth=1;for(let k=-1;k<=1;k++){ctx.beginPath();ctx.moveTo(0,mid+k*amp*1.25);ctx.lineTo(W,mid+k*amp*1.25);ctx.stroke()}
  const y=v=>mid-v*amp,x0=W-(pts.length-1)*STEP;
  ctx.beginPath();pts.forEach((v,i)=>i?ctx.lineTo(x0+i*STEP,y(v)):ctx.moveTo(x0,y(v)));ctx.lineTo(W,H);ctx.lineTo(x0,H);ctx.closePath();ctx.fillStyle=gl;ctx.fill();
  ctx.beginPath();pts.forEach((v,i)=>i?ctx.lineTo(x0+i*STEP,y(v)):ctx.moveTo(x0,y(v)));ctx.strokeStyle=g;ctx.lineWidth=2.4;ctx.shadowColor=g;ctx.shadowBlur=12;ctx.stroke();ctx.shadowBlur=0;
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(W-2,y(pts[pts.length-1]),4.5,0,7);ctx.fill();
  ctx.font='800 11px '+FONT;marks.forEach(m=>{ctx.strokeStyle=g;ctx.setLineDash([4,5]);ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(m.x,34);ctx.lineTo(m.x,H-26);ctx.stroke();ctx.setLineDash([]);
   ctx.fillStyle=g;ctx.fillText('▲ тревога',m.x-28,H-10)});
  P.textContent='P(разрыв) '+fmt(prob,2);P.classList.toggle('hot',prob>.5)}
 function frame(){if(live&&!document.hidden){pts.push(sample());pts.shift();marks.forEach(m=>m.x-=STEP);marks=marks.filter(m=>m.x>-60);draw()}requestAnimationFrame(frame)}
 size();addEventListener('resize',()=>{size();draw()});
 if(!ANIM){for(let k=0;k<pts.length*1.2;k++){pts.push(sample());pts.shift();marks.forEach(m=>m.x-=STEP)}if(!marks.length)marks.push({x:W*.62});draw();return}
 new IntersectionObserver(es=>{live=es[0].isIntersecting}).observe(cv);requestAnimationFrame(frame)}
function replay(){const c=(DATA.cases||[]).find(x=>/дамб/.test(x.note))||(DATA.cases||[])[0];if(!c)return;
 const M=['янв','фев','мар','апр','май','июн','июл','авг','сен','окт','ноя','дек'],lab=d=>M[+d.slice(5,7)-1]+' '+d.slice(0,4);
 const ST=['январ','феврал','март','апрел','ма[йя]','июн','июл','август','сентябр','октябр','ноябр','декабр'];
 const yr=(c.note.match(/(20\d\d)/)||[])[1],mi=ST.findIndex(r=>new RegExp(r).test(c.note));const newsAt=yr&&mi>=0?c.dates.indexOf(yr+'-'+String(mi+1).padStart(2,'0')):-1;
 document.getElementById('rpName').textContent=c.name;const feed=document.getElementById('rpFeed'),chip=document.getElementById('rpChip'),mo=document.getElementById('rpMonth');
 let chart=null,timer=null,i=0;
 const push=(cls,h)=>{const e=document.createElement('div');e.className='ev '+cls;e.innerHTML=h;feed.prepend(e);while(feed.children.length>5)feed.lastChild.remove()};
 const cfg=()=>{const o=O({yZero:false,unit:' ₽',xMax:8});o.animation=false;o.scales.y.suggestedMin=Math.min(...c.values)*.94;o.scales.y.suggestedMax=Math.max(...c.values)*1.04;
  return{type:'line',data:{labels:c.dates.map(lab),datasets:[
   {label:'маркетплейсы, ₽ на жителя',data:c.values.map((v,k)=>k<i?v:null),borderColor:css('--s2'),backgroundColor:css('--glow'),fill:true,borderWidth:2.6,pointRadius:0,tension:.3},
   {label:'тревога детектора',data:c.dates.map((d,k)=>k<i&&c.alarms.includes(d)?c.values[k]:null),showLine:false,pointRadius:8,pointHoverRadius:10,pointBackgroundColor:css('--green'),pointBorderColor:css('--card'),pointBorderWidth:2}]},options:o}};
 const step=()=>{const k=i++,d=c.dates[k],a=c.alarms.includes(d);chart.data.datasets[0].data[k]=c.values[k];if(a)chart.data.datasets[1].data[k]=c.values[k];chart.update('none');
  mo.textContent=lab(d);chip.textContent=a?'ТРЕВОГА':'норма';chip.classList.toggle('hot',a);
  if(k===newsAt)push('news','<b>Новость · '+lab(d)+'</b><br>'+c.note[0].toUpperCase()+c.note.slice(1));
  if(a)push('alarm','<b>Тревога · '+lab(d)+'</b><br>BOCPD: разрыв в отклонении МО');
  if(i>=c.dates.length){clearInterval(timer);timer=null;const f=c.alarms.slice().sort()[0];push('','<b>Итог</b><br>'+c.alarms.length+' тревоги, первая — '+lab(f)+(newsAt>=0?'; новость — '+lab(c.dates[newsAt]):''))}};
 function run(){clearInterval(timer);feed.innerHTML='';i=0;if(chart)chart.destroy();chart=new Chart(document.getElementById('chReplay'),cfg());
  if(!ANIM){while(i<c.dates.length)step();return}timer=setInterval(step,430)}
 window.rpRun=()=>{if(timer)run();else{const n=i;if(chart)chart.destroy();chart=new Chart(document.getElementById('chReplay'),cfg());i=n}};
 document.getElementById('rpBtn').onclick=run;if(!ANIM){run();return}
 const o=new IntersectionObserver(es=>{if(es[0].isIntersecting){run();o.disconnect()}},{threshold:.35});o.observe(document.getElementById('chReplay'))}
addEventListener('pointermove',e=>{const c=e.target.closest&&e.target.closest('.card');if(!c)return;const r=c.getBoundingClientRect();
 c.style.setProperty('--mx',(e.clientX-r.left)+'px');c.style.setProperty('--my',(e.clientY-r.top)+'px')},{passive:true});
fill();ticker();seismo();
const reduce=!ANIM;
const io=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)return;e.target.classList.add('in');
 e.target.querySelectorAll('[data-to]').forEach(el=>reduce?el.textContent=fmt(+el.dataset.to,el.dataset.suf==='×'?2:0)+(el.dataset.suf==='%'?' %':el.dataset.suf):countUp(el));io.unobserve(e.target)}),{threshold:.12});
document.querySelectorAll('.reveal').forEach(el=>io.observe(el));
// графики создаются, когда карточка попадает в экран, — так их анимация видна
const pending=new Map();const _mk=mk;mk=function(id,cfg){pending.set(id,cfg)};drawAll();mk=_mk;
document.fonts.ready.then(()=>{replay();
const cio=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)return;const id=e.target.id;if(pending.has(id)){mk(id,pending.get(id));pending.delete(id)}cio.unobserve(e.target)}),{threshold:.25});
pending.forEach((_,id)=>{const el=document.getElementById(id);if(el)cio.observe(el)});
});
document.getElementById('tgl').onclick=()=>{const t=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=t;try{localStorage.setItem('theme',t)}catch(e){}rebuild();if(window.rpRun)rpRun()};
const secs=[...document.querySelectorAll('section')],links=[...document.querySelectorAll('#nav a')];
addEventListener('scroll',()=>{const h=document.documentElement;document.getElementById('prog').style.width=(100*h.scrollTop/Math.max(1,h.scrollHeight-h.clientHeight))+'%';
 let cur='';secs.forEach(s=>{if(s.getBoundingClientRect().top<140)cur=s.id});links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+cur))},{passive:true});
</script></body></html>"""

SLIDES = r"""<!doctype html><html lang="ru" data-theme="__THEME__"><head><meta charset="utf-8"><title>Сейсмограф трат — презентация</title>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>__CSS__
@page{size:1280px 720px;margin:0}
html,body{background:var(--bg);-webkit-print-color-adjust:exact;print-color-adjust:exact}
.slide{width:1280px;height:720px;padding:52px 64px 60px;position:relative;overflow:hidden;break-after:page;page-break-after:always;background:var(--bg);display:flex;flex-direction:column}
.slide:nth-child(even){background:var(--bg2)}
.slide::before{content:'';position:absolute;right:-160px;top:-160px;width:420px;height:420px;border-radius:50%;background:radial-gradient(circle,var(--glow),transparent 70%)}
.slide>*{position:relative}
.tag{font-weight:800;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--green)}
h1{font-size:60px;line-height:1.04;letter-spacing:-.035em;font-weight:800;margin:20px 0 18px}
h1 em,h2 em{font-style:normal;color:var(--green)}
h2{font-size:38px;line-height:1.08;letter-spacing:-.03em;font-weight:800;margin:8px 0 4px}
.lead{font-size:20px;color:var(--ink2);max-width:960px}
.row{display:grid;gap:22px;flex:1;min-height:0;margin-top:20px}.c2{grid-template-columns:1fr 1fr}.c3{grid-template-columns:repeat(3,1fr)}.c4{grid-template-columns:repeat(4,1fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;display:flex;flex-direction:column;min-height:0}
.card h3{font-size:18px;margin-bottom:4px}.card>p{color:var(--ink2);font-size:14px}
.kpi .v{font-size:46px;font-weight:800;color:var(--green);letter-spacing:-.03em;line-height:1.05}.kpi .l{color:var(--ink2);font-size:15px;margin-top:8px}
.chart{position:relative;flex:1;min-height:0;margin-top:8px}
.tw{margin-top:8px}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
tr.hl td{font-weight:800}tr.hl td:first-child{color:var(--green)}.good{color:var(--green);font-weight:800}
ul.clean{list-style:none}ul.clean li{padding:7px 0 7px 24px;position:relative;color:var(--ink2);font-size:15px}
ul.clean li::before{content:'';position:absolute;left:2px;top:15px;width:9px;height:9px;border-radius:3px;background:var(--green)}ul.clean b{color:var(--ink)}
.num{width:38px;height:38px;border-radius:11px;background:var(--glow);color:var(--green);font-weight:800;display:grid;place-items:center;margin-bottom:12px}
.foot{position:absolute;bottom:22px;left:64px;right:64px;display:flex;justify-content:space-between;color:var(--ink3);font-size:12px}
.foot b{color:var(--green)}
</style></head><body>
<section class="slide"><span class="tag">Сейсмограф трат · СберИндекс 2026 · Задача № 2</span>
<h1>Сейсмограф трат:<br><em>ранние толчки</em> в расходах МО</h1>
<p class="lead">Прогноз расходов и раннее обнаружение шоков · 2 016 МО × 6 категорий трат · горизонты 1, 3, 6 и 12 месяцев · детектор структурных сдвигов · новости с привязкой к муниципалитетам</p>
<div class="row c4" id="sKpi" style="flex:none;margin-top:56px"></div><p class="lead" style="margin-top:22px;color:var(--ink)">Автор — <b>Максим Караханов</b></p><div class="foot"><span>Методологический отчёт — reports/methodology.md</span><b>1</b></div></section>

<section class="slide"><span class="tag">Методология</span><h2>Делим расходы на <em>две части</em></h2>
<p class="lead">Около 70 % колебаний — общая динамика страны. Её берём из панели и национальных рядов СберИндекса с 2018 г.; модели учим на отклонении МО.</p>
<div class="row c4"><div class="card"><div class="num">1</div><h3>log y = n + d</h3><p>n — медиана по МО: сезонность, инфляция, онлайн-покупки. d — отклонение муниципалитета.</p></div>
<div class="card"><div class="num">2</div><h3>Ансамбль на d</h3><p>TimesFM-2.5, Chronos-2, LightGBM и простые модели; веса по горизонтам на validation.</p></div>
<div class="card"><div class="num">3</div><h3>BOCPD на d</h3><p>Байесовский детектор разрывов; порог — 3 ложные тревоги на 100 ряд-месяцев.</p></div>
<div class="card"><div class="num">4</div><h3>Новости → hazard</h3><p>Сообщения о ЧС по месту и дате публикации повышают априорную вероятность разрыва.</p></div></div>
<div class="foot"><span>Максим Караханов · 18 точек прогноза · test — июль–декабрь 2024 · только прошлое на каждом шаге</span><b>2</b></div></section>

<section class="slide"><span class="tag">Прогноз</span><h2>Точнее Prophet на <em>всех горизонтах</em></h2>
<div class="row c2"><div class="card"><h3>MAE по горизонтам, вся панель (12 096 рядов)</h3><p>₽ на жителя в месяц; пунктир — Prophet</p><div class="chart"><canvas id="chMae"></canvas></div></div>
<div class="card"><h3>Значимость разницы</h3><p>95 % ДИ — двусторонний бутстреп по МО и месяцам</p><div id="sSig"></div></div></div>
<div class="foot"><span>Максим Караханов · На каждом горизонте 6 целевых месяцев · ансамбль лучше во всех 6 из 6</span><b>3</b></div></section>

<section class="slide"><span class="tag">Foundation models</span><h2>Решает <em>подготовка входа</em></h2>
<div class="row c2"><div class="card"><h3>Сырой ряд против отклонения МО</h3><p>MAE на 1 месяц</p><div class="chart"><canvas id="chFm"></canvas></div></div>
<div class="card"><h3>Выводы</h3><ul class="clean" id="sFm"></ul></div></div><div class="foot"><span>Максим Караханов · Chronos-2 · Chronos-Bolt · TimesFM-2.5 · TiRex · LoRA</span><b>4</b></div></section>

<section class="slide"><span class="tag">Детекция</span><h2>12 детекторов при <em>равных ложных тревогах</em></h2>
<div class="row c2"><div class="card"><h3>Полнота против частоты ложных тревог</h3><p>пунктир — случайные тревоги</p><div class="chart"><canvas id="chFar"></canvas></div></div>
<div class="card"><h3>Полнота по величине шока</h3><p>3 ложные тревоги на 100 ряд-месяцев</p><div class="chart"><canvas id="chMag"></canvas></div></div></div>
<div class="foot"><span>Максим Караханов · 108 полусинтетических сценариев: форма × величина × охват</span><b>5</b></div></section>

<section class="slide"><span class="tag">Реальные события</span><h2>Паводок 2024: тревоги в <em id="sLift">2,35</em> раза чаще фона</h2>
<div class="row c2"><div class="card"><h3>Детекторы на паводке апреля 2024</h3><div id="sReal"></div></div>
<div class="card"><h3>Рабочая схема</h3><ul class="clean"><li><b>Ряд МО:</b> BOCPD + новостной hazard — локальный разрыв.</li><li><b>МО / регион:</b> Stouffer по категориям и соседи по дорогам.</li><li><b>Категория по стране:</b> доля тревог больше 2 × FAR — системный сдвиг, не локальный шок.</li><li>32 региональных режима ЧС из актов (пожары, засуха) расходы не меняют — детекторы их правильно не замечают.</li></ul></div></div>
<div class="foot"><span id="sNc">Максим Караханов · Точный тест Фишера для заранее заявленного детектора</span><b>6</b></div></section>

<section class="slide"><span class="tag">Новости</span><h2>Где искать шок — <em>адрес из текста</em></h2>
<div class="row c2"><div class="card"><h3>Ошибка прогноза на 1 месяц, паводок апреля 2024</h3><div class="chart"><canvas id="chFlood"></canvas></div></div>
<div class="card"><h3>Согласование с данными СберИндекса</h3><ul class="clean" id="sNews"></ul></div></div>
<div class="foot"><span>Максим Караханов · Lenta.ru · ГУ МЧС · акты pravo.gov.ru · as-of по дате публикации</span><b>7</b></div></section>

<section class="slide"><span class="tag">Эпизоды</span><h2>Реальные эпизоды <em>2024 года</em></h2>
<div class="row c3" id="sCases"></div><div class="foot"><span>Максим Караханов · Маркетплейсы, ₽ на жителя · зелёные точки — тревоги детектора</span><b>8</b></div></section>

<section class="slide"><span class="tag">Итоги</span><h2>Что получилось и <em>что дальше</em></h2>
<div class="row c3"><div class="card"><h3>Проверки</h3><ul class="clean" id="sCheck"></ul></div>
<div class="card"><h3>Выводы</h3><ul class="clean"><li>Общая динамика страны — главный источник точности; Prophet проигрывает даже простой панельной модели.</li><li>Дальше 3 месяцев локальную динамику предсказать почти нечем.</li><li>Новости полезны детектору и как объяснение тревог, но не прогнозу.</li></ul></div>
<div class="card"><h3>Дальше</h3><ul class="clean"><li>Макромодель национальной части: ИПЦ, обороты, доля онлайн-покупок.</li><li>Новости ГУ МЧС: собраны по 52 регионам — подать их в hazard детектора.</li><li>Муниципальные данные 2025 г. для проверки на новом году.</li></ul></div></div>
<div class="foot"><span>Максим Караханов · python run_all.py · ноутбук Colab одной кнопкой · 20 тестов</span><b>9</b></div></section>

<script>const DATA=__DATA__;const ANIM=false;</script>
<script>__CHARTJS__
const K=DATA.kpi;
document.getElementById('sKpi').innerHTML=[[fmt(K.rel[0])+' %','MAE против Prophet, 1 месяц'],[fmt(K.rel[3])+' %','MAE против Prophet, 12 месяцев'],
 [fmt(K.recall)+' %','полнота детектора при 3 % ложных тревог'+(K.recall_big?' (на шоках 10–20 % — '+fmt(K.recall_big)+' %)':'')],[fmt(K.lift,2)+'×','тревоги на паводке чаще фона']].map(x=>'<div class="card kpi"><div class="v">'+x[0]+'</div><div class="l">'+x[1]+'</div></div>').join('');
document.getElementById('sSig').innerHTML=tbl(['сравнение','горизонт','ансамбль','Prophet','разница','95 % ДИ, ₽'],
 DATA.sig.map(s=>[s.vs.startsWith('Лучший')?'лучший Prophet':'Prophet, вся панель',s.h+' мес.',fmt(s.a),fmt(s.b),'<span class="good">'+fmt(s.rel)+' %</span>',fmt(s.lo)+' … '+fmt(s.hi)]));
const {L,Z}=LORA();
document.getElementById('sFm').innerHTML=li(['На сыром ряду из 7–23 точек модели не видят годового цикла; на отклонении МО они в <b>1,7–2 раза точнее</b>.',
 '<b>TimesFM-2.5</b> — лучшая одиночная модель; Chronos-2 совместно по 6 категориям точнее одномерного.',
 L&&Z?'<b>LoRA-дообучение</b> Chronos-2 (данные до 12.2023): '+fmt(L.mae[2])+' против '+fmt(Z.mae[2])+' на 6 месяцев.':'',
 'Вклад FM в ансамбль: <b>−2…−3 %</b> MAE на 1–3 месяца.','Выход факта за интервал Chronos-2 — сильный детектор шоков.']);
document.getElementById('sLift').textContent=fmt(K.lift,2);
if(DATA.noncirc)document.getElementById('sNc').textContent='Максим Караханов · Без круга (группа — ГУ МЧС, hazard — Lenta.ru и акты): '+fmt(DATA.noncirc.lift,2)+'×, p = '+fmt(DATA.noncirc.p,3)+' · точный тест Фишера';
document.getElementById('sReal').innerHTML=tbl(['детектор','МО с ЧС','страна','эффект','p'],DATA.real.map(r=>[r.name,fmt(r.named,1)+' %',fmt(r.base,1)+' %','<b>'+fmt(r.lift,2)+'×</b>',fmt(r.p,3)]),i=>i===0);
const A=DATA.audit;
document.getElementById('sNews').innerHTML=li(['<b>Геопривязка</b> к territory_id через справочник СберИндекса, омонимы — по региону.',
 '<b>Экспозиция МО × месяц</b> с долей месяца после публикации; только опубликованное к моменту прогноза.',
 '<b>Ручной аудит:</b> '+fmt(A['полностью верно, %'])+' % событий размечены полностью верно.','При шумовых новостях выигрыша нет — prior не «оракул».',
 'Новости не улучшают месячный прогноз — их роль в детекции.']);
document.getElementById('sCases').innerHTML=DATA.cases.map((c,i)=>'<div class="card"><h3>'+c.name+'</h3><p>'+c.note+'</p><div class="chart"><canvas id="chCase'+i+'"></canvas></div></div>').join('');
const E=DATA.external||{};
document.getElementById('sCheck').innerHTML=li(['<b>2025 г.:</b> прогноз роста +'+fmt(E.pred,1)+' % против факта +'+fmt(E.fact,1)+' %.',
 '<b>Бутстреп и знаковый тест:</b> лучше Prophet в 6 из 6 месяцев на каждом горизонте.','Независимый прогон в Colab (T4) повторил результаты.',
 'Журнал решений, принятых после просмотра результатов.']);
document.fonts.ready.then(drawAll);
</script></body></html>"""


def render(template, data, theme="light"):
    return (template.replace("__CSS__", CSS).replace("__CHARTJS__", CHART_JS)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__THEME__", theme))


def browser():
    cands = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    return next((p for p in cands if p and Path(p).exists()), None)


def print_pdf(html: Path, pdf: Path):
    exe = browser()
    if not exe:
        print("браузер для печати PDF не найден — откройте slides.html и распечатайте в PDF")
        return
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()], capture_output=True, timeout=180)
    print(pdf.relative_to(ROOT), f"{pdf.stat().st_size // 1024} КБ")


def main():
    data = build_data()
    (OUT / "index.html").write_text(render(SITE, data), encoding="utf-8")
    (OUT / "slides.html").write_text(render(SLIDES, data, "light"), encoding="utf-8")
    (OUT / "slides_dark.html").write_text(render(SLIDES, data, "dark"), encoding="utf-8")
    print("reports/site/index.html, slides.html, slides_dark.html")
    if "--pdf" in sys.argv:
        print_pdf(OUT / "slides.html", ROOT / "reports" / "presentation.pdf")
        print_pdf(OUT / "slides_dark.html", ROOT / "reports" / "presentation_dark.pdf")


if __name__ == "__main__":
    main()
