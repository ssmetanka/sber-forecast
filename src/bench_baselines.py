import sys, pandas as pd, numpy as np
p = sys.argv[1]
c = pd.read_parquet(p + '/consumption.parquet')
c['date'] = pd.PeriodIndex(c.date, freq='M')
w = c.pivot_table(index=['territory_id','category'], columns='date', values='value')
full = w.dropna()
print('series total', len(w), 'full', len(full), 'MO full', full.index.get_level_values(0).nunique())
cats = full.index.get_level_values(1)
print('categories:', list(pd.unique(cats)))
L = np.log(full)
# national factor = median log per category per month
n = L.groupby(level=1).median()
d = L - n.reindex(cats).values
var_tot = L.sub(L.mean(1), axis=0).pow(2).sum().sum()
var_d = d.sub(d.mean(1), axis=0).pow(2).sum().sum()
print('share of within-series variance explained by national factor: %.3f' % (1 - var_d/var_tot))
# check all-categories vs sum of 5
allc = full.xs('Все категории', level=1); s5 = full.drop('Все категории', level=1).groupby(level=0).sum()
print('sum5/all median ratio %.3f' % (s5/allc.reindex(s5.index)).median().median())
months = list(full.columns)
Y = full.values; LY = L.values; N = n.reindex(cats).values; D = d.values
res = {}
def add(name, h, yt, yp):
    res.setdefault((name, h), [[], []]); res[(name,h)][0].append(yt); res[(name,h)][1].append(yp)
for h in [1,3,6,12]:
    for o in range(11, 24-h):   # origin index; targets fall in 2023-12+h .. 2024-12
        t = o + h
        if months[t] < pd.Period('2024-01','M'): continue
        yt = Y[:, t]
        add('naive', h, yt, Y[:, o])
        add('snaive', h, yt, Y[:, t-12])
        if o >= 12:
            g = LY[:, o] - LY[:, o-12]; gn = N[:, o] - N[:, o-12]
        else:  # origin 2023-12: no yoy available -> annualised H2-vs-H1 trend of 2023
            g = 2*(LY[:, 6:12].mean(1) - LY[:, 0:6].mean(1)); gn = 2*(N[:, 6:12].mean(1) - N[:, 0:6].mean(1))
        add('snaive*own_growth', h, yt, np.exp(LY[:, t-12] + g))
        # panel: national factor forecast = n[t-12] + national yoy growth at origin; local d = mean of last 3
        nf = N[:, t-12] + gn
        add('panel_d_last', h, yt, np.exp(nf + D[:, o]))
        add('panel_d_mean3', h, yt, np.exp(nf + D[:, o-2:o+1].mean(1)))
        # panel with damped local yoy: d forecast = d[t-12] + 0.5*(d[o]-d[o-12])
        add('panel_d_seas_damped', h, yt, np.exp(nf + D[:, t-12] + 0.5*(D[:, o]-D[:, max(o-12,0)])))
        # oracle national factor (upper bound for local part)
        add('panel_n_nogrowth', h, yt, np.exp(N[:, t-12] + D[:, o-2:o+1].mean(1)))
        add('ORACLE_n + d_mean3', h, yt, np.exp(N[:, t] + D[:, o-2:o+1].mean(1)))
rows=[]
for (m,h),(a,b) in res.items():
    a=np.concatenate(a); b=np.concatenate(b)
    mae=np.abs(a-b).mean(); r2=1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    wape=np.abs(a-b).sum()/a.sum()
    rows.append((m,h,round(mae,1),round(r2,4),round(100*wape,2),len(a)))
r=pd.DataFrame(rows,columns=['model','h','MAE','R2','WAPE%','n'])
print(r.pivot(index='model',columns='h',values='MAE').round(0))
print(r.pivot(index='model',columns='h',values='WAPE%'))
print(r.pivot(index='model',columns='h',values='R2'))
# mass shifts: share of MO with |d jump| > 3 robust sigma per month per category
dd = d.diff(axis=1)
z = dd.sub(dd.median(1),axis=0).div(1.4826*dd.sub(dd.median(1),axis=0).abs().median(1)+1e-9,axis=0)
frac = (z.abs()>3).groupby(level=1).mean()
print('share of MO with |z|>3 in local-component jump, top months per category:')
for cat,row in frac.iterrows():
    print(cat, row.sort_values(ascending=False).head(3).round(3).to_dict())
# national factor yoy
print('national factor (median log) yoy growth 2024 vs 2023 (%):')
print(((np.exp(n.iloc[:,12:].values - n.iloc[:,:12].values)-1)*100).mean(1).round(1), list(n.index))
