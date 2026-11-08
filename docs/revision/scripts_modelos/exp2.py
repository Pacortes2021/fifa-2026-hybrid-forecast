from common import *
from lib import *
import pandas as pd, numpy as np, pickle, itertools, collections
def mult(gd):
    gd=abs(gd); return 1.0 if gd<=1 else (1.5 if gd==2 else (1.75 if gd==3 else 1.75+(gd-3)/8))
def elo_series(df,K,H,gdm=True):
    E=collections.defaultdict(lambda:1500.0); out=np.zeros(len(df))
    for i,(a,b,ga,gb) in enumerate(zip(df.local.values,df.visita.values,df.goles_local.values,df.goles_visita.values)):
        out[i]=E[a]-E[b]
        we=1/(1+10**(-(E[a]+H-E[b])/400)); w=1.0 if ga>gb else (0.0 if ga<gb else 0.5)
        d=K*(mult(ga-gb) if gdm else 1)*(w-we); E[a]+=d; E[b]-=d
    return out
def add_goals(df,l):
    if "goles_local" in df: return df
    p=pd.read_csv(f"{REPO}/{l}/data/partidos.csv",parse_dates=["fecha"])[["fecha","local","visita","goles_local","goles_visita"]].drop_duplicates(["fecha","local","visita"])
    d=df.merge(p,on=["fecha","local","visita"],how="left",validate="m:1"); assert d.goles_local.notna().all() and len(d)==len(df),l
    return d
res=[]; KS=[10,20,30,45,60]; HS=[30,55,80]
Pbest={}
for l in LIGAS:
    if l=="ucl": continue
    mo,M=load(l); df=getdf(M).reset_index(drop=True)
    df=add_goals(df,l)
    d2=df.sort_values("fecha",kind="stable").reset_index(drop=True) if "fecha" in df else df
    # check order == generacion
    ytr,ycal,yte=split_years(l); y=df.resultado.values.astype(int)
    tr=(df.temporada<=ytr).values; ca=(df.temporada==ycal).values; te=(df.temporada>=yte).values
    cur=elo_series(df,mo.K_LIGA,mo.HOME_ADV); corr=np.corrcoef(cur,df.elo_diff)[0,1]
    grid={}
    for K,H in itertools.product(KS,HS):
        e=elo_series(df,K,H)[:,None]; m=lr_elo(e[tr],y[tr])
        grid[(K,H)]=(logloss(y[ca],m.predict_proba(e[ca])), logloss(y[te],m.predict_proba(e[te])), logloss(y[tr|ca],lr_elo(e[tr|ca],y[tr|ca]).predict_proba(e[tr|ca])))
    kb=min(grid,key=lambda k:grid[k][0])
    cur_ll=None
    e=cur[:,None]; m=lr_elo(e[tr],y[tr]); cur_ll=logloss(y[te],m.predict_proba(e[te]))
    res.append(dict(liga=l,corr=corr,K=mo.K_LIGA,H=mo.HOME_ADV,cur_te=cur_ll,best_cal=kb,best_te=grid[kb][1],oracle=min(v[1] for v in grid.values()),grid=grid))
    print(l,round(corr,4),mo.K_LIGA,mo.HOME_ADV,round(cur_ll,4),kb,round(grid[kb][1],4),flush=True)
pickle.dump(res,open("res_elo.pkl","wb"))
# pooled: mejor (K,H) global por log-loss en cal
tot=collections.defaultdict(float); tote=collections.defaultdict(float); n=0; ne=0
for r in res:
    pass
import numpy as np
keys=list(res[0]["grid"])
w_cal=np.array([1]*len(res))
for k in keys:
    print(k, "cal %.4f  test %.4f"%(np.mean([r["grid"][k][0] for r in res]),np.mean([r["grid"][k][1] for r in res])))
print("actual cur_te mean",np.mean([r["cur_te"] for r in res]),"best_cal-per-liga test mean",np.mean([r["best_te"] for r in res]),"oracle",np.mean([r["oracle"] for r in res]))
