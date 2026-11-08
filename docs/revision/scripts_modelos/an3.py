import pandas as pd, numpy as np, pickle
from lib import *
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",500)
r=pd.read_csv("res1.csv"); st=pickle.load(open("res1.pkl","rb")); L=list(st)
for mname,key in [("stack",("full","stack")),("lasso",("full","lasso")),("rf",("full","rf")),("xgb",("full","xgb")),("elo",("base","elo"))]:
    s=r[(r["var"]==key[0])&(r.model==key[1])].set_index("liga")
    print("\n==",mname); print(s[["pd_pred","pd_real","ph_pred","ph_real","ph_argmax","pdraw_argmax","ece_a","ece_d","ece_h"]].round(3).assign(n=s.n).to_string())
    n=s.n; print("pooled: draw pred %.3f real %.3f | home pred %.3f real %.3f | ECE a/d/h %.3f %.3f %.3f"%tuple((s[c]*n).sum()/n.sum() for c in ["pd_pred","pd_real","ph_pred","ph_real","ece_a","ece_d","ece_h"]))
# curva de calibracion agrupada (stack)
for key in [("full","stack"),("base","elo")]:
    P=np.vstack([st[l]["P"][key] for l in L]); y=np.concatenate([st[l]["ytest"] for l in L])
    print("\nCURVA pooled",key, "N=",len(y))
    for k,nm in enumerate(["visita","empate","local"]):
        bins=np.quantile(P[:,k],np.linspace(0,1,9)); idx=np.clip(np.searchsorted(bins[1:-1],P[:,k],side="right"),0,7)
        print(nm,[(round(P[idx==b,k].mean(),3),round((y[idx==b]==k).mean(),3)) for b in range(8)])
    print("draw: rango pred", P[:,1].min().round(3), P[:,1].max().round(3), "std",P[:,1].std().round(3))
# Edge-case: partidos con P(local) alta
