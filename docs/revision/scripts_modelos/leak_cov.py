from common import *
from lib import *
import pandas as pd, numpy as np
rows=[]
for l in LIGAS:
    mo,M=load(l); df=getdf(M); 
    adv=pd.read_csv(f"{REPO}/{l}/data/advanced_features_historical.csv")
    norm=getattr(mo,"_norm_equipo",lambda x:x)
    have={}
    for e,t in zip(adv.equipo,adv.temporada): have.setdefault(e,set()).add(int(t))
    cnt={"exact":0,"past":0,"future":0,"missing":0,"tot":0}; ty=split_years(l)
    cnt_te=dict(cnt)
    for t_,lo,vi,temp in zip(range(len(df)),df.local,df.visita,df.temporada):
        for team in (lo,vi):
            ss=have.get(norm(team)) or have.get(team)
            tgt=cnt_te if temp>ty[2]-1 else cnt
            tgt["tot"]+=1
            if not ss: tgt["missing"]+=1
            elif temp in ss: tgt["exact"]+=1
            else:
                best=min(ss,key=lambda s:(abs(s-temp),s)); tgt["past" if best<temp else "future"]+=1
    # ucl override
    rows.append(dict(liga=l, tr_nonexact=1-cnt["exact"]/cnt["tot"], tr_future=cnt["future"]/cnt["tot"], tr_missing=cnt["missing"]/cnt["tot"],
        te_nonexact=1-cnt_te["exact"]/max(cnt_te["tot"],1), te_future=cnt_te["future"]/max(cnt_te["tot"],1), te_missing=cnt_te["missing"]/max(cnt_te["tot"],1)))
d=pd.DataFrame(rows).set_index("liga"); print(d.round(3).to_string()); d.to_csv("leak_cov.csv")
