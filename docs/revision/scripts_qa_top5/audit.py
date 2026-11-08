import sys,os; sys.path.insert(0,os.path.dirname(__file__))
from h import *
import numpy as np, pandas as pd, json
def num(s): return float(re.sub(r"[^0-9.\-]","",s))
def check_pred(at,l,tag,out):
    exc=[e.value for e in at.exception]
    if exc: out.append(f"[{l}|{tag}] EXC {exc[:1]}"); return
    md=[m.value for m in at.markdown]
    pr=[m for m in md if re.match(r"\*\*(Victoria .*|Empate): [0-9.]+%\*\* \(Cuota Justa: `([0-9.]+)`\)",m)]
    if len(pr)!=3:
        out.append(f"[{l}|{tag}] sin 3 probs: {texts(at)[:2]}"); return
    ps=[];
    for m in pr:
        g=re.match(r"\*\*(.*): ([0-9.]+)%\*\* \(Cuota Justa: `([0-9.]+)`\)",m); p=float(g[2])/100; c=float(g[3]); ps.append(p)
        if abs(c-1/p)>0.01*c+0.01: out.append(f"[{l}|{tag}] cuota!=1/p {m}")
    if abs(sum(ps)-1)>0.0015: out.append(f"[{l}|{tag}] 1X2 suma {sum(ps)}")
    # tablas mercados
    dfs=[d.value for d in at.dataframe]
    mk={}
    for d in dfs:
        if list(d.columns)==["Mercado","Prob.","Cuota justa"]:
            for r in d.itertuples(index=False): mk[r[0]]=(pct(r[1]),float(r[2]))
    if mk:
        for ln in ("1.5","2.5","3.5"):
            o=mk[f"Over {ln} goles"][0];u=mk[f"Under {ln} goles"][0]
            if abs(o+u-1)>0.0021: out.append(f"[{l}|{tag}] O/U {ln} suma {o+u}")
        if abs(mk["Ambos marcan: Sí"][0]+mk["Ambos marcan: No"][0]-1)>0.0021: out.append(f"[{l}|{tag}] BTTS no complementario")
        for k,(p,c) in mk.items():
            if p>0 and abs(c-1/p)>0.02*c+0.02: out.append(f"[{l}|{tag}] cuota {k} {p} {c}")
        o15,o25,o35=[mk[f"Over {x} goles"][0] for x in ("1.5","2.5","3.5")]
        if not(o15>=o25>=o35): out.append(f"[{l}|{tag}] Over no monotono")
        dnb=[v for k,v in mk.items() if k.startswith("Sin Empate")]
        if dnb and abs(dnb[0][0]+dnb[1][0]-1)>0.0021: out.append(f"[{l}|{tag}] DNB suma")
        d1=[v for k,v in mk.items() if "Doble" in k]
        if d1 and abs(d1[0][0]-(ps[0]+ps[1]))>0.0021: out.append(f"[{l}|{tag}] 1X incoherente")
    else: out.append(f"[{l}|{tag}] sin tablas mercado")
    # marcadores top
    top=[m for m in md if re.match(r"\*   `\d+ - \d+`",m)]
    if len(top)!=4: out.append(f"[{l}|{tag}] marcadores {len(top)}")
    return dict(ps=ps,mk=mk)
