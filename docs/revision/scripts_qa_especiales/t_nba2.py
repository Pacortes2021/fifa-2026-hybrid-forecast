from common import *
import numpy as np
at=abrir("NBA")
sb=at.sidebar.selectbox[1]
res={}
for op in sb.options:
    sb.select(op).run(); at.sidebar.selectbox[1]
    sb=at.sidebar.selectbox[1]
    print(op[:20], "EXC", [e.value[:200] for e in at.exception], [m.label+"="+str(m.value) for m in at.metric])
    res[op]=[ [d.value.copy() for d in at.dataframe][i] for i in (4,5)]
# sim independence from model
a=res[sb.options[0]]; 
for k,v in res.items():
    print(k[:15], all((v[i].values==a[i].values).all() for i in (0,1)))
d=a[0]; print(d.columns.tolist())
def pc(x): return float(str(x).strip('%'))/100
for i,(n,df) in enumerate(zip(("East","West"),a)):
    print(n, "n=",len(df), "champ",sum(df.P_Campeon.map(pc)), "finalConf",sum(df.P_Finalista_Conf.map(pc)),"playoff",sum(df.P_Playoff_Directo.map(pc)),"playin",sum(df.P_PlayIn.map(pc)))
    print((df.P_Playoff_Directo.map(pc)+df.P_PlayIn.map(pc)).max())
# spread table
at.sidebar.selectbox[1].select(sb.options[0]).run()

at.selectbox(key="sel_home_nba").options
for h,a_ in [("BOS · Boston Celtics","BOS · Boston Celtics"),("WSH · Washington Wizards","OKC · Oklahoma City Thunder"),("UTA · Utah Jazz","DET · Detroit Pistons"),("LAC · LA Clippers","LAL · Los Angeles Lakers")]:
    ops=at.selectbox(key="sel_home_nba").options
    h=next((o for o in ops if o.split(" · ")[1]==h.split(" · ")[1]),None)
    a_=next((o for o in ops if o.split(" · ")[1]==a_.split(" · ")[1]),None)
    print(h,a_)
    if not h or not a_: continue
    at.selectbox(key="sel_home_nba").select(h).run(); at.selectbox(key="sel_away_nba").select(a_).run()
    print([e.value[:150] for e in at.exception],[w.value[:100] for w in at.warning])
    if h!=a_:
        print([m.value.split('metric-val">')[1].split("<")[0] for m in at.markdown if "Victoria" in m.value and "metric-val" in m.value])
        print(at.dataframe[1].value.to_string())
