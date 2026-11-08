import pandas as pd, sys
for l in "ned por bel tur sco aut den gre nor swe".split():
    p=pd.read_csv(f"{l}/data/partidos.csv",parse_dates=["fecha"])
    p["ts"]=p.fecha.apply(lambda x: x.year if x.month>=7 else x.year-1)
    g=p.groupby("ts").agg(n=("fecha","size"),mn=("fecha","min"),mx=("fecha","max"),eq=("local","nunique"))
    print("==",l); print(g.assign(mn=g.mn.dt.date,mx=g.mx.dt.date).to_string())
    print("csv temporada vs ts mismatch:", int((p.temporada!=p.ts).sum()), "dups event_id", int(p.event_id.duplicated().sum()))
