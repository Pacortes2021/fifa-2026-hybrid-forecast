import pandas as pd, os
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for l in ["eng","esp","bund","ita","fra"]:
    d=f"{R}/{l}/data/"
    p=pd.read_csv(d+"partidos.csv",parse_dates=["fecha"]); f=pd.read_csv(d+"fixture.csv",parse_dates=["fecha"]); e=pd.read_csv(d+"equipos.csv")
    ts=lambda x: x.year if x.month>=7 else x.year-1
    p["t"]=p.fecha.apply(ts); f["t"]=f.fecha.apply(ts)
    cur=p.t.max()
    print("=====",l,"temp actual",cur,"partidos",len(p),"fixture",len(f), f.estado.value_counts().to_dict())
    print(" partidos por t:",p.groupby("t").size().to_dict())
    print(" fixture por t:",f.groupby("t").size().to_dict(), "fecha min/max",f.fecha.min(),f.fecha.max())
    eq_p=set(p[p.t==cur].local)|set(p[p.t==cur].visita); eq_f=set(f[f.t==cur].local)|set(f[f.t==cur].visita)
    print(" equipos temp en partidos",len(eq_p),"fixture",len(eq_f),"solo fixture",eq_f-eq_p,"solo partidos",eq_p-eq_f)
    # overlap fixture vs partidos
    k=set(zip(p.local,p.visita,p.t)); ov=[r for r in f.itertuples() if (r.local,r.visita,r.t) in k]
    print(" fixture ya jugados (misma pareja y temp):",len(ov), ov[:3])
    ids=set(p.event_id)&set(f.event_id); print(" event_id solapados:",len(ids))
    print(" fixture pasado (< hoy 2026-11-08):",(f.fecha<pd.Timestamp("2026-11-08")).sum(), f[f.fecha<pd.Timestamp("2026-11-08")].head(3)[["fecha","local","visita","estado"]].to_dict("records"))
    # partidos por equipo vs esperado
    allt=eq_p|eq_f
    ids_e=set(e.name)|set(e.norm_name)
    print(" sin escudo (nombre no está en equipos.name/norm_name):",sorted(allt-ids_e))
    cnt=p[p.t==cur].groupby("local").size().add(p[p.t==cur].groupby("visita").size(),fill_value=0)
    print(" pj min/max",cnt.min(),cnt.max())
    n=len(eq_f); print(" fixture esperado por equipo: partidos tot", len(p[p.t==cur])+len(f[f.t==cur]), "esperado", n*(n-1))
    print(" NaN goles partidos:",p[["goles_local","goles_visita"]].isna().sum().to_dict(), "estados:",p.estado.value_counts().to_dict())
    dup=p.duplicated(["fecha","local","visita"]).sum(); print(" duplicados partidos:",dup)
