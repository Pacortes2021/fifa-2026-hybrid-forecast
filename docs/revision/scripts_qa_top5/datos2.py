import pandas as pd, os, itertools
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ts=lambda x: x.year if x.month>=7 else x.year-1
def load(l):
    d=f"{R}/{l}/data/"
    p=pd.read_csv(d+"partidos.csv",parse_dates=["fecha"]); f=pd.read_csv(d+"fixture.csv",parse_dates=["fecha"])
    p["t"]=p.fecha.apply(ts); f["t"]=f.fecha.apply(ts); return p,f,pd.read_csv(d+"equipos.csv")
for l,t in [("esp",2026),("esp",2022),("ita",2022),("fra",2025)]:
    p,f,e=load(l); a=pd.concat([p[p.t==t],f[f.t==t]])
    eq=sorted(set(a.local)|set(a.visita)); 
    pares=a.groupby(["local","visita"]).size()
    falt=[(x,y) for x in eq for y in eq if x!=y and (x,y) not in pares.index]
    dup=pares[pares>1]
    print(l,t,"faltan",falt[:8],"dups",dup.head(5).to_dict(), "n eq",len(eq))
p,f,e=load("esp"); print(e[e.name.str.contains("Depor|Barcel|Alav")][["id","name","norm_name","short_name"]])
print(sorted(set(p[p.t==2026].local)))
# nombres duplicados entre equipos.csv name and norm_name (id lookup)
for l in ["eng","esp","bund","ita","fra"]:
    p,f,e=load(l); print(l,"equipos.csv filas",len(e),"activos",e.is_active.sum())
