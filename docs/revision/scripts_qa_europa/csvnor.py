import pandas as pd
for l in "nor swe".split():
    p=pd.read_csv(f"{l}/data/partidos.csv",parse_dates=["fecha"])
    g=p.groupby("temporada").agg(n=("fecha","size"),mn=("fecha","min"),mx=("fecha","max"),eq=("local","nunique"))
    print("==",l); print(g.to_string())
    print(p.groupby(["temporada",p.fecha.dt.month]).size().unstack(fill_value=0).to_string())
    print(p.estado.value_counts().to_dict(), p.status_name.value_counts().to_dict())
