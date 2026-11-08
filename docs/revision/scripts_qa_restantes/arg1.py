from common2 import *
at=abrir("Argentina"); resumen(at)
for i,d in enumerate(at.dataframe):
    v=d.value
    if "Escudo" in v.columns: v=v.drop(columns="Escudo")
    print("DF",i,v.shape); print(v.head(40).to_string()[:3500]); print(v.select_dtypes("number").sum().round(3).to_dict(), "NaN",v.isna().sum().sum())
for m in at.markdown:
    t=m.value
    if any(k in t for k in ("Zona","zona","Apertura","Clausura","Playoff","playoff","Promedio","promedio","Descenso","descenso","Copa","Libertadores","2025","2024","Sudamericana")): print("MD:",t[:400].replace("\n"," | "))
