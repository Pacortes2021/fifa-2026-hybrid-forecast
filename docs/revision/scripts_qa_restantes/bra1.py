from common2 import *
import sys
cl=sys.argv[1]
at=abrir(cl); resumen(at)
for i,d in enumerate(at.dataframe):
    v=d.value
    if "Escudo" in v.columns: v=v.drop(columns="Escudo")
    print("DF",i,v.shape); 
    if v.shape[0]<=40: print(v.to_string()[:3800])
    print(v.select_dtypes("number").sum().round(3).to_dict()  if v.shape[0]<=40 else "", "NaN",v.isna().sum().sum())
for m in at.markdown:
    t=m.value
    if any(k in t for k in ("Libertadores","Sudamericana","Descenso","descenso","2025","2024","2026","Serie","Apertura","Clausura","Campe")): print("MD:",t[:300].replace("\n"," | "))
