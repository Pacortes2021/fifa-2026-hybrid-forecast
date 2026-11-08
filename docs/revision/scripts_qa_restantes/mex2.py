from common import *
import pandas as pd
at=abrir("Liga MX"); resumen(at)
for i in (4,5):
    d=at.dataframe[i].value.drop(columns=["Escudo"]); print(d.to_string())
    print(d.select_dtypes("number").sum().round(3).to_dict(), d.isna().sum().sum())
print([ (m.label,m.value) for m in at.markdown][:0])
for m in at.markdown: 
    t=m.value
    if any(k in t for k in ("Apertura","Clausura","2025","2024","play","Play","19","Mazatl")): print("MD:",t[:400].replace("\n"," | "))
