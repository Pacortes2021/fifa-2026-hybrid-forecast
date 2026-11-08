from common import *
import numpy as np
at=abrir("Champions")
def E(tag):
    ex=[e.value[:300] for e in at.exception]
    if ex: print("EXC",tag,ex)
    return ex
# sums on sim table
df=[d.value for d in at.dataframe if "🏆 P(Campeón)" in d.value.columns][0]
print(df.shape, df.iloc[:,2:].sum().round(3).to_dict()); print("nan",df.isna().sum().sum())
print(sorted(df.Equipo.tolist()))
tabla=[d.value for d in at.dataframe if "GF" in d.value.columns and "PJ" in d.value.columns][0]
print(tabla.PJ.describe().to_dict(), tabla.shape, tabla.isna().sum().sum())
# models in sidebar
sb=at.sidebar.selectbox[1]
for op in sb.options:
    at.sidebar.selectbox[1].select(op).run(); E(op)
# in-tab model
m=[s for s in at.selectbox if s.label.startswith("🤖 Modelo para")][0]
for op in m.options:
    [s for s in at.selectbox if s.label.startswith("🤖 Modelo para")][0].select(op).run(); E(op)
    d=[d.value for d in at.dataframe][0]
# teams
ops=at.selectbox(key="sel_a_ucl").options
print(len(ops),ops[:3])
pairs=[(ops[0],ops[0]),(ops[0],ops[-1]),(ops[-1],ops[0]),(ops[5],ops[20])]
for a,b in pairs:
    at.selectbox(key="sel_a_ucl").select(a).run(); at.selectbox(key="sel_b_ucl").select(b).run(); 
    print(a,b,E("pair"),[w.value[:100] for w in at.warning], [m.label+"="+str(m.value) for m in at.metric][:1])
# checkbox
for lbl in ["Eliminatoria","Neutral"]:
    c=[c for c in at.checkbox if lbl in c.label][0]; c.check().run(); E(lbl)
    print(lbl,[x.value[:200] for x in list(at.caption) if "pase" in x.value], [m.label+"="+str(m.value) for m in at.metric][:3])
    for d in at.dataframe[:3]: print(d.value.head(3).to_string()[:400])
