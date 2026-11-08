from common2 import *
import re
at=abrir("Nations")
print([ (r.key,r.options) for r in at.radio])
for mo_ in at.sidebar.selectbox[1].options:
    at.sidebar.selectbox[1].select(mo_).run()
    print(mo_,[e.value[:100] for e in at.exception], [m.label+"="+str(m.value) for m in at.metric][:3])
for liga in ["Liga B","Liga C","Liga D"]:
    at.radio(key="unl_sel_liga").set_value(liga).run()
    print(liga,[e.value[:100] for e in at.exception])
    for df in at.dataframe:
        v=df.value
        if "P(Descenso)" in v.columns or "P_Ascenso" in v.columns: print(v.to_string()[:700]); print(v.select_dtypes("number").sum().round(2).to_dict())
    print([c.value[:150] for c in at.caption][-6:])
at.radio(key="unl_modo_vs").set_value('✍️ Selección Libre').run()
print([ (s.label,s.key,len(s.options)) for s in at.selectbox],[e.value[:100] for e in at.exception])
