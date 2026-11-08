from common2 import *
at=abrir("Liga MX")
for m in at.sidebar.selectbox[1].options:
    at.sidebar.selectbox[1].select(m).run()
    print(m, mds(at,"%")[:4] )
at.selectbox(key="sel_a_mex").select("Mazatlán FC").run()
at.selectbox(key="sel_b_mex").select("Mazatlán FC").run()
print("LOC=VIS",[e.value for e in at.exception],msgs(at)[:4], at.dataframe[0].value.to_string()[:300])
at.selectbox(key="sel_b_mex").select("América").run()
print("MZ vs AME",[e.value for e in at.exception],msgs(at)[:4])
f=at.selectbox(key="sel_fix_mex"); print(f.options)
f.select(f.options[1]).run(); print([e.value for e in at.exception], at.selectbox(key="sel_a_mex").value, at.selectbox(key="sel_b_mex").value)
