from common2 import *
import re
at=abrir("Argentina")
def probs(at): return [float(x) for m in at.markdown for x in re.findall(r"(?:Victoria [^:]+|Empate): ([\d.]+)%",m.value)]
pares=[("Boca Juniors","River Plate"),("Estudiantes de Río Cuarto","Racing Club"),("Aldosivi","Aldosivi"),("Central Córdoba","Lanús"),("Lanús","Central Córdoba")]
for mo_ in at.sidebar.selectbox[1].options:
    at.sidebar.selectbox[1].select(mo_).run()
    for a,b in pares:
        at.selectbox(key="sel_a_arg").select(a); at.selectbox(key="sel_b_arg").select(b); at.run()
        p=probs(at)[:3]
        print(mo_[:12],a,b,p,round(sum(p),1),[e.value[:100] for e in at.exception],msgs(at)[:2] if a==b else "")
for d in at.dataframe[:2]: pass
for df in at.dataframe[:3]: print(df.value.iloc[:,:4].to_string()[:600])
