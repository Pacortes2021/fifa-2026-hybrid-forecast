from common2 import *
import re,sys
cl,k=os.environ["CL"],os.environ["K"]
at=abrir(cl)
def probs(at): return [float(x) for m in at.markdown for x in re.findall(r"(?:Victoria [^:]+|Empate): ([\d.]+)%",m.value)][:3]
ops=at.selectbox(key=f"sel_a_{k}").options
pares=[(ops[0],ops[-1]),(ops[3],ops[3]),(ops[-1],ops[0])]
for mo_ in at.sidebar.selectbox[1].options:
    at.sidebar.selectbox[1].select(mo_).run()
    for a,b in pares:
        at.selectbox(key=f"sel_a_{k}").select(a); at.selectbox(key=f"sel_b_{k}").select(b); at.run()
        p=probs(at); print(mo_[:10],a,b,p,round(sum(p),1),[e.value[:100] for e in at.exception],msgs(at)[:1] if a==b else "")
f=at.selectbox(key=f"sel_fix_{k}"); print(f.options[:4], f.options[-2:])
for o in f.options[1:3]+f.options[-1:]:
    f.select(o).run(); print(o, at.selectbox(key=f"sel_a_{k}").value, at.selectbox(key=f"sel_b_{k}").value,[e.value[:100] for e in at.exception])
