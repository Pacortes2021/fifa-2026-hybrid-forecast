import sys; sys.path.insert(0,__import__("os").path.dirname(__file__))
from audit import *
l=sys.argv[1]; out=[]
at=open_liga(l)
for mdl in MODELOS:
    sel_modelo(at,mdl)
    r=check_pred(at,l,mdl,out)
    sub=[m.value for m in at.markdown if "main-subtitle" in m.value]
    t3=[m.value[:60] for m in at.markdown if "sec-title" in m.value and ("Explicabilidad" in m.value or "Importancia" in m.value)]
    t3b=[m.value[:100] for m in at.markdown if m.value.startswith("La ")]
    print(mdl,"| subt:",re.sub("<[^>]+>","",sub[0]) if sub else None,"| tab3:",[re.sub("<[^>]+>","",x) for x in t3], t3b[:1] and t3b[0][:50], "| texts:",[x[:60] for x in texts(at)])
    print("   ps",r and r["ps"], "cols t3", [list(d.value.columns) for d in at.dataframe if "Variable" in d.value.columns])
print("\n".join(out) or "OK sin hallazgos")
