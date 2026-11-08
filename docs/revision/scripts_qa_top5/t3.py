import sys; sys.path.insert(0,__import__("os").path.dirname(__file__))
from audit import *
import itertools
l=sys.argv[1]; out=[]; info=[]
at=open_liga(l)
def tabla_cols(at,col):
    for d in at.dataframe:
        if col in d.value.columns: return d.value
# --- Tab2 por modelo
tc={"eng":(8,3),"esp":(7,3),"bund":(6,2),"ita":(7,3),"fra":(6,3)}[l]
for mdl in MODELOS:
    t=time.time(); sel_modelo(at,mdl)
    if at.exception: out.append(f"[{l}|{mdl}] EXC {at.exception[0].value[:100]}"); continue
    act=tabla_cols(at,"PTS"); proy=tabla_cols(at,"🔻 P(Descenso)")
    cols=[c for c in proy.columns if c.startswith(("🏆","🌟","🇪🇺","🔻"))]
    s={c:round(proy[c].astype(float).sum(),3) for c in cols}
    info.append(f"{mdl} ({time.time()-t:.0f}s) sumas MC {s} filas proy {len(proy)} act {len(act)}")
    if abs(s["🏆 P(Campeón)"]-1)>0.01 if "🏆 P(Campeón)" in s else False: out.append(f"[{l}|{mdl}] campeon suma {s}")
    if abs(s["🇪🇺 P(Copas)"]-tc[0])>0.05: out.append(f"[{l}|{mdl}] copas suma {s['🇪🇺 P(Copas)']} esperado {tc[0]}")
    if abs(s["🔻 P(Descenso)"]-tc[1])>0.05: out.append(f"[{l}|{mdl}] descenso suma {s['🔻 P(Descenso)']} esperado {tc[1]}")
    if proy.isna().any().any() or act.isna().any().any(): out.append(f"[{l}|{mdl}] NaN en tablas")
    if set(proy["Equipo"])!=set(act["Equipo"]): out.append(f"[{l}|{mdl}] equipos tabla!=proy {set(proy.Equipo)^set(act.Equipo)}")
    if mdl.startswith("Random"):
        print(act.drop(columns="Escudo").to_string()); print(proy.drop(columns="Escudo").head(25).to_string())
        print("escudos faltantes:",list(act[act.Escudo.isna()].Equipo))
        # coherencia tabla: PJ
        print("PJ",sorted(set(act.PJ)))
print("\n".join(info))
print("\n".join(out) or "OK MC")
