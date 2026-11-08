from common2 import *
import socket
socket.socket.connect=lambda s,*a,**k:(_ for _ in ()).throw(OSError("net"))
at=abrir("Mundial")
at.button(key="wm_resim").click().run()
print("EXC",[e.value[:300] for e in at.exception])
print("MSG",[x.value[:300] for x in list(at.success)+list(at.info)+list(at.warning)+list(at.error)])
print("METRIC",[m.label+"="+str(m.value) for m in at.metric][:10])
for i,d in enumerate(at.dataframe):
    v=d.value
    if "P(Campeón)" in " ".join(map(str,v.columns)) or "🏆 P(Campeón)" in list(v.columns): print(i, v.head(5).to_string()[:500]); print(v.select_dtypes("number").sum().round(2).to_dict())
for r in at.radio: print(r.key, r.options, r.value)
at.radio(key="wm_fuente").set_value("✍️ Manual").run()
print("MANUAL EXC",[e.value[:300] for e in at.exception],[x.value[:200] for x in list(at.info)+list(at.warning)+list(at.error)])
for k in ("wm_model",):
    for o in at.radio(key=k).options:
        at.radio(key=k).set_value(o).run(); print(o,[e.value[:200] for e in at.exception],[m.value for m in at.metric][:6])
for c in ["Neutral","Local España","Local Argentina"]:
    at.radio(key="wm_cancha").set_value(c).run(); print(c,[e.value[:100] for e in at.exception])
for a,b in [("🇪🇸 España","🇪🇸 España"),("🇲🇽 México","🇿🇦 Sudáfrica")]:
    at.selectbox(key="m_a").select(a); at.selectbox(key="m_b").select(b); at.run()
    print(a,b,[e.value[:100] for e in at.exception],[x.value[:100] for x in list(at.error)], [m.value for m in at.metric][:5])
