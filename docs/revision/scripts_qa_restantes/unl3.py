from common2 import *
at=abrir("Nations")
at.radio(key="unl_modo_vs").set_value('✍️ Selección Libre').run()
L=at.selectbox(key="unl_free_loc"); print(L.options[:5])
for a,b in [("Spain","Spain"),("San Marino","Spain"),("Spain","San Marino")]:
    at.selectbox(key="unl_free_loc").select(a); at.selectbox(key="unl_free_vis").select(b); at.run()
    print(a,b,[e.value[:100] for e in at.exception],[m.label+"="+str(m.value) for m in at.metric][:6],msgs(at)[:2])
at.radio(key="unl_filtro_ia").set_value('🌐 Fixture Completo (148 Partidos)').run()
print([e.value[:100] for e in at.exception],[m.value[:150] for m in at.markdown if 'Partidos' in m.value][:3], [d.value.shape for d in at.dataframe][-4:])
