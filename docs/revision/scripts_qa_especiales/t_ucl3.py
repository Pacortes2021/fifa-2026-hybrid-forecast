from common import *
at=abrir("Champions")
for x in list(at.info)+list(at.success)+list(at.warning): print("MSG",x.value)
r=at.radio[0]
for op in r.options:
    at.radio[0].set_value(op).run()
    print(op,[e.value[:300] for e in at.exception], [x.value[:150] for x in list(at.info)+list(at.warning)+list(at.error)][-2:])
    print([(s.label,s.options[:3] if len(s.options)>3 else s.options, s.value) for s in at.selectbox if 'Torneo' not in s.label and 'Modelo' not in s.label and 'Club' not in s.label][:4])
    print([(s.label) for s in at.date_input], [(m.label,m.value) for m in at.metric if 'Partidos' in m.label])
at.checkbox[2].check().run() ; print("excl",[e.value[:300] for e in at.exception],[(m.label,m.value) for m in at.metric if 'Partidos' in m.label])
