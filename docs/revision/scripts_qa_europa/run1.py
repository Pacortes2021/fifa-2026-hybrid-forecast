import os, sys, warnings, time; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(repo); sys.path.insert(0, repo)
KEY = sys.argv[1]; NAME = {"ned":"Eredivisie","por":"Primeira","bel":"Jupiler","tur":"Süper","sco":"Scottish","aut":"Austrian","den":"Danish","gre":"Super League","nor":"Eliteserien","swe":"Allsvenskan"}[KEY]
t=time.time()
at = AppTest.from_file("app.py", default_timeout=900); at.run()
op = next(o for o in at.sidebar.selectbox[0].options if NAME in o)
at.sidebar.selectbox[0].select(op).run()
print("t=%.0fs"%(time.time()-t))
def dump(at, tag):
    print("---",tag)
    print("EXC", [e.value for e in at.exception])
    print("ERR", [x.value for x in at.error]); print("WARN",[x.value for x in at.warning])
    print("SIDEBAR sel", [(s.label, s.options if len(s.options)<8 else len(s.options)) for s in at.sidebar.selectbox])
    print("SEL", [(s.label, s.key, s.value) for s in at.selectbox])
    print("INFO",[x.value[:200] for x in list(at.info)+list(at.success)])
    print("METRIC",[m.label+"="+str(m.value)+"|"+str(m.delta) for m in at.metric])
dump(at,"default")
mod = at.sidebar.selectbox[0]
for sb in at.sidebar.selectbox:
    if "Modelo" in sb.label: mod=sb
for o in mod.options:
    mod.select(o).run(); 
    dump(at,o)
    print("MD", [m.value[:150] for m in at.markdown if "Modelo activo" in m.value])
