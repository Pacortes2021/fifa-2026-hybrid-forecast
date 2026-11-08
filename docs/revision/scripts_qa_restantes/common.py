import logging; logging.disable(logging.CRITICAL)
import os, sys, warnings, time; warnings.filterwarnings("ignore")
REPO = "/private/tmp/claude-501/-Users-pabloignaciocortesvielma-Desktop-portafolio-fintual--claude-worktrees-sports-forecast-repo-review-d72f96/221fe0a7-d084-49eb-bac5-2801723574ad/scratchpad/agentes/qa_restantes"
os.chdir(REPO); sys.path.insert(0, REPO)
from streamlit.testing.v1 import AppTest
def abrir(clave):
    at = AppTest.from_file("app.py", default_timeout=900); at.run()
    op = next(o for o in at.sidebar.selectbox[0].options if clave in o)
    t=time.time(); at.sidebar.selectbox[0].select(op).run(); print("t=%.1fs"%(time.time()-t))
    return at
def resumen(at, full=False):
    print("EXC:", [e.value for e in at.exception])
    print("SIDEBAR:", [(s.label, s.options if len(s.options)<12 else len(s.options), s.value) for s in at.sidebar.selectbox],
          [(r.label, r.options, r.value) for r in at.sidebar.radio],
          [(s.label, s.value) for s in at.sidebar.slider],[(s.label,s.value) for s in at.sidebar.checkbox])
    print("SEL:", [(s.label, s.key, len(s.options), s.value) for s in at.selectbox])
    print("RADIO:", [(r.label, r.key, r.options, r.value) for r in at.radio])
    print("SLIDER:", [(s.label, s.key, s.value) for s in at.slider], "NUM:", [(s.label,s.key,s.value) for s in at.number_input])
    print("TABS:", [t.label for t in at.tabs])
    print("BTN:", [(b.label,b.key) for b in at.button])
    print("CHK:", [(b.label,b.key,b.value) for b in at.checkbox])
    print("HEAD:", [h.value for h in list(at.header)+list(at.subheader)][:60])
    print("METRIC:", [m.label+"="+str(m.value) for m in at.metric])
    print("MSG:", [x.value[:200] for x in list(at.success)+list(at.info)+list(at.warning)+list(at.error)])
    print("CAP:", [c.value[:200] for c in at.caption])
    if full:
        for i,df in enumerate(at.dataframe): print("DF",i,df.value.shape); print(df.value.head(8).to_string()[:1200])
