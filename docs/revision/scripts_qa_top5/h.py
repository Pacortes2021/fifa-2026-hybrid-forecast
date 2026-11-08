import os, sys, warnings, time, re; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0,REPO)
KEY={"eng":"Premier","esp":"LaLiga","bund":"Bundesliga (Alemania)","ita":"Serie A","fra":"Ligue 1"}
MODELOS=["Random Forest","LASSO","Stacking","XGBoost"]
def open_liga(l):
    at=AppTest.from_file("app.py",default_timeout=900); at.run()
    op=next(o for o in at.sidebar.selectbox[0].options if KEY[l] in o)
    t=time.time(); at.sidebar.selectbox[0].select(op).run(); print(l,"carga",round(time.time()-t,1),"s")
    return at
def sel_modelo(at,name):
    sb=at.sidebar.selectbox[1]
    op=next(o for o in sb.options if name in o); sb.select(op).run()
def texts(at):
    return [x.value for g in (at.success,at.info,at.warning,at.error) for x in g]
def pct(s): return float(s.strip().rstrip("%"))/100
