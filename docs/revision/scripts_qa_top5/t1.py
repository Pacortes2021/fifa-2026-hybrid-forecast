import sys; sys.path.insert(0,__import__("os").path.dirname(__file__))
from h import *
l=sys.argv[1]
at=open_liga(l)
print("exc",[e.value for e in at.exception])
print("sidebar sb",[(s.label,s.options) for s in at.sidebar.selectbox])
print("sb",[s.label for s in at.selectbox], "tabs",[t.label for t in at.tabs])
print("title/md:",[m.value[:140] for m in at.markdown if "main-" in m.value])
print("sidebar md:",[m.value for m in at.sidebar.markdown]); print("caption sb",[c.value for c in at.sidebar.caption])
print("texts",texts(at))
print("metrics",[m.label+"="+str(m.value) for m in at.metric])
for i,df in enumerate(at.dataframe): print(i,df.value.shape,list(df.value.columns)[:8])
