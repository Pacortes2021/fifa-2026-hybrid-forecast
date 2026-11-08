import numpy as np, pandas as pd, re
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize, minimize_scalar
EPS=1e-7
ADV_PAT = re.compile(r"^(squad_value|avg_age|squad_size|pct_foreigners|foreigners|stadium_|avg_attendance|squad_val_per_player)")
ATT_PAT = re.compile(r"^(stadium_occupation|avg_attendance)")
def split_years(l): return (2024,2025,2026) if l=="ucl" else (2023,2024,2025)
def getdf(M): return M.get("df_features", M.get("df_dataset"))
def getcols(M): return list(M.get("cols", M.get("features")))
def logloss(y,p):
    p=np.clip(p,EPS,1); p=p/p.sum(1,keepdims=True); return float(-np.mean(np.log(p[np.arange(len(y)),y])))
def brier(y,p):
    Y=np.eye(3)[y]; return float(np.mean(((p-Y)**2).sum(1)))
def acc(y,p): return float(np.mean(p.argmax(1)==y))
def ece(y,p,k,nb=8):
    q=np.quantile(p[:,k],np.linspace(0,1,nb+1)); idx=np.clip(np.searchsorted(q[1:-1],p[:,k],side="right"),0,nb-1)
    e=0
    for b in range(nb):
        m=idx==b
        if m.sum(): e+=m.mean()*abs(p[m,k].mean()-(y[m]==k).mean())
    return float(e)
def fit_stack(pl,pr,px,y):
    def f(w):
        w=np.abs(w); w=w/w.sum(); return logloss(y,w[0]*pl+w[1]*pr+w[2]*px)
    r=minimize(f,[.4,.3,.3],method="Nelder-Mead"); w=np.abs(r.x); return w/w.sum()
def temp_fit(y,p):
    lp=np.log(np.clip(p,EPS,1))
    def f(T):
        z=lp/T; z-=z.max(1,keepdims=True); q=np.exp(z); q/=q.sum(1,keepdims=True); return logloss(y,q)
    return minimize_scalar(f,bounds=(.3,5),method="bounded").x
def temp_apply(p,T):
    z=np.log(np.clip(p,EPS,1))/T; z-=z.max(1,keepdims=True); q=np.exp(z); return q/q.sum(1,keepdims=True)
def iso_fit(y,p):
    return [IsotonicRegression(out_of_bounds="clip",y_min=1e-3,y_max=1-1e-3).fit(p[:,k],(y==k).astype(float)) for k in range(3)]
def iso_apply(ir,p):
    q=np.column_stack([ir[k].predict(p[:,k]) for k in range(3)]); return q/q.sum(1,keepdims=True)
def lr_elo(Xtr,ytr):
    return LogisticRegression(C=1e4,max_iter=2000).fit(Xtr,ytr)
def probs3(model,X):
    p=model.predict_proba(X); out=np.zeros((len(X),3)); out[:,model.classes_ if hasattr(model,'classes_') else model.named_steps[list(model.named_steps)[-1]].classes_]=p; return out
def fit_models(Xtr,ytr,M,which=("lasso","rf","xgb")):
    mods={}
    src={"lasso":"pipe_lasso","rf":"pipe_rf","xgb":"pipe_xgb"}
    for w in which:
        m=clone(M[src[w]]); m.fit(Xtr,ytr); mods[w]=m
    return mods
def pr(m,X):
    p=m.predict_proba(X); return p
