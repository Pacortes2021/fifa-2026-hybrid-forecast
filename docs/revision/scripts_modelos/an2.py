import pandas as pd, numpy as np, pickle
from lib import *
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",500)
r=pd.read_csv("res1.csv"); st=pickle.load(open("res1.pkl","rb")); L=list(st)
def ll_vec(y,p):
    p=np.clip(p,EPS,1); p=p/p.sum(1,keepdims=True); return -np.log(p[np.arange(len(y)),y])
rng=np.random.default_rng(0)
def pooled_delta(a,b,nb=1000,per=False):
    # a,b: (var,model); delta = ll(a)-ll(b) por partido; negativo=a mejor
    D=[np.concatenate([[0]])]; d_l=[]
    for l in L:
        y=st[l]["ytest"]; d=ll_vec(y,st[l]["P"][a])-ll_vec(y,st[l]["P"][b]); d_l.append(d)
    allv=np.concatenate(d_l)
    bs=[rng.choice(allv,len(allv)).mean() for _ in range(nb)]
    out=(allv.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5))
    if per:
        pl={}
        for l,d in zip(L,d_l):
            b_=[rng.choice(d,len(d)).mean() for _ in range(300)]; pl[l]=(d.mean(),np.percentile(b_,2.5),np.percentile(b_,97.5))
        return out,pl
    return out
print("=== ML vs Elo-LR (delta logloss por partido, pooled 20 ligas; neg = ML mejor)")
for a in [("full","lasso"),("full","rf"),("full","xgb"),("full","stack"),("core","lr"),("cal_temp","stack"),("deployed","stack"),("deployed","lasso")]:
    o,pl=pooled_delta(a,("base","elo"),per=True); print(a,"%.4f [%.4f, %.4f]"%o)
    if a==("full","stack"):
        sig=[(l,round(v[0],4),"SIG" if (v[2]<0 or v[1]>0) else "") for l,v in pl.items()]; print(sig)
print("=== elo vs freq", "%.4f [%.4f,%.4f]"%pooled_delta(("base","elo"),("base","freq")))
print("=== stack vs lasso", "%.4f [%.4f,%.4f]"%pooled_delta(("full","stack"),("full","lasso")))
print("=== stack vs rf", "%.4f [%.4f,%.4f]"%pooled_delta(("full","stack"),("full","rf")))
print("=== avg(1/3) vs stack", "%.4f [%.4f,%.4f]"%pooled_delta(("full","avg"),("full","stack")))
# --- fugas
print("\n=== FUGAS: logloss test, variantes (stack | lasso | rf)")
F=pd.DataFrame(index=L)
for v in ["full","no_adv","no_att"]:
    for m in ["lasso","rf","stack"]:
        F[f"{v}_{m}"]=r[(r["var"]==v)&(r.model==m)].set_index("liga").ll
F["elo"]=r[(r["var"]=="base")&(r.model=="elo")].set_index("liga").ll
F["n_adv"]=[st[l]["info"]["n_adv"] for l in L]; F["n_cols"]=[st[l]["info"]["n_cols"] for l in L]
F["d_stack(noadv-full)"]=F.no_adv_stack-F.full_stack; F["d_lasso"]=F.no_adv_lasso-F.full_lasso; F["d_rf"]=F.no_adv_rf-F.full_rf; F["d_stack_noatt"]=F.no_att_stack-F.full_stack
print(F.round(4).to_string()); print(F.mean().round(4))
n=np.array([len(st[l]["ytest"]) for l in L]); 
print("pooled weighted:", {c: round(float((F[c]*n).sum()/n.sum()),4) for c in ["full_lasso","no_adv_lasso","full_rf","no_adv_rf","full_stack","no_adv_stack","no_att_stack","elo"]})
for a,b in [(("no_adv","stack"),("full","stack")),(("no_adv","lasso"),("full","lasso")),(("no_adv","rf"),("full","rf")),(("no_att","stack"),("full","stack")),(("no_adv","stack"),("base","elo")),(("no_adv","lasso"),("base","elo"))]:
    print(a,"vs",b,"%.4f [%.4f, %.4f]"%pooled_delta(a,b))
# --- calibracion
print("\n=== CALIBRACION (stack full): logloss test")
C=pd.DataFrame(index=L)
C["raw"]=r[(r["var"]=="full")&(r.model=="stack")].set_index("liga").ll
for v in ["cal_temp","cal_iso","cal_draw"]: C[v]=r[(r["var"]==v)&(r.model=="stack")].set_index("liga").ll
C["T"]=[st[l]["info"].get("T_stack") for l in L]; C["draw_a"]=[st[l]["info"].get("draw_a") for l in L]
print(C.round(4).to_string()); print(C.mean().round(4))
for v in ["cal_temp","cal_iso","cal_draw"]: print(v,"%.4f [%.4f,%.4f]"%pooled_delta((v,"stack"),("full","stack")))
for m in ["rf","lasso","xgb"]:
    for v in ["cal_temp","cal_iso"]: print(v,m,"%.4f [%.4f,%.4f]"%pooled_delta((v,m),("full",m)))
