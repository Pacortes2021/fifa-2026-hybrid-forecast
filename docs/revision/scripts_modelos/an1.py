import pandas as pd, numpy as np, pickle
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",500)
r=pd.read_csv("res1.csv"); st=pickle.load(open("res1.pkl","rb"))
L=list(st)
def piv(var,model,col="ll"): 
    return r[(r["var"]==var)&(r.model==model)].set_index("liga")[col]
T=pd.DataFrame({"n":[len(st[l]["ytest"]) for l in L]},index=L)
for nm,(v,m) in {"freq":("base","freq"),"elo":("base","elo"),"lasso":("full","lasso"),"rf":("full","rf"),"xgb":("full","xgb"),"stack":("full","stack"),"avg":("full","avg"),"core":("core","lr"),"dep_stack":("deployed","stack"),"dep_rf":("deployed","rf"),"dep_lasso":("deployed","lasso"),"elo_tc":("base","elo_tc")}.items():
    T[nm]=piv(v,m)
print("LOGLOSS test"); print(T.round(4))
w=T.n/T.n.sum()
print("pooled (weighted by n):"); print((T.drop(columns="n").mul(w,axis=0)).sum().round(4))
print("simple mean:"); print(T.drop(columns="n").mean().round(4))
B=pd.DataFrame(index=L)
for nm,(v,m) in {"freq":("base","freq"),"elo":("base","elo"),"lasso":("full","lasso"),"rf":("full","rf"),"xgb":("full","xgb"),"stack":("full","stack")}.items(): B[nm]=piv(v,m,"brier")
print("BRIER"); print(B.round(4)); print(B.mean().round(4))
A=pd.DataFrame(index=L)
for nm,(v,m) in {"freq":("base","freq"),"elo":("base","elo"),"lasso":("full","lasso"),"rf":("full","rf"),"xgb":("full","xgb"),"stack":("full","stack")}.items(): A[nm]=piv(v,m,"acc")
print("ACC"); print((A*100).round(1)); print((A.mean()*100).round(2))
# mejor individual vs stack
print("stack - best individual (negativo = stack mejor):")
bi=T[["lasso","rf","xgb"]].min(axis=1); print((T.stack-bi).round(4).to_string()); print("mean",(T.stack-bi).mean())
print("n ligas donde ML(stack) < elo:", (T.stack<T.elo).sum(), " lasso<elo",(T.lasso<T.elo).sum()," rf<elo",(T.rf<T.elo).sum(), "xgb<elo",(T.xgb<T.elo).sum(), "core<elo",(T.core<T.elo).sum())
