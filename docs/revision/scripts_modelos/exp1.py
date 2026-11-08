import pickle, json, time
from common import *
from lib import *
rows=[]; store={}
t0=time.time()
for l in LIGAS:
    mo,M=load(l); df=getdf(M).reset_index(drop=True); cols=getcols(M)
    ytr_y,cal_y,te_y=split_years(l)
    tr=df.temporada<=ytr_y; ca=df.temporada==cal_y; te=df.temporada>=te_y
    y=df.resultado.values.astype(int)
    S={}
    # variantes de columnas
    adv=[c for c in cols if ADV_PAT.match(c)]; att=[c for c in cols if ATT_PAT.match(c)]
    V={"full":cols,"no_adv":[c for c in cols if c not in adv],"no_att":[c for c in cols if c not in att],
       "core":[c for c in ["elo_diff","pi_diff","pi_overall_diff","form_diff","ppg_diff","squad_value_diff"] if c in cols]}
    out={"n_tr":int(tr.sum()),"n_cal":int(ca.sum()),"n_te":int(te.sum()),"n_adv":len(adv),"n_cols":len(cols)}
    P={}  # probs de test por (variante, modelo); y cal
    PC={}
    ytest=y[te.values]; ycal=y[ca.values]
    for vn,vc in V.items():
        X=df[vc].fillna(0.0).values
        if vn=="core":
            m=Pipeline([("s",StandardScaler()),("lr",LogisticRegression(C=0.3,max_iter=2000))]).fit(X[tr.values],y[tr.values])
            P[(vn,"lr")]=m.predict_proba(X[te.values]); PC[(vn,"lr")]=m.predict_proba(X[ca.values]); continue
        mods=fit_models(X[tr.values],y[tr.values],M)
        for k,m in mods.items():
            P[(vn,k)]=m.predict_proba(X[te.values]); PC[(vn,k)]=m.predict_proba(X[ca.values])
        w=fit_stack(PC[(vn,"lasso")],PC[(vn,"rf")],PC[(vn,"xgb")],ycal)
        P[(vn,"stack")]=w[0]*P[(vn,"lasso")]+w[1]*P[(vn,"rf")]+w[2]*P[(vn,"xgb")]
        PC[(vn,"stack")]=w[0]*PC[(vn,"lasso")]+w[1]*PC[(vn,"rf")]+w[2]*PC[(vn,"xgb")]
        P[(vn,"avg")]=(P[(vn,"lasso")]+P[(vn,"rf")]+P[(vn,"xgb")])/3
        out[f"w_{vn}"]=[round(float(x),3) for x in w]
    # baselines
    freq=np.bincount(y[tr.values],minlength=3)/tr.sum()
    P[("base","freq")]=np.tile(freq,(te.sum(),1)); PC[("base","freq")]=np.tile(freq,(ca.sum(),1))
    Xe=df[["elo_diff"]].values
    me=lr_elo(Xe[tr.values],y[tr.values]); P[("base","elo")]=me.predict_proba(Xe[te.values]); PC[("base","elo")]=me.predict_proba(Xe[ca.values])
    # elo entrenado en train+cal (comparable al desplegado)
    me2=lr_elo(Xe[(tr|ca).values],y[(tr|ca).values]); P[("base","elo_tc")]=me2.predict_proba(Xe[te.values])
    # desplegado (train+cal)
    X=df[cols].fillna(0.0).values[te.values]
    dep={k:M[s].predict_proba(X) for k,s in [("lasso","pipe_lasso"),("rf","pipe_rf"),("xgb","pipe_xgb")]}
    sw=M.get("stack_w", M.get("weights_opt", M.get("weights_stacking")))
    if sw is not None and len(sw)==3:
        sw=np.array(sw,float); sw=sw/sw.sum()
        dep["stack"]=sw[0]*dep["lasso"]+sw[1]*dep["rf"]+sw[2]*dep["xgb"]; out["w_deployed"]=[round(float(x),3) for x in sw]
    for k,v in dep.items(): P[("deployed",k)]=v
    # calibracion (sobre stack full, cal 2024 con modelos train-only)
    for base in [("full","stack"),("full","rf"),("full","lasso"),("full","xgb")]:
        pc=PC[base]; pt=P[base]
        T=temp_fit(ycal,pc); P[("cal_temp",base[1])]=temp_apply(pt,T); out[f"T_{base[1]}"]=round(float(T),3)
        ir=iso_fit(ycal,pc); P[("cal_iso",base[1])]=iso_apply(ir,pt)
    # draw-boost: factor multiplicativo sobre empate ajustado en cal
    def dfac(pc):
        f=lambda a: logloss(ycal,(pc*np.array([1,np.exp(a),1])))
        return minimize_scalar(f,bounds=(-1,1),method="bounded").x
    a=dfac(PC[("full","stack")]); q=P[("full","stack")]*np.array([1,np.exp(a),1]); P[("cal_draw","stack")]=q/q.sum(1,keepdims=True); out["draw_a"]=round(float(a),3)
    out["y_freq_test"]=[round(float((ytest==k).mean()),3) for k in range(3)]; out["y_freq_train"]=[round(float(x),3) for x in freq]
    # metricas
    for key,p in P.items():
        r=dict(liga=l,var=key[0],model=key[1],n=len(ytest),ll=logloss(ytest,p),brier=brier(ytest,p),acc=acc(ytest,p),
               ece_a=ece(ytest,p,0),ece_d=ece(ytest,p,1),ece_h=ece(ytest,p,2),
               pd_pred=float(p[:,1].mean()),pd_real=float((ytest==1).mean()),ph_pred=float(p[:,2].mean()),ph_real=float((ytest==2).mean()),
               ph_argmax=float((p.argmax(1)==2).mean()),pdraw_argmax=float((p.argmax(1)==1).mean()))
        rows.append(r)
    store[l]=dict(P=P,PC=PC,ytest=ytest,ycal=ycal,info=out,te_idx=np.where(te.values)[0])
    print(l,round(time.time()-t0),out.get("w_full"),flush=True)
pd.DataFrame(rows).to_csv("res1.csv",index=False)
pickle.dump(store,open("res1.pkl","wb"))
