import sys; sys.path.insert(0,__import__("os").path.dirname(__file__))
from audit import *
l=sys.argv[1]; out=[]
at=open_liga(l)
sa=f"sel_a_{l}"; sb=f"sel_b_{l}"; sf=f"sel_fix_{l}"
opts=at.selectbox(key=sa).options
print("n equipos opciones",len(opts))
tab=next(d.value for d in at.dataframe if "PTS" in d.value.columns)
teams=list(tab.Equipo.str.replace("⚽ ","")); print("opciones vs tabla:",set(opts)^set(teams))
fav,weak=teams[0],teams[-1]; mid=teams[len(teams)//2]
pairs=[(fav,weak),(weak,fav),(teams[0],teams[1]),(mid,teams[len(teams)//2+1]),(teams[-1],teams[-2]),(fav,fav)]
# equipos con menos partidos en dataset
p=pd.read_csv(f"{l}/data/partidos.csv"); cnt=pd.concat([p.local,p.visita]).value_counts()
low=[t for t in teams if t in cnt.index]; low=sorted(low,key=lambda t:cnt[t])[:2]; print("menos partidos hist:",[(t,int(cnt[t])) for t in low])
pairs+= [(low[0],fav),(fav,low[0]),(low[0],low[1])]
for a,b in pairs:
    for mdl in ["Random","XGBoost"]:
        sel_modelo(at,mdl)
        at.selectbox(key=sa).select(a); at.selectbox(key=sb).select(b); at.run()
        if a==b:
            print("a==b ->",texts(at)[:1], "exc",[e.value[:80] for e in at.exception]); break
        r=check_pred(at,l,f"{a} v {b}|{mdl}",out)
        if r and mdl=="Random": print(f"{a:24s} v {b:24s} 1X2={r['ps']} alert={[x[:25] for x in texts(at) if 'ALERTA' in x or 'FAV' in x or 'VISITA' in x]}")
# fixtures
fx=at.selectbox(key=sf); print("fixture opts",len(fx.options),fx.options[:3])
for i in [1,2,5,10,25]:
    if i>=len(fx.options): continue
    at.selectbox(key=sf).select(fx.options[i]).run()
    m=re.search(r"\| (.*) vs (.*)$",fx.options[i]); a,b=m[1],m[2]
    ga,gb=at.selectbox(key=sa).value,at.selectbox(key=sb).value
    ok=(ga,gb)==(a,b); print("fix",fx.options[i],"->",ga,gb,"OK" if ok else "FALLA", "exc",[e.value[:60] for e in at.exception])
    if not ok: out.append(f"[{l}] fixture {fx.options[i]} -> {ga},{gb}")
    check_pred(at,l,f"fix{i}",out)
# todos los fixtures
bad=[]
for o in fx.options[1:]:
    m=re.search(r"\| (.*) vs (.*)$",o)
    if m[1] not in opts or m[2] not in opts: bad.append(o)
print("fixtures con equipo fuera de opciones:",bad)
print("\n".join(out) or "OK pares")
