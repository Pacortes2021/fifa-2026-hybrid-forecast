import os,re,ast,pandas as pd
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for l in ["eng","esp","bund","ita","fra"]:
    src=open(f"{R}/{l}/app_{l}.py").read()
    m=re.search(r"TEAM_DETAILS\s*=\s*\{(.*?)\n\}",src,re.S)
    keys=set(re.findall(r'^\s*"([^"]+)":\s*\{',m.group(1),re.M)) if m else set()
    p=pd.read_csv(f"{R}/{l}/data/partidos.csv",parse_dates=["fecha"])
    cur=p[p.fecha>="2026-07-01"]; eq=set(cur.local)|set(cur.visita)
    allt=set(p.local)|set(p.visita)
    print(l,"equipos 2026 sin emoji propio:",sorted(eq-keys),"| keys sin uso:",sorted(keys-allt))
