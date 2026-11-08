from common import *
at=abrir("NBA")
r=at.radio(key="nba_val_periodo"); r.set_value(r.options[1]).run()
print([e.value[:200] for e in at.exception],[m.label+"="+str(m.value) for m in at.metric])
print([c.value for c in at.caption][-6:], [x.value[:200] for x in list(at.info)+list(at.warning)])
f=at.selectbox(key="filtro_hist_nba"); print(f.options[:4]); f.select(f.options[3]).run()
print([e.value[:200] for e in at.exception], at.dataframe[-1].value.shape, at.dataframe[-1].value.head(3).to_string())
# train/cal/test accuracies
import nba.motor as mo
M=mo.cargar(); d=M["df_dataset"]
import numpy as np
p=M["pipe_lasso"].predict_proba(d[M["features"]].fillna(0))[:,1]
for lo,hi in ((2022,2024),(2025,2025),(2026,2027)):
    m=(d.temporada>=lo)&(d.temporada<=hi); y=d.target_win_local[m]
    print(lo,hi,m.sum(),((p[m]>.5)==y).mean())
print(d.temporada.value_counts().sort_index().to_dict(), M.get("sigma_spread"), M.get("sigma_totals"))
hi=p>=0.85
print("p>=.85 in-sample", hi.sum(), (d.target_win_local[hi]==1).mean())
t=d.temporada>=2026; h2=(p>=0.85)&t; print("OOS p>=.85",h2.sum(),(d.target_win_local[h2]==1).mean())
a=(1-p)>=0.8; print("visita>=.8 all",a.sum(),(d.target_win_local[a]==0).mean(), "oos",(a&t).sum(), (d.target_win_local[a&t]==0).mean())
