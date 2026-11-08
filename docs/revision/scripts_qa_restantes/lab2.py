from common2 import *
import socket
def blk(self,*a,**k): print("NET CONNECT", a); raise OSError("net bloqueada")
socket.socket.connect=blk
at=abrir("Mundial")
print("METRIC",[m.label+"="+str(m.value) for m in at.metric])
print("MSG",[x.value[:400] for x in list(at.success)+list(at.info)+list(at.warning)+list(at.error)])
print("CAP",[c.value[:300] for c in at.caption])
for m in at.markdown:
    t=m.value
    if len(t)<600 and any(k in t for k in ("Campe","campe","vivo","Final","final","Grupos","Fase","Octavos","pendiente","jugar","2026")): print("MD:",t[:500].replace("\n"," | "))
print([(t.label) for t in at.tabs])
for i,d in enumerate(at.dataframe[18:],18):
    print("DF",i,d.value.shape, list(d.value.columns)[:8]); print(d.value.head(5).to_string()[:600])
