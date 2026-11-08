from common2 import *
import socket
orig=socket.socket.connect
def blk(self,*a,**k): print("NET CONNECT", a); raise OSError("net bloqueada")
socket.socket.connect=blk
at=abrir("Mundial"); resumen(at)
for i,d in enumerate(at.dataframe):
    v=d.value; print("DF",i,v.shape); print(v.head(12).to_string()[:1500]); 
for m in at.markdown:
    t=m.value
    if any(k in t for k in ("2026","vivo","ESPN","Jun","jun","Julio","final","Final","Grupo","fase")): print("MD:",t[:300].replace("\n"," | "))
