from common2 import *
import socket,traceback
def blk(self,*a,**k): print("NET CONNECT", a, [f.name+":"+str(f.lineno) for f in traceback.extract_stack() if "qa_restantes" in f.filename and "_agente" not in f.filename][-3:]); raise OSError("net")
socket.socket.connect=blk
at=abrir("Mundial")
for m in at.markdown:
    t=m.value
    if "Campe" in t or "campe" in t: print("MD:",t[:700].replace("\n"," | "))
