from common import *
at=abrir("Liga MX"); resumen(at, True)
for i,t in enumerate(at.table): print("TBL",i,t.value.shape)
