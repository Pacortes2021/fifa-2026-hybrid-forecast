from common2 import *
at=abrir("Liga MX")
import sys; m=sys.modules['motor']; print(m.__file__, m.DATA)
import pandas as pd; print(len(pd.read_csv(m.DATA/"fixture.csv")))
f=at.selectbox(key="sel_fix_mex"); print(len(f.options))
