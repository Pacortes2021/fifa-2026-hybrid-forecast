from common import *
def msgs(at): return [x.value[:120] for x in list(at.error)+list(at.warning)+list(at.info)+list(at.success)]
def mds(at,*keys): return [m.value[:300].replace("\n"," ") for m in at.markdown if any(k in m.value for k in keys)]
