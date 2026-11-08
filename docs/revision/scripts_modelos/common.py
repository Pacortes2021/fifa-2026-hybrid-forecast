import sys, os, io, contextlib, warnings
warnings.filterwarnings("ignore")
REPO="/private/tmp/claude-501/-Users-pabloignaciocortesvielma-Desktop-portafolio-fintual--claude-worktrees-sports-forecast-repo-review-d72f96/221fe0a7-d084-49eb-bac5-2801723574ad/scratchpad/agentes/modelos"
LIGAS="eng esp bund ita fra ned por bel tur sco aut den gre nor swe bra mex chile arg ucl".split()
def load(liga):
    for k in list(sys.modules):
        if k=="motor": del sys.modules[k]
    sys.path[:] = [p for p in sys.path if not p.startswith(REPO)]
    sys.path.insert(0, f"{REPO}/{liga}")
    import motor
    pass
    with contextlib.redirect_stdout(io.StringIO()):
        M = motor.cargar()
    return motor, M
