"""
Recorrido headless de la UI: abre cada torneo del portal (app.py) con streamlit.testing.AppTest,
hace un rerun en cada uno (como al tocar un widget) y falla si alguna página lanza una excepción
o no muestra su contenido. Al final vuelve al primer torneo para comprobar que el enrutador no
mezcla módulos entre ligas.

Uso:
    python3 tests/ui_smoke.py                 # todos los torneos
    python3 tests/ui_smoke.py Premier NBA     # solo los que contengan esos textos
"""
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main():
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))  # como hace `streamlit run`
    at = AppTest.from_file("app.py", default_timeout=1200)
    at.run()
    if at.exception:
        print("FALLO al abrir el portal:", [e.value for e in at.exception])
        sys.exit(1)

    opciones = list(at.sidebar.selectbox[0].options)
    filtros = sys.argv[1:]
    if filtros:
        opciones = [o for o in opciones if any(f.lower() in o.lower() for f in filtros)]
    recorrido = opciones + opciones[:1]  # volver al primero al final

    fallos = 0
    for torneo in recorrido:
        t0 = time.time()
        errores = []
        try:
            at.sidebar.selectbox[0].select(torneo).run()
            errores += [str(e.value) for e in at.exception]
            at.run()
            errores += [str(e.value) for e in at.exception]
        except Exception as ex:  # timeout u otro error del propio AppTest
            errores.append(f"{type(ex).__name__}: {ex}")
        if not errores and len(at.markdown) + len(at.dataframe) == 0:
            errores.append("la página no mostró contenido")
        estado = "OK   " if not errores else "FALLO"
        fallos += bool(errores)
        print(f"{estado} {torneo[:40]:40} {time.time() - t0:6.1f}s {errores[:2] if errores else ''}", flush=True)

    print(f"\n{len(recorrido)} cargas, {fallos} con fallos")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
