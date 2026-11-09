"""
Recolecta el BOX SCORE detallado de cada partido de la Premier League desde ESPN (endpoint summary):
Recupera las 28 estadísticas disponibles en el boxscore (tiros, posesión, pases, etc.).
Usa un ThreadPoolExecutor para descargar en paralelo y acelerar la recolección.
Incremental: no re-pide partidos ya guardados.

Uso:  python3 recolectar_boxscore.py   -> esp/data/box_score.csv
"""
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comun.recolectar_comun import (
    STATS,
    eventos_terminados_anio,
    descargar_boxscore_evento,
    recolectar_boxscore_liga,
)

DATA = Path(__file__).resolve().parent / "data"
SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary"
TEMPORADAS = list(range(2021, int(datetime.now().year) + 1))


def _eventos(anio, intentos=3):
    return eventos_terminados_anio(SB, anio, intentos=intentos)


def _box(eid):
    return descargar_boxscore_evento(SUM, eid, stats=STATS)


def recolectar():
    return recolectar_boxscore_liga(DATA, SB, SUM, temporadas=TEMPORADAS)


if __name__ == "__main__":
    recolectar()
