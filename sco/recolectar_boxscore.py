"""
Recolecta el BOX SCORE detallado de cada partido de la Scottish Premiership desde ESPN:
Recupera las estadísticas completas de posesión, tiros, córners, faltas, etc.
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
SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/sco.1/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/sco.1/summary"
TEMPORADAS = list(range(2021, int(datetime.now().year) + 1))


def _eventos(anio, intentos=3):
    return eventos_terminados_anio(SB, anio, intentos=intentos)


def _box(eid):
    return descargar_boxscore_evento(SUM, eid, stats=STATS)


def recolectar():
    return recolectar_boxscore_liga(DATA, SB, SUM, temporadas=TEMPORADAS)


if __name__ == "__main__":
    recolectar()
