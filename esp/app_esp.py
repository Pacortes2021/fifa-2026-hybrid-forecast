"""
Portal de predicción LaLiga: interfaz estándar de las ligas (comun/ui_liga.py).
Lo propio de la liga va en CONFIG; el motor, los recolectores y los datos están en esta carpeta.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del repo (comun/)
import motor as mo  # noqa: E402
import recolectar as rec  # noqa: E402
import recolectar_boxscore as rec_box  # noqa: E402
from comun.ui_liga import run_liga_app  # noqa: E402

CONFIG = {
    "codigo": "esp",
    "nombre": "LaLiga",
    "bandera": "🇪🇸",
    "acento": "#c60b1e",
    "acento2": "#ffc400",
    "cmap": "Reds",
    "equipos_default": ('Real Madrid', 'FC Barcelona'),
    "columnas_proy": [('P_campeon', '🏆 P(Campeón)', 'Reds'), ('P_copas', '🇪🇺 P(Copas europeas)', None), ('P_descenso', '🔻 P(Descenso)', 'OrRd')],
}


def run_app():
    run_liga_app(mo, rec, rec_box, CONFIG)


if __name__ == "__main__":
    run_app()
