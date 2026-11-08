"""
Portal de predicción Liga MX: interfaz estándar de las ligas (comun/ui_liga.py).
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
    "codigo": "mex",
    "nombre": "Liga MX",
    "bandera": "🇲🇽",
    "acento": "#006847",
    "acento2": "#ce1126",
    "cmap": "Greens",
    "equipos_default": ("América", "Cruz Azul"),
    "n_sims": 5000,
    "columnas_proy": [
        ("P_campeon", "🏆 P(Campeón)", "Greens"),
        ("P_directo_QF", "🎯 P(Cuartos directo)", None),
        ("P_Liguilla_total", "🎟️ P(Liguilla / Play-in)", None),
    ],
    "reglas": "Liguilla: top 6 directo a cuartos de final, 7º al 10º juegan Play-in · En la Liga MX no hay descenso deportivo directo.",
}


def run_app():
    run_liga_app(mo, rec, rec_box, CONFIG)


if __name__ == "__main__":
    run_app()
