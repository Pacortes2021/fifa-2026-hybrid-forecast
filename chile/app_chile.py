"""
Portal de predicción Liga Chilena: interfaz estándar de las ligas (comun/ui_liga.py).
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
    "codigo": "chile",
    "nombre": "Liga Chilena",
    "bandera": "🇨🇱",
    "acento": "#0039a6",
    "acento2": "#d52b1e",
    "cmap": "Blues",
    "equipos_default": ("Colo Colo", "Universidad de Chile"),
    "n_sims": 4000,
    "columnas_proy": [
        ("P_campeon", "🏆 P(Campeón)", "Blues"),
        ("P_libertadores_directo", "⭐ P(Libertadores directa)", None),
        ("P_libertadores_total", "🌎 P(Libertadores total)", None),
        ("P_sudamericana", "🟡 P(Sudamericana)", None),
        ("P_descenso", "🔻 P(Descenso)", "OrRd"),
    ],
    "reglas": "Copa Libertadores: top 3 clasificados · Copa Sudamericana: 4º al 7º · Descenso: últimos 2 clasificados.",
}


def run_app():
    run_liga_app(mo, rec, rec_box, CONFIG)


if __name__ == "__main__":
    run_app()
