"""
Portal de predicción Brasileirão: interfaz estándar de las ligas (comun/ui_liga.py).
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
    "codigo": "bra",
    "nombre": "Brasileirão",
    "bandera": "🇧🇷",
    "acento": "#009c3b",
    "acento2": "#ffdf00",
    "cmap": "Greens",
    "equipos_default": ("Flamengo", "Palmeiras"),
    "n_sims": 4000,
    "columnas_proy": [
        ("P_campeon", "🏆 P(Campeón)", "Greens"),
        ("P_libertadores_directo", "⭐ P(Libertadores directa)", None),
        ("P_libertadores_total", "🌎 P(Libertadores total)", None),
        ("P_sudamericana", "🟡 P(Sudamericana)", None),
        ("P_descenso", "🔻 P(Descenso)", "OrRd"),
    ],
    "reglas": "Copa Libertadores: top 6 (G4 directo, G6 preliminar) · Copa Sudamericana: 7º al 12º · Descenso: últimos 4 clasificados.",
}


def run_app():
    run_liga_app(mo, rec, rec_box, CONFIG)


if __name__ == "__main__":
    run_app()
