"""
Portal de predicción Liga Profesional Argentina: interfaz estándar de las ligas (comun/ui_liga.py).
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
    "codigo": "arg",
    "nombre": "Liga Profesional Argentina",
    "bandera": "🇦🇷",
    "acento": "#74acdf",
    "acento2": "#1d4ed8",
    "cmap": "Blues",
    "equipos_default": ("Boca Juniors", "River Plate"),
    "n_sims": 4000,
    "columnas_proy": [
        ("P_campeon", "🏆 P(Campeón)", "Blues"),
        ("P_copas", "🌎 P(Copas internacionales)", None),
        ("P_descenso", "🔻 P(Descenso)", "OrRd"),
    ],
    "aviso_proyeccion": "Simulación sobre la tabla general del torneo vigente. Las zonas, los playoffs y el descenso por promedios no se simulan.",
    "reglas": "Copas internacionales: top 4 de la tabla general · Descenso: 2 equipos (tabla general y tabla de promedios).",
}


def run_app():
    run_liga_app(mo, rec, rec_box, CONFIG)


if __name__ == "__main__":
    run_app()
