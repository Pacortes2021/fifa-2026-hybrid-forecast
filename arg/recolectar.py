"""
Recolector de la Liga Profesional de Fútbol de Argentina desde la API pública de ESPN (liga 'arg.1').
Convierte la fecha UTC de ESPN a la hora local argentina ('America/Argentina/Buenos_Aires').
"""
from pathlib import Path
import time
import requests
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/arg.1/scoreboard"
TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]

NORM_MAP = {
    "Boca Juniors": "Boca Juniors",
    "River Plate": "River Plate",
    "Racing Club": "Racing Club",
    "Independiente": "Independiente",
    "San Lorenzo": "San Lorenzo",
    "Vélez Sarsfield": "Vélez Sarsfield",
    "Velez Sarsfield": "Vélez Sarsfield",
    "Estudiantes": "Estudiantes de La Plata",
    "Gimnasia": "Gimnasia La Plata",
    "Gimnasia La Plata": "Gimnasia La Plata",
    "Rosario Central": "Rosario Central",
    "Newell's Old Boys": "Newell's Old Boys",
    "Newells Old Boys": "Newell's Old Boys",
    "Talleres": "Talleres de Córdoba",
    "Talleres de Córdoba": "Talleres de Córdoba",
    "Belgrano": "Belgrano",
    "Instituto": "Instituto",
    "Godoy Cruz": "Godoy Cruz",
    "Defensa y Justicia": "Defensa y Justicia",
    "Argentinos Juniors": "Argentinos Juniors",
    "Huracán": "Huracán",
    "Huracan": "Huracán",
    "Lanús": "Lanús",
    "Lanus": "Lanús",
    "Banfield": "Banfield",
    "Platense": "Platense",
    "Tigre": "Tigre",
    "Central Córdoba": "Central Córdoba",
    "Central Cordoba": "Central Córdoba",
    "Atlético Tucumán": "Atlético Tucumán",
    "Atletico Tucuman": "Atlético Tucumán",
    "Unión": "Unión de Santa Fe",
    "Union": "Unión de Santa Fe",
    "Unión de Santa Fe": "Unión de Santa Fe",
    "Sarmiento": "Sarmiento",
    "Barracas Central": "Barracas Central",
    "Deportivo Riestra": "Deportivo Riestra",
    "Independiente Rivadavia": "Independiente Rivadavia"
}

def norm_team(name):
    return NORM_MAP.get(name, name)


def _temporada(anio):
    eventos = []
    for ini, fin in ((f"{anio}0101", f"{anio}0630"), (f"{anio}0701", f"{anio}1231")):
        try:
            r = requests.get(f"{SCOREBOARD}?dates={ini}-{fin}&limit=400", timeout=40)
            r.raise_for_status()
            eventos += r.json().get("events", [])
        except Exception as ex:
            print(f"  Error obteniendo rango {ini}-{fin}: {ex}")

    filas = []
    for e in eventos:
        try:
            comp = e["competitions"][0]
            cs = comp["competitors"]
            h = next(x for x in cs if x["homeAway"] == "home")
            a = next(x for x in cs if x["homeAway"] == "away")

            st_type = e.get("status", {}).get("type", {})
            status_name = st_type.get("name", "")

            is_finished = status_name in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_PEN", "STATUS_FINAL_AET")
            estado = "post" if is_finished else "pre"

            gl = None
            gv = None
            if is_finished:
                try:
                    gl = int(h["score"]) if h.get("score") not in (None, "") else None
                    gv = int(a["score"]) if a.get("score") not in (None, "") else None
                except (TypeError, ValueError):
                    gl = gv = None

            loc = norm_team(h["team"]["displayName"])
            vis = norm_team(a["team"]["displayName"])

            # Convertir fecha a hora local argentina (America/Argentina/Buenos_Aires)
            fecha_local = pd.to_datetime(e["date"]).tz_convert("America/Argentina/Buenos_Aires").tz_localize(None)

            filas.append({
                "event_id": str(e["id"]),
                "fecha": fecha_local,
                "temporada": anio,
                "local": loc,
                "visita": vis,
                "goles_local": gl,
                "goles_visita": gv,
                "estado": estado,
                "status_name": status_name
            })
        except (KeyError, IndexError, StopIteration):
            continue
    return filas


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    todo = []
    for anio in TEMPORADAS:
        try:
            f = _temporada(anio)
            todo += f
            jug = sum(1 for x in f if x["estado"] == "post")
            print(f"  {anio}: {len(f)} partidos ({jug} jugados)")
        except Exception as ex:
            print(f"  {anio}: error ({ex})")
        time.sleep(0.5)

    if not todo:
        print("No se pudieron recolectar partidos de Argentina.")
        return

    df = pd.DataFrame(todo).sort_values("fecha")
    df = df.drop_duplicates(subset=["fecha", "local", "visita"], keep="last")

    jugados = df[df.estado == "post"].dropna(subset=["goles_local", "goles_visita"]).reset_index(drop=True)
    fixture = df[df.estado == "pre"].reset_index(drop=True)

    jugados = jugados.drop(columns=["status_name"], errors="ignore")
    fixture = fixture.drop(columns=["status_name"], errors="ignore")

    jugados.to_csv(DATA / "partidos.csv", index=False)
    fixture.to_csv(DATA / "fixture.csv", index=False)
    print(f"\nGuardado Argentina: partidos.csv ({len(jugados)} jugados)")
    print(f"Guardado Argentina: fixture.csv ({len(fixture)} por jugar)")
    return jugados, fixture


if __name__ == "__main__":
    recolectar()
