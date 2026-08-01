"""
Recolector de la Liga Profesional de Fútbol de Argentina desde la API pública de ESPN (liga 'arg.1').
Mapeo exhaustivo de normalización de nombres y conversión a hora local argentina ('America/Argentina/Buenos_Aires').
"""
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/arg.1/scoreboard"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/arg.1/teams"
TEMPORADAS = list(range(2021, int(datetime.now().year) + 1))

NORM_MAP = {
    "Boca Juniors": "Boca Juniors",
    "River Plate": "River Plate",
    "Racing Club": "Racing Club",
    "Independiente": "Independiente",
    "San Lorenzo": "San Lorenzo",
    "Vélez Sarsfield": "Vélez Sarsfield",
    "Velez Sarsfield": "Vélez Sarsfield",
    "Estudiantes": "Estudiantes de La Plata",
    "Estudiantes de La Plata": "Estudiantes de La Plata",
    "Estudiantes (La Plata)": "Estudiantes de La Plata",
    "Estudiantes L.P.": "Estudiantes de La Plata",
    "Gimnasia": "Gimnasia La Plata",
    "Gimnasia La Plata": "Gimnasia La Plata",
    "Gimnasia (La Plata)": "Gimnasia La Plata",
    "Gimnasia L.P.": "Gimnasia La Plata",
    "Rosario Central": "Rosario Central",
    "Newell's Old Boys": "Newell's Old Boys",
    "Newells Old Boys": "Newell's Old Boys",
    "Talleres": "Talleres de Córdoba",
    "Talleres de Córdoba": "Talleres de Córdoba",
    "Talleres (Córdoba)": "Talleres de Córdoba",
    "Talleres (Cordoba)": "Talleres de Córdoba",
    "Belgrano": "Belgrano",
    "Belgrano (Córdoba)": "Belgrano",
    "Belgrano (Cordoba)": "Belgrano",
    "Instituto": "Instituto",
    "Instituto (Córdoba)": "Instituto",
    "Instituto (Cordoba)": "Instituto",
    "Godoy Cruz": "Godoy Cruz",
    "Godoy Cruz Antonio Tomba": "Godoy Cruz",
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
    "Central Córdoba (Santiago del Estero)": "Central Córdoba",
    "Central Cordoba (SdE)": "Central Córdoba",
    "Atlético Tucumán": "Atlético Tucumán",
    "Atletico Tucuman": "Atlético Tucumán",
    "Atl. Tucumán": "Atlético Tucumán",
    "Unión": "Unión de Santa Fe",
    "Union": "Unión de Santa Fe",
    "Unión de Santa Fe": "Unión de Santa Fe",
    "Unión (Santa Fe)": "Unión de Santa Fe",
    "Union (Santa Fe)": "Unión de Santa Fe",
    "Sarmiento": "Sarmiento",
    "Sarmiento (Junín)": "Sarmiento",
    "Sarmiento (Junin)": "Sarmiento",
    "Barracas Central": "Barracas Central",
    "Deportivo Riestra": "Deportivo Riestra",
    "Riestra": "Deportivo Riestra",
    "Independiente Rivadavia": "Independiente Rivadavia",
    "Colón": "Colón de Santa Fe",
    "Colon": "Colón de Santa Fe",
    "Colón de Santa Fe": "Colón de Santa Fe",
    "Colón (Santa Fe)": "Colón de Santa Fe",
    "Arsenal": "Arsenal Sarandí",
    "Arsenal Sarandí": "Arsenal Sarandí",
    "Arsenal Sarandi": "Arsenal Sarandí",
    "Aldosivi": "Aldosivi",
    "Patronato": "Patronato",
    "Gimnasia (Mendoza)": "Gimnasia Mendoza",
    "Gimnasia Mendoza": "Gimnasia Mendoza",
    "San Martín (San Juan)": "San Martín San Juan",
    "San Martin (San Juan)": "San Martín San Juan",
    "Estudiantes de Río Cuarto": "Estudiantes de Río Cuarto"
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

            # BUG 3 FIX: ignorar partidos que no se jugarán en la fecha prevista.
            # Sin este filtro, STATUS_POSTPONED y STATUS_CANCELED caen en el else→"pre"
            # y se guardan en fixture.csv como partidos futuros reales,
            # contaminando las 4000 iteraciones del simulador Monte Carlo.
            if status_name in ("STATUS_POSTPONED", "STATUS_CANCELED",
                               "STATUS_DELAYED",   "STATUS_SUSPENDED"):
                continue

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

            fecha_local = pd.to_datetime(e["date"]).tz_convert("America/Argentina/Buenos_Aires").tz_localize(None)

            filas.append({
                "event_id": str(e["id"]),
                "fecha": fecha_local,
                "temporada": anio,
                "local_id": str(h["team"]["id"]),
                "visita_id": str(a["team"]["id"]),
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


def _canonizar_por_id(df):
    """Reasigna los nombres usando el ID estable de ESPN: si un club aparece con
    nombres distintos entre temporadas (reformas de nombres, promociones, etc.),
    todo su historial queda bajo el nombre más reciente del ID."""
    if "local_id" not in df.columns:
        return df
    df = df.copy()
    id2name = {}
    for r in df.sort_values("fecha").itertuples(index=False):
        for idv, nm in ((r.local_id, r.local), (r.visita_id, r.visita)):
            if pd.notna(idv) and pd.notna(nm):
                id2name[str(idv)] = nm
    orig_l, orig_v = df["local"], df["visita"]
    df["local"] = df["local_id"].map(lambda x: id2name.get(str(x), np.nan) if pd.notna(x) else np.nan)
    df["visita"] = df["visita_id"].map(lambda x: id2name.get(str(x), np.nan) if pd.notna(x) else np.nan)
    df["local"] = df["local"].where(df["local"].notna(), orig_l)
    df["visita"] = df["visita"].where(df["visita"].notna(), orig_v)
    return df


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

    df = _canonizar_por_id(pd.DataFrame(todo).sort_values("fecha"))
    df = df.drop_duplicates(subset=["event_id"], keep="last")

    jugados = df[df.estado == "post"].dropna(subset=["goles_local", "goles_visita"]).reset_index(drop=True)
    fixture = df[df.estado == "pre"]
    fixture = fixture[fixture.temporada == fixture.temporada.max()].reset_index(drop=True)

    jugados = jugados.drop(columns=["status_name"], errors="ignore")
    fixture = fixture.drop(columns=["status_name"], errors="ignore")

    jugados.to_csv(DATA / "partidos.csv", index=False)
    fixture.to_csv(DATA / "fixture.csv", index=False)
    print(f"\nGuardado Argentina: partidos.csv ({len(jugados)} jugados)")
    print(f"Guardado Argentina: fixture.csv ({len(fixture)} por jugar)")
    return jugados, fixture


if __name__ == "__main__":
    recolectar()
