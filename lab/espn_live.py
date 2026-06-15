"""
Conector con la API pública de ESPN para traer los resultados REALES del Mundial 2026
día a día. Alimenta el modo "torneo en vivo" de app_lab.py.

Endpoint: site.api.espn.com (scoreboard del torneo 'fifa.world'). No requiere API key.
"""
from datetime import date
import requests
import pandas as pd

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
INICIO_MUNDIAL = "20260611"

# ESPN usa algunos nombres distintos a team_states.csv -> normalización
NORM = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "USA": "United States",
}


def _norm(nombre):
    return NORM.get(nombre, nombre)


def traer_resultados(desde=INICIO_MUNDIAL, hasta=None, solo_finalizados=True):
    """Devuelve un DataFrame de partidos del Mundial 2026 con columnas
       fecha, local, visita, goles_local, goles_visita, estado, ronda.
       Nombres ya normalizados a los de team_states.csv."""
    if hasta is None:
        hasta = date.today().strftime("%Y%m%d")
    url = f"{SCOREBOARD}?dates={desde}-{hasta}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    eventos = r.json().get("events", [])

    filas = []
    for e in eventos:
        comp = e["competitions"][0]
        estado = e["status"]["type"]["state"]   # pre | in | post
        if solo_finalizados and estado != "post":
            continue
        cs = comp["competitors"]
        h = next(x for x in cs if x["homeAway"] == "home")
        a = next(x for x in cs if x["homeAway"] == "away")
        try:
            gl, gv = int(h.get("score")), int(a.get("score"))
        except (TypeError, ValueError):
            continue
        filas.append({
            "fecha": pd.to_datetime(e["date"]).date(),
            "local": _norm(h["team"]["displayName"]),
            "visita": _norm(a["team"]["displayName"]),
            "goles_local": gl,
            "goles_visita": gv,
            "estado": estado,
            "ronda": e.get("season", {}).get("slug", "") or comp.get("notes", [{}])[0].get("headline", "") if comp.get("notes") else "",
        })
    df = pd.DataFrame(filas)
    if len(df):
        df = df.sort_values("fecha").reset_index(drop=True)
    return df


def partidos_en_vivo(desde=INICIO_MUNDIAL, hasta=None):
    """Partidos actualmente en juego (estado 'in'), para mostrarlos aparte."""
    if hasta is None:
        hasta = date.today().strftime("%Y%m%d")
    r = requests.get(f"{SCOREBOARD}?dates={desde}-{hasta}", timeout=20)
    r.raise_for_status()
    filas = []
    for e in r.json().get("events", []):
        if e["status"]["type"]["state"] != "in":
            continue
        comp = e["competitions"][0]; cs = comp["competitors"]
        h = next(x for x in cs if x["homeAway"] == "home")
        a = next(x for x in cs if x["homeAway"] == "away")
        filas.append({"local": _norm(h["team"]["displayName"]), "visita": _norm(a["team"]["displayName"]),
                      "marcador": f"{h.get('score','?')}-{a.get('score','?')}",
                      "minuto": e["status"].get("displayClock", "")})
    return pd.DataFrame(filas)
