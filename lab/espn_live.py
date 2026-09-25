"""
Conector con la API pública de ESPN para traer los resultados REALES del Mundial 2026
día a día. Alimenta el modo "torneo en vivo" de app_lab.py.

Endpoint: site.api.espn.com (scoreboard del torneo 'fifa.world'). No requiere API key.
"""
from datetime import date
import json
from pathlib import Path
import re
import requests
import pandas as pd

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
INICIO_MUNDIAL = "20260611"
INICIO_KO      = "20260628"   # inicio fase eliminatoria (fin grupos = 27-jun)
FIN_GRUPOS     = "20260627"   # último día de la fase de grupos

LOCAL_CSV = Path(__file__).resolve().parent.parent / "data" / "partidos_mundial_2026.csv"
LOCAL_JSON = Path(__file__).resolve().parent.parent / "data" / "espn_mundial_2026_events.json"

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
    "Congo DR": "DR Congo",     # ESPN invierte el orden; el modelo usa "DR Congo"
}


def _norm(nombre):
    return NORM.get(nombre, nombre)


def _fetch_events():
    """Obtiene los 104 eventos del Mundial 2026 desde ESPN (dates=2026&limit=1000).
    Si la API de ESPN falla o no hay conexión, recurre al respaldo local persistente."""
    try:
        url = f"{SCOREBOARD}?dates=2026&limit=1000"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        evs = r.json().get("events", [])
        if len(evs) >= 1:
            try:
                LOCAL_JSON.parent.mkdir(parents=True, exist_ok=True)
                with open(LOCAL_JSON, "w", encoding="utf-8") as f:
                    json.dump(r.json(), f)
            except Exception:
                pass
            return evs
    except Exception:
        pass
    if LOCAL_JSON.exists():
        try:
            with open(LOCAL_JSON, "r", encoding="utf-8") as f:
                return json.load(f).get("events", [])
        except Exception:
            pass
    return []


def traer_resultados(desde=INICIO_MUNDIAL, hasta=None, solo_finalizados=True):
    """Devuelve un DataFrame de partidos del Mundial 2026 con columnas
       fecha, local, visita, goles_local, goles_visita, estado, ronda.
       Nombres ya normalizados a los de team_states.csv."""
    eventos = _fetch_events()
    filas = []
    for e in eventos:
        comp = e["competitions"][0]
        estado = e["status"]["type"]["state"]   # pre | in | post
        if solo_finalizados and estado != "post":
            continue
        cs = comp["competitors"]
        h = next((x for x in cs if x["homeAway"] == "home"), None)
        a = next((x for x in cs if x["homeAway"] == "away"), None)
        if not h or not a:
            continue
        try:
            gl, gv = int(h.get("score")), int(a.get("score"))
        except (TypeError, ValueError):
            continue
        f_dt = pd.to_datetime(e["date"]).date()
        ronda = (e.get("season", {}).get("slug", "") or 
                 (comp.get("notes", [{}])[0].get("headline", "") if comp.get("notes") else ""))
        filas.append({
            "fecha": f_dt,
            "local": _norm(h["team"]["displayName"]),
            "visita": _norm(a["team"]["displayName"]),
            "goles_local": gl,
            "goles_visita": gv,
            "estado": estado,
            "ronda": ronda,
        })
    df = pd.DataFrame(filas)
    if len(df) == 0 and LOCAL_CSV.exists():
        try:
            df = pd.read_csv(LOCAL_CSV)
            df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        except Exception:
            pass
    if len(df):
        df = df.sort_values("fecha").reset_index(drop=True)
        if desde:
            try:
                d_desde = pd.to_datetime(desde).date()
                df = df[df["fecha"] >= d_desde]
            except Exception:
                pass
        if hasta:
            try:
                d_hasta = pd.to_datetime(hasta).date()
                df = df[df["fecha"] <= d_hasta]
            except Exception:
                pass
        try:
            LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(LOCAL_CSV, index=False)
        except Exception:
            pass
    return df


_KO_RE = re.compile(r"(Round of 32|Round of 16|Quarterfinal|Semifinal) (\d+) (Winner|Loser)")
_RONDA_KEY = {"Round of 32": "R32", "Round of 16": "R16", "Quarterfinal": "QF", "Semifinal": "SF"}
_ORDEN_RONDAS = ["R32", "R16", "QF", "SF", "3RD", "FINAL"]
_TAM_RONDA = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "3RD": 1, "FINAL": 1}


def _ko_evento(e):
    comp = e["competitions"][0]
    cs = comp["competitors"]
    h = next(x for x in cs if x["homeAway"] == "home")
    a = next(x for x in cs if x["homeAway"] == "away")

    def num(x, campo):
        try:
            return int(x.get(campo))
        except (TypeError, ValueError):
            return None
    # ganador REAL (incluye penales): ESPN marca winner=True en el competidor que avanza
    if h.get("winner"):
        ganador = _norm(h["team"]["displayName"])
    elif a.get("winner"):
        ganador = _norm(a["team"]["displayName"])
    else:
        ganador = None
    soh, soa = num(h, "shootoutScore"), num(a, "shootoutScore")
    pens = (soh, soa) if (soh is not None and soa is not None) else None
    return {"id": int(e["id"]), "dt": pd.to_datetime(e["date"]),
            "home": _norm(h["team"]["displayName"]), "away": _norm(a["team"]["displayName"]),
            "state": e["status"]["type"]["state"], "gh": num(h, "score"), "ga": num(a, "score"),
            "winner": ganador, "pens": pens}


def _ronda_placeholder(m):
    """Ronda a la que pertenece un partido a partir de su placeholder (la de ARRIBA de la
       referencia): 'Round of 32 N' -> R16, etc. 'Semifinal N Loser' -> 3er puesto."""
    for nombre in (m["home"], m["away"]):
        mt = _KO_RE.match(nombre)
        if mt:
            ref = _RONDA_KEY[mt.group(1)]
            sube = {"R32": "R16", "R16": "QF", "QF": "SF", "SF": None}[ref]
            if sube is None:
                return "3RD" if mt.group(3) == "Loser" else "FINAL"
            return sube
    return None


def bracket_eliminatorias(desde="20260628", hasta="20260720"):
    """Los 16 CRUCES REALES de dieciseisavos (R32) del Mundial 2026 según ESPN.

    Devuelve {'R32': {n: {home, away, state, gh, ga}}} con los emparejamientos verdaderos
    (incluida la asignación de terceros, que ESPN ya resolvió). Estos cruces son HECHOS y son lo
    único fiable que se necesita de ESPN: el ÁRBOL (qué llave alimenta a cuál) lo arma
    `motor.bracket_real`, colocando cada cruce en la plantilla FIFA por el equipo 1°/2° conocido.

    OJO: la numeración interna de ESPN para R32 NO es cronológica (es por posición de llave:
    N de ESPN = nº de partido FIFA − 72), así que aquí solo se exponen los emparejamientos, sin
    intentar reconstruir el árbol desde esa numeración."""
    evs = _fetch_events()
    ko_evs = evs[72:] if len(evs) >= 88 else evs
    E = sorted((_ko_evento(e) for e in ko_evs), key=lambda x: (x["dt"], x["id"]))

    por_ronda = {r: [] for r in _ORDEN_RONDAS}
    sin_ph = []
    for m in E:
        r = _ronda_placeholder(m)
        (por_ronda[r].append if r else sin_ph.append)(m)
    # partidos con ambos equipos reales (R32 o rondas ya jugadas) rellenan las rondas tempranas
    sin_ph.sort(key=lambda x: (x["dt"], x["id"]))
    i = 0
    for r in _ORDEN_RONDAS:
        for _ in range(max(0, _TAM_RONDA[r] - len(por_ronda[r]))):
            if i < len(sin_ph):
                por_ronda[r].append(sin_ph[i]); i += 1
    por_ronda["R32"].sort(key=lambda x: (x["dt"], x["id"]))

    R32 = {n: {"home": m["home"], "away": m["away"], "state": m["state"], "gh": m["gh"], "ga": m["ga"],
               "winner": m["winner"], "pens": m["pens"]}
           for n, m in enumerate(por_ronda["R32"], 1)}
    return {"R32": R32}


def ganadores_ko(desde="20260628", hasta="20260720"):
    """{frozenset({local, visita}): equipo que AVANZÓ} para cada partido de eliminatorias ya
       resuelto. Usa el flag `winner` de ESPN, así que **respeta los penales** (un 1-1 que se define
       en la tanda queda fijado al que realmente pasó, no al marcador). La ventana arranca el 28-jun,
       cuando empiezan las eliminatorias (la fase de grupos terminó el 27-jun)."""
    evs = _fetch_events()
    ko_evs = evs[72:] if len(evs) >= 88 else evs
    out = {}
    for e in ko_evs:
        m = _ko_evento(e)
        if m["state"] == "post" and m["winner"]:
            out[frozenset({m["home"], m["away"]})] = m["winner"]
    return out


def partidos_en_vivo(desde=INICIO_MUNDIAL, hasta=None):
    """Partidos actualmente en juego (estado 'in'), para mostrarlos aparte."""
    evs = _fetch_events()
    filas = []
    for e in evs:
        if e["status"]["type"]["state"] != "in":
            continue
        comp = e["competitions"][0]; cs = comp["competitors"]
        h = next(x for x in cs if x["homeAway"] == "home")
        a = next(x for x in cs if x["homeAway"] == "away")
        filas.append({"local": _norm(h["team"]["displayName"]), "visita": _norm(a["team"]["displayName"]),
                      "marcador": f"{h.get('score','?')}-{a.get('score','?')}",
                      "minuto": e["status"].get("displayClock", "")})
    return pd.DataFrame(filas)

