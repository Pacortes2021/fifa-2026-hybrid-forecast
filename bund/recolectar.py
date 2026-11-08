"""
Recolector de la Bundesliga (Alemania) desde la API pública de ESPN (liga 'ger.1').
Convierte la fecha UTC de ESPN a la hora local alemana ('Europe/Berlin') para coincidir exactamente con los días del calendario real.
"""
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/scoreboard"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/teams"
# Temporada julio–junio: desde julio también se pide el año siguiente, para que el
# fixture incluya la segunda rueda (ene–may) y la simulación cubra la temporada completa.
_HOY = datetime.now()
TEMPORADAS = list(range(2021, _HOY.year + (2 if _HOY.month >= 7 else 1)))

NORM_MAP = {
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Bayer Leverkusen"
}

def norm_team(name):
    return NORM_MAP.get(name, name)


def _eventos_anio(anio, intentos=3):
    """Eventos del año calendario `anio` desde el scoreboard de ESPN.

    Usa `dates=AAAA`: desde sep-2026 ESPN responde HTTP 400 a los rangos
    `dates=AAAAMMDD-AAAAMMDD`. Si tras los reintentos sigue fallando, lanza la
    excepción para que recolectar() no sobrescriba los CSV con datos incompletos.
    """
    for intento in range(1, intentos + 1):
        try:
            r = requests.get(f"{SCOREBOARD}?dates={anio}&limit=1000", timeout=40)
            r.raise_for_status()
            return r.json().get("events", [])
        except Exception as ex:
            if intento == intentos:
                raise RuntimeError(f"ESPN no respondió para {anio}: {ex}") from ex
            time.sleep(2 * intento)


def _temporada(anio):
    eventos = _eventos_anio(anio)

    filas = []
    for e in eventos:
        try:
            comp = e["competitions"][0]
            cs = comp["competitors"]
            h = next(x for x in cs if x["homeAway"] == "home")
            a = next(x for x in cs if x["homeAway"] == "away")

            st_type = e.get("status", {}).get("type", {})
            status_name = st_type.get("name", "")

            # Ignorar partidos que no se jugarán en la fecha prevista
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

            # Convertir hora UTC a la zona horaria local de Alemania (Berlín)
            fecha_local = pd.to_datetime(e["date"]).tz_convert("Europe/Berlin").tz_localize(None)

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
    errores = []
    for anio in TEMPORADAS:
        try:
            f = _temporada(anio)
            todo += f
            jug = sum(1 for x in f if x["estado"] == "post")
            print(f"  {anio}: {len(f)} partidos ({jug} jugados)")
        except Exception as ex:
            print(f"  {anio}: error ({ex})")
            errores.append(f"{anio}: {ex}")
        time.sleep(0.5)

    # Si falla algún año no se escriben los CSV (quedarían incompletos) y el error
    # deja el workflow en rojo, en vez de seguir en verde con datos congelados.
    if errores or not todo:
        detalle = "; ".join(errores) or "ESPN no devolvió partidos"
        raise RuntimeError(f"No se pudieron recolectar partidos de Alemania: {detalle}")

    df = _canonizar_por_id(pd.DataFrame(todo).sort_values("fecha"))
    df = df.drop_duplicates(subset=["event_id"], keep="last")

    jugados = df[df.estado == "post"].dropna(subset=["goles_local", "goles_visita"]).reset_index(drop=True)
    fixture = df[df.estado == "pre"].copy()
    # Etiqueta de temporada julio–junio (ene–may 2027 = temporada 2026), la que usan los motores
    fixture["temporada"] = fixture["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    fixture = fixture[fixture.temporada == fixture.temporada.max()].reset_index(drop=True)

    jugados = jugados.drop(columns=["status_name"], errors="ignore")
    fixture = fixture.drop(columns=["status_name"], errors="ignore")

    jugados.to_csv(DATA / "partidos.csv", index=False)
    fixture.to_csv(DATA / "fixture.csv", index=False)
    print(f"\nGuardado Alemania: partidos.csv ({len(jugados)} jugados)")
    print(f"Guardado Alemania: fixture.csv ({len(fixture)} por jugar)")
    return jugados, fixture


if __name__ == "__main__":
    recolectar()
