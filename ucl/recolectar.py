"""
Recolector de la UEFA Champions League desde la API pública de ESPN (liga "uefa.champions").
Convierte la fecha UTC de ESPN a la hora central europea CET/CEST ("Europe/Paris").
Etiqueta etapas (Fase de Liga / Grupos vs Eliminatorias / Final) y campos neutrales.
"""
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard"
YEAR_RANGES = list(range(2021, int(datetime.now().year) + 2))

NORM_MAP = {
    "Barcelona": "FC Barcelona",
    "Internazionale": "Inter",
    "Paris Saint-Germain": "PSG",
    "Ajax Amsterdam": "Ajax",
    "PSV Eindhoven": "PSV",
    "Feyenoord Rotterdam": "Feyenoord",
    "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "Malmö FF": "Malmo FF",
    "RB Salzburg": "Salzburg",
    "Red Star Belgrade": "Crvena Zvezda",
    "SK Sturm Graz": "Sturm Graz",
    "FC Porto": "Porto",
    "FK Qarabag": "Qarabag",
    "Sparta Prague": "Sparta Praga",
    "Slavia Prague": "Slavia Praga",
    "Sheriff Tiraspol": "Sheriff",
    "Bodo/Glimt": "Bodo/Glimt",
    "Union St.-Gilloise": "Union Saint-Gilloise",
    "1. FC Union Berlin": "Union Berlin",
    "Atlético Madrid": "Atletico Madrid",
    "Sporting CP": "Sporting CP",
    "Sporting Lisbon": "Sporting CP"
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

            # Ignorar partidos pospuestos, cancelados o suspendidos
            # Los cancelados no se juegan; los aplazados/suspendidos sí, más adelante: quedan
            # en el fixture marcados como aplazados para que la simulación los cuente.
            if status_name == "STATUS_CANCELED":
                continue
            aplazado = status_name in ("STATUS_POSTPONED", "STATUS_DELAYED", "STATUS_SUSPENDED")

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

            # Convertir hora UTC a la zona horaria local europea (París / CET / CEST)
            fecha_local = pd.to_datetime(e["date"]).tz_convert("Europe/Paris").tz_localize(None)

            # En el calendario europeo, temporadas de julio a junio (ej. sep 2026 -> 2027)
            temp_partido = fecha_local.year + 1 if fecha_local.month >= 7 else fecha_local.year

            slug_fase = e.get("season", {}).get("slug", "league-phase")
            is_knockout = 1 if slug_fase in ("knockout-round-playoffs", "round-of-16", "quarterfinals", "semifinals", "final") else 0
            is_neutral = 1 if (comp.get("neutralSite") is True or slug_fase == "final") else 0

            filas.append({
                "event_id": str(e["id"]),
                "fecha": fecha_local,
                "temporada": temp_partido,
                "stage": slug_fase,
                "is_knockout": is_knockout,
                "is_neutral": is_neutral,
                "local_id": str(h["team"]["id"]),
                "visita_id": str(a["team"]["id"]),
                "local": loc,
                "visita": vis,
                "goles_local": gl,
                "goles_visita": gv,
                "estado": estado,
                "status_name": status_name, "aplazado": aplazado
            })
        except (KeyError, IndexError, StopIteration):
            continue
    return filas

def _canonizar_por_id(df):
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
    for anio in YEAR_RANGES:
        try:
            f = _temporada(anio)
            todo += f
            jug = sum(1 for x in f if x["estado"] == "post")
            print(f"  Año {anio}: {len(f)} partidos ({jug} jugados)")
        except Exception as ex:
            print(f"  Año {anio}: error ({ex})")
            errores.append(f"{anio}: {ex}")
        time.sleep(0.5)

    # Si falla algún año no se escriben los CSV (quedarían incompletos) y el error
    # deja el workflow en rojo, en vez de seguir en verde con datos congelados.
    if errores or not todo:
        detalle = "; ".join(errores) or "ESPN no devolvió partidos"
        raise RuntimeError(f"No se pudieron recolectar partidos de Champions League: {detalle}")

    df = _canonizar_por_id(pd.DataFrame(todo).sort_values("fecha"))
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    # Un aplazado que ESPN ya reprogramó (mismo local y visita, fecha posterior, otro event_id)
    # es el mismo partido: se descarta el evento original para no contarlo dos veces.
    if df["aplazado"].any():
        def _reprogramado(r):
            otros = df[(df.local == r.local) & (df.visita == r.visita) & (df.event_id != r.event_id) & (~df.aplazado)]
            dias = (otros.fecha - r.fecha).dt.days
            return bool(((dias > 0) & (dias <= 300)).any())
        reprogramados = df[df.aplazado].apply(_reprogramado, axis=1)
        df = df.drop(index=reprogramados[reprogramados].index)

    # Guardar partidos finalizados
    df_jugados = df[df["estado"] == "post"].dropna(subset=["goles_local", "goles_visita"]).copy().drop(columns=["aplazado"], errors="ignore")
    df_jugados["goles_local"] = df_jugados["goles_local"].astype(int)
    df_jugados["goles_visita"] = df_jugados["goles_visita"].astype(int)
    out_p = DATA / "partidos.csv"
    df_jugados.to_csv(out_p, index=False)
    print(f"Guardado {out_p} ({len(df_jugados)} partidos jugados de Champions League)")

    # Guardar fixture futuro
    df_fix = df[df["estado"] == "pre"].copy()
    out_f = DATA / "fixture.csv"
    df_fix.to_csv(out_f, index=False)
    print(f"Guardado {out_f} ({len(df_fix)} partidos programados en fixture)")

if __name__ == "__main__":
    recolectar()
