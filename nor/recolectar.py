"""
Recolector de la Norwegian Eliteserien (Noruega) desde la API pública de ESPN (liga "nor.1").
Convierte la fecha UTC de ESPN a la hora local noruega ("Europe/Oslo").
Temporadas anuales (primavera - otoño).
"""
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/nor.1/scoreboard"
TEMPORADAS = list(range(2021, int(datetime.now().year) + 1))

NORM_MAP = {
    "Bodø/Glimt": "Bodo/Glimt",
    "FK Bodø/Glimt": "Bodo/Glimt",
    "FK Bodo/Glimt": "Bodo/Glimt",
    "Bodo Glimt": "Bodo/Glimt",
    "Brann": "SK Brann",
    "Brann Bergen": "SK Brann",
    "Rosenborg BK": "Rosenborg",
    "Molde FK": "Molde",
    "Viking": "Viking FK",
    "Tromsø": "Tromso",
    "Tromsø IL": "Tromso",
    "Lillestrøm": "Lillestrom",
    "Lillestrøm SK": "Lillestrom",
    "Valerenga": "Vålerenga",
    "Vålerenga Fotball": "Vålerenga",
    "Strømsgodset": "Stromsgodset",
    "Strømsgodset IF": "Stromsgodset",
    "Sarpsborg": "Sarpsborg FK",
    "Sarpsborg 08": "Sarpsborg FK",
    "Sandefjord Fotball": "Sandefjord",
    "HamKam": "Hamarkameratene",
    "Ham-Kam": "Hamarkameratene",
    "Kristiansund": "Kristiansund BK",
    "KFUM": "KFUM Oslo",
    "KFUM-Kameratene": "KFUM Oslo",
    "Aalesunds FK": "Aalesund",
    "Stabæk": "Stabaek",
    "Stabæk Fotball": "Stabaek",
    "Odd": "Odds BK",
    "Mjøndalen": "Mjondalen IF",
    "Jerv": "FK Jerv",
    "Start": "IK Start"
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

            # Ignorar partidos pospuestos, cancelados o suspendidos
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

            # Convertir hora UTC a la zona horaria local de Noruega (Oslo / CET / CEST)
            fecha_local = pd.to_datetime(e["date"]).tz_convert("Europe/Oslo").tz_localize(None)

            # En Noruega la liga se juega en año calendario (primavera - otoño)
            temp_partido = fecha_local.year

            filas.append({
                "event_id": str(e["id"]),
                "fecha": fecha_local,
                "temporada": temp_partido,
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
        print("No se pudieron recolectar partidos de Noruega.")
        return

    df = _canonizar_por_id(pd.DataFrame(todo).sort_values("fecha"))
    df = df.drop_duplicates(subset=["event_id"], keep="last")

    # Guardar partidos finalizados
    df_jugados = df[df["estado"] == "post"].dropna(subset=["goles_local", "goles_visita"]).copy()
    df_jugados["goles_local"] = df_jugados["goles_local"].astype(int)
    df_jugados["goles_visita"] = df_jugados["goles_visita"].astype(int)
    out_p = DATA / "partidos.csv"
    df_jugados.to_csv(out_p, index=False)
    print(f"Guardado {out_p} ({len(df_jugados)} partidos jugados de Noruega)")

    # Guardar fixture futuro
    df_fix = df[df["estado"] == "pre"].copy()
    out_f = DATA / "fixture.csv"
    df_fix.to_csv(out_f, index=False)
    print(f"Guardado {out_f} ({len(df_fix)} partidos programados)")

if __name__ == "__main__":
    recolectar()
