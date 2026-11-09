"""
Módulo compartido para la recolección incremental de Box Scores detallados desde ESPN.
Centraliza la descarga paralela (ThreadPoolExecutor), reintentos HTTP y persistencia
en `box_score.csv` para las 19 ligas domésticas.
"""
from datetime import datetime
from pathlib import Path
import concurrent.futures
import time
import pandas as pd
import requests

STATS = [
    "foulsCommitted", "yellowCards", "redCards", "offsides", "wonCorners", "saves",
    "possessionPct", "totalShots", "shotsOnTarget", "shotPct", "penaltyKickGoals",
    "penaltyKickShots", "accuratePasses", "totalPasses", "passPct", "accurateCrosses",
    "totalCrosses", "crossPct", "totalLongBalls", "accurateLongBalls", "longballPct",
    "blockedShots", "effectiveTackles", "totalTackles", "tacklePct", "interceptions",
    "effectiveClearance", "totalClearance",
]


def eventos_terminados_anio(sb_url, anio, intentos=3):
    """Partidos terminados del año `anio` desde el endpoint scoreboard de ESPN (`dates=AAAA`)."""
    for intento in range(1, intentos + 1):
        try:
            r = requests.get(f"{sb_url}?dates={anio}&limit=1000", timeout=40)
            r.raise_for_status()
            break
        except Exception as ex:
            if intento == intentos:
                raise RuntimeError(f"ESPN no respondió para {anio}: {ex}") from ex
            time.sleep(2 * intento)
    out = []
    for e in r.json().get("events", []):
        if e.get("status", {}).get("type", {}).get("state") == "post":
            out.append((e["id"], pd.to_datetime(e["date"]).tz_localize(None), anio))
    return out


def descargar_boxscore_evento(sum_url, eid, stats=STATS):
    """Descarga y normaliza las 28 estadísticas del resumen (`summary?event=`) de ESPN."""
    s = requests.get(f"{sum_url}?event={eid}", timeout=15).json()
    teams = s.get("boxscore", {}).get("teams", [])
    if len(teams) != 2:
        return None
    fila = {}
    for t in teams:
        pref = "local" if t.get("homeAway") == "home" else "visita"
        fila[f"{pref}_equipo"] = t["team"]["displayName"]
        sd = {st["name"]: st.get("displayValue") for st in t.get("statistics", [])}
        for k in stats:
            try:
                val = sd.get(k)
                if val is not None and val != "":
                    val_str = str(val).replace("%", "").strip()
                    fila[f"{pref}_{k}"] = float(val_str)
                else:
                    fila[f"{pref}_{k}"] = None
            except (TypeError, ValueError):
                fila[f"{pref}_{k}"] = None
    return fila


def recolectar_boxscore_liga(data_dir, sb_url, sum_url, temporadas=None, max_workers=15):
    """Descarga incrementalmente los box scores faltantes de una liga y actualiza `box_score.csv`."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if temporadas is None:
        temporadas = list(range(2021, int(datetime.now().year) + 1))

    path = data_dir / "box_score.csv"
    ya = pd.read_csv(path) if path.exists() else pd.DataFrame()
    hechos = set(ya["event_id"].astype(str)) if len(ya) else set()
    eventos = [ev for a in temporadas for ev in eventos_terminados_anio(sb_url, a)]
    print(f"{len(eventos)} partidos jugados en total; {len(hechos)} ya recolectados. Pidiendo el resto...")

    a_procesar = [ev for ev in eventos if str(ev[0]) not in hechos]
    if not a_procesar:
        print("No se encontraron partidos nuevos para descargar.")
        return ya

    print(f"Descargando {len(a_procesar)} partidos nuevos en paralelo con {max_workers} workers...")
    nuevos_resultados = []

    def process_event(ev):
        eid, fecha, anio = ev
        try:
            b = descargar_boxscore_evento(sum_url, eid)
            if b:
                b.update({"event_id": eid, "fecha": fecha, "temporada": anio})
                return b
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ev = {executor.submit(process_event, ev): ev for ev in a_procesar}
        count = 0
        for future in concurrent.futures.as_completed(future_to_ev):
            res = future.result()
            if res:
                nuevos_resultados.append(res)
            count += 1
            if count % 100 == 0:
                print(f"  Procesados {count}/{len(a_procesar)} partidos...")
                if nuevos_resultados:
                    df_new = pd.DataFrame(nuevos_resultados)
                    df_combined = pd.concat([ya, df_new], ignore_index=True).drop_duplicates(subset=["event_id"])
                    df_combined.to_csv(path, index=False)

    if nuevos_resultados:
        df_new = pd.DataFrame(nuevos_resultados)
        df_total = pd.concat([ya, df_new], ignore_index=True)
        df_total["fecha"] = pd.to_datetime(df_total["fecha"])
        df_total = df_total.drop_duplicates(subset=["event_id"]).sort_values("fecha").reset_index(drop=True)
        df_total.to_csv(path, index=False)
        ya = df_total
        print(f"Completado: {len(df_new)} nuevos partidos añadidos a box_score.csv. Total registros: {len(df_total)}.")
    else:
        print("No se agregaron partidos nuevos.")

    return pd.read_csv(path) if path.exists() else pd.DataFrame()
