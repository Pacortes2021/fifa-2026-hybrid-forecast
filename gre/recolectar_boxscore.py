"""
Recolecta el BOX SCORE detallado de cada partido de la Greek Super League desde ESPN:
Recupera las estadísticas completas de posesión, tiros, córners, faltas, etc.
"""
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import concurrent.futures

DATA = Path(__file__).resolve().parent / "data"
SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/gre.1/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/gre.1/summary"
TEMPORADAS = list(range(2021, int(datetime.now().year) + 1))

STATS = [
    "foulsCommitted", "yellowCards", "redCards", "offsides", "wonCorners", "saves",
    "possessionPct", "totalShots", "shotsOnTarget", "shotPct", "penaltyKickGoals",
    "penaltyKickShots", "accuratePasses", "totalPasses", "passPct", "accurateCrosses",
    "totalCrosses", "crossPct", "totalLongBalls", "accurateLongBalls", "longballPct",
    "blockedShots", "effectiveTackles", "totalTackles", "tacklePct", "interceptions",
    "effectiveClearance", "totalClearance"
]

def _eventos(anio, intentos=3):
    """Partidos terminados del año `anio`. Usa dates=AAAA: desde sep-2026 ESPN responde
    HTTP 400 a los rangos AAAAMMDD-AAAAMMDD. Si falla tras los reintentos, lanza la
    excepción y recolectar() no escribe nada."""
    for intento in range(1, intentos + 1):
        try:
            r = requests.get(f"{SB}?dates={anio}&limit=1000", timeout=40)
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

def _box(eid):
    s = requests.get(f"{SUM}?event={eid}", timeout=15).json()
    teams = s.get("boxscore", {}).get("teams", [])
    if len(teams) != 2:
        return None
    fila = {}
    for t in teams:
        pref = "local" if t.get("homeAway") == "home" else "visita"
        fila[f"{pref}_equipo"] = t["team"]["displayName"]
        sd = {st["name"]: st.get("displayValue") for st in t.get("statistics", [])}
        for k in STATS:
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

def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "box_score.csv"
    ya = pd.read_csv(path) if path.exists() else pd.DataFrame()
    hechos = set(ya["event_id"].astype(str)) if len(ya) else set()
    eventos = [ev for a in TEMPORADAS for ev in _eventos(a)]
    print(f"{len(eventos)} partidos jugados en total; {len(hechos)} ya recolectados. Pidiendo el resto...")

    a_procesar = [ev for ev in eventos if str(ev[0]) not in hechos]
    if not a_procesar:
        print("No se encontraron partidos nuevos para descargar.")
        return ya

    print(f"Descargando {len(a_procesar)} partidos nuevos en paralelo con 20 workers...")

    nuevos_resultados = []

    def process_event(ev):
        eid, fecha, anio = ev
        try:
            b = _box(eid)
            if b:
                b.update({"event_id": eid, "fecha": fecha, "temporada": anio})
                return b
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ev = {executor.submit(process_event, ev): ev for ev in a_procesar}

        count = 0
        for future in concurrent.futures.as_completed(future_to_ev):
            res = future.result()
            if res:
                nuevos_resultados.append(res)
            count += 1
            if count % 200 == 0:
                print(f"  Progreso: {count}/{len(a_procesar)} partidos procesados...")

    if nuevos_resultados:
        df_nuevos = pd.DataFrame(nuevos_resultados)
        if len(ya) > 0:
            df_final = pd.concat([ya, df_nuevos], ignore_index=True)
        else:
            df_final = df_nuevos
        df_final = df_final.drop_duplicates(subset=["event_id"], keep="last")
        df_final.sort_values("fecha", inplace=True)
        df_final.to_csv(path, index=False)
        print(f"Éxito: {len(df_final)} partidos totales guardados en {path}")
        return df_final
    else:
        print("No se pudieron obtener nuevos box scores.")
        return ya

if __name__ == "__main__":
    recolectar()
