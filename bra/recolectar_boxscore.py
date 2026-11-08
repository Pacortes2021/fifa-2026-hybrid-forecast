"""
Recolecta el BOX SCORE detallado de cada partido del Brasileirão desde ESPN (endpoint summary):
Recupera las 28 estadísticas disponibles en el boxscore (tiros, posesión, pases, intercepciones, etc.).
Usa un ThreadPoolExecutor para descargar en paralelo y acelerar la recolección.
Incremental: no re-pide partidos ya guardados.

Uso:  python3 recolectar_boxscore.py   -> bra/data/box_score.csv
"""
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import concurrent.futures

DATA = Path(__file__).resolve().parent / "data"
SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/summary"
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
        
    print(f"Descargando {len(a_procesar)} partidos nuevos en paralelo con 15 workers...")
    
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

    # Paralelismo controlado para evitar ser bloqueados por ESPN
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
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
        df_combined = pd.concat([ya, df_new], ignore_index=True).drop_duplicates(subset=["event_id"])
        df_combined.to_csv(path, index=False)
        ya = df_combined

    cob = ya[[c for c in ya.columns if c.endswith("totalShots")]].notna().mean().mean()
    print(f"\nGuardado definitivo: box_score.csv ({len(ya)} partidos en total, cobertura de stats ~{cob:.0%})")
    return ya


if __name__ == "__main__":
    recolectar()
