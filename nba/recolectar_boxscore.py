"""
Recolector de BOX SCORE detallado de la NBA desde la API de ESPN (endpoint summary):
Pérdidas (TOV), rebotes ofensivos (OREB), rebotes defensivos (DREB), robos (STL),
bloqueos (BLK), puntos en pintura, puntos de contraataque y cuotas de casas de apuestas (Pickcenter).
Soporta recolección concurrente incremental (no re-pide partidos ya procesados).

Uso: python3 recolectar_boxscore.py -> nba/data/box_score.csv
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SUM_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

STATS_DESEADAS = [
    "turnovers", "totalTurnovers", "offensiveRebounds", "defensiveRebounds",
    "steals", "blocks", "fastBreakPoints", "pointsInPaint", "turnoverPoints",
    "fouls", "technicalFouls", "largestLead", "leadChanges"
]


def _obtener_box(eid):
    try:
        r = requests.get(f"{SUM_URL}?event={eid}", timeout=15)
        if r.status_code != 200:
            return None
        s = r.json()
        teams = s.get("boxscore", {}).get("teams", [])
        if len(teams) != 2:
            return None
            
        fila = {"event_id": str(eid)}
        for t in teams:
            pref = "local" if t.get("homeAway") == "home" else "visita"
            fila[f"{pref}_equipo"] = t["team"]["displayName"]
            stats_dict = {st["name"]: st.get("displayValue") for st in t.get("statistics", [])}
            
            for k in STATS_DESEADAS:
                val = stats_dict.get(k)
                try:
                    fila[f"{pref}_{k}"] = float(val) if val not in (None, "") else None
                except (ValueError, TypeError):
                    fila[f"{pref}_{k}"] = None

        # Pickcenter / Odds si existen
        pickcenter = s.get("pickcenter", [])
        if pickcenter:
            pc0 = pickcenter[0]
            fila["spread"] = pc0.get("spread")
            fila["over_under"] = pc0.get("overUnder")
            fila["provider"] = pc0.get("provider", {}).get("name")
        else:
            fila["spread"] = None
            fila["over_under"] = None
            fila["provider"] = None

        return fila
    except Exception:
        return None


def recolectar(max_workers=8, limite=None):
    DATA.mkdir(parents=True, exist_ok=True)
    box_path = DATA / "box_score.csv"
    partidos_path = DATA / "partidos.csv"
    
    if not partidos_path.exists():
        print("No se encontró partidos.csv. Ejecute recolectar.py primero.")
        return None

    df_partidos = pd.read_csv(partidos_path)
    ya = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame()
    hechos = set(ya["event_id"].astype(str)) if len(ya) and "event_id" in ya.columns else set()

    pendientes = [eid for eid in df_partidos["event_id"].astype(str) if eid not in hechos]
    if limite:
        pendientes = pendientes[-limite:]

    print(f"NBA Box Scores: {len(df_partidos)} partidos totales, {len(hechos)} ya recolectados, {len(pendientes)} pendientes.")

    if not pendientes:
        print("Todos los box scores ya se encuentran recolectados.")
        return ya

    filas_nuevas = []
    print(f"Descargando {len(pendientes)} box scores con {max_workers} hilos...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_obtener_box, eid): eid for eid in pendientes}
        count = 0
        for f in as_completed(futures):
            res = f.result()
            if res:
                filas_nuevas.append(res)
            count += 1
            if count % 200 == 0:
                print(f"  ... {count}/{len(pendientes)} procesados ({len(filas_nuevas)} exitosos)")
                # Guardado intermedio seguro
                df_temp = pd.concat([ya, pd.DataFrame(filas_nuevas)], ignore_index=True).drop_duplicates(subset=["event_id"])
                df_temp.to_csv(box_path, index=False)

    df_final = pd.concat([ya, pd.DataFrame(filas_nuevas)], ignore_index=True).drop_duplicates(subset=["event_id"])
    df_final.to_csv(box_path, index=False)
    print(f"\n✅ Guardado NBA: box_score.csv ({len(df_final)} partidos)")
    return df_final


if __name__ == "__main__":
    recolectar()
