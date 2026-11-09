"""
Recolector de estadísticas individuales de jugadores de la NBA (Box Scores de Atletas)
desde la API pública de ESPN (endpoint summary).
Extrae minutos, puntos, tiros de campo, triples, tiros libres, rebotes (ofensivos/defensivos),
asistencias, robos, bloqueos, pérdidas, faltas y +/- de cada jugador en cada partido.
Soporta recolección incremental multihilo.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time
import urllib.request
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SUM_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def _obtener_boxscore_jugadores(eid):
    """Descarga y normaliza las planillas individuales de ambos equipos para un partido."""
    url = f"{SUM_URL}?event={eid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            d = json.loads(resp.read().decode())
    except Exception:
        return []

    header = d.get("header", {})
    competitions = header.get("competitions", [{}])[0]
    competitors = {
        c.get("homeAway"): c.get("team", {}).get("displayName")
        for c in competitions.get("competitors", [])
    }
    date = header.get("competitions", [{}])[0].get("date")
    season = header.get("season", {}).get("year")
    home_name = competitors.get("home")
    away_name = competitors.get("away")

    filas = []
    for team_box in d.get("boxscore", {}).get("players", []):
        team_name = team_box.get("team", {}).get("displayName")
        is_home = (home_name == team_name)
        rival = away_name if is_home else home_name

        for stat_group in team_box.get("statistics", []):
            names = stat_group.get("names", [])
            for ath in stat_group.get("athletes", []):
                athlete = ath.get("athlete", {})
                stats = ath.get("stats", [])
                if not stats or ath.get("didNotPlay"):
                    continue

                row = {
                    "event_id": str(eid),
                    "fecha": date,
                    "temporada": season,
                    "equipo": team_name,
                    "rival": rival,
                    "es_local": 1 if is_home else 0,
                    "player_id": athlete.get("id"),
                    "player_name": athlete.get("displayName"),
                    "short_name": athlete.get("shortName"),
                    "jersey": athlete.get("jersey"),
                    "position": athlete.get("position", {}).get("abbreviation"),
                    "starter": 1 if ath.get("starter") else 0,
                }
                for k, v in zip(names, stats):
                    row[k] = v
                filas.append(row)
    return filas


def recolectar(max_workers=12, limite_partidos=None, min_temporada=2026):
    DATA.mkdir(parents=True, exist_ok=True)
    out_path = DATA / "jugadores_boxscore.csv"
    p_path = DATA / "partidos.csv"
    if not p_path.exists():
        print("No se encontró partidos.csv.")
        return None

    df_partidos = pd.read_csv(p_path)
    if min_temporada is not None and "temporada" in df_partidos.columns:
        df_partidos = df_partidos[df_partidos["temporada"] >= int(min_temporada)]
    ya = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    hechos = set(ya["event_id"].astype(str)) if len(ya) and "event_id" in ya.columns else set()

    # Ordenar por fecha cronológica inversa para priorizar las temporadas más recientes
    df_partidos = df_partidos.sort_values("fecha", ascending=False)
    todos_eids = [str(x) for x in df_partidos["event_id"].dropna().unique()]
    pendientes = [x for x in todos_eids if x not in hechos]

    if limite_partidos:
        pendientes = pendientes[:limite_partidos]

    print(f"NBA Jugadores Box Score: {len(todos_eids)} partidos totales, {len(hechos)} ya procesados, {len(pendientes)} por descargar.")
    if not pendientes:
        print("Todos los partidos ya tienen estadísticas de jugadores.")
        return ya

    filas_nuevas = []
    print(f"Descargando {len(pendientes)} partidos en paralelo ({max_workers} hilos)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_obtener_boxscore_jugadores, eid): eid for eid in pendientes}
        count = 0
        for f in as_completed(futures):
            res = f.result()
            if res:
                filas_nuevas.extend(res)
            count += 1
            if count % 100 == 0:
                elapsed = time.time() - t0
                rate = count / max(elapsed, 0.1)
                print(f"  ... {count}/{len(pendientes)} partidos procesados ({len(filas_nuevas)} registros de jugadores, {rate:.1f} partidos/s)")
                # Guardado intermedio
                df_temp = pd.concat([ya, pd.DataFrame(filas_nuevas)], ignore_index=True)
                df_temp = df_temp.drop_duplicates(subset=["event_id", "player_id"])
                df_temp.to_csv(out_path, index=False)

    df_final = pd.concat([ya, pd.DataFrame(filas_nuevas)], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["event_id", "player_id"]).reset_index(drop=True)
    df_final.to_csv(out_path, index=False)
    print(f"\n✅ Guardado completado: {out_path} ({len(df_final)} registros de jugadores en total)")
    return df_final


if __name__ == "__main__":
    recolectar()
