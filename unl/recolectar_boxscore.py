"""
Recolector de estadísticas detalladas (Box Scores) de partidos de UEFA Nations League desde ESPN.
Guarda data/box_score.csv con tiros, tiros al arco, corners, posesión, faltas, etc.
"""
from pathlib import Path
import requests
import pandas as pd
import concurrent.futures

DATA = Path(__file__).resolve().parent / "data"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.nations/summary"

STATS = [
    "foulsCommitted", "yellowCards", "redCards", "offsides", "wonCorners", "saves",
    "possessionPct", "totalShots", "shotsOnTarget", "shotPct", "penaltyKickGoals",
    "penaltyKickShots", "blockedShots"
]


def _box(eid):
    try:
        r = requests.get(f"{SUM}?event={eid}", timeout=10)
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
    except Exception:
        return None


def recolectar(max_partidos=200):
    DATA.mkdir(parents=True, exist_ok=True)
    out_path = DATA / "box_score.csv"
    
    partidos_path = DATA / "partidos.csv"
    if not partidos_path.exists():
        print("No se encontró data/partidos.csv")
        return pd.DataFrame()

    df_part = pd.read_csv(partidos_path)
    ya = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    hechos = set(ya["event_id"].astype(str)) if len(ya) and "event_id" in ya.columns else set()

    # Priorizar partidos más recientes (2026, 2024, etc.)
    pendientes = [
        str(row["event_id"])
        for _, row in df_part.sort_values("fecha", ascending=False).iterrows()
        if str(row["event_id"]) not in hechos
    ][:max_partidos]

    if not pendientes:
        print("Todos los box scores solicitados ya están descargados.")
        return ya

    print(f"Descargando {len(pendientes)} box scores en paralelo...")
    filas = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        resultados = list(ex.map(_box, pendientes))

    nuevos = [r for r in resultados if r is not None]
    df_nuevos = pd.DataFrame(nuevos)

    df_final = pd.concat([ya, df_nuevos], ignore_index=True) if len(ya) else df_nuevos
    if len(df_final):
        df_final = df_final.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
        df_final.to_csv(out_path, index=False)
        print(f"Guardado {out_path} ({len(df_final)} registros totales)")
    return df_final


if __name__ == "__main__":
    recolectar(max_partidos=150)
