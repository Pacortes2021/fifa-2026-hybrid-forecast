"""
Recolecta el BOX SCORE detallado de cada partido de la liga chilena desde ESPN (endpoint summary):
tiros, tiros al arco, córners, posesión, faltas, tarjetas amarillas/rojas, offsides, atajadas,
tiros bloqueados. Cobertura ~96%. Incremental: no re-pide partidos ya guardados.

Uso:  python3 recolectar_boxscore.py   -> chile/data/box_score.csv  (una fila por equipo-partido)
"""
from pathlib import Path
import time
import requests
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/chi.1/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/chi.1/summary"
TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]
STATS = ["totalShots", "shotsOnTarget", "wonCorners", "possessionPct", "foulsCommitted",
         "yellowCards", "redCards", "offsides", "saves", "blockedShots"]


def _eventos(anio):
    out = []
    for ini, fin in ((f"{anio}0101", f"{anio}0630"), (f"{anio}0701", f"{anio}1231")):
        try:
            r = requests.get(f"{SB}?dates={ini}-{fin}&limit=400", timeout=40); r.raise_for_status()
            for e in r.json().get("events", []):
                if e.get("status", {}).get("type", {}).get("state") == "post":
                    out.append((e["id"], pd.to_datetime(e["date"]).tz_localize(None), anio))
        except Exception as ex:
            print(f"  scoreboard {anio} {ini}: {ex}")
    return out


def _box(eid):
    s = requests.get(f"{SUM}?event={eid}", timeout=25).json()
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
                fila[f"{pref}_{k}"] = float(sd.get(k)) if sd.get(k) not in (None, "") else None
            except (TypeError, ValueError):
                fila[f"{pref}_{k}"] = None
    return fila


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "box_score.csv"
    ya = pd.read_csv(path) if path.exists() else pd.DataFrame()
    hechos = set(ya["event_id"].astype(str)) if len(ya) else set()
    eventos = [ev for a in TEMPORADAS for ev in _eventos(a)]
    print(f"{len(eventos)} partidos jugados; {len(hechos)} ya recolectados. Pidiendo el resto...")
    filas = ya.to_dict("records") if len(ya) else []
    nuevos = 0
    for i, (eid, fecha, anio) in enumerate(eventos):
        if str(eid) in hechos:
            continue
        try:
            b = _box(eid)
            if b:
                b.update({"event_id": eid, "fecha": fecha, "temporada": anio})
                filas.append(b); nuevos += 1
        except Exception:
            pass
        time.sleep(0.12)
        if nuevos and nuevos % 100 == 0:
            pd.DataFrame(filas).to_csv(path, index=False)
            print(f"  ...{nuevos} nuevos guardados")
    df = pd.DataFrame(filas)
    df.to_csv(path, index=False)
    cob = df[[c for c in df.columns if c.endswith("totalShots")]].notna().mean().mean()
    print(f"\nGuardado: box_score.csv ({len(df)} partidos, cobertura de stats ~{cob:.0%})")
    return df


if __name__ == "__main__":
    recolectar()
