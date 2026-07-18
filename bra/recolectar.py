"""
Recolector de la Serie A de Brasil (Brasileirão) desde la API pública de ESPN (liga 'bra.1').
Construye el histórico de partidos (para entrenar el modelo) y el fixture restante de 2026
(para simular el campeonato). Sin API key.

Uso:  python3 recolectar.py        -> bra/data/partidos.csv y bra/data/fixture.csv
"""
from pathlib import Path
import time
import requests
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/scoreboard"
TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]


def _temporada(anio):
    # Traemos por semestres con limit moderado para evitar error 500 en la API de ESPN
    eventos = []
    # El Brasileirão usualmente corre de abril a diciembre, pero para estar seguros cubrimos todo el año
    for ini, fin in ((f"{anio}0101", f"{anio}0630"), (f"{anio}0701", f"{anio}1231")):
        r = requests.get(f"{SCOREBOARD}?dates={ini}-{fin}&limit=400", timeout=40)
        r.raise_for_status()
        eventos += r.json().get("events", [])
    filas = []
    for e in eventos:
        try:
            comp = e["competitions"][0]; cs = comp["competitors"]
            h = next(x for x in cs if x["homeAway"] == "home")
            a = next(x for x in cs if x["homeAway"] == "away")
            estado = e["status"]["type"]["state"]      # pre | in | post
            try:
                gl = int(h["score"]) if h.get("score") not in (None, "") else None
                gv = int(a["score"]) if a.get("score") not in (None, "") else None
            except (TypeError, ValueError):
                gl = gv = None
            filas.append({
                "event_id": str(e["id"]),
                "fecha": pd.to_datetime(e["date"]).tz_localize(None),
                "temporada": anio, "local": h["team"]["displayName"], "visita": a["team"]["displayName"],
                "goles_local": gl, "goles_visita": gv, "estado": estado,
            })
        except (KeyError, IndexError, StopIteration):
            continue   # evento mal formado / sin equipos
    return filas


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
        print("No se pudieron recolectar partidos.")
        return
        
    df = pd.DataFrame(todo).drop_duplicates(subset=["fecha", "local", "visita"]).sort_values("fecha")

    # Separar jugados y fixture
    jugados = df[df.estado == "post"].dropna(subset=["goles_local", "goles_visita"]).reset_index(drop=True)
    fixture = df[df.estado == "pre"].reset_index(drop=True)
    
    jugados.to_csv(DATA / "partidos.csv", index=False)
    fixture.to_csv(DATA / "fixture.csv", index=False)
    print(f"\nGuardado: partidos.csv ({len(jugados)} jugados)")
    print(f"Guardado: fixture.csv ({len(fixture)} por jugar)")
    return jugados, fixture


if __name__ == "__main__":
    recolectar()
