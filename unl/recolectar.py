"""
Recolector oficial de partidos y fixture de la UEFA Nations League desde ESPN API.
Genera data/partidos.csv (historial 2018-2026) y data/fixture.csv (partidos pendientes 2026-27).
"""
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
from recolectar_equipos import DATA, norm_team

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.nations/scoreboard"
YEAR_RANGES = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    
    # Cargar equipos para mapeo de grupo y liga
    equipos_path = DATA / "equipos.csv"
    group_map = {}
    league_map = {}
    if equipos_path.exists():
        df_eq = pd.read_csv(equipos_path)
        for _, r in df_eq.iterrows():
            group_map[r["norm_name"]] = r.get("group", "")
            league_map[r["norm_name"]] = r.get("league", "")

    seen_ids = set()
    eventos = []
    print("Descargando eventos de UEFA Nations League desde ESPN...")
    for anio in YEAR_RANGES:
        url = f"{SCOREBOARD}?dates={anio}&limit=1000"
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                evs = r.json().get("events", [])
                for e in evs:
                    if e["id"] not in seen_ids:
                        seen_ids.add(e["id"])
                        eventos.append((anio, e))
        except Exception as ex:
            print(f"  Error en año {anio}: {ex}")

    partidos_filas = []
    fixture_filas = []

    for anio, e in eventos:
        try:
            comp = e["competitions"][0]
            cs = comp["competitors"]
            h = next((x for x in cs if x["homeAway"] == "home"), None)
            a = next((x for x in cs if x["homeAway"] == "away"), None)
            if not h or not a:
                continue

            loc_raw = h["team"]["displayName"]
            vis_raw = a["team"]["displayName"]
            loc = norm_team(loc_raw)
            vis = norm_team(vis_raw)

            estado = e["status"]["type"]["state"]
            status_name = e["status"]["type"]["name"]

            # Ignorar cancelados / pospuestos
            if status_name in ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED"):
                continue

            dt_utc = pd.to_datetime(e["date"])
            # Convertir a CET / CEST (horario europeo central)
            try:
                dt_cet = dt_utc.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_cet = dt_utc.strftime("%Y-%m-%d %H:%M:%S")

            note = comp.get("notes", [{}])[0].get("headline", "") if comp.get("notes") else ""
            slug = e.get("season", {}).get("slug", "") or note
            is_neutral = 1 if comp.get("neutralSite", False) else 0
            is_knockout = 1 if any(k in slug.lower() for k in ["semifinal", "final", "playoff", "3rd"]) else 0

            # Temporada de la Nations League
            temp = 2018 if anio <= 2019 else (2020 if anio <= 2021 else (2022 if anio <= 2023 else (2024 if anio <= 2025 else 2026)))

            if estado == "post":
                try:
                    gl = int(h.get("score"))
                    gv = int(a.get("score"))
                except (TypeError, ValueError):
                    continue
                partidos_filas.append({
                    "event_id": e["id"],
                    "fecha": dt_cet,
                    "temporada": temp,
                    "stage": slug,
                    "is_knockout": is_knockout,
                    "is_neutral": is_neutral,
                    "local_id": h["team"].get("id", ""),
                    "visita_id": a["team"].get("id", ""),
                    "local": loc,
                    "visita": vis,
                    "goles_local": gl,
                    "goles_visita": gv,
                    "estado": estado,
                    "status_name": status_name
                })
            else:
                # Partido pendiente (fixture)
                grp = group_map.get(loc) or group_map.get(vis) or ""
                lg = league_map.get(loc) or league_map.get(vis) or ""
                fixture_filas.append({
                    "event_id": e["id"],
                    "fecha": dt_cet,
                    "temporada": temp,
                    "round": slug or grp,
                    "local": loc,
                    "visita": vis,
                    "local_id": h["team"].get("id", ""),
                    "visita_id": a["team"].get("id", ""),
                    "group": grp,
                    "league": lg,
                    "estado": estado
                })
        except Exception:
            continue

    df_partidos = pd.DataFrame(partidos_filas)
    if len(df_partidos):
        df_partidos = df_partidos.sort_values("fecha").reset_index(drop=True)
        out_part = DATA / "partidos.csv"
        df_partidos.to_csv(out_part, index=False)
        print(f"Guardado {out_part} ({len(df_partidos)} partidos finalizados)")

    df_fixture = pd.DataFrame(fixture_filas)
    if len(df_fixture):
        df_fixture = df_fixture.sort_values("fecha").reset_index(drop=True)
        out_fix = DATA / "fixture.csv"
        df_fixture.to_csv(out_fix, index=False)
        print(f"Guardado {out_fix} ({len(df_fixture)} partidos programados en fixture)")

    return df_partidos, df_fixture


if __name__ == "__main__":
    recolectar()
