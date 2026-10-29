"""
Recolector de la NBA desde la API pública de ESPN ('basketball/nba').
Extrae partidos oficiales (temporada regular, play-in y playoffs),
puntuaciones por cuarto, estadísticas de tiro, rebotes y asistencias.
Genera nba/data/partidos.csv (jugados) y nba/data/fixture.csv (programados).
"""
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"

# Temporadas a recolectar: 2023 (2022-23), 2024 (2023-24), 2025 (2024-25), 2026 (2025-26), 2027 (2026-27 actual)
TEMPORADAS = [2023, 2024, 2025, 2026, 2027]

NORM_MAP = {
    "LA Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers"
}


def norm_team(name):
    return NORM_MAP.get(name, name)


def _parse_stat(competitor, stat_name):
    for st in competitor.get("statistics", []):
        if st.get("name") == stat_name:
            v = st.get("displayValue")
            try:
                return float(v) if v not in (None, "") else None
            except (ValueError, TypeError):
                return None
    return None


def _parse_linescores(competitor):
    scores = {"q1": None, "q2": None, "q3": None, "q4": None, "ot": 0.0}
    for ls in competitor.get("linescores", []):
        try:
            period = int(ls.get("period", 0))
            val = float(ls.get("value", 0.0))
            if period == 1:
                scores["q1"] = val
            elif period == 2:
                scores["q2"] = val
            elif period == 3:
                scores["q3"] = val
            elif period == 4:
                scores["q4"] = val
            elif period > 4:
                scores["ot"] += val
        except (ValueError, TypeError):
            continue
    return scores


def _meses_temporada(season_year):
    """Devuelve la lista de meses YYYYMM que componen la temporada NBA (Oct a Jun)."""
    prev_year = season_year - 1
    # Oct, Nov, Dic del año previo
    meses = [f"{prev_year}10", f"{prev_year}11", f"{prev_year}12"]
    # Ene, Feb, Mar, Abr, May, Jun del año de cierre
    for m in range(1, 7):
        meses.append(f"{season_year}{m:02d}")
    return meses


def _obtener_eventos_mes(ym):
    try:
        r = requests.get(f"{SCOREBOARD}?dates={ym}&limit=500", timeout=30)
        if r.status_code == 200:
            return r.json().get("events", [])
    except Exception as ex:
        print(f"  Error obteniendo mes {ym}: {ex}")
    return []


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    todo = []
    
    print("Iniciando recolección de partidos NBA desde ESPN...")
    for anio in TEMPORADAS:
        meses = _meses_temporada(anio)
        eventos_anio = []
        for ym in meses:
            evs = _obtener_eventos_mes(ym)
            eventos_anio.extend(evs)
            time.sleep(0.08)
            
        filas_temporada = []
        for e in eventos_anio:
            try:
                season_info = e.get("season", {})
                stype = season_info.get("type", 2)
                # Ignorar pretemporada (1) y All-Star (4); aceptar regular (2), playoffs (3) y play-in (5)
                if stype not in (2, 3, 5):
                    continue

                comp = e["competitions"][0]
                cs = comp["competitors"]
                h = next(x for x in cs if x["homeAway"] == "home")
                a = next(x for x in cs if x["homeAway"] == "away")

                st_type = e.get("status", {}).get("type", {})
                status_name = st_type.get("name", "")

                if status_name in ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED"):
                    continue

                is_finished = status_name in ("STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FULL_TIME")
                estado = "post" if is_finished else "pre"

                pl = None
                pv = None
                if is_finished:
                    try:
                        pl = float(h["score"]) if h.get("score") not in (None, "") else None
                        pv = float(a["score"]) if a.get("score") not in (None, "") else None
                    except (ValueError, TypeError):
                        pl = pv = None

                loc = norm_team(h["team"]["displayName"])
                vis = norm_team(a["team"]["displayName"])

                # Ignorar placeholders de llaves no definidas aún (ej. cuartos de final de Copa NBA)
                if loc in ("TBD", "TBA", "") or vis in ("TBD", "TBA", "") or loc == vis:
                    continue

                fecha_dt = pd.to_datetime(e["date"]).tz_convert("America/New_York").tz_localize(None)

                # Estadísticas de tiro y líneas de anotación
                ls_h = _parse_linescores(h)
                ls_a = _parse_linescores(a)

                fga_h = _parse_stat(h, "fieldGoalsAttempted")
                fgm_h = _parse_stat(h, "fieldGoalsMade")
                tpa_h = _parse_stat(h, "threePointFieldGoalsAttempted")
                tpm_h = _parse_stat(h, "threePointFieldGoalsMade")
                fta_h = _parse_stat(h, "freeThrowsAttempted")
                ftm_h = _parse_stat(h, "freeThrowsMade")
                reb_h = _parse_stat(h, "rebounds")
                ast_h = _parse_stat(h, "assists")

                fga_a = _parse_stat(a, "fieldGoalsAttempted")
                fgm_a = _parse_stat(a, "fieldGoalsMade")
                tpa_a = _parse_stat(a, "threePointFieldGoalsAttempted")
                tpm_a = _parse_stat(a, "threePointFieldGoalsMade")
                fta_a = _parse_stat(a, "freeThrowsAttempted")
                ftm_a = _parse_stat(a, "freeThrowsMade")
                reb_a = _parse_stat(a, "rebounds")
                ast_a = _parse_stat(a, "assists")

                # Métricas Four Factors aproximadas de partido
                efg_h = (fgm_h + 0.5 * tpm_h) / fga_h if (fga_h and fga_h > 0 and fgm_h is not None and tpm_h is not None) else None
                efg_a = (fgm_a + 0.5 * tpm_a) / fga_a if (fga_a and fga_a > 0 and fgm_a is not None and tpm_a is not None) else None

                ft_rate_h = fta_h / fga_h if (fga_h and fga_h > 0 and fta_h is not None) else None
                ft_rate_a = fta_a / fga_a if (fga_a and fga_a > 0 and fta_a is not None) else None

                # Estimación de posesiones básicas del partido
                est_pace = None
                if fga_h and fta_h and fga_a and fta_a:
                    est_poss_h = fga_h + 0.44 * fta_h
                    est_poss_a = fga_a + 0.44 * fta_a
                    est_pace = 0.5 * (est_poss_h + est_poss_a)

                filas_temporada.append({
                    "event_id": str(e["id"]),
                    "fecha": fecha_dt,
                    "temporada": anio,
                    "tipo_temporada": stype,
                    "local_id": str(h["team"]["id"]),
                    "visita_id": str(a["team"]["id"]),
                    "local": loc,
                    "visita": vis,
                    "puntos_local": pl,
                    "puntos_visita": pv,
                    "diferencial": (pl - pv) if (pl is not None and pv is not None) else None,
                    "total_puntos": (pl + pv) if (pl is not None and pv is not None) else None,
                    "ganador": ("local" if pl > pv else "visita") if (pl is not None and pv is not None) else None,
                    "estado": estado,
                    "status_name": status_name,
                    # Cuartos
                    "q1_local": ls_h["q1"], "q2_local": ls_h["q2"], "q3_local": ls_h["q3"], "q4_local": ls_h["q4"], "ot_local": ls_h["ot"],
                    "q1_visita": ls_a["q1"], "q2_visita": ls_a["q2"], "q3_visita": ls_a["q3"], "q4_visita": ls_a["q4"], "ot_visita": ls_a["ot"],
                    # Estadísticas de tiro
                    "fgm_local": fgm_h, "fga_local": fga_h, "fg_pct_local": (fgm_h / fga_h) if (fga_h and fgm_h is not None) else None,
                    "tpm_local": tpm_h, "tpa_local": tpa_h, "tp_pct_local": (tpm_h / tpa_h) if (tpa_h and tpm_h is not None) else None,
                    "ftm_local": ftm_h, "fta_local": fta_h, "ft_pct_local": (ftm_h / fta_h) if (fta_h and ftm_h is not None) else None,
                    "reb_local": reb_h, "ast_local": ast_h,
                    "fgm_visita": fgm_a, "fga_visita": fga_a, "fg_pct_visita": (fgm_a / fga_a) if (fga_a and fgm_a is not None) else None,
                    "tpm_visita": tpm_a, "tpa_visita": tpa_a, "tp_pct_visita": (tpm_a / tpa_a) if (tpa_a and tpm_a is not None) else None,
                    "ftm_visita": ftm_a, "fta_visita": fta_a, "ft_pct_visita": (ftm_a / fta_a) if (fta_a and ftm_a is not None) else None,
                    "reb_visita": reb_a, "ast_visita": ast_a,
                    # Factores avanzados
                    "efg_local": efg_h, "efg_visita": efg_a,
                    "ft_rate_local": ft_rate_h, "ft_rate_visita": ft_rate_a,
                    "est_pace": est_pace
                })
            except (KeyError, IndexError, StopIteration):
                continue
                
        jugados_t = sum(1 for x in filas_temporada if x["estado"] == "post")
        pre_t = sum(1 for x in filas_temporada if x["estado"] == "pre")
        print(f"  Temporada {anio-1}-{str(anio)[-2:]}: {len(filas_temporada)} partidos ({jugados_t} jugados, {pre_t} programados)")
        todo.extend(filas_temporada)

    if not todo:
        print("No se pudieron recolectar partidos de la NBA.")
        return None, None

    df = pd.DataFrame(todo).sort_values("fecha").drop_duplicates(subset=["event_id"], keep="last")
    
    jugados = df[df.estado == "post"].dropna(subset=["puntos_local", "puntos_visita"]).reset_index(drop=True)
    fixture = df[df.estado == "pre"].reset_index(drop=True)

    # Conservar el fixture de la temporada en curso
    temporada_actual = df["temporada"].max()
    fixture_actual = fixture[fixture.temporada == temporada_actual].reset_index(drop=True)

    jugados.to_csv(DATA / "partidos.csv", index=False)
    fixture_actual.to_csv(DATA / "fixture.csv", index=False)

    print(f"\n✅ Guardado NBA: partidos.csv ({len(jugados)} jugados {jugados.fecha.min().date()} – {jugados.fecha.max().date()})")
    print(f"✅ Guardado NBA: fixture.csv ({len(fixture_actual)} partidos por jugar de la temporada {temporada_actual-1}-{str(temporada_actual)[-2:]})")
    return jugados, fixture_actual


if __name__ == "__main__":
    recolectar()
