"""
Motor predictivo de rendimiento individual de jugadores (Player Props & Matchup Lab)
para la NBA. Modela proyecciones de Puntos, Rebotes, Asistencias, Triples y PRA (PTS+REB+AST)
considerando:
  1. Eficiencia por minuto y tasa de uso (Usage Rate).
  2. Ajuste dinámico por ritmo del partido (Pace del rival).
  3. Redistribución de volumen por ausencia de compañeros clave ("Vacuum Effect").
  4. Historial directo contra el rival (H2H individual).
  5. Cálculo analítico de probabilidades Over/Under y cuotas justas de apuestas.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

DATA = Path(__file__).resolve().parent / "data"
LEAGUE_AVG_PACE = 99.5


def _parse_dash(val):
    """Parsea formatos 'M-A' de ESPN (ej. '8-15' -> 8 hechos, 15 intentados)."""
    if pd.isna(val) or not str(val).strip():
        return 0.0, 0.0
    partes = str(val).strip().split("-")
    if len(partes) == 2:
        try:
            return float(partes[0]), float(partes[1])
        except ValueError:
            return 0.0, 0.0
    return 0.0, 0.0


def _parse_min(val):
    """Parsea minutos jugados asegurando tipo flotante."""
    if pd.isna(val) or str(val).strip() in ("--", "", "DNP", "NWT"):
        return 0.0
    try:
        return float(str(val).strip())
    except ValueError:
        return 0.0


def cargar_dataset_jugadores():
    """Carga y normaliza el dataset de box scores individuales y rosters activos."""
    box_p = DATA / "jugadores_boxscore.csv"
    rost_p = DATA / "rosters_activos.csv"
    if not box_p.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(box_p)
    df["min_num"] = df["MIN"].apply(_parse_min)
    df["pts_num"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0.0)
    df["reb_num"] = pd.to_numeric(df["REB"], errors="coerce").fillna(0.0)
    df["ast_num"] = pd.to_numeric(df["AST"], errors="coerce").fillna(0.0)
    df["stl_num"] = pd.to_numeric(df["STL"], errors="coerce").fillna(0.0)
    df["blk_num"] = pd.to_numeric(df["BLK"], errors="coerce").fillna(0.0)
    df["tov_num"] = pd.to_numeric(df["TO"], errors="coerce").fillna(0.0)
    df["plus_minus_num"] = pd.to_numeric(df["+/-"], errors="coerce").fillna(0.0)

    # Tiros
    fg_m_a = df["FG"].apply(_parse_dash)
    tp_m_a = df["3PT"].apply(_parse_dash)
    ft_m_a = df["FT"].apply(_parse_dash)

    df["fgm"] = [x[0] for x in fg_m_a]
    df["fga"] = [x[1] for x in fg_m_a]
    df["tpm"] = [x[0] for x in tp_m_a]
    df["tpa"] = [x[1] for x in tp_m_a]
    df["ftm"] = [x[0] for x in ft_m_a]
    df["fta"] = [x[1] for x in ft_m_a]

    df["pra"] = df["pts_num"] + df["reb_num"] + df["ast_num"]
    df["fecha_dt"] = pd.to_datetime(df["fecha"])

    df_rosters = pd.read_csv(rost_p) if rost_p.exists() else pd.DataFrame()
    return df, df_rosters


def obtener_perfil_jugador(df_box, df_rosters, player_name):
    """Extrae las métricas históricas, per-36, tendencias y varianza de un jugador."""
    p_df = df_box[df_box["player_name"] == player_name].sort_values("fecha_dt", ascending=False)
    if p_df.empty:
        return None

    # Solo partidos con minutos reales (> 5 min) para promedios de ritmo
    p_active = p_df[p_df["min_num"] >= 5.0]
    if p_active.empty:
        p_active = p_df

    pj = len(p_df)
    min_tot = p_active["min_num"].sum()
    min_avg = float(p_active["min_num"].mean()) if pj else 0.0

    # Estadísticas por partido
    pts_avg = float(p_active["pts_num"].mean())
    reb_avg = float(p_active["reb_num"].mean())
    ast_avg = float(p_active["ast_num"].mean())
    tpm_avg = float(p_active["tpm_num"].mean()) if "tpm_num" in p_active else float(p_active["tpm"].mean())
    pra_avg = float(p_active["pra"].mean())

    # Desviación estándar (para cálculo de probabilidades de apuestas)
    pts_std = max(float(p_active["pts_num"].std(ddof=0)), 2.5) if len(p_active) > 1 else 4.0
    reb_std = max(float(p_active["reb_num"].std(ddof=0)), 1.5) if len(p_active) > 1 else 2.5
    ast_std = max(float(p_active["ast_num"].std(ddof=0)), 1.2) if len(p_active) > 1 else 2.0
    tpm_std = max(float(p_active["tpm"].std(ddof=0)), 0.8) if len(p_active) > 1 else 1.2
    pra_std = max(float(p_active["pra"].std(ddof=0)), 3.5) if len(p_active) > 1 else 6.0

    # Tasas por 36 minutos
    factor_36 = (36.0 / min_avg) if min_avg > 0 else 1.0
    pts_per36 = pts_avg * factor_36
    reb_per36 = reb_avg * factor_36
    ast_per36 = ast_avg * factor_36
    tpm_per36 = tpm_avg * factor_36

    # Forma reciente (Últimos 5 partidos)
    l5 = p_active.head(5)
    pts_l5 = float(l5["pts_num"].mean()) if len(l5) else pts_avg
    reb_l5 = float(l5["reb_num"].mean()) if len(l5) else reb_avg
    ast_l5 = float(l5["ast_num"].mean()) if len(l5) else ast_avg
    tpm_l5 = float(l5["tpm"].mean()) if len(l5) else tpm_avg
    min_l5 = float(l5["min_num"].mean()) if len(l5) else min_avg

    # Tiros y Usage proxy
    fga_avg = float(p_active["fga"].mean())
    fta_avg = float(p_active["fta"].mean())
    tov_avg = float(p_active["tov_num"].mean())
    usage_proxy = (fga_avg + 0.44 * fta_avg + tov_avg) / max(min_avg, 1.0)

    # Info de roster actual
    meta = {}
    if not df_rosters.empty:
        r_match = df_rosters[df_rosters["player_name"] == player_name]
        if not r_match.empty:
            meta = r_match.iloc[0].to_dict()

    equipo_actual = meta.get("team_name") or p_df["equipo"].iloc[0]

    return {
        "player_name": player_name,
        "equipo": equipo_actual,
        "position": meta.get("position") or p_df["position"].iloc[0] or "G",
        "jersey": meta.get("jersey", ""),
        "age": meta.get("age", ""),
        "height": meta.get("height", ""),
        "weight": meta.get("weight", ""),
        "headshot": meta.get("headshot", ""),
        "pj": pj,
        "min_avg": min_avg,
        "min_l5": min_l5,
        "pts_avg": pts_avg,
        "pts_l5": pts_l5,
        "pts_std": pts_std,
        "reb_avg": reb_avg,
        "reb_l5": reb_l5,
        "reb_std": reb_std,
        "ast_avg": ast_avg,
        "ast_l5": ast_l5,
        "ast_std": ast_std,
        "tpm_avg": tpm_avg,
        "tpm_l5": tpm_l5,
        "tpm_std": tpm_std,
        "pra_avg": pra_avg,
        "pra_std": pra_std,
        "pts_per36": pts_per36,
        "reb_per36": reb_per36,
        "ast_per36": ast_per36,
        "tpm_per36": tpm_per36,
        "fga_avg": fga_avg,
        "usage_proxy": usage_proxy,
        "df_games": p_df,
    }


def obtener_companeros_equipo(df_box, df_rosters, team_name, exclude_player=None):
    """Obtiene los compañeros de equipo ordenados por minutos/puntos para simular bajas."""
    companeros = []
    if not df_rosters.empty:
        sub_r = df_rosters[df_rosters["team_name"] == team_name]
        for _, r in sub_r.iterrows():
            nom = r["player_name"]
            if nom != exclude_player:
                p_games = df_box[df_box["player_name"] == nom]
                pts = float(p_games["pts_num"].mean()) if len(p_games) else 5.0
                minutos = float(p_games["min_num"].mean()) if len(p_games) else 15.0
                companeros.append({
                    "player_name": nom,
                    "position": r.get("position", ""),
                    "pts_avg": pts,
                    "min_avg": minutos,
                    "headshot": r.get("headshot", "")
                })
    if not companeros:
        # Fallback desde el historial de box score
        sub_b = df_box[df_box["equipo"] == team_name]
        for nom in sub_b["player_name"].unique():
            if nom != exclude_player:
                p_games = sub_b[sub_b["player_name"] == nom]
                companeros.append({
                    "player_name": nom,
                    "position": p_games["position"].iloc[0] if "position" in p_games else "",
                    "pts_avg": float(p_games["pts_num"].mean()),
                    "min_avg": float(p_games["min_num"].mean()),
                    "headshot": ""
                })

    return sorted(companeros, key=lambda x: (x["min_avg"], x["pts_avg"]), reverse=True)


def obtener_estrellas_franquicia(df_box, df_rosters, team_name, top_n=2):
    """Devuelve dinámicamente las figuras clave de una franquicia según minutos y puntos recientes."""
    comps = obtener_companeros_equipo(df_box, df_rosters, team_name)
    if comps:
        return [c["player_name"] for c in comps[:top_n]]
    return ["Jugador Franquicia"]


def proyectar_jugador(perfil, rival, minutos_proy=None, es_local=True,
                      bajas_companeros=None, pace_rival=LEAGUE_AVG_PACE,
                      drtg_rival=114.0, h2h_games=None):
    """
    Calcula la proyección analítica del jugador para un partido determinado,
    incorporando minutos, rival, localía y la redistribución de uso por bajas de compañeros.
    """
    if perfil is None:
        return {}

    bajas_companeros = bajas_companeros or []
    minutos = float(minutos_proy if minutos_proy is not None else perfil["min_l5"])
    minutos = max(minutos, 5.0)

    # 1. Ponderación de tasa base: 60% forma reciente (L5) + 40% temporada completa
    p_per_min = (0.60 * (perfil["pts_l5"] / max(perfil["min_l5"], 1)) +
                 0.40 * (perfil["pts_avg"] / max(perfil["min_avg"], 1)))
    r_per_min = (0.60 * (perfil["reb_l5"] / max(perfil["min_l5"], 1)) +
                 0.40 * (perfil["reb_avg"] / max(perfil["min_avg"], 1)))
    a_per_min = (0.60 * (perfil["ast_l5"] / max(perfil["min_l5"], 1)) +
                 0.40 * (perfil["ast_avg"] / max(perfil["min_avg"], 1)))
    t_per_min = (0.60 * (perfil["tpm_l5"] / max(perfil["min_l5"], 1)) +
                 0.40 * (perfil["tpm_avg"] / max(perfil["min_avg"], 1)))

    # 2. Ajuste por Ritmo del Partido (Pace)
    factor_pace = 1.0 + (pace_rival - LEAGUE_AVG_PACE) / LEAGUE_AVG_PACE

    # 3. Ajuste por Calidad Defensiva del Rival (DRtg)
    # Media liga DRtg ~ 114. Rival con 120 es defensa permisiva (+5%), rival con 108 es férrea (-5%)
    factor_def = 1.0 + (drtg_rival - 114.0) / 150.0

    # 4. Ventaja de Localía individual (ligeramente más anotación y acierto en casa)
    factor_casa = 1.035 if es_local else 0.970

    # 5. Redistribución de Uso por Bajas de Compañeros ("Vacuum Effect")
    # Cada baja de peso deja un volumen de tiros y creación vacante
    boost_pts_baja = 1.0
    boost_ast_baja = 1.0
    boost_reb_baja = 1.0
    impactos_bajas = []

    for baja in bajas_companeros:
        # Estimar el peso del ausente
        fga_baja = baja.get("pts_avg", 10.0) / 1.3
        # El jugador absorbe entre el 10% y el 25% del volumen del compañero ausente
        delta_p = min(0.25, max(0.06, (fga_baja / 30.0) * (perfil["usage_proxy"] / 0.70)))
        boost_pts_baja += delta_p
        boost_ast_baja += delta_p * 0.8
        boost_reb_baja += delta_p * 0.4
        impactos_bajas.append({
            "baja": baja["player_name"],
            "delta_pts_pct": f"+{delta_p*100:.1f}%"
        })

    # Proyecciones finales
    pts_proy = p_per_min * minutos * factor_pace * factor_def * factor_casa * boost_pts_baja
    reb_proy = r_per_min * minutos * factor_pace * factor_casa * boost_reb_baja
    ast_proy = a_per_min * minutos * factor_pace * factor_def * boost_ast_baja
    tpm_proy = t_per_min * minutos * factor_pace * factor_def * factor_casa * boost_pts_baja

    # Ajuste suave por historial H2H si existen al menos 2 partidos previos contra el rival
    if h2h_games is not None and len(h2h_games) >= 2:
        h2h_pts = float(h2h_games["pts_num"].mean())
        # Peso del 15% al historial directo específico
        pts_proy = 0.85 * pts_proy + 0.15 * h2h_pts

    pra_proy = pts_proy + reb_proy + ast_proy

    # Desviación ajustada para la calculadora de cuotas
    scale = np.sqrt(minutos / max(perfil["min_avg"], 10.0))
    pts_sigma = perfil["pts_std"] * scale
    reb_sigma = perfil["reb_std"] * scale
    ast_sigma = perfil["ast_std"] * scale
    tpm_sigma = perfil["tpm_std"] * scale
    pra_sigma = perfil["pra_std"] * scale

    return {
        "player_name": perfil["player_name"],
        "equipo": perfil["equipo"],
        "rival": rival,
        "minutos": round(minutos, 1),
        "pts": round(pts_proy, 1),
        "reb": round(reb_proy, 1),
        "ast": round(ast_proy, 1),
        "tpm": round(tpm_proy, 1),
        "pra": round(pra_proy, 1),
        "sigmas": {
            "pts": pts_sigma, "reb": reb_sigma, "ast": ast_sigma,
            "tpm": tpm_sigma, "pra": pra_sigma
        },
        "factores": {
            "pace": factor_pace,
            "defensa_rival": factor_def,
            "localia": factor_casa,
            "boost_bajas_pts": boost_pts_baja,
            "impactos_bajas": impactos_bajas,
        }
    }


def calcular_probabilidades_prop(media, sigma, linea):
    """
    Calcula P(Over) y P(Under) y cuotas justas para una línea de apuestas dada (ej. 27.5 puntos).
    Usa distribución normal continua corregida para variables continuas/discretas.
    """
    linea = float(linea)
    if sigma <= 0:
        sigma = 1.0
    # Probabilidad de superar la línea
    p_under = float(norm.cdf(linea, loc=media, scale=sigma))
    p_over = 1.0 - p_under

    p_over = float(np.clip(p_over, 0.02, 0.98))
    p_under = float(np.clip(p_under, 0.02, 0.98))

    cuota_over = round(1.0 / p_over, 2)
    cuota_under = round(1.0 / p_under, 2)

    return {
        "linea": linea,
        "media_proy": media,
        "prob_over": p_over,
        "prob_under": p_under,
        "cuota_over": cuota_over,
        "cuota_under": cuota_under,
        "recomendacion": "🔥 OVER" if p_over >= 0.58 else ("🧊 UNDER" if p_under >= 0.58 else "⚖️ LÍNEA JUSTA")
    }
