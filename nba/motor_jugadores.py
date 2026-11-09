"""
Motor predictivo de rendimiento individual de jugadores (Player Props & Matchup Lab v2.0)
para la NBA. Modela proyecciones de Puntos, Rebotes, Asistencias, Triples y PRA (PTS+REB+AST)
considerando:
  1. Tasas por minuto con encogimiento bayesiano (temporada actual en su franquicia + racha L5).
  2. Ajuste dinámico real por ritmo del partido (Pace) y Eficiencia Defensiva del rival (DRtg).
  3. Defensa contra la Posición (DvP: Defense vs. Position para G, F y C) extraída de los box scores.
  4. Motor Híbrido On/Off Empírico (partidos reales CON vs. SIN compañeros) + Redistribución cóncava
     de uso ("Usage Vacuum") con rendimientos decrecientes y techo fisiológico.
  5. Simulación de Bajas en el Equipo Rival (ausencia de protectores de aro o defensores perimetrales)
     y alerta de riesgo de Blowout (partido roto en el 4° cuarto).
  6. Distribuciones discretas (Binomial Negativa / Poisson) para conteos bajos (3PM, REB, AST) y
     Normal truncada con corrección de continuidad para PTS y PRA, más Hit Rate histórico real.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, nbinom

DATA = Path(__file__).resolve().parent / "data"
LEAGUE_AVG_PACE = 99.2
LEAGUE_AVG_DRTG = 114.5


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
    s = str(val).strip()
    if ":" in s:
        partes = s.split(":")
        try:
            return float(partes[0]) + float(partes[1]) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _norm_pos(pos):
    """Normaliza la posición a uno de los 3 grupos estándar: G (Guard), F (Forward), C (Center)."""
    if pd.isna(pos) or not str(pos).strip():
        return "G"
    p = str(pos).strip().upper()
    if p in ("PG", "SG", "G"):
        return "G"
    if p in ("SF", "PF", "F"):
        return "F"
    if p == "C":
        return "C"
    if "G" in p:
        return "G"
    if "C" in p:
        return "C"
    return "F"


def cargar_dataset_jugadores():
    """Carga y normaliza el dataset de box scores individuales y rosters activos."""
    box_p = DATA / "jugadores_boxscore.csv"
    rost_p = DATA / "rosters_activos.csv"
    if not box_p.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(box_p)
    df["event_id"] = df["event_id"].astype(str)
    df["min_num"] = df["MIN"].apply(_parse_min)
    df["pts_num"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0.0)
    df["reb_num"] = pd.to_numeric(df["REB"], errors="coerce").fillna(0.0)
    df["ast_num"] = pd.to_numeric(df["AST"], errors="coerce").fillna(0.0)
    df["stl_num"] = pd.to_numeric(df["STL"], errors="coerce").fillna(0.0)
    df["blk_num"] = pd.to_numeric(df["BLK"], errors="coerce").fillna(0.0)
    df["tov_num"] = pd.to_numeric(df["TO"], errors="coerce").fillna(0.0)
    df["plus_minus_num"] = pd.to_numeric(df["+/-"], errors="coerce").fillna(0.0)
    df["pos_group"] = df["position"].apply(_norm_pos)

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
    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")

    df_rosters = pd.read_csv(rost_p) if rost_p.exists() else pd.DataFrame()
    if not df_rosters.empty and "position" in df_rosters.columns:
        df_rosters["pos_group"] = df_rosters["position"].apply(_norm_pos)
    return df, df_rosters


def calcular_defensa_vs_posicion(df_box):
    """
    Calcula la matriz de Defensa contra la Posición (DvP: Defense vs. Position)
    para cada franquicia de la NBA frente a Bases (G), Aleros (F) y Pívots (C).
    Retorna multiplicadores suavizados con encogimiento bayesiano y el ranking (1=mejor defensa, 30=más permisiva).
    """
    if df_box is None or df_box.empty:
        return {}

    valid = df_box[df_box["min_num"] >= 8.0].copy()
    if valid.empty:
        return {}

    if "pos_group" not in valid.columns:
        valid["pos_group"] = valid["position"].apply(_norm_pos)

    # Tasas promedio de la liga por minuto para cada grupo posicional
    league_rates = {}
    for pg in ("G", "F", "C"):
        sub = valid[valid["pos_group"] == pg]
        tot_m = max(sub["min_num"].sum(), 1.0)
        league_rates[pg] = {
            "pts": sub["pts_num"].sum() / tot_m,
            "reb": sub["reb_num"].sum() / tot_m,
            "ast": sub["ast_num"].sum() / tot_m,
            "tpm": sub["tpm"].sum() / tot_m,
        }

    dvp = {}
    equipos = sorted(valid["rival"].dropna().unique())
    for pg in ("G", "F", "C"):
        sub_pg = valid[valid["pos_group"] == pg]
        lr = league_rates[pg]
        raw_scores = []
        for eq in equipos:
            sub_eq = sub_pg[sub_pg["rival"] == eq]
            n_obs = len(sub_eq)
            tot_m = max(sub_eq["min_num"].sum(), 1.0)
            # Encogimiento bayesiano (kappa = 80 apariciones de jugadores rivales en esa posición)
            w = n_obs / (n_obs + 80.0) if n_obs > 0 else 0.0
            f_pts = 1.0 + w * (((sub_eq["pts_num"].sum() / tot_m) / max(lr["pts"], 1e-6)) - 1.0) if n_obs else 1.0
            f_reb = 1.0 + w * (((sub_eq["reb_num"].sum() / tot_m) / max(lr["reb"], 1e-6)) - 1.0) if n_obs else 1.0
            f_ast = 1.0 + w * (((sub_eq["ast_num"].sum() / tot_m) / max(lr["ast"], 1e-6)) - 1.0) if n_obs else 1.0
            f_tpm = 1.0 + w * (((sub_eq["tpm"].sum() / tot_m) / max(lr["tpm"], 1e-6)) - 1.0) if n_obs else 1.0

            f_pts = float(np.clip(f_pts, 0.90, 1.10))
            f_reb = float(np.clip(f_reb, 0.90, 1.10))
            f_ast = float(np.clip(f_ast, 0.90, 1.10))
            f_tpm = float(np.clip(f_tpm, 0.88, 1.12))

            dvp.setdefault(eq, {})[pg] = {
                "pts": f_pts,
                "reb": f_reb,
                "ast": f_ast,
                "tpm": f_tpm,
                "n_obs": n_obs,
                "rank_pts": 15,
            }
            raw_scores.append((eq, f_pts))

        # Ordenar de menor factor (1 = defensa más dura) a mayor factor (30 = defensa más permisiva)
        raw_scores.sort(key=lambda x: x[1])
        for rk, (eq, _) in enumerate(raw_scores, start=1):
            dvp[eq][pg]["rank_pts"] = rk

    return dvp


def obtener_perfil_jugador(df_box, df_rosters, player_name):
    """Extrae las métricas históricas, per-36, tendencias y varianza de un jugador."""
    p_df_all = df_box[df_box["player_name"] == player_name].sort_values("fecha_dt", ascending=False)
    if p_df_all.empty:
        return None

    # Info de roster actual
    meta = {}
    if not df_rosters.empty:
        r_match = df_rosters[df_rosters["player_name"] == player_name]
        if not r_match.empty:
            meta = r_match.iloc[0].to_dict()

    equipo_actual = meta.get("team_name") or p_df_all["equipo"].iloc[0]

    # Si el jugador cambió de equipo y ya tiene >= 5 partidos en su equipo actual,
    # priorizar sus estadísticas en el rol de su equipo actual para los promedios base
    p_df_team = p_df_all[p_df_all["equipo"] == equipo_actual]
    p_df = p_df_team if len(p_df_team[p_df_team["min_num"] >= 5.0]) >= 5 else p_df_all

    # Solo partidos con minutos reales (>= 5 min) para promedios de ritmo
    p_active = p_df[p_df["min_num"] >= 5.0]
    if p_active.empty:
        p_active = p_df

    pj = len(p_active)
    min_avg = float(p_active["min_num"].mean()) if pj else 0.0

    # Estadísticas por partido
    pts_avg = float(p_active["pts_num"].mean())
    reb_avg = float(p_active["reb_num"].mean())
    ast_avg = float(p_active["ast_num"].mean())
    tpm_avg = float(p_active["tpm"].mean())
    pra_avg = float(p_active["pra"].mean())
    plus_minus_avg = float(p_active["plus_minus_num"].mean())

    # Desviación estándar (para cálculo de probabilidades de apuestas)
    pts_std = max(float(p_active["pts_num"].std(ddof=0)), 2.5) if len(p_active) > 1 else 4.5
    reb_std = max(float(p_active["reb_num"].std(ddof=0)), 1.4) if len(p_active) > 1 else 2.2
    ast_std = max(float(p_active["ast_num"].std(ddof=0)), 1.2) if len(p_active) > 1 else 2.0
    tpm_std = max(float(p_active["tpm"].std(ddof=0)), 0.85) if len(p_active) > 1 else 1.2
    pra_std = max(float(p_active["pra"].std(ddof=0)), 3.5) if len(p_active) > 1 else 6.0

    # Tasas por 36 minutos
    factor_36 = (36.0 / min_avg) if min_avg > 0 else 1.0
    pts_per36 = pts_avg * factor_36
    reb_per36 = reb_avg * factor_36
    ast_per36 = ast_avg * factor_36
    tpm_per36 = tpm_avg * factor_36

    # Forma reciente (Últimos 5 y Últimos 10 partidos)
    l5 = p_active.head(5)
    l10 = p_active.head(10)
    pts_l5 = float(l5["pts_num"].mean()) if len(l5) else pts_avg
    reb_l5 = float(l5["reb_num"].mean()) if len(l5) else reb_avg
    ast_l5 = float(l5["ast_num"].mean()) if len(l5) else ast_avg
    tpm_l5 = float(l5["tpm"].mean()) if len(l5) else tpm_avg
    min_l5 = float(l5["min_num"].mean()) if len(l5) else min_avg

    pts_l10 = float(l10["pts_num"].mean()) if len(l10) else pts_avg
    min_l10 = float(l10["min_num"].mean()) if len(l10) else min_avg

    # Tiros y Usage proxy (posesiones usadas por minuto en cancha)
    fga_avg = float(p_active["fga"].mean())
    fta_avg = float(p_active["fta"].mean())
    tov_avg = float(p_active["tov_num"].mean())
    usage_proxy = (fga_avg + 0.44 * fta_avg + tov_avg) / max(min_avg, 1.0)

    pos_raw = meta.get("position") or p_df["position"].iloc[0] or "G"
    pos_group = _norm_pos(pos_raw)

    return {
        "player_name": player_name,
        "equipo": equipo_actual,
        "position": pos_raw,
        "pos_group": pos_group,
        "jersey": meta.get("jersey", ""),
        "age": meta.get("age", ""),
        "height": meta.get("height", ""),
        "weight": meta.get("weight", ""),
        "headshot": meta.get("headshot", ""),
        "pj": pj,
        "min_avg": min_avg,
        "min_l5": min_l5,
        "min_l10": min_l10,
        "pts_avg": pts_avg,
        "pts_l5": pts_l5,
        "pts_l10": pts_l10,
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
        "plus_minus_avg": plus_minus_avg,
        "pts_per36": pts_per36,
        "reb_per36": reb_per36,
        "ast_per36": ast_per36,
        "tpm_per36": tpm_per36,
        "fga_avg": fga_avg,
        "fta_avg": fta_avg,
        "usage_proxy": usage_proxy,
        "df_games": p_df_all,
        "df_games_team": p_active,
    }


def obtener_companeros_equipo(df_box, df_rosters, team_name, exclude_player=None, solo_con_partidos=False):
    """
    Obtiene los jugadores de una franquicia ordenados por importancia en la rotación
    (minutos y puntos recientes) para selectores y simulación de bajas.
    """
    companeros = []
    if not df_rosters.empty:
        sub_r = df_rosters[df_rosters["team_name"] == team_name]
        for _, r in sub_r.iterrows():
            nom = r["player_name"]
            if nom == exclude_player:
                continue
            p_games_all = df_box[(df_box["player_name"] == nom) & (df_box["min_num"] >= 5.0)]
            p_games_team = p_games_all[p_games_all["equipo"] == team_name]
            p_games = p_games_team if len(p_games_team) >= 5 else p_games_all
            pj = len(p_games)
            if solo_con_partidos and pj == 0:
                continue
            pts = float(p_games["pts_num"].mean()) if pj else 0.0
            reb = float(p_games["reb_num"].mean()) if pj else 0.0
            ast = float(p_games["ast_num"].mean()) if pj else 0.0
            minutos = float(p_games["min_num"].mean()) if pj else 0.0
            pm = float(p_games["plus_minus_num"].mean()) if pj else 0.0
            pos_raw = r.get("position", "G")
            companeros.append({
                "player_name": nom,
                "position": pos_raw,
                "pos_group": _norm_pos(pos_raw),
                "pts_avg": pts,
                "reb_avg": reb,
                "ast_avg": ast,
                "min_avg": minutos,
                "plus_minus_avg": pm,
                "pj": pj,
                "has_games": 1 if pj > 0 else 0,
                "headshot": r.get("headshot", "")
            })
    if not companeros and not df_box.empty:
        sub_b = df_box[(df_box["equipo"] == team_name) & (df_box["min_num"] >= 5.0)]
        for nom in sub_b["player_name"].unique():
            if nom != exclude_player:
                p_games = sub_b[sub_b["player_name"] == nom]
                pos_raw = p_games["position"].iloc[0] if "position" in p_games else "G"
                companeros.append({
                    "player_name": nom,
                    "position": pos_raw,
                    "pos_group": _norm_pos(pos_raw),
                    "pts_avg": float(p_games["pts_num"].mean()),
                    "reb_avg": float(p_games["reb_num"].mean()),
                    "ast_avg": float(p_games["ast_num"].mean()),
                    "min_avg": float(p_games["min_num"].mean()),
                    "plus_minus_avg": float(p_games["plus_minus_num"].mean()),
                    "pj": len(p_games),
                    "has_games": 1,
                    "headshot": ""
                })

    # Ordenar primero jugadores activos con partidos por impacto (minutos + producción), y al final los sin partidos
    return sorted(
        companeros,
        key=lambda x: (x["has_games"], round(x["min_avg"] * 0.7 + x["pts_avg"] * 0.6, 2), x["pts_avg"]),
        reverse=True
    )


def obtener_estrellas_franquicia(df_box, df_rosters, team_name, top_n=2):
    """Devuelve dinámicamente las figuras clave de una franquicia según minutos y puntos recientes."""
    comps = obtener_companeros_equipo(df_box, df_rosters, team_name, solo_con_partidos=True)
    if comps:
        return [c["player_name"] for c in comps[:top_n]]
    return ["Jugador Franquicia"]


def calcular_impacto_bajas_equipo(df_box, df_rosters, team_name, nombres_bajas):
    """
    Cuantifica el impacto colectivo de las bajas seleccionadas sobre el rendimiento del equipo:
    - Penalización de Rating Elo (delta_elo)
    - Penalización de Net Rating (delta_netrtg), Offensive Rating (delta_ortg) y Defensive Rating (delta_drtg)
    Usa rendimientos decrecientes para evitar colapsos irreales cuando faltan múltiples jugadores.
    """
    if not nombres_bajas:
        return {
            "delta_elo": 0.0,
            "delta_netrtg": 0.0,
            "delta_ortg": 0.0,
            "delta_drtg": 0.0,
            "detalle": []
        }

    plantel = {c["player_name"]: c for c in obtener_companeros_equipo(df_box, df_rosters, team_name)}
    raw_net_loss = 0.0
    raw_ortg_loss = 0.0
    raw_drtg_loss = 0.0
    detalle = []

    for nom in nombres_bajas:
        info = plantel.get(nom)
        if not info or info["pj"] == 0:
            imp_net = 1.5
            imp_ortg = 1.0
            imp_drtg = 0.5
        else:
            min_share = min(info["min_avg"] / 36.0, 1.05)
            # Valor ofensivo por minuto basado en puntos y creación
            off_val = (info["pts_avg"] + 1.4 * info["ast_avg"]) / 38.0
            # Valor defensivo/rebotes según posición y rebotes
            def_val = (info["reb_avg"] / 11.0) * (1.25 if info["pos_group"] == "C" else 1.0)
            imp_ortg = float(np.clip(min_share * off_val * 4.2, 0.6, 4.5))
            imp_drtg = float(np.clip(min_share * def_val * 2.2, 0.4, 2.8))
            imp_net = imp_ortg + imp_drtg

        raw_net_loss += imp_net
        raw_ortg_loss += imp_ortg
        raw_drtg_loss += imp_drtg
        detalle.append({
            "player_name": nom,
            "pts_avg": info["pts_avg"] if info else 0.0,
            "min_avg": info["min_avg"] if info else 0.0,
            "delta_netrtg": round(imp_net, 2)
        })

    # Curva cóncava de saturación: los suplentes absorben parte de los minutos vacantes ("Next Man Up")
    eff_net_loss = 10.5 * (1.0 - np.exp(-raw_net_loss / 9.5))
    scale = (eff_net_loss / raw_net_loss) if raw_net_loss > 0 else 1.0
    eff_ortg_loss = raw_ortg_loss * scale
    eff_drtg_loss = raw_drtg_loss * scale
    eff_elo_loss = eff_net_loss * 13.5  # ~13.5 puntos Elo por cada punto de Net Rating en la NBA

    return {
        "delta_elo": round(float(eff_elo_loss), 1),
        "delta_netrtg": round(float(eff_net_loss), 2),
        "delta_ortg": round(float(eff_ortg_loss), 2),
        "delta_drtg": round(float(eff_drtg_loss), 2),
        "detalle": detalle,
    }


def calcular_on_off_companeros(df_box, player_name, equipo, bajas_companeros):
    """
    Cruza empíricamente en jugadores_boxscore.csv los partidos que el jugador disputó
    CON vs. SIN cada uno de los compañeros seleccionados en bajas_companeros.
    Devuelve la tabla comparativa real y los multiplicadores empíricos por 36 minutos
    suavizados mediante encogimiento bayesiano (Bayesian Shrinkage).
    """
    if df_box is None or df_box.empty or not bajas_companeros:
        return {"splits": [], "emp_mult": {"pts": 1.0, "reb": 1.0, "ast": 1.0, "tpm": 1.0}, "n_off_total": 0}

    # Filtrar partidos del jugador en su equipo actual (si tiene >= 5) o globales
    p_all = df_box[(df_box["player_name"] == player_name) & (df_box["min_num"] >= 8.0)]
    p_team = p_all[p_all["equipo"] == equipo]
    p_games = p_team if len(p_team) >= 5 else p_all

    if p_games.empty:
        return {"splits": [], "emp_mult": {"pts": 1.0, "reb": 1.0, "ast": 1.0, "tpm": 1.0}, "n_off_total": 0}

    splits = []
    # Pesos acumulados para el multiplicador empírico
    w_pts_acc, w_reb_acc, w_ast_acc, w_tpm_acc = 0.0, 0.0, 0.0, 0.0
    n_off_total = 0

    for baja in bajas_companeros:
        b_name = baja["player_name"] if isinstance(baja, dict) else str(baja)
        # Filtrar únicamente las temporadas en las que el compañero perteneció a este mismo equipo
        b_team_df = df_box[
            (df_box["player_name"] == b_name) &
            (df_box["equipo"] == equipo) &
            (df_box["min_num"] >= 5.0)
        ]
        temps_compartidas = set(b_team_df["temporada"].unique())
        b_events = set(b_team_df["event_id"].astype(str))

        # Solo evaluar partidos de las temporadas en que ambos coincidieron en la franquicia
        p_scope = p_games[p_games["temporada"].isin(temps_compartidas)] if temps_compartidas else p_games.iloc[0:0]
        df_con = p_scope[p_scope["event_id"].astype(str).isin(b_events)]
        df_sin = p_scope[~p_scope["event_id"].astype(str).isin(b_events)]

        n_con = len(df_con)
        n_sin = len(df_sin)
        n_off_total += n_sin

        if n_con > 0 and n_sin > 0:
            min_c = max(float(df_con["min_num"].mean()), 1.0)
            min_s = max(float(df_sin["min_num"].mean()), 1.0)

            pts_c, pts_s = float(df_con["pts_num"].mean()), float(df_sin["pts_num"].mean())
            reb_c, reb_s = float(df_con["reb_num"].mean()), float(df_sin["reb_num"].mean())
            ast_c, ast_s = float(df_con["ast_num"].mean()), float(df_sin["ast_num"].mean())
            tpm_c, tpm_s = float(df_con["tpm"].mean()), float(df_sin["tpm"].mean())
            fga_c, fga_s = float(df_con["fga"].mean()), float(df_sin["fga"].mean())

            # Tasas por minuto CON vs SIN
            r_pts_c, r_pts_s = pts_c / min_c, pts_s / min_s
            r_reb_c, r_reb_s = reb_c / min_c, reb_s / min_s
            r_ast_c, r_ast_s = ast_c / min_c, ast_s / min_s
            r_tpm_c, r_tpm_s = tpm_c / min_c, tpm_s / min_s

            # Ratio empírico acotado y encogido según tamaño de muestra (kappa = 10 partidos sin el compañero)
            w_emp = n_sin / (n_sin + 10.0)
            ratio_pts = np.clip((r_pts_s / max(r_pts_c, 0.05)) - 1.0, -0.04, 0.22)
            ratio_reb = np.clip((r_reb_s / max(r_reb_c, 0.02)) - 1.0, -0.04, 0.20)
            ratio_ast = np.clip((r_ast_s / max(r_ast_c, 0.02)) - 1.0, -0.04, 0.24)
            ratio_tpm = np.clip((r_tpm_s / max(r_tpm_c, 0.01)) - 1.0, -0.05, 0.22)

            w_pts_acc += w_emp * ratio_pts
            w_reb_acc += w_emp * ratio_reb
            w_ast_acc += w_emp * ratio_ast
            w_tpm_acc += w_emp * ratio_tpm

            splits.append({
                "companero": b_name,
                "pj_con": n_con,
                "pj_sin": n_sin,
                "min_con": round(min_c, 1),
                "min_sin": round(min_s, 1),
                "pts_con": round(pts_c, 1),
                "pts_sin": round(pts_s, 1),
                "delta_pts": round(pts_s - pts_c, 1),
                "reb_con": round(reb_c, 1),
                "reb_sin": round(reb_s, 1),
                "ast_con": round(ast_c, 1),
                "ast_sin": round(ast_s, 1),
                "fga_con": round(fga_c, 1),
                "fga_sin": round(fga_s, 1),
                "peso_empirico": round(w_emp, 2),
            })
        else:
            splits.append({
                "companero": b_name,
                "pj_con": n_con,
                "pj_sin": n_sin,
                "min_con": round(float(df_con["min_num"].mean()), 1) if n_con else 0.0,
                "min_sin": round(float(df_sin["min_num"].mean()), 1) if n_sin else 0.0,
                "pts_con": round(float(df_con["pts_num"].mean()), 1) if n_con else 0.0,
                "pts_sin": round(float(df_sin["pts_num"].mean()), 1) if n_sin else 0.0,
                "delta_pts": 0.0,
                "reb_con": round(float(df_con["reb_num"].mean()), 1) if n_con else 0.0,
                "reb_sin": round(float(df_sin["reb_num"].mean()), 1) if n_sin else 0.0,
                "ast_con": round(float(df_con["ast_num"].mean()), 1) if n_con else 0.0,
                "ast_sin": round(float(df_sin["ast_num"].mean()), 1) if n_sin else 0.0,
                "fga_con": round(float(df_con["fga"].mean()), 1) if n_con else 0.0,
                "fga_sin": round(float(df_sin["fga"].mean()), 1) if n_sin else 0.0,
                "peso_empirico": 0.0,
            })

    return {
        "splits": splits,
        "emp_mult": {
            "pts": 1.0 + float(np.clip(w_pts_acc, -0.10, 0.22)),
            "reb": 1.0 + float(np.clip(w_reb_acc, -0.10, 0.20)),
            "ast": 1.0 + float(np.clip(w_ast_acc, -0.10, 0.24)),
            "tpm": 1.0 + float(np.clip(w_tpm_acc, -0.12, 0.22)),
        },
        "n_off_total": n_off_total,
    }


def proyectar_jugador(perfil, rival, minutos_proy=None, es_local=True,
                      bajas_companeros=None, bajas_rivales=None,
                      pace_rival=LEAGUE_AVG_PACE, pace_equipo=LEAGUE_AVG_PACE,
                      drtg_rival=LEAGUE_AVG_DRTG, h2h_games=None,
                      dvp_data=None, df_box=None, spread_esperado=0.0):
    """
    Calcula la proyección analítica del jugador para un partido determinado (v2.0):
      1. Tasa por minuto con encogimiento bayesiano (65% base del equipo actual + 35% forma L5).
      2. Ajuste por ritmo real del partido esperado (Pace combinado) y Defensa del Rival (DRtg).
      3. Ajuste por Defensa contra la Posición (DvP para G, F o C).
      4. Redistribución Híbrida (On/Off Empírico en boxscore + Usage Vacuum cóncavo con saturación).
      5. Impacto de Bajas en el Equipo Rival (protección de aro / defensa perimetral) y alerta de Blowout.
      6. Encogimiento H2H directo en todas las métricas si existen antecedentes.
    """
    if perfil is None:
        return {}

    bajas_companeros = bajas_companeros or []
    bajas_rivales = bajas_rivales or []
    minutos_base = float(minutos_proy if minutos_proy is not None else (0.55 * perfil["min_l5"] + 0.45 * perfil["min_avg"]))
    minutos = max(minutos_base, 5.0)

    # 1. Alerta y ajuste suave por Riesgo de Blowout (partido roto cuando |spread| >= 14.5)
    riesgo_blowout = False
    penalizacion_min_blowout = 0.0
    if abs(float(spread_esperado)) >= 14.5 and minutos >= 28.0:
        riesgo_blowout = True
        exceso = min(abs(float(spread_esperado)) - 13.5, 10.0)
        penalizacion_min_blowout = round(0.22 * exceso, 1)

    # 2. Tasa base por minuto: 65% media estable + 35% racha L5 (evita sobre-reaccionar a rachas extremas de 5 juegos)
    rate_season_p = perfil["pts_avg"] / max(perfil["min_avg"], 1.0)
    rate_l5_p = perfil["pts_l5"] / max(perfil["min_l5"], 1.0)
    p_per_min = 0.65 * rate_season_p + 0.35 * rate_l5_p

    rate_season_r = perfil["reb_avg"] / max(perfil["min_avg"], 1.0)
    rate_l5_r = perfil["reb_l5"] / max(perfil["min_l5"], 1.0)
    r_per_min = 0.65 * rate_season_r + 0.35 * rate_l5_r

    rate_season_a = perfil["ast_avg"] / max(perfil["min_avg"], 1.0)
    rate_l5_a = perfil["ast_l5"] / max(perfil["min_l5"], 1.0)
    a_per_min = 0.65 * rate_season_a + 0.35 * rate_l5_a

    rate_season_t = perfil["tpm_avg"] / max(perfil["min_avg"], 1.0)
    rate_l5_t = perfil["tpm_l5"] / max(perfil["min_l5"], 1.0)
    t_per_min = 0.65 * rate_season_t + 0.35 * rate_l5_t

    # 3. Ajuste por Ritmo del Partido (Pace esperado del cruce frente al Pace habitual del equipo del jugador)
    exp_pace = ( float(pace_equipo) + float(pace_rival) ) / 2.0
    factor_pace = float(np.clip(1.0 + (exp_pace - LEAGUE_AVG_PACE) / (1.5 * LEAGUE_AVG_PACE), 0.93, 1.07))

    # 4. Ajuste por Calidad Defensiva del Rival (DRtg global + Defensa vs Posición DvP)
    factor_drtg = float(np.clip(1.0 + (float(drtg_rival) - LEAGUE_AVG_DRTG) / 135.0, 0.90, 1.10))
    pos_grp = perfil.get("pos_group", _norm_pos(perfil.get("position", "G")))

    dvp_pos = {"pts": 1.0, "reb": 1.0, "ast": 1.0, "tpm": 1.0, "rank_pts": 15}
    if dvp_data and rival in dvp_data and pos_grp in dvp_data[rival]:
        dvp_pos = dvp_data[rival][pos_grp]

    # Combinamos el DRtg colectivo del rival (55%) con su DvP específico contra la posición del jugador (45%)
    factor_def_pts = 0.55 * factor_drtg + 0.45 * dvp_pos["pts"]
    factor_def_reb = 0.35 * factor_drtg + 0.65 * dvp_pos["reb"]
    factor_def_ast = 0.50 * factor_drtg + 0.50 * dvp_pos["ast"]
    factor_def_tpm = 0.45 * factor_drtg + 0.55 * dvp_pos["tpm"]

    # 5. Ventaja de Localía individual (~+2.2% en casa, -1.8% en ruta)
    factor_casa = 1.022 if es_local else 0.982

    # 6. Motor Híbrido de Bajas de Compañeros: On/Off Empírico + Usage Vacuum con Rendimientos Decrecientes
    on_off_info = {"splits": [], "emp_mult": {"pts": 1.0, "reb": 1.0, "ast": 1.0, "tpm": 1.0}, "n_off_total": 0}
    if df_box is not None and not df_box.empty and bajas_companeros:
        on_off_info = calcular_on_off_companeros(df_box, perfil["player_name"], perfil["equipo"], bajas_companeros)

    raw_vac_pts = 0.0
    raw_vac_ast = 0.0
    raw_vac_reb = 0.0
    impactos_bajas = []

    for idx_b, baja in enumerate(bajas_companeros):
        pts_b = float(baja.get("pts_avg", 10.0))
        min_b = float(baja.get("min_avg", 15.0))
        reb_b = float(baja.get("reb_avg", 4.0))
        ast_b = float(baja.get("ast_avg", 2.5))
        pos_b = baja.get("pos_group", _norm_pos(baja.get("position", "G")))

        # Peso del volumen vacante ponderado por minutos del ausente y rol del jugador proyectado
        share_afinidad_reb = 1.35 if (pos_b in ("C", "F") and pos_grp in ("C", "F")) else 0.75
        share_afinidad_ast = 1.30 if (pos_b in ("G", "F") and pos_grp in ("G", "F")) else 0.70

        u_factor = float(np.clip(perfil["usage_proxy"] / 0.65, 0.55, 1.35))
        d_p = (pts_b / 135.0) * (min_b / 32.0) * u_factor
        d_a = (ast_b / 42.0) * (min_b / 32.0) * share_afinidad_ast
        d_r = (reb_b / 52.0) * (min_b / 32.0) * share_afinidad_reb

        sp = on_off_info["splits"][idx_b] if idx_b < len(on_off_info["splits"]) else None
        if sp and sp["pj_sin"] > 0 and sp["pj_con"] > 0:
            w_i = min(0.45, sp["pj_sin"] / (sp["pj_sin"] + 12.0))
            r_p_emp = max(0.25 * d_p, ((sp["pts_sin"] / max(sp["min_sin"], 1.0)) / max(sp["pts_con"] / max(sp["min_con"], 1.0), 0.05)) - 1.0)
            r_r_emp = max(0.25 * d_r, ((sp["reb_sin"] / max(sp["min_sin"], 1.0)) / max(sp["reb_con"] / max(sp["min_con"], 1.0), 0.02)) - 1.0)
            r_a_emp = max(0.25 * d_a, ((sp["ast_sin"] / max(sp["min_sin"], 1.0)) / max(sp["ast_con"] / max(sp["min_con"], 1.0), 0.02)) - 1.0)
            d_p = (1.0 - w_i) * d_p + w_i * float(np.clip(r_p_emp, 0.0, 0.18))
            d_r = (1.0 - w_i) * d_r + w_i * float(np.clip(r_r_emp, 0.0, 0.18))
            d_a = (1.0 - w_i) * d_a + w_i * float(np.clip(r_a_emp, 0.0, 0.20))

        raw_vac_pts += d_p
        raw_vac_ast += d_a
        raw_vac_reb += d_r

        impactos_bajas.append({
            "baja": baja["player_name"],
            "pj_sin": sp["pj_sin"] if sp else 0,
            "pts_con": sp["pts_con"] if sp else perfil["pts_avg"],
            "pts_sin": sp["pts_sin"] if sp else perfil["pts_avg"],
        })

    # Saturación cóncava (rendimientos decrecientes + penalización de eficiencia por doble marca defensiva)
    # Techo máximo realista: +20% PTS, +24% AST, +19% REB aunque falten 3 titulares
    boost_pts_baja = 1.0 + 0.20 * (1.0 - np.exp(-raw_vac_pts / 0.16))
    boost_ast_baja = 1.0 + 0.24 * (1.0 - np.exp(-raw_vac_ast / 0.18))
    boost_reb_baja = 1.0 + 0.19 * (1.0 - np.exp(-raw_vac_reb / 0.16))
    boost_tpm_baja = 1.0 + 0.85 * (boost_pts_baja - 1.0)

    # 7. Impacto de Bajas en el Equipo Rival (ej. ausencia del pívot protector de aro o defensor perimetral rival)
    boost_rival_pts = 1.0
    boost_rival_reb = 1.0
    boost_rival_ast = 1.0
    for b_riv in bajas_rivales:
        min_r = float(b_riv.get("min_avg", 15.0))
        reb_r = float(b_riv.get("reb_avg", 4.0))
        pos_r = b_riv.get("pos_group", _norm_pos(b_riv.get("position", "G")))
        share_m = min(min_r / 34.0, 1.0)
        if pos_r == "C":
            # Si falta el pívot rival, se abre la pintura y quedan más rebotes disponibles
            boost_rival_pts += 0.022 * share_m
            boost_rival_reb += 0.045 * share_m * (1.3 if pos_grp in ("C", "F") else 0.7)
            boost_rival_ast += 0.018 * share_m
        elif pos_r == "F":
            boost_rival_pts += 0.018 * share_m
            boost_rival_reb += 0.028 * share_m
            boost_rival_ast += 0.015 * share_m
        else:
            boost_rival_pts += 0.016 * share_m * (1.25 if pos_grp == "G" else 0.85)
            boost_rival_ast += 0.020 * share_m * (1.25 if pos_grp == "G" else 0.85)

    boost_rival_pts = float(np.clip(boost_rival_pts, 1.0, 1.07))
    boost_rival_reb = float(np.clip(boost_rival_reb, 1.0, 1.10))
    boost_rival_ast = float(np.clip(boost_rival_ast, 1.0, 1.07))

    # 8. Proyecciones combinadas antes de H2H
    pts_proy = p_per_min * minutos * factor_pace * factor_def_pts * factor_casa * boost_pts_baja * boost_rival_pts
    reb_proy = r_per_min * minutos * factor_pace * factor_def_reb * factor_casa * boost_reb_baja * boost_rival_reb
    ast_proy = a_per_min * minutos * factor_pace * factor_def_ast * factor_casa * boost_ast_baja * boost_rival_ast
    tpm_proy = t_per_min * minutos * factor_pace * factor_def_tpm * factor_casa * boost_tpm_baja * boost_rival_pts

    # 9. Ajuste Bayesiano por Historial Directo (H2H contra el rival, normalizado a los minutos proyectados)
    n_h2h = 0
    if h2h_games is not None and not h2h_games.empty:
        h2h_valid = h2h_games[h2h_games["min_num"] >= 10.0]
        n_h2h = len(h2h_valid)
        if n_h2h >= 1:
            w_h2h = min(0.20, n_h2h / (n_h2h + 10.0))
            m_h2h = max(float(h2h_valid["min_num"].sum()), 1.0)
            h2h_pts_scaled = (float(h2h_valid["pts_num"].sum()) / m_h2h) * minutos * boost_pts_baja
            h2h_reb_scaled = (float(h2h_valid["reb_num"].sum()) / m_h2h) * minutos * boost_reb_baja
            h2h_ast_scaled = (float(h2h_valid["ast_num"].sum()) / m_h2h) * minutos * boost_ast_baja
            h2h_tpm_scaled = (float(h2h_valid["tpm"].sum()) / m_h2h) * minutos * boost_tpm_baja

            pts_proy = (1.0 - w_h2h) * pts_proy + w_h2h * h2h_pts_scaled
            reb_proy = (1.0 - w_h2h) * reb_proy + w_h2h * h2h_reb_scaled
            ast_proy = (1.0 - w_h2h) * ast_proy + w_h2h * h2h_ast_scaled
            tpm_proy = (1.0 - w_h2h) * tpm_proy + w_h2h * h2h_tpm_scaled

    pra_proy = pts_proy + reb_proy + ast_proy

    # 10. Desviación ajustada para la calculadora de cuotas
    scale = np.sqrt(minutos / max(perfil["min_avg"], 12.0))
    pts_sigma = perfil["pts_std"] * scale
    reb_sigma = perfil["reb_std"] * scale
    ast_sigma = perfil["ast_std"] * scale
    tpm_sigma = perfil["tpm_std"] * scale
    pra_sigma = perfil["pra_std"] * scale

    return {
        "player_name": perfil["player_name"],
        "equipo": perfil["equipo"],
        "rival": rival,
        "pos_group": pos_grp,
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
            "pace": round(factor_pace, 3),
            "exp_pace": round(exp_pace, 1),
            "drtg_rival": round(float(drtg_rival), 1),
            "defensa_rival_pts": round(factor_def_pts, 3),
            "dvp_rank_pts": dvp_pos.get("rank_pts", 15),
            "dvp_pts_mult": round(dvp_pos.get("pts", 1.0), 3),
            "localia": factor_casa,
            "boost_bajas_pts": round(boost_pts_baja, 3),
            "boost_bajas_reb": round(boost_reb_baja, 3),
            "boost_bajas_ast": round(boost_ast_baja, 3),
            "boost_rival_pts": round(boost_rival_pts, 3),
            "boost_rival_reb": round(boost_rival_reb, 3),
            "n_h2h": n_h2h,
            "riesgo_blowout": riesgo_blowout,
            "penalizacion_min_blowout": penalizacion_min_blowout,
            "impactos_bajas": impactos_bajas,
            "on_off_splits": on_off_info["splits"],
        }
    }


def calcular_probabilidades_prop(media, sigma, linea, stat_key="pts", df_games=None):
    """
    Calcula P(Over) y P(Under), cuotas justas y el Hit Rate histórico del jugador para una línea dada.
    - Para conteos discretos bajos (tpm, reb, ast): utiliza Binomial Negativa (sobredispersa) o Poisson
      evaluada exactamente en k = floor(linea), evitando asignar densidad a valores negativos.
    - Para agregados de alto puntaje (pts, pra): utiliza distribución Normal truncada en >= 0 con
      corrección de continuidad.
    """
    linea = float(linea)
    media = max(float(media), 0.1)
    sigma = max(float(sigma), 0.5)
    k_floor = int(np.floor(linea))

    es_discreta = stat_key in ("tpm", "reb", "ast")
    if es_discreta:
        var = sigma ** 2
        # En baloncesto los conteos suelen tener sobredispersión (varianza > media)
        if var > media * 1.08:
            p_nb = float(np.clip(media / var, 0.05, 0.95))
            n_nb = max(media * p_nb / (1.0 - p_nb), 0.5)
            p_under = float(nbinom.cdf(k_floor, n_nb, p_nb))
            max_k = int(max(k_floor + 8, np.ceil(media + 3.5 * sigma)))
            x_grid = np.arange(0, max_k + 1)
            y_grid = nbinom.pmf(x_grid, n_nb, p_nb)
            dist_nombre = "Binomial Negativa (Conteo Discreto)"
        else:
            p_under = float(poisson.cdf(k_floor, mu=media))
            max_k = int(max(k_floor + 8, np.ceil(media + 3.5 * sigma)))
            x_grid = np.arange(0, max_k + 1)
            y_grid = poisson.pmf(x_grid, mu=media)
            dist_nombre = "Poisson (Conteo Discreto)"
        p_over = 1.0 - p_under
    else:
        # Normal truncada en 0 con corrección de continuidad en líneas .5
        cdf_0 = float(norm.cdf(0.0, loc=media, scale=sigma))
        cdf_l = float(norm.cdf(linea, loc=media, scale=sigma))
        p_under = max(0.0, (cdf_l - cdf_0) / max(1.0 - cdf_0, 1e-6))
        p_over = 1.0 - p_under
        x_grid = np.linspace(max(0.0, media - 3.5 * sigma), media + 3.5 * sigma, 200)
        y_grid = norm.pdf(x_grid, loc=media, scale=sigma) / max(1.0 - cdf_0, 1e-6)
        dist_nombre = "Gaussiana Continua"

    p_over = float(np.clip(p_over, 0.02, 0.98))
    p_under = float(np.clip(p_under, 0.02, 0.98))

    cuota_over = round(1.0 / p_over, 2)
    cuota_under = round(1.0 / p_under, 2)

    # Calcular Hit Rate histórico real del jugador sobre esta línea
    col_map = {
        "pts": "pts_num",
        "reb": "reb_num",
        "ast": "ast_num",
        "tpm": "tpm",
        "pra": "pra"
    }
    col_name = col_map.get(stat_key, "pts_num")
    hit_season, n_season, hit_l10, n_l10 = None, 0, None, 0
    ultimos_12 = []

    if df_games is not None and not df_games.empty and col_name in df_games.columns:
        sub_g = df_games[df_games["min_num"] >= 8.0].sort_values("fecha_dt", ascending=False)
        n_season = len(sub_g)
        if n_season > 0:
            hits_s = int((sub_g[col_name] > linea).sum())
            hit_season = hits_s / n_season
            sub_10 = sub_g.head(10)
            n_l10 = len(sub_10)
            hits_10 = int((sub_10[col_name] > linea).sum())
            hit_l10 = hits_10 / n_l10

            sub_12 = sub_g.head(12).iloc[::-1]  # Orden cronológico para gráfico de barras
            for _, r in sub_12.iterrows():
                f_lbl = pd.to_datetime(r["fecha"]).strftime("%d/%m") if pd.notna(r["fecha"]) else ""
                riv_ab = str(r["rival"]).split()[-1][:3].upper()
                ultimos_12.append({
                    "etiqueta": f"{f_lbl}\n{riv_ab}",
                    "valor": float(r[col_name]),
                    "over": bool(float(r[col_name]) > linea)
                })

    return {
        "linea": linea,
        "media_proy": media,
        "prob_over": p_over,
        "prob_under": p_under,
        "cuota_over": cuota_over,
        "cuota_under": cuota_under,
        "es_discreta": es_discreta,
        "dist_nombre": dist_nombre,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "hit_season": hit_season,
        "n_season": n_season,
        "hit_l10": hit_l10,
        "n_l10": n_l10,
        "ultimos_12": ultimos_12,
        "recomendacion": "🔥 OVER CON VALOR" if p_over >= 0.57 else ("🧊 UNDER CON VALOR" if p_under >= 0.57 else "⚖️ LÍNEA JUSTA")
    }
