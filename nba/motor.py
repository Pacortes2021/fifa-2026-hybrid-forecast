"""
Motor predictivo de Machine Learning y simulación de Monte Carlo para la NBA.
Modelado adaptado al baloncesto:
- Sistema Elo NBA (metodología FiveThirtyEight con multiplicador MOV, ventaja de localía de +100 puntos y ajuste por fatiga/back-to-back).
- Factores Four Factors de Dean Oliver (eFG%, TOV%, ORB%, FTRate ofensivos y defensivos).
- Ritmo de juego (Pace) y Ratings de Eficiencia Ofensiva (ORTG), Defensiva (DRTG) y Neta (NetRTG).
- Clasificadores de victoria (Moneyline): LASSO L1, Random Forest, XGBoost y Ensemble Stacking.
- Regresores de hándicap (Spread) y total de puntos (Over/Under).
- Simulador de Monte Carlo para partidos (probabilidades de líneas, hándicaps y cuartos).
- Simulador de Monte Carlo de temporada regular, Play-In y Playoffs de Conferencia/Finales NBA.
"""
from collections import defaultdict, deque
from datetime import datetime
import hashlib
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA = Path(__file__).resolve().parent / "data"
EQUIPOS_PATH = DATA / "equipos.csv"

# Constantes del baloncesto NBA
ELO_INIT = 1500.0
ELO_HOME_ADV = 100.0        # Ventaja estándar de localía en la NBA (~3.0 - 3.5 puntos)
ELO_B2B_PENALTY = 0.0       # El impacto de Back-to-Back y fatiga es aprendido empíricamente por los modelos ML vía variables binarias y rest_diff
LEAGUE_AVG_PACE = 99.2       # Posesiones promedio por 48 minutos en la NBA moderna
LEAGUE_AVG_ORTG = 114.5      # Puntos promedio por 100 posesiones en la NBA moderna


def _default_elo():
    return ELO_INIT


def _default_deque():
    return deque(maxlen=20)


def _default_deque_10():
    return deque(maxlen=10)


def _default_h2h():
    return deque(maxlen=10)


def _default_record():
    return [0, 0]


# Figuras clave / Franquicia Stars de la NBA para simulación de bajas de peso
NBA_FRANCHISE_STARS = {
    "Atlanta Hawks": ["Trae Young", "Jalen Johnson"],
    "Boston Celtics": ["Jayson Tatum", "Jaylen Brown"],
    "Brooklyn Nets": ["Cam Thomas", "Dennis Schröder"],
    "Charlotte Hornets": ["LaMelo Ball", "Brandon Miller"],
    "Chicago Bulls": ["Zach LaVine", "Nikola Vučević"],
    "Cleveland Cavaliers": ["Donovan Mitchell", "Darius Garland"],
    "Dallas Mavericks": ["Luka Dončić", "Kyrie Irving"],
    "Denver Nuggets": ["Nikola Jokić", "Jamal Murray"],
    "Detroit Pistons": ["Cade Cunningham", "Jaden Ivey"],
    "Golden State Warriors": ["Stephen Curry", "Draymond Green"],
    "Houston Rockets": ["Alperen Şengün", "Jalen Green"],
    "Indiana Pacers": ["Tyrese Haliburton", "Pascal Siakam"],
    "LA Clippers": ["Kawhi Leonard", "James Harden"],
    "Los Angeles Lakers": ["LeBron James", "Anthony Davis"],
    "Memphis Grizzlies": ["Ja Morant", "Jaren Jackson Jr."],
    "Miami Heat": ["Jimmy Butler", "Bam Adebayo"],
    "Milwaukee Bucks": ["Giannis Antetokounmpo", "Damian Lillard"],
    "Minnesota Timberwolves": ["Anthony Edwards", "Rudy Gobert"],
    "New Orleans Pelicans": ["Zion Williamson", "Brandon Ingram"],
    "New York Knicks": ["Jalen Brunson", "Karl-Anthony Towns"],
    "Oklahoma City Thunder": ["Shai Gilgeous-Alexander", "Chet Holmgren"],
    "Orlando Magic": ["Paolo Banchero", "Franz Wagner"],
    "Philadelphia 76ers": ["Joel Embiid", "Paul George"],
    "Phoenix Suns": ["Kevin Durant", "Devin Booker"],
    "Portland Trail Blazers": ["Anfernee Simons", "Jerami Grant"],
    "Sacramento Kings": ["De'Aaron Fox", "Domantas Sabonis"],
    "San Antonio Spurs": ["Victor Wembanyama", "Chris Paul"],
    "Toronto Raptors": ["Scottie Barnes", "RJ Barrett"],
    "Utah Jazz": ["Lauri Markkanen", "Collin Sexton"],
    "Washington Wizards": ["Jordan Poole", "Kyle Kuzma"]
}


def _cargar_equipos():
    """Carga los metadatos de las 30 franquicias de la NBA desde equipos.csv."""
    if EQUIPOS_PATH.exists():
        try:
            df = pd.read_csv(EQUIPOS_PATH).set_index("name")
            return df.to_dict("index")
        except Exception:
            return {}
    return {}


def cuota(p, margen=0.0):
    """Calcula cuota decimal justa o con margen."""
    if p is None or p <= 0:
        return 99.0
    p_eff = min(max(p, 0.005), 0.995)
    return round((1.0 / p_eff) * (1.0 + margen), 2)


def cuota_americana(p):
    """Calcula la cuota en formato americano (+150, -130)."""
    if p is None or p <= 0:
        return "+9900"
    if p >= 0.5:
        american = -int(round((p / (1.0 - p)) * 100.0))
        return f"{american}"
    else:
        american = int(round(((1.0 - p) / p) * 100.0))
        return f"+{american}"


class FeatureTrackerNBA:
    """
    Rastreador dinámico de variables estadísticas para la NBA.
    Actualiza de manera causal (sin data leakage) el rating Elo,
    los Four Factors de Dean Oliver, el Pace, ORTG/DRTG y los días de descanso.
    """
    def __init__(self):
        self.elo = defaultdict(_default_elo)
        # Historial reciente de partidos por equipo
        self.history = defaultdict(_default_deque)
        # Historial específico de partidos jugados como local y como visitante
        self.home_history = defaultdict(_default_deque_10)
        self.away_history = defaultdict(_default_deque_10)
        # Fecha del último partido disputado
        self.last_game_date = {}
        # Historial de enfrentamientos directos H2H
        self.h2h = defaultdict(_default_h2h)
        # Racha de victorias (positivo) o derrotas (negativo)
        self.streaks = defaultdict(int)
        # Récord como local y visitante
        self.home_records = defaultdict(_default_record) # [W, L]
        self.away_records = defaultdict(_default_record) # [W, L]
        # Temporada actual para detectar regresión inter-temporada
        self.current_season = None

    def _regresion_temporada(self, season):
        if self.current_season is not None and season != self.current_season:
            # Regresión al promedio histórico de la NBA (FiveThirtyEight methodology: 75% Elo + 25% 1505)
            for eq in list(self.elo.keys()):
                self.elo[eq] = 0.75 * self.elo[eq] + 0.25 * 1505.0
            # Resetear rachas y récords de temporada
            self.streaks.clear()
            self.home_records.clear()
            self.away_records.clear()
            self.home_history.clear()
            self.away_history.clear()
        self.current_season = season

    def get_features_for_match(self, local, visita, fecha=None, temporada=None):
        """Calcula el vector de variables predictivas para un enfrentamiento hipotético o programado."""
        if temporada is not None:
            self._regresion_temporada(temporada)

        # 1. Días de descanso y banderas Back-to-Back (B2B)
        if fecha is not None:
            fecha_dt = pd.to_datetime(fecha)
            last_h = self.last_game_date.get(local)
            last_a = self.last_game_date.get(visita)
            
            rest_h = min((fecha_dt - last_h).days, 7) if last_h is not None else 3
            rest_a = min((fecha_dt - last_a).days, 7) if last_a is not None else 3
        else:
            rest_h = 2
            rest_a = 2

        b2b_h = 1.0 if rest_h <= 1 else 0.0
        b2b_a = 1.0 if rest_a <= 1 else 0.0
        rest_diff = float(rest_h - rest_a)

        # 2. Rating Elo con ventaja de localía y Elo puro
        elo_h_base = self.elo[local]
        elo_a_base = self.elo[visita]
        
        elo_h_adj = elo_h_base + ELO_HOME_ADV - (ELO_B2B_PENALTY if b2b_h else 0.0)
        elo_a_adj = elo_a_base - (ELO_B2B_PENALTY if b2b_a else 0.0)
        elo_diff = elo_h_adj - elo_a_adj
        elo_diff_pure = elo_h_base - elo_a_base
        elo_prob_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        # 3. Métricas rodantes (Four Factors y Ratings)
        def _resumir(hist):
            if not hist:
                return {
                    "ortg": LEAGUE_AVG_ORTG, "drtg": LEAGUE_AVG_ORTG, "netrtg": 0.0,
                    "efg": 0.54, "opp_efg": 0.54, "ft_rate": 0.22, "tpa_rate": 0.38,
                    "pace": LEAGUE_AVG_PACE, "win_pct": 0.50, "ast_to_tov": 1.7
                }
            df_h = pd.DataFrame(hist)
            return {
                "ortg": df_h["ortg"].mean(),
                "drtg": df_h["drtg"].mean(),
                "netrtg": df_h["netrtg"].mean(),
                "efg": df_h["efg"].mean(),
                "opp_efg": df_h["opp_efg"].mean(),
                "ft_rate": df_h["ft_rate"].mean(),
                "tpa_rate": df_h["tpa_rate"].mean(),
                "pace": df_h["pace"].mean(),
                "win_pct": df_h["win"].mean(),
                "ast_to_tov": df_h["ast_to_tov"].mean()
            }

        st_h = _resumir(self.history[local])
        st_a = _resumir(self.history[visita])

        # Ritmo de juego esperado para el partido
        exp_pace = st_h["pace"] + st_a["pace"] - LEAGUE_AVG_PACE

        # 4. Récords en casa y visita globales
        hw, hl = self.home_records[local]
        home_win_pct = hw / (hw + hl) if (hw + hl) > 0 else 0.55
        aw, al = self.away_records[visita]
        away_win_pct = aw / (aw + al) if (aw + al) > 0 else 0.45

        # 5. Medias móviles específicas de localía reciente (L10 en casa vs L10 en ruta)
        hh = self.home_history[local]
        ah = self.away_history[visita]
        l5_home_win_pct = float(np.mean([x["win"] for x in hh])) if len(hh) > 0 else 0.55
        l5_away_win_pct = float(np.mean([x["win"] for x in ah])) if len(ah) > 0 else 0.45
        l5_home_netrtg = float(np.mean([x["netrtg"] for x in hh])) if len(hh) > 0 else 2.5
        l5_away_netrtg = float(np.mean([x["netrtg"] for x in ah])) if len(ah) > 0 else -2.5

        # 6. Tendencias y Momentum (Rendimiento L5 vs L20 partidos)
        all_h = list(self.history[local])
        all_a = list(self.history[visita])
        l5_h = all_h[-5:] if len(all_h) >= 5 else all_h
        l5_a = all_a[-5:] if len(all_a) >= 5 else all_a
        trend_local = (float(np.mean([x["netrtg"] for x in l5_h])) - float(np.mean([x["netrtg"] for x in all_h]))) if len(all_h) >= 5 else 0.0
        trend_visita = (float(np.mean([x["netrtg"] for x in l5_a])) - float(np.mean([x["netrtg"] for x in all_a]))) if len(all_a) >= 5 else 0.0

        # 7. Historial directo (H2H)
        h2h_matches = self.h2h.get((local, visita), deque())
        if h2h_matches:
            h2h_h_wins = sum(1 for w in h2h_matches if w == local)
            h2h_win_pct = h2h_h_wins / len(h2h_matches)
        else:
            h2h_win_pct = 0.5

        return {
            # Ratings Elo
            "elo_local": elo_h_adj,
            "elo_visita": elo_a_adj,
            "elo_diff": elo_diff,
            "elo_diff_pure": elo_diff_pure,
            "elo_prob_local": elo_prob_h,
            # Descanso y fatiga
            "rest_local": float(rest_h),
            "rest_visita": float(rest_a),
            "b2b_local": b2b_h,
            "b2b_visita": b2b_a,
            "rest_diff": rest_diff,
            # Medias móviles y tendencias de localía/visita
            "l5_home_win_pct": l5_home_win_pct,
            "l5_away_win_pct": l5_away_win_pct,
            "l5_home_netrtg": l5_home_netrtg,
            "l5_away_netrtg": l5_away_netrtg,
            "trend_local": trend_local,
            "trend_visita": trend_visita,
            # Indicadores de ausencia de jugadores estrella
            "star_missing_local": 0.0,
            "star_missing_visita": 0.0,
            # Dean Oliver Four Factors & Ratings
            "ortg_local": st_h["ortg"],
            "drtg_local": st_h["drtg"],
            "netrtg_local": st_h["netrtg"],
            "ortg_visita": st_a["ortg"],
            "drtg_visita": st_a["drtg"],
            "netrtg_visita": st_a["netrtg"],
            "netrtg_diff": st_h["netrtg"] - st_a["netrtg"],
            "efg_local": st_h["efg"],
            "opp_efg_local": st_h["opp_efg"],
            "efg_visita": st_a["efg"],
            "opp_efg_visita": st_a["opp_efg"],
            "efg_diff": st_h["efg"] - st_a["opp_efg"],
            "ft_rate_local": st_h["ft_rate"],
            "ft_rate_visita": st_a["ft_rate"],
            "tpa_rate_local": st_h["tpa_rate"],
            "tpa_rate_visita": st_a["tpa_rate"],
            # Ritmo (Pace)
            "pace_local": st_h["pace"],
            "pace_visita": st_a["pace"],
            "exp_game_pace": exp_pace,
            # Forma reciente y rachas
            "win_pct_local": st_h["win_pct"],
            "win_pct_visita": st_a["win_pct"],
            "win_pct_diff": st_h["win_pct"] - st_a["win_pct"],
            "streak_local": float(self.streaks[local]),
            "streak_visita": float(self.streaks[visita]),
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "h2h_win_pct": h2h_win_pct
        }

    def update(self, row):
        """Actualiza el estado interno con el resultado de un partido finalizado."""
        l = row["local"]
        v = row["visita"]
        pl = float(row["puntos_local"])
        pv = float(row["puntos_visita"])
        fecha = pd.to_datetime(row["fecha"])
        season = row.get("temporada")
        
        self._regresion_temporada(season)

        # 1. Actualización de Elo con multiplicador MOV (FiveThirtyEight)
        elo_h = self.elo[l]
        elo_v = self.elo[v]
        home_adv = ELO_HOME_ADV
        elo_diff = (elo_h + home_adv) - elo_v
        e_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        e_v = 1.0 - e_h

        win_h = 1.0 if pl > pv else 0.0
        mov = abs(pl - pv)
        elo_winner_diff = abs(elo_h - elo_v) if win_h else abs(elo_v - elo_h)
        k_factor = 20.0 * ((mov + 3.0) ** 0.8) / (7.5 + 0.006 * elo_winner_diff)

        self.elo[l] += k_factor * (win_h - e_h)
        self.elo[v] += k_factor * ((1.0 - win_h) - e_v)

        # 2. Posesiones y Ratings
        fga_h = float(row.get("fga_local") or 88.0)
        fta_h = float(row.get("fta_local") or 22.0)
        fga_v = float(row.get("fga_visita") or 88.0)
        fta_v = float(row.get("fta_visita") or 22.0)
        pace = float(row.get("est_pace") or (0.5 * ((fga_h + 0.44 * fta_h) + (fga_v + 0.44 * fta_v))))
        if pace <= 60 or pace >= 140:
            pace = LEAGUE_AVG_PACE

        ortg_h = (pl / pace) * 100.0
        drtg_h = (pv / pace) * 100.0
        ortg_v = (pv / pace) * 100.0
        drtg_v = (pl / pace) * 100.0

        efg_h = float(row.get("efg_local") or 0.53)
        efg_v = float(row.get("efg_visita") or 0.53)
        ft_rate_h = float(row.get("ft_rate_local") or (fta_h / max(fga_h, 1.0)))
        ft_rate_v = float(row.get("ft_rate_visita") or (fta_v / max(fga_v, 1.0)))
        
        tpa_h = float(row.get("tpa_local") or 35.0)
        tpa_v = float(row.get("tpa_visita") or 35.0)
        tpa_rate_h = tpa_h / max(fga_h, 1.0)
        tpa_rate_v = tpa_v / max(fga_v, 1.0)

        # Registrar métricas rodantes
        self.history[l].append({
            "ortg": ortg_h, "drtg": drtg_h, "netrtg": ortg_h - drtg_h,
            "efg": efg_h, "opp_efg": efg_v, "ft_rate": ft_rate_h,
            "tpa_rate": tpa_rate_h, "pace": pace, "win": win_h,
            "ast_to_tov": float(row.get("ast_local") or 25.0) / 14.0
        })
        self.history[v].append({
            "ortg": ortg_v, "drtg": drtg_v, "netrtg": ortg_v - drtg_v,
            "efg": efg_v, "opp_efg": efg_h, "ft_rate": ft_rate_v,
            "tpa_rate": tpa_rate_v, "pace": pace, "win": 1.0 - win_h,
            "ast_to_tov": float(row.get("ast_visita") or 25.0) / 14.0
        })

        self.home_history[l].append({"win": win_h, "netrtg": ortg_h - drtg_h})
        self.away_history[v].append({"win": 1.0 - win_h, "netrtg": ortg_v - drtg_v})

        # Fechas, rachas y récords
        self.last_game_date[l] = fecha
        self.last_game_date[v] = fecha
        self.h2h[(l, v)].append(l if win_h else v)
        self.h2h[(v, l)].append(l if win_h else v)

        if win_h:
            self.streaks[l] = self.streaks[l] + 1 if self.streaks[l] > 0 else 1
            self.streaks[v] = self.streaks[v] - 1 if self.streaks[v] < 0 else -1
            self.home_records[l][0] += 1
            self.away_records[v][1] += 1
        else:
            self.streaks[l] = self.streaks[l] - 1 if self.streaks[l] < 0 else -1
            self.streaks[v] = self.streaks[v] + 1 if self.streaks[v] > 0 else 1
            self.home_records[l][1] += 1
            self.away_records[v][0] += 1


def construir_dataset(partidos_df):
    """Genera la matriz de características X e y procesando secuencialmente cada partido."""
    tracker = FeatureTrackerNBA()
    filas_feat = []

    partidos_sorted = partidos_df.sort_values("fecha").reset_index(drop=True)
    for r in partidos_sorted.itertuples(index=False):
        row_dict = r._asdict()
        feats = tracker.get_features_for_match(
            r.local, r.visita, r.fecha, getattr(r, "temporada", None)
        )
        feats.update({
            "event_id": str(r.event_id),
            "fecha": r.fecha,
            "temporada": r.temporada,
            "local": r.local,
            "visita": r.visita,
            "puntos_local": float(r.puntos_local),
            "puntos_visita": float(r.puntos_visita),
            "diferencial": float(r.diferencial),
            "total_puntos": float(r.total_puntos),
            "target_win_local": 1 if r.puntos_local > r.puntos_visita else 0
        })
        filas_feat.append(feats)
        tracker.update(row_dict)

    df_out = pd.DataFrame(filas_feat)
    return df_out, tracker


def entrenar_modelos():
    """Entrena los clasificadores de victoria y los regresores de margen y totales."""
    partidos_path = DATA / "partidos.csv"
    if not partidos_path.exists():
        raise FileNotFoundError("partidos.csv no encontrado en nba/data")

    df_partidos = pd.read_csv(partidos_path)
    print(f"Cargados {len(df_partidos)} partidos. Extrayendo variables de baloncesto...")
    df_dataset, tracker = construir_dataset(df_partidos)

    cols_excluir = {
        "event_id", "fecha", "temporada", "local", "visita",
        "puntos_local", "puntos_visita", "diferencial", "total_puntos", "target_win_local"
    }
    features = [c for c in df_dataset.columns if c not in cols_excluir]

    # Partición temporal de validación out-of-sample:
    # Train: temporadas <= 2024 (2022-23 y 2023-24)
    # Calibración: temporada 2025 (2024-25)
    # Test out-of-sample: temporadas >= 2026 (2025-26 y actual 2026-27)
    train_mask = df_dataset["temporada"] <= 2024
    cal_mask = df_dataset["temporada"] == 2025
    test_mask = df_dataset["temporada"] >= 2026

    X_train = df_dataset.loc[train_mask, features].fillna(0.0)
    y_train = df_dataset.loc[train_mask, "target_win_local"]
    diff_train = df_dataset.loc[train_mask, "diferencial"]
    tot_train = df_dataset.loc[train_mask, "total_puntos"]

    X_cal = df_dataset.loc[cal_mask, features].fillna(0.0)
    y_cal = df_dataset.loc[cal_mask, "target_win_local"]
    diff_cal = df_dataset.loc[cal_mask, "diferencial"]
    tot_cal = df_dataset.loc[cal_mask, "total_puntos"]

    X_test = df_dataset.loc[test_mask, features].fillna(0.0)
    y_test = df_dataset.loc[test_mask, "target_win_local"]
    diff_test = df_dataset.loc[test_mask, "diferencial"]
    tot_test = df_dataset.loc[test_mask, "total_puntos"]

    print(f"Particiones temporales: Train={len(X_train)}, Calibración={len(X_cal)}, Test={len(X_test)}")

    # 1. LASSO Logistic Regression (L1) con optimización de regularización
    best_c = 0.1
    best_score = float("inf")
    for c_val in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="saga", C=c_val, max_iter=2000, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds_cal = np.clip(pipe.predict_proba(X_cal), 1e-6, 1.0 - 1e-6)
        ll = log_loss(y_cal, preds_cal)
        if ll < best_score:
            best_score = ll
            best_c = c_val

    # Entrenar clasificadores sobre Train + Calibración
    X_full = df_dataset.loc[train_mask | cal_mask, features].fillna(0.0)
    y_full = df_dataset.loc[train_mask | cal_mask, "target_win_local"]
    diff_full = df_dataset.loc[train_mask | cal_mask, "diferencial"]
    tot_full = df_dataset.loc[train_mask | cal_mask, "total_puntos"]

    pipe_lasso = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=2500, random_state=42))
    ])
    pipe_lasso.fit(X_full, y_full)

    # 2. Random Forest Classifier
    pipe_rf = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=6, random_state=42, n_jobs=-1))
    ])
    pipe_rf.fit(X_full, y_full)

    # 3. XGBoost Classifier
    pipe_xgb = Pipeline([
        ("sc", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="logloss"
        ))
    ])
    pipe_xgb.fit(X_full, y_full)

    # 4. Ensamble Stacking Óptimo (minimización de Log-Loss)
    p_lasso_cal = pipe_lasso.predict_proba(X_cal)[:, 1]
    p_rf_cal = pipe_rf.predict_proba(X_cal)[:, 1]
    p_xgb_cal = pipe_xgb.predict_proba(X_cal)[:, 1]

    def _loss_stack(weights):
        w = np.abs(weights)
        w = w / w.sum()
        p_mix = w[0] * p_lasso_cal + w[1] * p_rf_cal + w[2] * p_xgb_cal
        p_mix = np.clip(p_mix, 1e-6, 1.0 - 1e-6)
        return log_loss(y_cal, p_mix)

    res_opt = minimize(_loss_stack, [0.35, 0.35, 0.30], method="Nelder-Mead")
    stack_w = np.abs(res_opt.x)
    stack_w = stack_w / stack_w.sum()

    # 5. Regresores de Margen (Spread) y Totales (Over/Under)
    pipe_spread = Pipeline([
        ("sc", StandardScaler()),
        ("ridge", Ridge(alpha=15.0, random_state=42))
    ])
    pipe_spread.fit(X_full, diff_full)

    pipe_totals = Pipeline([
        ("sc", StandardScaler()),
        ("ridge", Ridge(alpha=15.0, random_state=42))
    ])
    pipe_totals.fit(X_full, tot_full)

    # Evaluación y métricas Out-of-Sample (Test >= 2026)
    def _eval_clf(pipe):
        prob = np.clip(pipe.predict_proba(X_test), 1e-6, 1.0 - 1e-6)
        pred = (prob[:, 1] >= 0.5).astype(int)
        return {
            "logloss": round(float(log_loss(y_test, prob)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred) * 100.0), 2),
            "brier": round(float(brier_score_loss(y_test, prob[:, 1])), 4)
        }

    met_lasso = _eval_clf(pipe_lasso)
    met_rf = _eval_clf(pipe_rf)
    met_xgb = _eval_clf(pipe_xgb)

    p_lasso_t = pipe_lasso.predict_proba(X_test)[:, 1]
    p_rf_t = pipe_rf.predict_proba(X_test)[:, 1]
    p_xgb_t = pipe_xgb.predict_proba(X_test)[:, 1]
    p_stack_t = np.clip(stack_w[0] * p_lasso_t + stack_w[1] * p_rf_t + stack_w[2] * p_xgb_t, 1e-6, 1.0 - 1e-6)
    pred_stack_t = (p_stack_t >= 0.5).astype(int)

    # Baseline de frecuencias en Test
    p_base = float(y_test.mean())
    ll_base = float(log_loss(y_test, [p_base] * len(y_test)))

    pred_diff_test = pipe_spread.predict(X_test)
    pred_tot_test = pipe_totals.predict(X_test)

    # Desviaciones estándar empíricas del error de predicción
    sigma_spread = float(np.std(diff_test - pred_diff_test))
    sigma_totals = float(np.std(tot_test - pred_tot_test))

    met_stack = {
        "logloss": round(float(log_loss(y_test, p_stack_t)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred_stack_t) * 100.0), 2),
        "brier": round(float(brier_score_loss(y_test, p_stack_t)), 4),
        "logloss_base": round(ll_base, 4),
        "w": [round(float(w), 3) for w in stack_w],
        "mae_spread": round(float(mean_absolute_error(diff_test, pred_diff_test)), 2),
        "rmse_spread": round(float(np.sqrt(mean_squared_error(diff_test, pred_diff_test))), 2),
        "mae_totals": round(float(mean_absolute_error(tot_test, pred_tot_test)), 2),
        "rmse_totals": round(float(np.sqrt(mean_squared_error(tot_test, pred_tot_test))), 2),
        "sigma_spread": round(sigma_spread, 2),
        "sigma_totals": round(sigma_totals, 2)
    }

    # Importancia de variables desde LASSO y RF
    coefs_lasso = pipe_lasso.named_steps["lr"].coef_[0]
    importancias_rf = pipe_rf.named_steps["rf"].feature_importances_
    features_rank = []
    for f, c_val, imp in sorted(zip(features, coefs_lasso, importancias_rf), key=lambda x: abs(x[1]), reverse=True):
        features_rank.append({
            "feature": f,
            "coef_lasso": round(float(c_val), 4),
            "importancia_rf": round(float(imp), 4)
        })

    metricas = {
        "lasso": met_lasso,
        "rf": met_rf,
        "xgb": met_xgb,
        "stacking": met_stack,
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
        "features_rank": features_rank
    }

    print(f"Métricas Test (2025-26+): LASSO={met_lasso} | RF={met_rf} | XGB={met_xgb} | Stacking={met_stack}")

    return {
        "tracker": tracker,
        "equipos": _cargar_equipos(),
        "features": features,
        "pipe_lasso": pipe_lasso,
        "pipe_rf": pipe_rf,
        "pipe_xgb": pipe_xgb,
        "stack_w": stack_w,
        "pipe_spread": pipe_spread,
        "pipe_totals": pipe_totals,
        "sigma_spread": sigma_spread,
        "sigma_totals": sigma_totals,
        "metricas": metricas,
        "df_dataset": df_dataset
    }


def _cache_key():
    """Genera hash único dependiente del código del motor y de los archivos CSV de datos."""
    h = hashlib.sha256()
    h.update(hashlib.sha256(Path(__file__).read_bytes()).hexdigest().encode())
    for f in sorted(DATA.glob("*.csv")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode())
    return h.hexdigest()


def cargar(use_cache=True):
    """Carga el modelo entrenado desde el cache de disco o re-entrena si los datos han cambiado."""
    cache_path = Path(__file__).resolve().parent / ".model_cache.pkl"
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("key") == _cache_key():
                return saved["M"]
        except Exception as ex:
            print(f"Cache inválido ({ex}); re-entrenando...")

    M = entrenar_modelos()
    if use_cache:
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump({"key": _cache_key(), "M": M}, fh)
        except Exception as ex:
            print(f"No se pudo guardar el cache: {ex}")
    return M


def predecir_match(M, local, visita, fecha=None, temporada=None, modelo="stacking",
                   baja_estrella_local=False, baja_estrella_visita=False):
    """
    Predice el ganador, cuotas justas, margen esperado y totales de puntos
    para un enfrentamiento entre local y visita.
    Permite simular el impacto analítico de la baja de figuras de peso (superstar absence).
    """
    tracker = M["tracker"]
    features = M["features"]
    
    feats = tracker.get_features_for_match(local, visita, fecha, temporada)

    # Impacto de ausencia de figura estrella (ajuste empírico de ratings y Four Factors)
    if baja_estrella_local:
        feats["star_missing_local"] = 1.0
        feats["elo_local"] = feats["elo_local"] - 75.0
        feats["elo_diff"] = feats["elo_diff"] - 75.0
        feats["elo_diff_pure"] = feats["elo_diff_pure"] - 75.0
        feats["netrtg_local"] = feats["netrtg_local"] - 5.5
        feats["netrtg_diff"] = feats["netrtg_diff"] - 5.5
        feats["ortg_local"] = feats["ortg_local"] - 5.0
        feats["l5_home_netrtg"] = feats["l5_home_netrtg"] - 5.5
        feats["efg_local"] = max(0.40, feats["efg_local"] - 0.025)

    if baja_estrella_visita:
        feats["star_missing_visita"] = 1.0
        feats["elo_visita"] = feats["elo_visita"] - 75.0
        feats["elo_diff"] = feats["elo_diff"] + 75.0
        feats["elo_diff_pure"] = feats["elo_diff_pure"] + 75.0
        feats["netrtg_visita"] = feats["netrtg_visita"] - 5.5
        feats["netrtg_diff"] = feats["netrtg_diff"] + 5.5
        feats["ortg_visita"] = feats["ortg_visita"] - 5.0
        feats["l5_away_netrtg"] = feats["l5_away_netrtg"] - 5.5
        feats["efg_visita"] = max(0.40, feats["efg_visita"] - 0.025)

    df_feat = pd.DataFrame([feats])[features].fillna(0.0)

    # Probabilidades de victoria
    if modelo == "rf":
        p_home = float(M["pipe_rf"].predict_proba(df_feat)[0, 1])
    elif modelo == "lasso":
        p_home = float(M["pipe_lasso"].predict_proba(df_feat)[0, 1])
    elif modelo == "xgb":
        p_home = float(M["pipe_xgb"].predict_proba(df_feat)[0, 1])
    else: # Stacking
        w = M["stack_w"]
        p_l = M["pipe_lasso"].predict_proba(df_feat)[0, 1]
        p_r = M["pipe_rf"].predict_proba(df_feat)[0, 1]
        p_x = M["pipe_xgb"].predict_proba(df_feat)[0, 1]
        p_home = float(np.clip(w[0] * p_l + w[1] * p_r + w[2] * p_x, 0.005, 0.995))

    p_away = 1.0 - p_home

    # Proyecciones numéricas de Puntos y Spread
    proj_spread = float(M["pipe_spread"].predict(df_feat)[0]) # Local - Visita
    proj_total = float(M["pipe_totals"].predict(df_feat)[0])  # Local + Visita

    proj_pts_local = round((proj_total + proj_spread) / 2.0, 1)
    proj_pts_visita = round((proj_total - proj_spread) / 2.0, 1)

    return {
        "p_local": p_home,
        "p_visita": p_away,
        "cuota_local": cuota(p_home),
        "cuota_visita": cuota(p_away),
        "cuota_am_local": cuota_americana(p_home),
        "cuota_am_visita": cuota_americana(p_away),
        "proj_pts_local": proj_pts_local,
        "proj_pts_visita": proj_pts_visita,
        "proj_spread": round(proj_spread, 1),
        "proj_total": round(proj_total, 1),
        "feats": feats,
        "baja_estrella_local": baja_estrella_local,
        "baja_estrella_visita": baja_estrella_visita
    }


def simular_partido_mc(M, local, visita, n_sims=5000, modelo="stacking", seed=42,
                       baja_estrella_local=False, baja_estrella_visita=False):
    """
    Simulación Monte Carlo de un partido de la NBA.
    Genera la distribución exacta de puntuaciones, probabilidades para
    líneas de Hándicap (Spread) y totales Over/Under.
    """
    if seed is not None:
        np.random.seed(seed)

    pred = predecir_match(
        M, local, visita, modelo=modelo,
        baja_estrella_local=baja_estrella_local,
        baja_estrella_visita=baja_estrella_visita
    )
    mu_spread = pred["proj_spread"]
    mu_total = pred["proj_total"]
    sigma_spread = M.get("sigma_spread", 12.5)
    sigma_tot = M.get("sigma_totals", 18.5)

    # Simular diferenciales y totales de puntos con ruido gaussiano
    diff_sims = np.random.normal(loc=mu_spread, scale=sigma_spread, size=n_sims)
    tot_sims = np.random.normal(loc=mu_total, scale=sigma_tot, size=n_sims)

    pts_h_sims = np.round((tot_sims + diff_sims) / 2.0)
    pts_a_sims = np.round((tot_sims - diff_sims) / 2.0)

    # En caso de empate en tiempo regular, simular prórroga (+5 pts aprox por equipo)
    empates = (pts_h_sims == pts_a_sims)
    if np.any(empates):
        n_emp = np.sum(empates)
        ot_res = np.random.choice([1, -1], size=n_emp, p=[pred["p_local"], pred["p_visita"]])
        ot_h = np.random.randint(6, 15, size=n_emp)
        ot_a = ot_h - ot_res * np.random.randint(1, 5, size=n_emp)
        pts_h_sims[empates] += ot_h
        pts_a_sims[empates] += ot_a

    # Probabilidades de líneas de hándicap comunes en NBA
    lineas_spread = [-11.5, -9.5, -7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5]
    probs_spread = {}
    for line in lineas_spread:
        # Cubrir línea: (pts_local - pts_visita) > -line
        cubierto = np.mean((pts_h_sims - pts_a_sims) > line)
        probs_spread[line] = round(float(cubierto), 3)

    # Probabilidades de líneas de Over/Under comunes en NBA
    lineas_ou = [210.5, 215.5, 220.5, 225.5, 230.5, 235.5, 240.5, 245.5]
    probs_ou = {}
    for ou in lineas_ou:
        over = np.mean((pts_h_sims + pts_a_sims) > ou)
        probs_ou[ou] = round(float(over), 3)

    # Proyección de anotación por cuartos (distribución proporcional)
    q_weights = [0.255, 0.250, 0.245, 0.250]
    cuartos_h = [round(pred["proj_pts_local"] * w, 1) for w in q_weights]
    cuartos_a = [round(pred["proj_pts_visita"] * w, 1) for w in q_weights]

    return {
        "pred": pred,
        "n_sims": n_sims,
        "win_pct_local_mc": round(float(np.mean(pts_h_sims > pts_a_sims)), 3),
        "win_pct_visita_mc": round(float(np.mean(pts_a_sims > pts_h_sims)), 3),
        "probs_spread": probs_spread,
        "probs_ou": probs_ou,
        "cuartos_local": cuartos_h,
        "cuartos_visita": cuartos_a,
        "pts_h_sims": pts_h_sims,
        "pts_a_sims": pts_a_sims
    }


def simular_temporada(M, n_sims=2000, seed=42):
    """
    Simulación Monte Carlo completa de la temporada NBA, Play-In y Playoffs.
    Proyecta las victorias esperadas en Conferencia Este y Oeste,
    posibilidades de Play-In (7-10), Playoff directo (1-6) y Campeón de la NBA.
    """
    cache_path = Path(__file__).resolve().parent / "simulacion_mc.pkl"
    fp = hashlib.sha256()
    fp.update(str(n_sims).encode())
    fp.update(_cache_key().encode())
    cache_key = fp.hexdigest()

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("key") == cache_key:
                return saved["resultados"]
        except Exception:
            pass

    if seed is not None:
        np.random.seed(seed)

    fixture_path = DATA / "fixture.csv"
    partidos_path = DATA / "partidos.csv"
    equipos = M.get("equipos", {})

    df_partidos = pd.read_csv(partidos_path)
    temp_actual = df_partidos["temporada"].max()
    partidos_curr = df_partidos[df_partidos["temporada"] == temp_actual]
    df_fix = pd.read_csv(fixture_path) if fixture_path.exists() else pd.DataFrame()

    # Récords actuales reales de la temporada
    wins_actual = defaultdict(int)
    losses_actual = defaultdict(int)
    for r in partidos_curr.itertuples():
        if r.puntos_local > r.puntos_visita:
            wins_actual[r.local] += 1
            losses_actual[r.visita] += 1
        else:
            wins_actual[r.visita] += 1
            losses_actual[r.local] += 1

    # Todos los equipos activos
    todos_equipos = sorted(list(equipos.keys()))
    if not todos_equipos:
        todos_equipos = sorted(list(set(df_partidos["local"].unique())))

    # Pre-computar probabilidades entre todos los equipos para simulación ultrarrápida
    h2h_probs = {}
    for eq1 in todos_equipos:
        for eq2 in todos_equipos:
            if eq1 != eq2:
                h2h_probs[(eq1, eq2)] = predecir_match(M, eq1, eq2)["p_local"]

    # Predecir probabilidades de cada partido restante del fixture (filtrando equipos válidos)
    preds_fixture = []
    for r in df_fix.itertuples():
        loc = getattr(r, "local", "")
        vis = getattr(r, "visita", "")
        if loc in todos_equipos and vis in todos_equipos:
            p_local = h2h_probs.get((loc, vis), 0.55)
            preds_fixture.append((loc, vis, p_local))

    conferencias = {}
    for eq in todos_equipos:
        conf = equipos.get(eq, {}).get("conference", "Eastern Conference")
        conferencias[eq] = "East" if "East" in conf else "West"

    # Acumuladores de Monte Carlo
    tot_wins = defaultdict(list)
    playin_counts = defaultdict(int)       # 7º a 10º
    playoff_counts = defaultdict(int)      # 1º a 6º directo + clasificación Play-In
    conf_final_counts = defaultdict(int)   # Finalistas de Conferencia
    champ_counts = defaultdict(int)        # Campeón NBA

    for s_idx in range(n_sims):
        sim_wins = {e: wins_actual[e] for e in todos_equipos}
        
        # Simular partidos del fixture
        if preds_fixture:
            probs = np.array([p for _, _, p in preds_fixture])
            rnd = np.random.uniform(size=len(probs))
            home_wins = rnd < probs
            for i, (loc, vis, _) in enumerate(preds_fixture):
                if home_wins[i]:
                    sim_wins[loc] += 1
                else:
                    sim_wins[vis] += 1

        for e in todos_equipos:
            tot_wins[e].append(sim_wins[e])

        # Ranquear por conferencia (East y West)
        east_teams = sorted([e for e in todos_equipos if conferencias[e] == "East"], key=lambda x: sim_wins[x], reverse=True)
        west_teams = sorted([e for e in todos_equipos if conferencias[e] == "West"], key=lambda x: sim_wins[x], reverse=True)

        for conf_ranked in (east_teams, west_teams):
            for i, eq in enumerate(conf_ranked):
                if i < 6:
                    playoff_counts[eq] += 1
                elif i < 10:
                    playin_counts[eq] += 1

        # Simular Play-In Tournament
        def _sim_playin(ranked):
            t7, t8, t9, t10 = ranked[6], ranked[7], ranked[8], ranked[9]
            # 7 vs 8 (ganador = seed 7)
            p78 = h2h_probs.get((t7, t8), 0.55)
            w_78 = t7 if np.random.random() < p78 else t8
            l_78 = t8 if w_78 == t7 else t7
            # 9 vs 10 (perdedor eliminado)
            p910 = h2h_probs.get((t9, t10), 0.55)
            w_910 = t9 if np.random.random() < p910 else t10
            # Partido final por seed 8: l_78 vs w_910
            p_final = h2h_probs.get((l_78, w_910), 0.55)
            w_seed8 = l_78 if np.random.random() < p_final else w_910
            return ranked[:6] + [w_78, w_seed8]

        east_playoffs = _sim_playin(east_teams)
        west_playoffs = _sim_playin(west_teams)

        for eq in [east_playoffs[6], east_playoffs[7], west_playoffs[6], west_playoffs[7]]:
            playoff_counts[eq] += 1

        # Simular Serie al Mejor de 7
        def _sim_serie_7(t_home, t_away):
            p_h = h2h_probs.get((t_home, t_away), 0.55)
            p_a_opp = 1.0 - h2h_probs.get((t_away, t_home), 0.55)
            # Formato 2-2-1-1-1 (Home juega 1, 2, 5, 7 en casa)
            w_h = 0
            w_a = 0
            for g in range(7):
                is_home = g in (0, 1, 4, 6)
                p = p_h if is_home else p_a_opp
                if np.random.random() < p:
                    w_h += 1
                else:
                    w_a += 1
                if w_h == 4:
                    return t_home
                if w_a == 4:
                    return t_away
            return t_home

        def _sim_conf_playoffs(seeds):
            # 1 vs 8, 4 vs 5, 2 vs 7, 3 vs 6
            r1 = _sim_serie_7(seeds[0], seeds[7])
            r2 = _sim_serie_7(seeds[3], seeds[4])
            r3 = _sim_serie_7(seeds[1], seeds[6])
            r4 = _sim_serie_7(seeds[2], seeds[5])
            # Semifinales
            sem1 = _sim_serie_7(r1, r2)
            sem2 = _sim_serie_7(r3, r4)
            # Final de Conferencia
            conf_champ = _sim_serie_7(sem1, sem2)
            return conf_champ, sem1, sem2

        east_champ, e_f1, e_f2 = _sim_conf_playoffs(east_playoffs)
        west_champ, w_f1, w_f2 = _sim_conf_playoffs(west_playoffs)

        conf_final_counts[e_f1] += 1
        conf_final_counts[e_f2] += 1
        conf_final_counts[w_f1] += 1
        conf_final_counts[w_f2] += 1

        # Finales de la NBA: East Champion vs West Champion
        # Ventaja de localía para el de más victorias
        if sim_wins[east_champ] >= sim_wins[west_champ]:
            nba_champ = _sim_serie_7(east_champ, west_champ)
        else:
            nba_champ = _sim_serie_7(west_champ, east_champ)

        champ_counts[nba_champ] += 1

    # Compilar resultados finales
    filas = []
    for eq in todos_equipos:
        avg_w = float(np.mean(tot_wins[eq]))
        filas.append({
            "Equipo": eq,
            "Conferencia": conferencias[eq],
            "Victorias Esperadas": round(avg_w, 1),
            "Derrotas Esperadas": round(82.0 - avg_w, 1),
            "P_Playoff_Directo": round(playoff_counts[eq] / n_sims, 3),
            "P_PlayIn": round(playin_counts[eq] / n_sims, 3),
            "P_Finalista_Conf": round(conf_final_counts[eq] / n_sims, 3),
            "P_Campeon": round(champ_counts[eq] / n_sims, 3)
        })

    df_res = pd.DataFrame(filas).sort_values(by="Victorias Esperadas", ascending=False).reset_index(drop=True)
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": cache_key, "resultados": df_res}, fh)
    except Exception:
        pass

    return df_res


def validacion_en_vivo(M, temporada=2027):
    """
    Evalúa las predicciones del modelo frente a los partidos reales disputados
    en la temporada actual (validación continua out-of-sample).
    """
    df_dataset = M["df_dataset"]
    df_val = df_dataset[df_dataset["temporada"] == temporada].copy()
    if df_val.empty:
        # Fallback a última temporada si 2027 aún no tiene partidos
        df_val = df_dataset[df_dataset["temporada"] == df_dataset["temporada"].max()].copy()

    features = M["features"]
    pipe = M["pipe_lasso"] # Modelo base de referencia calibrado temporalmente

    X_val = df_val[features].fillna(0.0)
    y_val = df_val["target_win_local"]

    probs = pipe.predict_proba(X_val)
    preds = pipe.predict(X_val)

    ll = log_loss(y_val, probs)
    acc = accuracy_score(y_val, preds)

    # Baseline empírico
    freq = y_val.mean()
    base_probs = np.full_like(probs, 0.5)
    base_probs[:, 1] = freq
    base_probs[:, 0] = 1.0 - freq
    ll_base = log_loss(y_val, base_probs)

    met = {
        "n": len(df_val),
        "acierto": round(float(acc * 100.0), 2),
        "logloss": round(float(ll), 4),
        "logloss_base": round(float(ll_base), 4),
        "brier": round(float(brier_score_loss(y_val, probs[:, 1])), 4)
    }

    df_val["Prob_Local"] = probs[:, 1]
    df_val["Prob_Visita"] = probs[:, 0]
    df_val["Prediccion"] = preds

    # Evolución temporal acumulada de Log-Loss
    log_losses = []
    for i in range(1, len(df_val) + 1):
        sub_y = df_val["target_win_local"].iloc[:i]
        sub_p = probs[:i]
        try:
            log_losses.append(log_loss(sub_y, sub_p))
        except Exception:
            log_losses.append(np.nan)

    evol = pd.DataFrame({
        "partido": range(1, len(df_val) + 1),
        "logloss_acum": log_losses
    })

    return df_val, met, evol


if __name__ == "__main__":
    M = cargar(use_cache=False)
    print("\n--- PRUEBA DE PREDICCIÓN ---")
    p = predecir_match(M, "Boston Celtics", "Los Angeles Lakers")
    print("Boston Celtics vs Los Angeles Lakers:")
    print(f"Probabilidades: Local {p['p_local']:.1%} | Visita {p['p_visita']:.1%}")
    print(f"Cuotas: {p['cuota_local']} ({p['cuota_am_local']}) vs {p['cuota_visita']} ({p['cuota_am_visita']})")
    print(f"Puntos proyectados: {p['proj_pts_local']} - {p['proj_pts_visita']} (Spread: {p['proj_spread']}, Total: {p['proj_total']})")
