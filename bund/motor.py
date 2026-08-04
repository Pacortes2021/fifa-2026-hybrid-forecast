import sys
import math
import pickle
from pathlib import Path
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize_scalar, minimize

DATA = Path(__file__).resolve().parent / "data"
EQUIPOS_PATH = DATA / "equipos.csv"

def _cargar_equipos():
    """Lee data/equipos.csv (id estable de ESPN -> info del club: nombre, abreviatura, colores, logo)."""
    if EQUIPOS_PATH.exists():
        try:
            df = pd.read_csv(EQUIPOS_PATH).set_index("id")
            return df.to_dict("index")
        except Exception:
            return {}
    return {}



# Constantes de la Bundesliga de Alemania
DESCIENDEN = 2
CUPOS_COPA = 6  # Champions League (1º a 4º) + Europa League (5º y 6º, una vía copa) + Conference

SQUAD_VALUES = {
    "Bayern Munich": 1080.0, "RB Leipzig": 604.8, "Bayer Leverkusen": 513.45,
    "Borussia Dortmund": 464.2, "VfB Stuttgart": 420.15, "Eintracht Frankfurt": 338.85,
    "TSG Hoffenheim": 294.95, "SC Freiburg": 242.55, "Mainz": 169.2,
    "FC Augsburg": 164.25, "Werder Bremen": 149.83, "Borussia Mönchengladbach": 136.0,
    "FC Cologne": 123.9, "1. FC Union Berlin": 109.88, "Hamburg SV": 106.15,
    "Schalke 04": 59.13, "SV Elversberg": 51.6, "SC Paderborn 07": 37.88,
    "VfL Wolfsburg": 120.0, "St. Pauli": 95.0, "VfL Bochum": 80.0,
    "1. FC Heidenheim 1846": 75.0, "Holstein Kiel": 55.0, "SV Darmstadt 98": 50.0,
    "SpVgg Greuther Fürth": 45.0, "Arminia Bielefeld": 40.0, "Hertha Berlin": 90.0
}

ADV_FEATURES_PATH = DATA / "advanced_features_historical.csv"
if ADV_FEATURES_PATH.exists():
    DF_ADV_FEATURES = pd.read_csv(ADV_FEATURES_PATH)
else:
    DF_ADV_FEATURES = pd.DataFrame(columns=[
        "temporada", "equipo", "squad_size", "avg_age", "foreigners",
        "pct_foreigners", "squad_value", "stadium_capacity", "avg_attendance", "stadium_occupation"
    ])
DF_SQUAD_VALUES = DF_ADV_FEATURES

STATS = ["totalShots", "shotsOnTarget", "wonCorners", "possessionPct", "foulsCommitted",
         "yellowCards", "redCards", "offsides", "saves", "blockedShots"]

ELO_INIT = 1500.0

def _elo_default():
    return ELO_INIT


def _none_default():
    return None

K_LIGA = 35.0      # ELO K-factor para la Bundesliga (las ligas top tienen más volatilidad)
HOME_ADV = 60.0    # Ventaja de local típica de 60 puntos ELO


COORDS_GERMANY = {
    "Bayern Munich": (48.1372, 11.5755), "RB Leipzig": (51.3397, 12.3731),
    "Bayer Leverkusen": (51.0384, 7.0028), "Borussia Dortmund": (51.4927, 7.4511),
    "VfB Stuttgart": (48.7758, 9.1829), "Eintracht Frankfurt": (50.1109, 8.6821),
    "TSG Hoffenheim": (49.2531, 8.8838), "SC Freiburg": (48.0211, 7.8534),
    "Mainz": (49.9929, 8.2473), "FC Augsburg": (48.3705, 10.8978),
    "Werder Bremen": (53.0793, 8.8017), "Borussia Mönchengladbach": (51.1777, 6.4374),
    "FC Cologne": (50.9375, 6.9603), "1. FC Union Berlin": (52.5076, 13.4681),
    "Hamburg SV": (53.5871, 9.8987), "Schalke 04": (51.5548, 7.0673),
    "SV Elversberg": (49.3167, 7.1167), "SC Paderborn 07": (51.7189, 8.7551),
    "1. FC Heidenheim 1846": (48.6764, 10.1522), "Holstein Kiel": (54.3233, 10.1228),
    "VfL Bochum": (51.4818, 7.2162), "SV Darmstadt 98": (49.8748, 8.6539),
    "SpVgg Greuther Fürth": (49.4771, 10.9887), "Arminia Bielefeld": (52.0302, 8.5164),
    "Hertha Berlin": (52.5146, 13.2398), "St. Pauli": (53.5871, 9.8987),
    "VfL Wolfsburg": (52.4303, 10.7882)
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def get_distance_km(local, visita):
    c_loc = COORDS_GERMANY.get(local)
    c_vis = COORDS_GERMANY.get(visita)
    if not c_loc or not c_vis:
        return 0.0
    return haversine_km(c_vis[0], c_vis[1], c_loc[0], c_loc[1])


NOMBRES_ADV = {
    "FC Bayern München": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "1. FC Köln": "FC Cologne",
    "FC Köln": "FC Cologne",
    "1. FC Union Berlin": "1. FC Union Berlin",
    "1. FSV Mainz 05": "Mainz",
    "Mainz 05": "Mainz",
    "SV Werder Bremen": "Werder Bremen",
    "FC Schalke 04": "Schalke 04",
    "Hamburger SV": "Hamburg SV",
    "FC St. Pauli": "St. Pauli",
    "TSG 1899 Hoffenheim": "TSG Hoffenheim",
    "DSC Arminia Bielefeld": "Arminia Bielefeld",
    "Hertha BSC": "Hertha Berlin",
    "SpVgg Greuther Fürth": "SpVgg Greuther Fürth",
    "SV 07 Elversberg": "SV Elversberg",
    "SC Paderborn 07": "SC Paderborn 07"
}


def _norm_equipo(team):
    return NOMBRES_ADV.get(team, team)


def get_advanced_features(team, season):
    if len(DF_ADV_FEATURES) > 0:
        df_eq = DF_ADV_FEATURES[DF_ADV_FEATURES.equipo == _norm_equipo(team)]
        if len(df_eq) > 0:
            row = df_eq[df_eq.temporada == season]
            if len(row) > 0:
                return row.iloc[0]
            diffs = (df_eq["temporada"] - season).abs()
            best_idx = diffs.idxmin()
            return df_eq.loc[best_idx]
    return pd.Series({
        "squad_size": 25, "avg_age": 25.0, "foreigners": 0, "pct_foreigners": 0.0,
        "stadium_capacity": 35000, "avg_attendance": 20000, "stadium_occupation": 0.5,
        "squad_value": SQUAD_VALUES.get(_norm_equipo(team), 50.0)
    })


def get_squad_value(team, season):
    """Lee el valor real de plantilla desde TM. Fallback al dict estático si no hay datos."""
    feat = get_advanced_features(team, season)
    val = feat.get("squad_value", None)
    if val is not None and float(val) > 0:
        return float(val)
    return SQUAD_VALUES.get(team, 50.0)




def actualizar_elo(ea, eb, ga, gb):
    we = 1 / (1 + 10 ** (-(ea - eb) / 400))
    w = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
    gd = abs(ga - gb)
    mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8))
    return K_LIGA * mult * (w - we)


class PiRatingTracker:
    def __init__(self, lmb=0.035, gamma=0.70):
        self.r_home = defaultdict(float)
        self.r_away = defaultdict(float)
        self.lmb = lmb
        self.gamma = gamma

    def get_features(self, home, away):
        rh = self.r_home[home]
        ra = self.r_away[away]
        pi_overall_l = (self.r_home[home] + self.r_away[home]) / 2.0
        pi_overall_v = (self.r_home[away] + self.r_away[away]) / 2.0
        return {
            "pi_diff": rh - ra,
            "pi_overall_diff": pi_overall_l - pi_overall_v
        }

    def registrar_partido(self, home, away, ga, gb):
        e_actual = ga - gb
        rh = self.r_home[home]
        ra = self.r_away[away]
        err = e_actual - (rh - ra)
        self.r_home[home] += self.lmb * err
        self.r_away[home] += self.gamma * self.lmb * err
        self.r_away[away] -= self.lmb * err
        self.r_home[away] -= self.gamma * self.lmb * err


class StateTracker:

    def __init__(self):
        self.elos = defaultdict(_elo_default)
        self.history = defaultdict(deque)
        self.home_history = defaultdict(deque)
        self.away_history = defaultdict(deque)
        self.h2h_goles = defaultdict(float)
        self.recent_results = defaultdict(deque)
        self.recent_gf = defaultdict(deque)
        self.recent_ga = defaultdict(deque)
        self.match_count = defaultdict(int)
        self.season_pts = defaultdict(int)
        self.season_matches = defaultdict(int)
        self.curr_season = None
        self.last_match_date = defaultdict(_none_default)
        self.recent_dates = defaultdict(deque)
        self.pi_tracker = PiRatingTracker()




    def get_features_for_match(self, local, visita, temporada, fecha=None, reset_season=True):
        feats = {}

        feats["elo_diff"] = self.elos[local] - self.elos[visita]
        feats["elo_home"] = self.elos[local] + HOME_ADV
        feats["elo_away"] = self.elos[visita]
        vl = get_squad_value(local, temporada)
        vv = get_squad_value(visita, temporada)
        feats["squad_value_diff"] = np.log(max(vl, 0.1)) - np.log(max(vv, 0.1))
        feats["squad_value_home_log"] = np.log(max(vl, 0.1))
        feats["squad_value_away_log"] = np.log(max(vv, 0.1))
        feats["h2h_diff"] = self.h2h_goles[(local, visita)]

        # Características avanzadas de TM
        feat_l = get_advanced_features(local, temporada)
        feat_v = get_advanced_features(visita, temporada)
        feats["avg_age_diff"] = feat_l["avg_age"] - feat_v["avg_age"]
        feats["avg_age_home"] = float(feat_l["avg_age"])
        feats["avg_age_away"] = float(feat_v["avg_age"])
        feats["squad_size_diff"] = feat_l["squad_size"] - feat_v["squad_size"]
        feats["squad_size_home"] = float(feat_l["squad_size"])
        feats["squad_size_away"] = float(feat_v["squad_size"])
        feats["pct_foreigners_diff"] = feat_l["pct_foreigners"] - feat_v["pct_foreigners"]
        feats["pct_foreigners_home"] = float(feat_l["pct_foreigners"])
        feats["pct_foreigners_away"] = float(feat_v["pct_foreigners"])
        feats["foreigners_diff"] = feat_l["foreigners"] - feat_v["foreigners"]
        feats["stadium_capacity"] = np.log(max(float(feat_l["stadium_capacity"]), 1.0))
        feats["stadium_occupation"] = float(feat_l["stadium_occupation"])
        feats["avg_attendance"] = np.log(max(float(feat_l["avg_attendance"]), 1.0))
        feats["stadium_capacity_diff"] = np.log(max(float(feat_l["stadium_capacity"]), 1.0)) - np.log(max(float(feat_v["stadium_capacity"]), 1.0))
        feats["stadium_occupation_diff"] = float(feat_l["stadium_occupation"]) - float(feat_v["stadium_occupation"])
        feats["avg_attendance_diff"] = np.log(max(float(feat_l["avg_attendance"]), 1.0)) - np.log(max(float(feat_v["avg_attendance"]), 1.0))

        N = 5
        rl = list(self.recent_results[local]); rv = list(self.recent_results[visita])
        feats["form_diff"] = (np.mean(rl[-N:]) if rl else 0.333) - (np.mean(rv[-N:]) if rv else 0.333)
        gfl = list(self.recent_gf[local]); gfv = list(self.recent_gf[visita])
        gal = list(self.recent_ga[local]); gav = list(self.recent_ga[visita])
        feats["gf_diff"] = (np.mean(gfl[-N:]) if gfl else 1.0) - (np.mean(gfv[-N:]) if gfv else 1.0)
        feats["ga_diff"] = (np.mean(gal[-N:]) if gal else 1.0) - (np.mean(gav[-N:]) if gav else 1.0)

        if reset_season and temporada != self.curr_season:
            self.curr_season = temporada
            self.season_pts.clear()
            self.season_matches.clear()

        pl = self.season_pts[local]; ml = self.season_matches[local]
        pv = self.season_pts[visita]; mv = self.season_matches[visita]
        ppg_l = (pl / ml) if ml > 0 else 1.33
        ppg_v = (pv / mv) if mv > 0 else 1.33
        feats["ppg_diff"] = ppg_l - ppg_v

        if fecha is not None:
            f = pd.to_datetime(fecha)
            dl = self.last_match_date[local]; dv = self.last_match_date[visita]
            rl = float(np.clip((f - dl).days if dl is not None else 7.0, 3.0, 14.0))
            rv = float(np.clip((f - dv).days if dv is not None else 7.0, 3.0, 14.0))
            feats["rest_days_diff"] = rl - rv
            cl = float(sum(1 for d in self.recent_dates[local] if 0 <= (f - d).days <= 14))
            cv = float(sum(1 for d in self.recent_dates[visita] if 0 <= (f - d).days <= 14))
            feats["congestion_14d_diff"] = cl - cv
        else:
            feats["rest_days_diff"] = 0.0
            feats["congestion_14d_diff"] = 0.0

        pfeats = self.pi_tracker.get_features(local, visita)
        feats["pi_diff"] = pfeats["pi_diff"]
        feats["pi_overall_diff"] = pfeats["pi_overall_diff"]

        d_km = get_distance_km(local, visita)
        feats["travel_dist_log_km"] = float(np.log1p(d_km))





        for s in STATS:
            hl = self.history[local]
            hv = self.history[visita]
            vsl = [h[s] for h in hl if h[s] is not None]
            vsv = [h[s] for h in hv if h[s] is not None]
            feats[f"{s}_total_diff"] = (np.mean(vsl) if vsl else 0.0) - (np.mean(vsv) if vsv else 0.0)

            hhl = self.home_history[local]
            ahv = self.away_history[visita]
            vhl = [h[s] for h in hhl if h[s] is not None]
            vav = [h[s] for h in ahv if h[s] is not None]
            feats[f"{s}_sede_diff"] = (np.mean(vhl) if vhl else 0.0) - (np.mean(vav) if vav else 0.0)

        return feats

    def registrar_partido(self, local, visita, ga, gb, stats_l=None, stats_v=None, fecha=None):
        self.h2h_goles[(local, visita)] += (ga - gb)

        self.h2h_goles[(visita, local)] -= (ga - gb)

        delta = actualizar_elo(self.elos[local] + HOME_ADV, self.elos[visita], ga, gb)
        self.elos[local] += delta
        self.elos[visita] -= delta

        sl = stats_l if stats_l else {s: None for s in STATS}
        sv = stats_v if stats_v else {s: None for s in STATS}

        self.history[local].append(sl)
        if len(self.history[local]) > 6: self.history[local].popleft()
        self.history[visita].append(sv)
        if len(self.history[visita]) > 6: self.history[visita].popleft()

        self.home_history[local].append(sl)
        if len(self.home_history[local]) > 4: self.home_history[local].popleft()
        self.away_history[visita].append(sv)
        if len(self.away_history[visita]) > 4: self.away_history[visita].popleft()

        w_l = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
        w_v = 1.0 - w_l if w_l != 0.5 else 0.5
        self.recent_results[local].append(w_l); self.recent_results[visita].append(w_v)
        self.recent_gf[local].append(ga);       self.recent_gf[visita].append(gb)
        self.recent_ga[local].append(gb);       self.recent_ga[visita].append(ga)
        self.match_count[local] += 1; self.match_count[visita] += 1

        if ga > gb:
            self.season_pts[local] += 3
        elif ga == gb:
            self.season_pts[local] += 1
            self.season_pts[visita] += 1
        else:
            self.season_pts[visita] += 3
        self.season_matches[local] += 1
        self.season_matches[visita] += 1

        if fecha is not None:
            f = pd.to_datetime(fecha)
            self.last_match_date[local] = f
            self.last_match_date[visita] = f
            self.recent_dates[local].append(f)
            self.recent_dates[visita].append(f)
            while self.recent_dates[local] and (f - self.recent_dates[local][0]).days > 30:
                self.recent_dates[local].popleft()
            while self.recent_dates[visita] and (f - self.recent_dates[visita][0]).days > 30:
                self.recent_dates[visita].popleft()

        self.pi_tracker.registrar_partido(local, visita, ga, gb)





def cargar_y_entrenar():
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    box_path = DATA / "box_score.csv"
    box = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame(columns=["event_id"])
    
    box_dict = {}
    for r in box.itertuples(index=False):
        sl = {}
        sv = {}
        for s in STATS:
            sl[s] = getattr(r, f"local_{s}", None)
            sv[s] = getattr(r, f"visita_{s}", None)
        box_dict[str(r.event_id)] = (sl, sv)
        
    tracker = StateTracker()
    filas_X = []
    y = []
    
    for r in partidos.itertuples(index=False):
        local, visita, ga, gb = r.local, r.visita, int(r.goles_local), int(r.goles_visita)
        
        feats = tracker.get_features_for_match(local, visita, r.temporada, fecha=r.fecha)
        feats["event_id"] = str(r.event_id) if hasattr(r, "event_id") else ""
        feats["fecha"] = r.fecha
        feats["temporada"] = r.temporada
        feats["local"] = local
        feats["visita"] = visita
        feats["goles_local"] = ga
        feats["goles_visita"] = gb
        
        res = 2 if ga > gb else (1 if ga == gb else 0)
        feats["resultado"] = res
        
        filas_X.append(feats)
        y.append(res)
        
        eb_id = str(getattr(r, "event_id")) if hasattr(r, "event_id") else ""
        stats_l, stats_v = box_dict.get(eb_id, (None, None))
        tracker.registrar_partido(local, visita, ga, gb, stats_l, stats_v, fecha=r.fecha)

        
    df_features = pd.DataFrame(filas_X)
    
    # Modelo 1: L1 Regularized SAGA logistic regression
    cols_feat = [
        "elo_diff", "elo_home", "elo_away",
        "squad_value_diff", "squad_value_home_log", "squad_value_away_log",
        "h2h_diff",
        "avg_age_diff", "avg_age_home", "avg_age_away",
        "squad_size_diff", "squad_size_home", "squad_size_away",
        "pct_foreigners_diff", "pct_foreigners_home", "pct_foreigners_away",
        "foreigners_diff",
        "stadium_capacity", "stadium_occupation", "avg_attendance",
        "stadium_capacity_diff", "stadium_occupation_diff", "avg_attendance_diff",
        "form_diff", "gf_diff", "ga_diff", "rest_days_diff", "congestion_14d_diff",
        "pi_diff", "pi_overall_diff", "travel_dist_log_km"
    ] + [f"{s}_total_diff" for s in STATS] + [f"{s}_sede_diff" for s in STATS]





                
    train_mask = df_features["temporada"] <= 2023
    cal_mask   = df_features["temporada"] == 2024
    test_mask  = df_features["temporada"] >= 2025

    X_train = df_features.loc[train_mask, cols_feat].fillna(0.0)
    y_train = df_features.loc[train_mask, "resultado"]
    X_cal = df_features.loc[cal_mask, cols_feat].fillna(0.0)
    y_cal = df_features.loc[cal_mask, "resultado"]
    X_test = df_features.loc[test_mask, cols_feat].fillna(0.0)
    y_test = df_features.loc[test_mask, "resultado"]

    # C-search con CV temporal sobre train (sin mirar el test)
    best_c = 0.04; best_loss = 999.0
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        _losses = []
        for tr, va in tscv.split(X_train):
            _p = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=4000, random_state=42))])
            _p.fit(X_train.iloc[tr], y_train.iloc[tr])
            _losses.append(log_loss(y_train.iloc[va], _p.predict_proba(X_train.iloc[va]), labels=[0,1,2]))
        _l = float(np.mean(_losses))
        if _l < best_loss: best_loss = _l; best_c = C
    print(f"Mejor C para Lasso (CV temporal en train): {best_c} | Log-Loss CV: {best_loss:.4f}")

    pipe_lasso_base = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=4000, random_state=42))])
    pipe_lasso_base.fit(X_train, y_train)
    pipe_rf_base = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(max_depth=3, n_estimators=100, min_samples_split=5, random_state=42, n_jobs=-1))])
    pipe_rf_base.fit(X_train, y_train)
    pipe_xgb_base = Pipeline([("scale", StandardScaler()), ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="mlogloss"))])
    pipe_xgb_base.fit(X_train, y_train)

    if len(X_cal) >= 10:
        pipe_lasso_cal = pipe_lasso_base
        pipe_rf_cal = pipe_rf_base
        pipe_xgb_cal = pipe_xgb_base
    else:
        pipe_lasso_cal = pipe_lasso_base; pipe_rf_cal = pipe_rf_base

    if len(X_cal) >= 10:
        p_l_cal = pipe_lasso_cal.predict_proba(X_cal)
        p_r_cal = pipe_rf_cal.predict_proba(X_cal)
        p_x_cal = pipe_xgb_cal.predict_proba(X_cal)
        def _stk3(w):
            w = np.abs(w); w = w / w.sum()
            blend = np.clip(w[0]*p_l_cal + w[1]*p_r_cal + w[2]*p_x_cal, 1e-7, 1-1e-7)
            return log_loss(y_cal, blend)
        res = minimize(_stk3, [0.4, 0.3, 0.3], method="Nelder-Mead")
        w_opt = np.abs(res.x); w_opt = w_opt / w_opt.sum()
        alpha_opt = float(w_opt[0])
    else:
        w_opt = np.array([0.4, 0.3, 0.3])
        alpha_opt = 0.4

    X_full = df_features.loc[train_mask | cal_mask, cols_feat].fillna(0.0)
    y_full = df_features.loc[train_mask | cal_mask, "resultado"]
    pipe_lasso = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=4000, random_state=42))])
    pipe_lasso.fit(X_full, y_full)
    pipe_rf = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(max_depth=3, n_estimators=100, min_samples_split=5, random_state=42, n_jobs=-1))])
    pipe_rf.fit(X_full, y_full)
    pipe_xgb = Pipeline([("scale", StandardScaler()), ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="mlogloss"))])
    pipe_xgb.fit(X_full, y_full)

    def _met(proba, y):
        proba = np.clip(proba, 1e-7, 1-1e-7)
        return {"logloss": round(log_loss(y, proba), 4), "accuracy": round(accuracy_score(y, proba.argmax(axis=1))*100, 2)}
    met_lasso = _met(pipe_lasso.predict_proba(X_test), y_test)
    met_rf = _met(pipe_rf.predict_proba(X_test), y_test)
    met_xgb = _met(pipe_xgb.predict_proba(X_test), y_test)
    w_opt = w_opt / w_opt.sum()
    p_st = np.clip(w_opt[0]*pipe_lasso.predict_proba(X_test)+w_opt[1]*pipe_rf.predict_proba(X_test)+w_opt[2]*pipe_xgb.predict_proba(X_test), 1e-7, 1-1e-7)
    met_stack = {"logloss": round(log_loss(y_test,p_st),4), "accuracy": round(accuracy_score(y_test,p_st.argmax(axis=1))*100,2), "w": [round(float(x),3) for x in w_opt]}
    metricas = {"lasso": met_lasso, "rf": met_rf, "xgb": met_xgb, "stacking": met_stack}
    print(f"Metricas ENG Test>=2025: LASSO={met_lasso} RF={met_rf} Stacking={met_stack}")
    print(f"LASSO (L1): {len(cols_feat)} features totales -> {int((np.abs(pipe_lasso.named_steps['lr'].coef_) > 1e-8).sum())} seleccionadas")
    
    # Ajuste Poisson para goles esperados
    # Estimamos goles promedio en función del ELO diferencial + localidad (bias ±0.172 sin is_home).
    # Solo train+cal (temporada <= 2024): los partidos de test no entrenan el modelo.
    df_goles = df_features[df_features["temporada"] <= 2024]
    largo = pd.concat([
        pd.DataFrame({"g": df_goles["goles_local"].values,  "d": df_goles["elo_diff"].values,  "is_home": 1}),
        pd.DataFrame({"g": df_goles["goles_visita"].values, "d": -df_goles["elo_diff"].values, "is_home": 0})
    ])

    gp = sm.GLM(largo["g"], sm.add_constant(largo[["d", "is_home"]]), family=sm.families.Poisson()).fit()
    g_const, g_d, g_home = float(gp.params["const"]), float(gp.params["d"]), float(gp.params["is_home"])
    
    return {
        "pipe_lasso": pipe_lasso,
        "pipe_rf": pipe_rf,
        "cols": cols_feat,
        "tracker": tracker,
        "equipos": _cargar_equipos(),
        "g_const": g_const,
        "g_d": g_d,
        "g_home": g_home,
        "df_features": df_features,
        "alpha_stack": alpha_opt,
        "stack_w": w_opt,
        "pipe_xgb": pipe_xgb,
        "metricas": metricas
    }


def _cache_key():
    # El modelo depende de los datos (data/*.csv) y del propio código del motor
    import hashlib
    h = hashlib.sha256()
    h.update(hashlib.sha256(Path(__file__).read_bytes()).hexdigest().encode())
    for f in sorted(DATA.glob("*.csv")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode())
    return h.hexdigest()


def cargar(use_cache=True):
    # Retorna un diccionario con modelos entrenados y estados finales
    import pickle
    cache_path = Path(__file__).resolve().parent / ".model_cache.pkl"
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("key") == _cache_key():
                print("Modelo cargado desde cache de disco")
                return saved["M"]
        except Exception as ex:
            print(f"Cache inválido ({ex}); re-entrenando...")
    M = cargar_y_entrenar()
    if use_cache:
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump({"key": _cache_key(), "M": M}, fh)
            print("Modelo guardado en cache de disco")
        except Exception as ex:
            print(f"No se pudo guardar el cache: {ex}")
    return M


def _temporada_actual():
    # En la Bundesliga la temporada va de julio a junio (2026 = temporada 2026/27)
    hoy = pd.Timestamp.now()
    return int(hoy.year) if hoy.month >= 7 else int(hoy.year) - 1


def predecir_match(M, local, visita, temporada=None, modelo_tipo="rf"):
    # Retorna P(Local), P(Empate), P(Visita) en base al modelo seleccionado
    tracker = M["tracker"]
    cols = M["cols"]
    if temporada is None:
        temporada = _temporada_actual()
    
    # Obtener features en el estado final del tracker
    feats = tracker.get_features_for_match(local, visita, temporada, reset_season=False)
    df_test = pd.DataFrame([feats])[cols]
    
    if modelo_tipo == "stacking":
        w = M.get("stack_w", np.array([M.get("alpha_stack", 0.4), 1-M.get("alpha_stack", 0.4), 0.0]))
        p_l = M["pipe_lasso"].predict_proba(df_test)[0]
        p_r = M["pipe_rf"].predict_proba(df_test)[0]
        p_x = M["pipe_xgb"].predict_proba(df_test)[0]
        p_raw = w[0]*p_l + w[1]*p_r + w[2]*p_x; p_raw /= p_raw.sum()
    elif modelo_tipo == "xgb":
        p_raw = M["pipe_xgb"].predict_proba(df_test)[0]
    else:
        pipe = M["pipe_rf"] if modelo_tipo == "rf" else M["pipe_lasso"]
        p_raw = pipe.predict_proba(df_test)[0]
    p = np.array([p_raw[2], p_raw[1], p_raw[0]])
    return p  # Orden: [Local, Empate, Visita]


def grilla_goles(M, local, visita, modelo_tipo="rf"):
    # Retorna matriz 10x10 de goles esperados usando modelo Poisson y Dixon-Coles
    p_1x2 = predecir_match(M, local, visita, modelo_tipo=modelo_tipo)
    
    tracker = M["tracker"]
    elo_diff = tracker.elos[local] - tracker.elos[visita]
    
    # Calcular lambdas esperados
    la = np.exp(M["g_const"] + M["g_d"] * elo_diff + M.get("g_home", 0.0) * 1)
    lb = np.exp(M["g_const"] - M["g_d"] * elo_diff + M.get("g_home", 0.0) * 0)
    
    # Construir grilla Poisson
    max_g = 10
    pa = np.array([la**i * np.exp(-la) / math.factorial(i) for i in range(max_g)])
    pb = np.array([lb**j * np.exp(-lb) / math.factorial(j) for j in range(max_g)])
    
    grid = np.outer(pa, pb)
    
    # Escalar para que sume exactamente a las probabilidades del modelo 1X2
    p_local_g = grid[np.triu_indices(max_g, 1)].sum() + grid.diagonal().sum() * 0.0 # Gana Local (arriba de diagonal, pero dependiente de cómo ordenes)
    # Corrección para ordenar fila=goles_local, col=goles_visita:
    # Gana local: i > j (abajo de la diagonal)
    # Gana visita: i < j (arriba de la diagonal)
    p_l = 0.0; p_d = 0.0; p_v = 0.0
    for i in range(max_g):
        for j in range(max_g):
            if i > j: p_l += grid[i, j]
            elif i == j: p_d += grid[i, j]
            else: p_v += grid[i, j]
            
    # Escalamiento multiplicativo suave para calibrar la distribución
    grid_adj = grid.copy()
    if p_l > 0: grid_adj[np.tril_indices(max_g, -1)] *= (p_1x2[0] / p_l)
    if p_d > 0: grid_adj[np.diag_indices(max_g)] *= (p_1x2[1] / p_d)
    if p_v > 0: grid_adj[np.triu_indices(max_g, 1)] *= (p_1x2[2] / p_v)
    
    # Normalizar para que sume 1.0 en total
    s = grid_adj.sum()
    if s > 0:
        grid_adj /= s
        
    return grid_adj


def cuota(p):
    return 1 / p if p > 0 else 99.0


def mercados(mix):
    # Extrae Over/Under, BTTS a partir de la grilla de goles
    mk = {}
    max_g = mix.shape[0]
    
    # Over/Under Goles
    for line in [1.5, 2.5, 3.5]:
        p_over = 0.0
        for i in range(max_g):
            for j in range(max_g):
                if i + j > line:
                    p_over += mix[i, j]
        mk[f"Over {line}"] = p_over
        mk[f"Under {line}"] = 1.0 - p_over
        
    # Ambos Marcan (BTTS)
    p_btts_si = 0.0
    for i in range(1, max_g):
        for j in range(1, max_g):
            p_btts_si += mix[i, j]
    mk["Ambos marcan (BTTS sí)"] = p_btts_si
    mk["BTTS no"] = 1.0 - p_btts_si
    
    # Top Marcadores
    list_m = []
    for i in range(max_g):
        for j in range(max_g):
            list_m.append((i, j, mix[i, j]))
    list_m = sorted(list_m, key=lambda x: x[2], reverse=True)
    mk["_top_marcadores"] = list_m
    
    return mk


def handicap_asiatico(mix, line):
    # Handicap asiático de goles (e.g. +1, -1)
    max_g = mix.shape[0]
    p_cubre = 0.0
    p_push = 0.0
    for i in range(max_g):
        for j in range(max_g):
            diff = i - j
            if diff + line > 0:
                p_cubre += mix[i, j]
            elif diff + line == 0:
                p_push += mix[i, j]
    return {"A cubre": p_cubre, "Push": p_push, "B cubre": 1.0 - p_cubre - p_push}


def simular_fixture_regular(M, PREDS, fijos=None, modelo_tipo="rf"):
    # Lee fixture.csv y simula el fin de temporada
    fix_path = DATA / "fixture.csv"
    if not fix_path.exists():
        return pd.DataFrame()
        
    fix = pd.read_csv(fix_path)
    if len(fix) == 0:
        return pd.DataFrame()
        
    fix["temporada"] = pd.to_datetime(fix["fecha"]).apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    temporada_sim = fix["temporada"].mode().iloc[0]
    
    # Filtrar solo el fixture de la temporada activa
    fix = fix[fix.temporada == temporada_sim].copy()
    
    # Inicializar tabla de posiciones actual
    tabla = obtener_tabla_actual(M, temporada=temporada_sim)
    
    pts = dict(zip(tabla["equipo"], tabla["puntos"]))
    gf = dict(zip(tabla["equipo"], tabla["goles_favor"]))
    gc = dict(zip(tabla["equipo"], tabla["goles_contra"]))
    pj = dict(zip(tabla["equipo"], tabla["pj"]))
    
    # Simular partidos
    for r in fix.itertuples(index=False):
        local, visita = r.local, r.visita
        if fijos and (local, visita) in fijos:
            gl, gv = fijos[(local, visita)]
        else:
            p = PREDS.get((local, visita))
            if p is None:
                p = grilla_goles(M, local, visita, modelo_tipo=modelo_tipo)
            # Muestrear marcador realista desde la grilla 10x10 calibrada (Dixon-Coles + 1X2)
            idx = np.random.choice(p.size, p=p.ravel())
            gl, gv = divmod(int(idx), p.shape[1])
                
        # Registrar stats
        pj[local] += 1; pj[visita] += 1
        gf[local] += gl; gf[visita] += gv
        gc[local] += gv; gc[visita] += gl
        if gl > gv:
            pts[local] += 3
        elif gl == gv:
            pts[local] += 1; pts[visita] += 1
        else:
            pts[visita] += 3
            
    # Re-armar DataFrame de tabla simulada
    filas = []
    for eq in pts.keys():
        diff = gf[eq] - gc[eq]
        filas.append({
            "equipo": eq, "pj": pj[eq], "puntos": pts[eq],
            "goles_favor": gf[eq], "goles_contra": gc[eq], "dif_goles": diff
        })
    df_tabla = pd.DataFrame(filas)
    return ordenar_tabla(df_tabla)


def ordenar_tabla(tabla):
    # LaLiga desempata por puntos -> diferencia de goles -> goles a favor
    return tabla.sort_values(by=["puntos", "dif_goles", "goles_favor"], ascending=False).reset_index(drop=True)



def obtener_tabla_actual(M, temporada=None):
    # Calcula la tabla de posiciones real a partir de partidos.csv
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"])
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    
    if temporada is None:
        temporada = partidos["temporada"].max()
        
    # Identificar todos los equipos de la temporada activa
    # Para ser robustos, extraemos los equipos de la temporada del fixture y de partidos jugados
    fix_path = DATA / "fixture.csv"
    eqs_active = set()
    if fix_path.exists():
        fix = pd.read_csv(fix_path)
        fix["temporada"] = pd.to_datetime(fix["fecha"]).apply(lambda x: x.year if x.month >= 7 else x.year - 1)
        fix_temp = fix[fix.temporada == temporada]
        if len(fix_temp):
            eqs_active.update(fix_temp["local"].unique())
            eqs_active.update(fix_temp["visita"].unique())
            
    partidos_temp = partidos[partidos.temporada == temporada]
    if len(partidos_temp):
        eqs_active.update(partidos_temp["local"].unique())
        eqs_active.update(partidos_temp["visita"].unique())
        
    if not eqs_active:
        # Fallback si no hay temporada
        eqs_active = set(partidos["local"].unique())
        
    pts = {t: 0 for t in eqs_active}
    gf = {t: 0 for t in eqs_active}
    gc = {t: 0 for t in eqs_active}
    pj = {t: 0 for t in eqs_active}
    
    for r in partidos_temp.itertuples(index=False):
        l, v, gl, gv = r.local, r.visita, int(r.goles_local), int(r.goles_visita)
        if l not in eqs_active or v not in eqs_active:
            continue
        pj[l] += 1; pj[v] += 1
        gf[l] += gl; gf[v] += gv
        gc[l] += gv; gc[v] += gl
        if gl > gv:
            pts[l] += 3
        elif gl == gv:
            pts[l] += 1; pts[v] += 1
        else:
            pts[v] += 3
            
    filas = []
    for eq in eqs_active:
        filas.append({
            "equipo": eq, "pj": pj[eq], "puntos": pts[eq],
            "goles_favor": gf[eq], "goles_contra": gc[eq], "dif_goles": gf[eq] - gc[eq]
        })
    return ordenar_tabla(pd.DataFrame(filas))


def _simular_fixture_vec(M, PREDS, fijos, n_sims, modelo_tipo="rf"):
    fix_path = DATA / "fixture.csv"
    fix = pd.read_csv(fix_path)
    fix["temporada"] = pd.to_datetime(fix["fecha"]).apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    temporada_sim = fix["temporada"].mode().iloc[0]
    fix = fix[fix.temporada == temporada_sim]
    tabla = obtener_tabla_actual(M, temporada=temporada_sim)
    seed = dict(zip(tabla["equipo"], zip(tabla["puntos"], tabla["goles_favor"], tabla["goles_contra"])))
    eqs = sorted(set(seed).union(set(fix["local"])).union(set(fix["visita"])))
    n_eq = len(eqs)
    ix = {e: i for i, e in enumerate(eqs)}
    PTS = np.zeros((n_eq, n_sims))
    GF = np.zeros_like(PTS)
    GC = np.zeros_like(PTS)
    for e in eqs:
        if e in seed:
            pts, gf_, gc_ = seed[e]
            PTS[ix[e]] = pts
            GF[ix[e]] = gf_
            GC[ix[e]] = gc_
    for r in fix.itertuples(index=False):
        local, visita = r.local, r.visita
        if fijos and (local, visita) in fijos:
            gl = np.full(n_sims, float(fijos[(local, visita)][0]))
            gv = np.full(n_sims, float(fijos[(local, visita)][1]))
        else:
            p = PREDS.get((local, visita))
            if p is None:
                p = grilla_goles(M, local, visita, modelo_tipo=modelo_tipo)
            k = np.random.choice(p.size, size=n_sims, p=p.ravel())
            gl = (k // p.shape[1]).astype(float)
            gv = (k % p.shape[1]).astype(float)
        li, vi = ix[local], ix[visita]
        win_l = gl > gv
        draw = gl == gv
        PTS[li] += np.where(win_l, 3.0, np.where(draw, 1.0, 0.0))
        PTS[vi] += np.where(gv > gl, 3.0, np.where(draw, 1.0, 0.0))
        GF[li] += gl
        GF[vi] += gv
        GC[li] += gv
        GC[vi] += gl
    DG = GF - GC
    key = (PTS * 1_000_000 + DG * 1_000 + GF).astype(np.int64)
    order = np.argsort(-key, axis=0)
    return eqs, order


def simular_campeonato(M, n_sims=3000, fijos=None, modelo_tipo="rf", seed=42):
    # Corre simulación de Monte Carlo para obtener probabilidades de campeón, copas y descenso.
    # El resultado se persiste a disco (eng/data/simulacion_mc.pkl) y se reutiliza mientras
    # el fixture, el modelo y n_sims no cambien: la simulación solo se ejecuta una vez.
    if seed is not None:
        np.random.seed(seed)
    fix_path = DATA / "fixture.csv"
    if not fix_path.exists() or len(pd.read_csv(fix_path)) == 0:
        # Si no hay fixture por jugar, la tabla actual es la definitiva
        tab = obtener_tabla_actual(M)
        res = tab[["equipo"]].copy()
        res["P_campeon"] = 0.0
        res.loc[0, "P_campeon"] = 1.0
        res["P_copas"] = 0.0
        res.loc[:CUPOS_COPA - 1, "P_copas"] = 1.0
        res["P_descenso"] = 0.0
        res.loc[res.index[-DESCIENDEN:], "P_descenso"] = 1.0
        return res

    import hashlib
    fp = hashlib.sha256()
    fp.update(hashlib.sha256(fix_path.read_bytes()).hexdigest().encode())
    fp.update(str(fix_path.stat().st_size).encode())
    fp.update(str(n_sims).encode())
    fp.update(_cache_key().encode())
    if fijos is not None:
        fp.update(repr(sorted(fijos.items())).encode())
    cache_path = Path(__file__).resolve().parent / "simulacion_mc.pkl"
    resultados = {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("key") == fp.hexdigest():
                resultados = saved.get("results", {})
                if modelo_tipo in resultados:
                    print("Simulación Monte Carlo cargada desde cache de disco")
                    return resultados[modelo_tipo]
        except Exception as ex:
            print(f"Cache de simulación inválido ({ex}); re-simulando...")

    fix = pd.read_csv(fix_path)
    fix["temporada"] = pd.to_datetime(fix["fecha"]).apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    temporada_sim = fix["temporada"].mode().iloc[0]
    
    # Filtrar solo el fixture de la temporada activa
    fix = fix[fix.temporada == temporada_sim].copy()
    eqs = set(fix["local"].unique()).union(set(fix["visita"].unique()))
    
    # Pre-calcular predicciones fijas para acelerar
    print("Precalculando predicciones de fixture...")
    PREDS = {}
    for r in fix.itertuples(index=False):
        PREDS[(r.local, r.visita)] = grilla_goles(M, r.local, r.visita, modelo_tipo=modelo_tipo)
        
    counts_campeon = defaultdict(int)
    counts_copas = defaultdict(int)
    counts_descenso = defaultdict(int)
    
    eqs, order = _simular_fixture_vec(M, PREDS, fijos, n_sims, modelo_tipo=modelo_tipo)
    n_eq = len(eqs)
    campeon_c = np.bincount(order[0], minlength=n_eq)
    copas_c = np.bincount(order[:CUPOS_COPA].ravel(), minlength=n_eq)
    descenso_c = np.bincount(order[-DESCIENDEN:].ravel(), minlength=n_eq)
    counts_campeon = {e: int(campeon_c[i]) for i, e in enumerate(eqs)}
    counts_copas = {e: int(copas_c[i]) for i, e in enumerate(eqs)}
    counts_descenso = {e: int(descenso_c[i]) for i, e in enumerate(eqs)}
            
    # Armar DataFrame resumen
    res = []
    for eq in sorted(list(eqs)):
        res.append({
            "equipo": eq,
            "P_campeon": counts_campeon[eq] / n_sims,
            "P_copas": counts_copas[eq] / n_sims,
            "P_descenso": counts_descenso[eq] / n_sims
        })
    df_res = pd.DataFrame(res).sort_values("P_campeon", ascending=False).reset_index(drop=True)

    try:
        resultados[modelo_tipo] = df_res
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": fp.hexdigest(), "results": resultados}, fh)
        print("Simulación Monte Carlo guardada en cache de disco")
    except Exception as ex:
        print(f"No se pudo guardar el cache de simulación: {ex}")
    return df_res


def validacion_en_vivo(M, temporada_val=None, modelo_tipo="rf"):
    # Mismo reporte de validación en vivo para la Bundesliga
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    # Re-etiquetar temporada como en cargar_y_entrenar (jul-jun), para que coincida
    # con el selector de la app (df_features["temporada"])
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    if temporada_val is None:
        temporada_val = partidos["temporada"].max()
    val_df = partidos[partidos.temporada == temporada_val]
    
    if len(val_df) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()
        
    # Re-inicializar tracker para evitar filtrado de futuro
    box_path = DATA / "box_score.csv"
    box = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame(columns=["event_id"])
    box_dict = {}
    for r in box.itertuples(index=False):
        sl = {}
        sv = {}
        for s in STATS:
            sl[s] = getattr(r, f"local_{s}", None)
            sv[s] = getattr(r, f"visita_{s}", None)
        box_dict[str(r.event_id)] = (sl, sv)
        
    tracker = StateTracker()
    filas = []
    P_list = []
    y_list = []
    
    for r in partidos.itertuples(index=False):
        local, visita, ga, gb = r.local, r.visita, int(r.goles_local), int(r.goles_visita)
        eb_id = str(getattr(r, "event_id")) if hasattr(r, "event_id") else ""
        
        if r.temporada == temporada_val:
            feats = tracker.get_features_for_match(local, visita, r.temporada)
            df_test = pd.DataFrame([feats])[M["cols"]]
            
            pipe = M["pipe_xgb"] if modelo_tipo == "xgb" else (M["pipe_rf"] if modelo_tipo == "rf" else M["pipe_lasso"])
            p = pipe.predict_proba(df_test)[0]
            p_1x2 = np.array([p[2], p[1], p[0]])  # local, empate, visita
            
            real = 2 if ga > gb else (1 if ga == gb else 0)
            pred = 2 if (p_1x2[0] >= p_1x2[1] and p_1x2[0] >= p_1x2[2]) else (1 if p_1x2[1] >= p_1x2[2] else 0)
            
            P_list.append([p_1x2[2], p_1x2[1], p_1x2[0]])  # orden clases sklearn: [0, 1, 2] -> [visita, empate, local]
            y_list.append(real)
            
            filas.append({
                "fecha": r.fecha.date().strftime("%Y-%m-%d"),
                "local": local,
                "visita": visita,
                "goles_local": ga,
                "goles_visita": gb,
                "resultado": ["Gana visita", "Empate", "Gana local"][real],
                "Prediccion": ["Gana visita", "Empate", "Gana local"][pred],
                "Prob_Local": p_1x2[0],
                "Prob_Empate": p_1x2[1],
                "Prob_Visita": p_1x2[2]
            })
            
        stats_l, stats_v = box_dict.get(eb_id, (None, None))
        tracker.registrar_partido(local, visita, ga, gb, stats_l, stats_v)
        
    df_val = pd.DataFrame(filas)
    if len(df_val) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()
        
    P_arr = np.array(P_list)
    y_arr = np.array(y_list)
    n = len(y_arr)
    
    # Baseline: frecuencias históricas típicas (44% local, 27% empate, 29% visita)
    base = np.tile([0.29, 0.27, 0.44], (n, 1))
    
    aciertos = sum(1 for i in range(n) if np.argmax(P_arr[i]) == y_arr[i])
    met = {
        "n": n,
        "acierto": aciertos / n,
        "logloss": log_loss(y_arr, P_arr, labels=[0, 1, 2]),
        "logloss_base": log_loss(y_arr, base, labels=[0, 1, 2])
    }
    
    # Evolución
    evol = [log_loss(y_arr[:i+1], P_arr[:i+1], labels=[0, 1, 2]) for i in range(n)]
    df_evol = pd.DataFrame({
        "partido": range(1, n + 1),
        "logloss_acum": evol,
        "baseline": [met["logloss_base"]] * n
    })
    
    return df_val, met, df_evol
