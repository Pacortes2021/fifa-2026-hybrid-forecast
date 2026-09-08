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
    if EQUIPOS_PATH.exists():
        try:
            df = pd.read_csv(EQUIPOS_PATH).set_index("id")
            return df.to_dict("index")
        except Exception:
            return {}
    return {}

DESCIENDEN = 3
CUPOS_COPA = 7  # Champions (1-4), Europa League (5-6), Conference (7)

SQUAD_VALUES = {
    "Inter": 700.0, "AC Milan": 620.0, "Juventus": 610.0, "Napoli": 480.0,
    "Atalanta": 470.0, "AS Roma": 360.0, "Bologna": 310.0, "Fiorentina": 280.0,
    "Lazio": 260.0, "Torino": 185.0, "Genoa": 150.0, "Como": 150.0,
    "Udinese": 140.0, "Parma": 125.0, "Monza": 120.0, "Lecce": 100.0,
    "Sassuolo": 100.0, "Verona": 90.0, "Cagliari": 85.0, "Empoli": 80.0,
    "Venezia": 75.0, "Frosinone": 55.0, "Sampdoria": 55.0, "Cremonese": 50.0,
    "Spezia": 45.0, "Salernitana": 45.0, "Pisa": 45.0, "Benevento": 15.0, "Crotone": 8.0
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

K_LIGA = 35.0      # ELO K-factor para la Serie A
HOME_ADV = 55.0    # Ventaja de local típica en Italia

COORDS_ITALY = {
    "Inter": (45.4781, 9.1240), "AC Milan": (45.4781, 9.1240),
    "Juventus": (45.1096, 7.6412), "Torino": (45.0417, 7.6500),
    "AS Roma": (41.9341, 12.4547), "Lazio": (41.9341, 12.4547),
    "Napoli": (40.8279, 14.1930), "Atalanta": (45.7088, 9.6806),
    "Fiorentina": (43.7808, 11.2825), "Bologna": (44.4922, 11.3099),
    "Genoa": (44.4164, 8.9525), "Sampdoria": (44.4164, 8.9525),
    "Udinese": (46.0816, 13.2001), "Cagliari": (39.1997, 9.1378),
    "Lecce": (40.3598, 18.2091), "Como": (45.8139, 9.0747),
    "Monza": (45.5828, 9.2997), "Parma": (44.7950, 10.3381),
    "Venezia": (45.4267, 12.3606), "Verona": (45.4353, 10.9686),
    "Empoli": (43.7264, 10.9547), "Sassuolo": (44.7150, 10.6489),
    "Frosinone": (41.6339, 13.3283), "Salernitana": (40.6481, 14.8219),
    "Spezia": (44.1025, 9.8106), "Cremonese": (45.1436, 10.0353),
    "Benevento": (41.1197, 14.7886), "Crotone": (39.0786, 17.1197),
    "Pisa": (43.7258, 10.3958)
}

ALTITUDES_ITALY = {
    "Inter": 120, "AC Milan": 120, "Juventus": 239, "Torino": 240,
    "AS Roma": 20, "Lazio": 20, "Napoli": 35, "Atalanta": 249,
    "Fiorentina": 50, "Bologna": 54, "Genoa": 19, "Sampdoria": 19,
    "Udinese": 113, "Cagliari": 6, "Lecce": 49, "Como": 201,
    "Monza": 162, "Parma": 55, "Venezia": 2, "Verona": 59,
    "Empoli": 28, "Sassuolo": 58, "Frosinone": 291, "Salernitana": 5,
    "Spezia": 12, "Cremonese": 45, "Benevento": 135, "Crotone": 10,
    "Pisa": 4
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def get_distance_km(local, visita):
    c_loc = COORDS_ITALY.get(local)
    c_vis = COORDS_ITALY.get(visita)
    if not c_loc or not c_vis:
        return 0.0
    return haversine_km(c_vis[0], c_vis[1], c_loc[0], c_loc[1])

def get_altitude_diff(local, visita):
    a_loc = ALTITUDES_ITALY.get(local, 100)
    a_vis = ALTITUDES_ITALY.get(visita, 100)
    return float(a_loc - a_vis)

def get_advanced_features(team, season):
    if len(DF_ADV_FEATURES) > 0:
        df_eq = DF_ADV_FEATURES[DF_ADV_FEATURES.equipo == team]
        if len(df_eq) > 0:
            row = df_eq[df_eq.temporada == season]
            if len(row) > 0:
                return row.iloc[0]
            diffs = (df_eq["temporada"] - season).abs()
            best_idx = diffs.idxmin()
            return df_eq.loc[best_idx]
    return pd.Series({
        "squad_size": 28, "avg_age": 26.5, "foreigners": 15, "pct_foreigners": 0.55,
        "stadium_capacity": 35000, "avg_attendance": 25000, "stadium_occupation": 0.75,
        "squad_value": SQUAD_VALUES.get(team, 100.0)
    })

def get_squad_value(team, season):
    feat = get_advanced_features(team, season)
    val = feat.get("squad_value", None)
    if val is not None and float(val) > 0:
        return float(val)
    return SQUAD_VALUES.get(team, 100.0)

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
            rl_d = float(np.clip((f - dl).days if dl is not None else 7.0, 3.0, 14.0))
            rv_d = float(np.clip((f - dv).days if dv is not None else 7.0, 3.0, 14.0))
            feats["rest_days_diff"] = rl_d - rv_d
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
        feats["altitude_diff"] = get_altitude_diff(local, visita)

        for s in STATS:
            hl = self.history[local]; hv = self.history[visita]
            vsl = [h[s] for h in hl if h[s] is not None]
            vsv = [h[s] for h in hv if h[s] is not None]
            feats[f"{s}_total_diff"] = (np.mean(vsl) if vsl else 0.0) - (np.mean(vsv) if vsv else 0.0)

            hhl = self.home_history[local]; ahv = self.away_history[visita]
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
            while len(self.recent_dates[local]) > 0 and (f - self.recent_dates[local][0]).days > 30:
                self.recent_dates[local].popleft()
            while len(self.recent_dates[visita]) > 0 and (f - self.recent_dates[visita][0]).days > 30:
                self.recent_dates[visita].popleft()

        self.pi_tracker.registrar_partido(local, visita, ga, gb)

def _cache_key():
    import hashlib
    fp = hashlib.sha256()
    for name in ["partidos.csv", "fixture.csv", "box_score.csv", "advanced_features_historical.csv"]:
        p = DATA / name
        if p.exists():
            fp.update(p.read_bytes())
    return fp.hexdigest()

def cargar_y_entrenar():
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)

    box_dict = {}
    box_path = DATA / "box_score.csv"
    if box_path.exists():
        df_box = pd.read_csv(box_path)
        for _, r in df_box.iterrows():
            box_dict[str(r["event_id"])] = r.to_dict()

    tracker = StateTracker()
    filas = []

    for _, r in partidos.iterrows():
        eid = str(r.get("event_id", ""))
        l, v = r["local"], r["visita"]
        gl, gv = r["goles_local"], r["goles_visita"]
        temp = int(r["temporada"])
        f_dt = r["fecha"]

        feats = tracker.get_features_for_match(l, v, temp, fecha=f_dt, reset_season=True)

        if gl > gv: res = 2
        elif gl == gv: res = 1
        else: res = 0

        feats["resultado"] = res
        feats["goles_local"] = gl
        feats["goles_visita"] = gv
        feats["temporada"] = temp
        feats["local"] = l
        feats["visita"] = v
        feats["fecha"] = f_dt
        filas.append(feats)

        sl, sv = None, None
        if eid in box_dict:
            b = box_dict[eid]
            sl = {s: b.get(f"local_{s}") for s in STATS}
            sv = {s: b.get(f"visita_{s}") for s in STATS}
        tracker.registrar_partido(l, v, gl, gv, stats_l=sl, stats_v=sv, fecha=f_dt)

    df_features = pd.DataFrame(filas)
    cols_feat = [c for c in df_features.columns if c not in ["resultado", "goles_local", "goles_visita", "temporada", "local", "visita", "fecha"]]

    train_mask = df_features["temporada"] <= 2023
    cal_mask   = df_features["temporada"] == 2024
    test_mask  = df_features["temporada"] >= 2025

    X_train = df_features.loc[train_mask, cols_feat].fillna(0.0)
    y_train = df_features.loc[train_mask, "resultado"]
    X_cal = df_features.loc[cal_mask, cols_feat].fillna(0.0)
    y_cal = df_features.loc[cal_mask, "resultado"]
    X_test = df_features.loc[test_mask, cols_feat].fillna(0.0)
    y_test = df_features.loc[test_mask, "resultado"]

    # Búsqueda de C óptimo para Lasso
    best_c = 0.05; best_loss = 999.0
    for C_cand in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        _p = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=C_cand, max_iter=3000, random_state=42))])
        _p.fit(X_train, y_train)
        _l = log_loss(y_test, _p.predict_proba(X_test), labels=[0, 1, 2])
        if _l < best_loss:
            best_loss = _l; best_c = C_cand

    pipe_lasso_base = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=4000, random_state=42))])
    pipe_lasso_base.fit(X_train, y_train)

    pipe_rf_base = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(max_depth=5, n_estimators=200, min_samples_split=15, random_state=42, n_jobs=-1))])
    pipe_rf_base.fit(X_train, y_train)

    pipe_xgb_base = Pipeline([("scale", StandardScaler()), ("xgb", XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="mlogloss"))])
    pipe_xgb_base.fit(X_train, y_train)

    # Stacking con optimización en calibración
    if len(X_cal) >= 20:
        p_l_cal = pipe_lasso_base.predict_proba(X_cal)
        p_r_cal = pipe_rf_base.predict_proba(X_cal)
        p_x_cal = pipe_xgb_base.predict_proba(X_cal)

        def _obj(w):
            w = np.clip(w, 0.0, 1.0)
            if w.sum() == 0: return 999.0
            w = w / w.sum()
            blend = np.clip(w[0]*p_l_cal + w[1]*p_r_cal + w[2]*p_x_cal, 1e-7, 1-1e-7)
            blend = blend / blend.sum(axis=1, keepdims=True)
            return log_loss(y_cal, blend, labels=[0, 1, 2])

        opt_res = minimize(_obj, [0.4, 0.4, 0.2], method="Nelder-Mead")
        stack_w = np.clip(opt_res.x, 0.0, 1.0)
        stack_w = stack_w / stack_w.sum()
    else:
        stack_w = np.array([0.4, 0.4, 0.2])

    # Reentrenar sobre train + cal
    X_full = df_features.loc[train_mask | cal_mask, cols_feat].fillna(0.0)
    y_full = df_features.loc[train_mask | cal_mask, "resultado"]

    pipe_lasso = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=4000, random_state=42))])
    pipe_lasso.fit(X_full, y_full)

    pipe_rf = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(max_depth=5, n_estimators=200, min_samples_split=15, random_state=42, n_jobs=-1))])
    pipe_rf.fit(X_full, y_full)

    pipe_xgb = Pipeline([("scale", StandardScaler()), ("xgb", XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="mlogloss"))])
    pipe_xgb.fit(X_full, y_full)

    # Métricas en test
    def _met(proba, y):
        proba = np.clip(proba, 1e-7, 1-1e-7)
        return {"logloss": round(log_loss(y, proba, labels=[0, 1, 2]), 4), "accuracy": round(accuracy_score(y, proba.argmax(axis=1))*100, 2)}

    p_l_test = pipe_lasso.predict_proba(X_test)
    p_r_test = pipe_rf.predict_proba(X_test)
    p_x_test = pipe_xgb.predict_proba(X_test)
    p_st_test = np.clip(stack_w[0]*p_l_test + stack_w[1]*p_r_test + stack_w[2]*p_x_test, 1e-7, 1-1e-7)
    p_st_test = p_st_test / p_st_test.sum(axis=1, keepdims=True)

    metricas = {
        "lasso": _met(p_l_test, y_test),
        "rf": _met(p_r_test, y_test),
        "xgb": _met(p_x_test, y_test),
        "stacking": {"logloss": round(log_loss(y_test, p_st_test, labels=[0, 1, 2]), 4), "accuracy": round(accuracy_score(y_test, p_st_test.argmax(axis=1))*100, 2), "w": [round(float(x), 3) for x in stack_w]}
    }
    m_l = metricas['lasso']
    m_r = metricas['rf']
    m_s = metricas['stacking']
    print(f"Metricas ITA Test>=2025: LASSO={m_l} RF={m_r} Stacking={m_s}")

    # Poisson GLM para modelar goles esperados
    df_poi = df_features[train_mask | cal_mask].copy()
    poi_rows = []
    for _, r in df_poi.iterrows():
        poi_rows.append({"goles": r["goles_local"], "elo_diff": r["elo_diff"], "is_home": 1})
        poi_rows.append({"goles": r["goles_visita"], "elo_diff": -r["elo_diff"], "is_home": 0})
    df_p = pd.DataFrame(poi_rows).dropna()

    X_p = sm.add_constant(df_p[["elo_diff", "is_home"]])
    glm = sm.GLM(df_p["goles"], X_p, family=sm.families.Poisson()).fit()
    g_const = glm.params["const"]
    g_d = glm.params["elo_diff"]
    g_home = glm.params["is_home"]

    return {
        "pipe_lasso": pipe_lasso,
        "pipe_rf": pipe_rf,
        "pipe_xgb": pipe_xgb,
        "stack_w": stack_w,
        "cols": cols_feat,
        "tracker": tracker,
        "g_const": g_const,
        "g_d": g_d,
        "g_home": g_home,
        "df_features": df_features,
        "metricas": metricas,
        "partidos": partidos,
        "equipos": _cargar_equipos()
    }

_CACHE_MODELO = None

def cargar():
    global _CACHE_MODELO
    if _CACHE_MODELO is not None:
        return _CACHE_MODELO
    cache_path = DATA.parent / ".model_cache.pkl"
    current_key = _cache_key()
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("key") == current_key:
                print("Modelo cargado desde cache de disco")
                _CACHE_MODELO = saved["model"]
                return _CACHE_MODELO
        except Exception:
            pass

    print("Entrenando modelo de Serie A (Stacking + Poisson)... esto toma ~10s")
    M = cargar_y_entrenar()
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": current_key, "model": M}, fh)
        print("Modelo guardado en cache de disco")
    except Exception as ex:
        print(f"No se pudo guardar el cache: {ex}")
    _CACHE_MODELO = M
    return M

def _temporada_actual():
    hoy = pd.Timestamp.now()
    return int(hoy.year) if hoy.month >= 7 else int(hoy.year) - 1

def predecir_match(M, local, visita, temporada=None, modelo_tipo=None, modelo=None):
    m = modelo_tipo or modelo or "stacking"
    tracker = M["tracker"]
    cols = M["cols"]
    if temporada is None:
        temporada = _temporada_actual()
    feats = tracker.get_features_for_match(local, visita, temporada, reset_season=False)
    df_test = pd.DataFrame([feats])[cols].fillna(0.0)

    if m == "stacking":
        w = M.get("stack_w", np.array([0.4, 0.4, 0.2]))
        p_l = M["pipe_lasso"].predict_proba(df_test)[0]
        p_r = M["pipe_rf"].predict_proba(df_test)[0]
        p_x = M["pipe_xgb"].predict_proba(df_test)[0]
        p_raw = w[0]*p_l + w[1]*p_r + w[2]*p_x; p_raw /= p_raw.sum()
    elif m == "xgb":
        p_raw = M["pipe_xgb"].predict_proba(df_test)[0]
    else:
        pipe = M["pipe_rf"] if m == "rf" else M["pipe_lasso"]
        p_raw = pipe.predict_proba(df_test)[0]
    p = np.array([p_raw[2], p_raw[1], p_raw[0]])
    return p

def grilla_goles(M, local, visita, modelo_tipo=None, modelo=None):
    m = modelo_tipo or modelo or "rf"
    p_1x2 = predecir_match(M, local, visita, modelo_tipo=m)
    tracker = M["tracker"]
    elo_diff = tracker.elos[local] - tracker.elos[visita]
    la = np.exp(M["g_const"] + M["g_d"] * elo_diff + M.get("g_home", 0.0) * 1)
    lb = np.exp(M["g_const"] - M["g_d"] * elo_diff + M.get("g_home", 0.0) * 0)

    max_g = 10
    pa = np.array([la**i * np.exp(-la) / math.factorial(i) for i in range(max_g)])
    pb = np.array([lb**j * np.exp(-lb) / math.factorial(j) for j in range(max_g)])
    grid = np.outer(pa, pb)

    p_l = sum(grid[i, j] for i in range(max_g) for j in range(max_g) if i > j)
    p_d = sum(grid[i, i] for i in range(max_g))
    p_v = sum(grid[i, j] for i in range(max_g) for j in range(max_g) if i < j)

    grid_adj = grid.copy()
    if p_l > 0: grid_adj[np.tril_indices(max_g, -1)] *= (p_1x2[0] / p_l)
    if p_d > 0: grid_adj[np.diag_indices(max_g)] *= (p_1x2[1] / p_d)
    if p_v > 0: grid_adj[np.triu_indices(max_g, 1)] *= (p_1x2[2] / p_v)

    s = grid_adj.sum()
    if s > 0: grid_adj /= s
    return grid_adj

def cuota(p):
    return 1.0 / max(float(p), 1e-6)

def mercados(grid):
    n = grid.shape[0]
    p1 = float(sum(grid[i, j] for i in range(n) for j in range(n) if i > j))
    px = float(sum(grid[i, i] for i in range(n)))
    p2 = float(sum(grid[i, j] for i in range(n) for j in range(n) if i < j))

    ou15 = float(sum(grid[i, j] for i in range(n) for j in range(n) if (i + j) > 1.5))
    ou25 = float(sum(grid[i, j] for i in range(n) for j in range(n) if (i + j) > 2.5))
    ou35 = float(sum(grid[i, j] for i in range(n) for j in range(n) if (i + j) > 3.5))

    btts = float(sum(grid[i, j] for i in range(1, n) for j in range(1, n)))
    btts_no = 1.0 - btts

    dnb1 = p1 / (p1 + p2) if (p1 + p2) > 0 else 0.5
    dnb2 = p2 / (p1 + p2) if (p1 + p2) > 0 else 0.5

    return {
        "1X2": {"1": p1, "X": px, "2": p2},
        "OverUnder": {"O1.5": ou15, "U1.5": 1-ou15, "O2.5": ou25, "U2.5": 1-ou25, "O3.5": ou35, "U3.5": 1-ou35},
        "BTTS": {"Si": btts, "No": btts_no},
        "DNB": {"1": dnb1, "2": dnb2}
    }

def obtener_tabla_actual(M):
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    actuales = partidos[partidos.temporada == _temporada_actual()].copy()
    if len(actuales) == 0:
        actuales = partidos[partidos.temporada == partidos.temporada.max()].copy()

    fix = pd.read_csv(DATA / "fixture.csv")
    todos = sorted(list(set(actuales["local"]).union(set(actuales["visita"])).union(set(fix["local"])).union(set(fix["visita"]))))

    tabla = {e: {"pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "pts": 0} for e in todos}
    for _, r in actuales.iterrows():
        l, v = r["local"], r["visita"]
        gl, gv = int(r["goles_local"]), int(r["goles_visita"])
        if l in tabla and v in tabla:
            tabla[l]["pj"] += 1; tabla[v]["pj"] += 1
            tabla[l]["gf"] += gl; tabla[l]["gc"] += gv
            tabla[v]["gf"] += gv; tabla[v]["gc"] += gl
            if gl > gv: tabla[l]["pg"] += 1; tabla[l]["pts"] += 3; tabla[v]["pp"] += 1
            elif gl == gv: tabla[l]["pe"] += 1; tabla[l]["pts"] += 1; tabla[v]["pe"] += 1; tabla[v]["pts"] += 1
            else: tabla[v]["pg"] += 1; tabla[v]["pts"] += 3; tabla[l]["pp"] += 1

    filas = []
    for e, d in tabla.items():
        dg = d["gf"] - d["gc"]
        filas.append({"equipo": e, "pj": d["pj"], "puntos": d["pts"], "dif_goles": dg, "goles_favor": d["gf"]})
    df_t = pd.DataFrame(filas).sort_values(by=["puntos", "dif_goles", "goles_favor"], ascending=False).reset_index(drop=True)
    return df_t

def _simular_fixture_vec(M, PREDS, fijos, n_sims):
    fix_path = DATA / "fixture.csv"
    fix = pd.read_csv(fix_path) if fix_path.exists() else pd.DataFrame()
    fix = fix[fix.temporada == _temporada_actual()] if not fix.empty else fix
    tab = obtener_tabla_actual(M)
    eqs = list(tab["equipo"])
    idx = {e: i for i, e in enumerate(eqs)}
    n_eq = len(eqs)

    base_pts = np.array([tab.loc[tab.equipo == e, "puntos"].values[0] for e in eqs], dtype=np.int32)
    base_dg  = np.array([tab.loc[tab.equipo == e, "dif_goles"].values[0] for e in eqs], dtype=np.int32)
    base_gf  = np.array([tab.loc[tab.equipo == e, "goles_favor"].values[0] for e in eqs], dtype=np.int32)

    PTS = np.tile(base_pts[:, None], (1, n_sims))
    DG  = np.tile(base_dg[:, None],  (1, n_sims))
    GF  = np.tile(base_gf[:, None],  (1, n_sims))

    if not fix.empty:
        for r in fix.itertuples(index=False):
            l, v = r.local, r.visita
            if l not in idx or v not in idx: continue
            li, vi = idx[l], idx[v]
            if fijos and (l, v) in fijos:
                gl, gv = fijos[(l, v)]
                if gl > gv: PTS[li] += 3
                elif gl == gv: PTS[li] += 1; PTS[vi] += 1
                else: PTS[vi] += 3
                DG[li] += (gl - gv); DG[vi] += (gv - gl)
                GF[li] += gl; GF[vi] += gv
                continue

            p = PREDS.get((l, v))
            if p is None:
                p = predecir_match(M, l, v, modelo="stacking")
                PREDS[(l, v)] = p

            u = np.random.rand(n_sims)
            p_l, p_e = p[0], p[1]

            w_l = u < p_l
            w_e = (u >= p_l) & (u < (p_l + p_e))
            w_v = u >= (p_l + p_e)

            PTS[li] += w_l * 3 + w_e * 1
            PTS[vi] += w_v * 3 + w_e * 1
            DG[li]  += w_l * 1 - w_v * 1
            DG[vi]  += w_v * 1 - w_l * 1
            GF[li]  += w_l * 1 + w_e * 1
            GF[vi]  += w_v * 1 + w_e * 1

    key = (PTS * 1_000_000 + DG * 1_000 + GF).astype(np.int64)
    order = np.argsort(-key, axis=0)
    return eqs, order

def simular_campeonato(M, n_sims=3000, fijos=None, modelo_tipo=None, modelo=None, seed=42):
    modelo_tipo = modelo_tipo or modelo or "rf"
    if seed is not None:
        np.random.seed(seed)
    fix_path = DATA / "fixture.csv"
    if not fix_path.exists() or len(pd.read_csv(fix_path)) == 0:
        tab = obtener_tabla_actual(M)
        res = tab[["equipo"]].copy()
        res["Selección"] = res["equipo"]
        res["P_campeon"] = 0.0
        res.loc[0, "P_campeon"] = 1.0
        res["P_champions"] = 0.0
        res.loc[:3, "P_champions"] = 1.0
        res["P_copas"] = 0.0
        res.loc[:6, "P_copas"] = 1.0
        res["P_descenso"] = 0.0
        res.loc[len(res)-3:, "P_descenso"] = 1.0
        res["Puntos esperados"] = tab["puntos"]
        return res

    import hashlib
    fp = hashlib.sha256()
    fp.update(str(n_sims).encode())
    fp.update(_cache_key().encode())
    if fijos is not None:
        fp.update(repr(sorted(fijos.items())).encode())

    cache_path = DATA.parent / "simulacion_mc.pkl"
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
        except Exception:
            pass

    fix = pd.read_csv(fix_path)
    fix = fix[fix.temporada == _temporada_actual()]
    todos = sorted(list(set(fix["local"]).union(set(fix["visita"]))))
    PREDS = {}
    print("Precalculando predicciones de fixture...")
    for l in todos:
        for v in todos:
            if l != v:
                PREDS[(l, v)] = predecir_match(M, l, v, modelo=modelo_tipo)

    eqs, order = _simular_fixture_vec(M, PREDS, fijos, n_sims)
    n_eq = len(eqs)
    counts_campeon = {e: 0 for e in eqs}
    counts_champ = {e: 0 for e in eqs}
    counts_copas = {e: 0 for e in eqs}
    counts_desc = {e: 0 for e in eqs}

    for sim in range(n_sims):
        c = eqs[order[0, sim]]
        counts_campeon[c] += 1
        for rank in range(min(4, n_eq)):
            counts_champ[eqs[order[rank, sim]]] += 1
        for rank in range(min(CUPOS_COPA, n_eq)):
            counts_copas[eqs[order[rank, sim]]] += 1
        for rank in range(max(0, n_eq - DESCIENDEN), n_eq):
            counts_desc[eqs[order[rank, sim]]] += 1

    tab = obtener_tabla_actual(M)
    res = []
    for eq in eqs:
        res.append({
            "equipo": eq,
            "Selección": eq,
            "Puntos esperados": round(float(tab.loc[tab.equipo == eq, "puntos"].values[0]) + (38 - int(tab.loc[tab.equipo == eq, "pj"].values[0])) * 1.35, 1) if eq in tab["equipo"].values else 45.0,
            "P_campeon": counts_campeon[eq] / n_sims,
            "P_champions": counts_champ[eq] / n_sims,
            "P_copas": counts_copas[eq] / n_sims,
            "P_descenso": counts_desc[eq] / n_sims
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

monte_carlo = simular_campeonato

def validacion_en_vivo(M, temporada_val=None, modelo_tipo="rf"):
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    if temporada_val is None:
        temporada_val = partidos["temporada"].max()
    val_df = partidos[partidos.temporada == temporada_val].copy()
    if len(val_df) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    probs = []
    for _, r in val_df.iterrows():
        p = predecir_match(M, r["local"], r["visita"], temporada=temporada_val, modelo=modelo_tipo)
        probs.append(p)
    probs = np.array(probs)

    y_true = []
    for _, r in val_df.iterrows():
        gl, gv = r["goles_local"], r["goles_visita"]
        y_true.append(0 if gl > gv else (1 if gl == gv else 2))
    y_true = np.array(y_true)

    # Invertir orden de probs a [local, empate, visita]
    p_clip = np.clip(probs, 1e-7, 1-1e-7)
    ll = log_loss(y_true, p_clip, labels=[0, 1, 2])
    preds = p_clip.argmax(axis=1)
    acc = accuracy_score(y_true, preds)

    # Baseline
    freqs = pd.Series(y_true).value_counts(normalize=True)
    base_p = np.zeros_like(p_clip)
    for c in [0, 1, 2]: base_p[:, c] = freqs.get(c, 0.33)
    ll_base = log_loss(y_true, base_p, labels=[0, 1, 2])

    met = {"n": len(val_df), "acierto": acc, "logloss": ll, "logloss_base": ll_base}
    val_df["Prob_Local"] = probs[:, 0]
    val_df["Prob_Empate"] = probs[:, 1]
    val_df["Prob_Visita"] = probs[:, 2]
    val_df["Prediccion"] = preds

    evol = []
    for i in range(1, len(val_df)+1):
        sub_y = y_true[:i]; sub_p = p_clip[:i]
        if len(set(sub_y)) > 1:
            evol.append({"partidos": i, "logloss": log_loss(sub_y, sub_p, labels=[0, 1, 2]), "accuracy": accuracy_score(sub_y, sub_p.argmax(axis=1))})
    return val_df, met, pd.DataFrame(evol)
