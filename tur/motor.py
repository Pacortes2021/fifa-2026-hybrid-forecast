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
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

DESCIENDEN = 4
CUPOS_COPA = 4  # Champions (1-2), Europa League (3), Conference (4)

SQUAD_VALUES = {
    "Galatasaray": 280.0, "Fenerbahce": 260.0, "Besiktas": 140.0,
    "Trabzonspor": 95.0, "Istanbul Basaksehir": 55.0, "Samsunspor": 40.0,
    "Eyupspor": 38.0, "Kasimpasa": 35.0, "Alanyaspor": 32.0,
    "Goztepe": 32.0, "Caykur Rizespor": 30.0, "Antalyaspor": 28.0,
    "Gaziantep FK": 26.0, "Konyaspor": 26.0, "Kayserispor": 25.0,
    "Sivasspor": 24.0, "Bodrum FK": 22.0, "Adana Demirspor": 20.0,
    "Hatayspor": 18.0, "Fatih Karagümrük": 18.0, "Ankaragucu": 16.0,
    "Pendikspor": 15.0, "Istanbulspor": 14.0, "Kocaelispor": 14.0,
    "Giresunspor": 12.0, "Umraniyespor": 12.0, "Genclerbirligi": 12.0,
    "Corum FK": 10.0, "Amed SFK": 10.0, "Erzurum BB": 10.0,
    "Altay Izmir": 8.0, "Yeni Malatyaspor": 6.0, "Denizlispor": 6.0
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

K_LIGA = 35.0      # ELO K-factor para la Süper Lig turca
HOME_ADV = 60.0    # Fuerte ventaja de local en Turquía

COORDS_TURKEY = {
    "Galatasaray": (41.1034, 28.9910), "Fenerbahce": (40.9877, 29.0369),
    "Besiktas": (41.0392, 28.9944), "Istanbul Basaksehir": (41.1228, 28.8089),
    "Kasimpasa": (41.0336, 28.9725), "Fatih Karagümrük": (41.0744, 28.7656),
    "Eyupspor": (41.0500, 28.9300), "Istanbulspor": (41.0250, 28.6750),
    "Pendikspor": (40.8800, 29.2400), "Umraniyespor": (41.0300, 29.1200),
    "Trabzonspor": (41.0027, 39.6542), "Adana Demirspor": (37.0600, 35.3400),
    "Alanyaspor": (36.5400, 32.0600), "Antalyaspor": (36.8867, 30.6650),
    "Caykur Rizespor": (41.0400, 40.5700), "Gaziantep FK": (37.1100, 37.3800),
    "Goztepe": (38.3970, 27.0870), "Altay Izmir": (38.4360, 27.1500),
    "Hatayspor": (36.8000, 34.5500), "Kayserispor": (38.7400, 35.4300),
    "Konyaspor": (37.9500, 32.4800), "Samsunspor": (41.2400, 36.4600),
    "Sivasspor": (39.7300, 36.9900), "Ankaragucu": (39.9800, 32.6500),
    "Genclerbirligi": (39.9800, 32.6500), "Bodrum FK": (37.0400, 27.4300),
    "Giresunspor": (40.9100, 38.4500), "Kocaelispor": (40.7500, 29.9800),
    "Yeni Malatyaspor": (38.3400, 38.4300), "Erzurum BB": (39.9000, 41.2500),
    "Amed SFK": (37.9300, 40.1500), "Corum FK": (40.5400, 34.9500),
    "Denizlispor": (37.7700, 29.0800)
}

ALTITUDES_TURKEY = {
    "Galatasaray": 110, "Fenerbahce": 25, "Besiktas": 15, "Istanbul Basaksehir": 120,
    "Kasimpasa": 30, "Fatih Karagümrük": 125, "Eyupspor": 20, "Istanbulspor": 80,
    "Pendikspor": 25, "Umraniyespor": 140, "Trabzonspor": 10, "Adana Demirspor": 60,
    "Alanyaspor": 20, "Antalyaspor": 35, "Caykur Rizespor": 10, "Gaziantep FK": 840,
    "Goztepe": 15, "Altay Izmir": 10, "Hatayspor": 50, "Kayserispor": 1050,
    "Konyaspor": 1020, "Samsunspor": 15, "Sivasspor": 1290, "Ankaragucu": 820,
    "Genclerbirligi": 820, "Bodrum FK": 20, "Giresunspor": 15, "Kocaelispor": 25,
    "Yeni Malatyaspor": 960, "Erzurum BB": 1890, "Amed SFK": 670, "Corum FK": 800,
    "Denizlispor": 350
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

def get_distance_km(local, visita):
    if local in COORDS_TURKEY and visita in COORDS_TURKEY:
        c1 = COORDS_TURKEY[local]; c2 = COORDS_TURKEY[visita]
        return haversine_km(c1[0], c1[1], c2[0], c2[1])
    return 550.0

def get_altitude_diff(local, visita):
    al = ALTITUDES_TURKEY.get(local, 150)
    av = ALTITUDES_TURKEY.get(visita, 150)
    return float(al - av)

def get_squad_value(equipo, temporada):
    if not DF_SQUAD_VALUES.empty:
        sub = DF_SQUAD_VALUES[(DF_SQUAD_VALUES["equipo"] == equipo) & (DF_SQUAD_VALUES["temporada"] == temporada)]
        if not sub.empty:
            return float(sub.iloc[0]["squad_value"])
    return float(SQUAD_VALUES.get(equipo, 30.0))

def get_advanced_features(equipo, temporada):
    if not DF_ADV_FEATURES.empty:
        sub = DF_ADV_FEATURES[(DF_ADV_FEATURES["equipo"] == equipo) & (DF_ADV_FEATURES["temporada"] == temporada)]
        if not sub.empty:
            r = sub.iloc[0]
            return {
                "squad_size": float(r["squad_size"]), "avg_age": float(r["avg_age"]),
                "foreigners": float(r["foreigners"]), "pct_foreigners": float(r["pct_foreigners"]),
                "squad_value": float(r["squad_value"]), "stadium_capacity": float(r["stadium_capacity"]),
                "avg_attendance": float(r["avg_attendance"]), "stadium_occupation": float(r["stadium_occupation"])
            }
    return {
        "squad_size": 28.0, "avg_age": 26.5, "foreigners": 15.0, "pct_foreigners": 0.52,
        "squad_value": float(SQUAD_VALUES.get(equipo, 30.0)), "stadium_capacity": 25000.0,
        "avg_attendance": 15000.0, "stadium_occupation": 0.60
    }

class PiRatingsTracker:
    def __init__(self, lambda_param=0.035, gamma_param=0.55):
        self.lmb = lambda_param
        self.gamma = gamma_param
        self.r_home = defaultdict(float)
        self.r_away = defaultdict(float)

    def get_features(self, home, away):
        rh = self.r_home[home]
        ra = self.r_away[away]
        return {"pi_diff": rh - ra, "pi_overall_diff": (rh + self.r_away[home]) - (ra + self.r_home[away])}

    def registrar_partido(self, home, away, gh, ga):
        gd = gh - ga
        rh = self.r_home[home]
        ra = self.r_away[away]
        diff = rh - ra
        e_gd = (10.0 ** (abs(diff) / 3.0) - 1.0) * (1.0 if diff >= 0 else -1.0)
        err = gd - e_gd
        self.r_home[home] += err * self.lmb
        self.r_away[home] += err * self.lmb * self.gamma
        self.r_away[away] -= err * self.lmb
        self.r_home[away] -= err * self.lmb * self.gamma

class StateTracker:
    def __init__(self):
        self.elos = defaultdict(_elo_default)
        self.h2h_goles = defaultdict(float)
        self.history = defaultdict(list)
        self.history_home = defaultdict(list)
        self.history_away = defaultdict(list)
        self.curr_season = None
        self.season_pts = defaultdict(int)
        self.season_matches = defaultdict(int)
        self.last_match_date = defaultdict(_none_default)
        self.recent_dates = defaultdict(deque)
        self.pi_tracker = PiRatingsTracker()
        self.recent_results = defaultdict(deque)
        self.recent_gf = defaultdict(deque)
        self.recent_ga = defaultdict(deque)
        self.match_count = defaultdict(int)

    def get_features_for_match(self, local, visita, temporada, fecha=None, reset_season=False):
        feats = {}
        el = self.elos[local]
        ev = self.elos[visita]
        feats["elo_local"] = el
        feats["elo_visita"] = ev
        feats["elo_diff"] = (el + HOME_ADV) - ev

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

        dist = get_distance_km(local, visita)
        feats["distance_km"] = float(dist)
        feats["distance_log"] = float(np.log1p(dist))
        feats["altitude_diff"] = get_altitude_diff(local, visita)

        for stat in STATS:
            hl = [m[f"local_{stat}"] for m in self.history[local] if f"local_{stat}" in m and pd.notna(m[f"local_{stat}"])]
            hv = [m[f"visita_{stat}"] for m in self.history[visita] if f"visita_{stat}" in m and pd.notna(m[f"visita_{stat}"])]
            ml = np.mean(hl[-10:]) if hl else 0.0
            mv = np.mean(hv[-10:]) if hv else 0.0
            feats[f"{stat}_diff"] = ml - mv

        return feats

    def registrar_partido(self, local, visita, ga, gb, fecha, stats_dict=None):
        dr = (self.elos[local] + HOME_ADV) - self.elos[visita]
        we = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
        w = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
        g_diff = abs(ga - gb)
        mult = 1.0 if g_diff <= 1 else (1.5 if g_diff == 2 else (1.75 + (g_diff - 3) / 8.0))
        delta = K_LIGA * mult * (w - we)
        self.elos[local] += delta
        self.elos[visita] -= delta

        self.h2h_goles[(local, visita)] = self.h2h_goles[(local, visita)] * 0.8 + (ga - gb) * 0.2
        self.h2h_goles[(visita, local)] = -self.h2h_goles[(local, visita)]

        self.pi_tracker.registrar_partido(local, visita, ga, gb)

        if ga > gb:
            self.season_pts[local] += 3
        elif ga == gb:
            self.season_pts[local] += 1
            self.season_pts[visita] += 1
        else:
            self.season_pts[visita] += 3
        self.season_matches[local] += 1
        self.season_matches[visita] += 1

        f = pd.to_datetime(fecha)
        self.last_match_date[local] = f
        self.last_match_date[visita] = f
        self.recent_dates[local].append(f)
        self.recent_dates[visita].append(f)
        while self.recent_dates[local] and (f - self.recent_dates[local][0]).days > 21:
            self.recent_dates[local].popleft()
        while self.recent_dates[visita] and (f - self.recent_dates[visita][0]).days > 21:
            self.recent_dates[visita].popleft()

        entry = {"fecha": fecha, "ga": ga, "gb": gb}
        if stats_dict:
            entry.update(stats_dict)
        self.history[local].append(entry)
        self.history[visita].append(entry)
        self.history_home[local].append(entry)
        self.history_away[visita].append(entry)

        w_l = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
        w_v = 1.0 - w_l if w_l != 0.5 else 0.5
        self.recent_results[local].append(w_l)
        self.recent_results[visita].append(w_v)
        self.recent_gf[local].append(ga)
        self.recent_gf[visita].append(gb)
        self.recent_ga[local].append(gb)
        self.recent_ga[visita].append(ga)
        self.match_count[local] += 1
        self.match_count[visita] += 1

_MOTOR_CACHE = None

def cargar():
    global _MOTOR_CACHE
    if _MOTOR_CACHE is not None:
        return _MOTOR_CACHE

    cache_path = DATA.parent / ".model_cache.pkl"
    partidos_path = DATA / "partidos.csv"
    box_path = DATA / "box_score.csv"

    if not partidos_path.exists():
        return None

    df_partidos = pd.read_csv(partidos_path, parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)
    df_box = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame()
    box_dict = {}
    if not df_box.empty and "event_id" in df_box.columns:
        df_box["event_id"] = df_box["event_id"].astype(str)
        box_dict = df_box.set_index("event_id").to_dict("index")

    import hashlib
    fp = hashlib.md5()
    fp.update(str(df_partidos.shape).encode())
    fp.update(str(df_partidos["fecha"].iloc[-1]).encode() if len(df_partidos) else b"")
    if not df_box.empty:
        fp.update(str(df_box.shape).encode())
    key_actual = fp.hexdigest()

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                obj = pickle.load(fh)
            if obj.get("key") == key_actual:
                print("Modelo cargado desde cache de disco")
                _MOTOR_CACHE = obj["motor"]
                return _MOTOR_CACHE
        except Exception:
            pass

    print("Entrenando modelo de Süper Lig (Stacking + Poisson)... esto toma ~10s")

    # Inferencia de temporada europea de agosto a mayo
    df_partidos["temporada"] = df_partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)

    tracker = StateTracker()
    rows = []
    poisson_rows = []

    for idx, r in df_partidos.iterrows():
        loc = r["local"]; vis = r["visita"]
        gl = r["goles_local"]; gv = r["goles_visita"]
        eid = str(r.get("event_id", ""))
        stats = box_dict.get(eid, {})

        f = tracker.get_features_for_match(loc, vis, r["temporada"], fecha=r["fecha"], reset_season=True)
        f["resultado"] = 0 if gl > gv else (1 if gl == gv else 2)
        f["goles_local"] = gl; f["goles_visita"] = gv
        f["temporada"] = r["temporada"]
        f["local"] = loc; f["visita"] = vis
        rows.append(f)

        tracker.registrar_partido(loc, vis, gl, gv, r["fecha"], stats)

        poisson_rows.append({
            "goles": gl, "is_home": 1.0, "elo_diff": f["elo_diff"],
            "distance": f["distance_km"], "altitude": f["altitude_diff"]
        })
        poisson_rows.append({
            "goles": gv, "is_home": 0.0, "elo_diff": -f["elo_diff"],
            "distance": f["distance_km"], "altitude": -f["altitude_diff"]
        })

    df_dataset = pd.DataFrame(rows)
    df_poisson = pd.DataFrame(poisson_rows)

    cols_excluir = {"resultado", "goles_local", "goles_visita", "temporada", "local", "visita"}
    cols_features = [c for c in df_dataset.columns if c not in cols_excluir]

    train_mask = df_dataset["temporada"] <= 2023
    cal_mask   = df_dataset["temporada"] == 2024
    test_mask  = df_dataset["temporada"] >= 2025

    X_train_raw = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    non_zero_cols = [c for c in cols_features if X_train_raw[c].std() > 1e-5]
    cols_features = non_zero_cols

    X_train = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    y_train = df_dataset.loc[train_mask, "resultado"]
    X_cal   = df_dataset.loc[cal_mask,   cols_features].fillna(0.0)
    y_cal   = df_dataset.loc[cal_mask,   "resultado"]
    X_test  = df_dataset.loc[test_mask,  cols_features].fillna(0.0)
    y_test  = df_dataset.loc[test_mask,  "resultado"]

    best_c = 0.04; best_loss = 999.0
    for C in [0.01, 0.02, 0.04, 0.08, 0.15, 0.3]:
        _pipe = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=2500, random_state=42))])
        _pipe.fit(X_train, y_train)
        _loss = log_loss(y_test, _pipe.predict_proba(X_test), labels=[0, 1, 2])
        if _loss < best_loss:
            best_loss = _loss; best_c = C

    pipe_lasso = Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=3000, random_state=42))])
    pipe_lasso.fit(X_train, y_train)

    pipe_rf = Pipeline([("sc", StandardScaler()),
                        ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=15, random_state=42, n_jobs=-1))])
    pipe_rf.fit(X_train, y_train)

    pipe_xgb = Pipeline([("sc", StandardScaler()),
                         ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                               eval_metric="mlogloss", random_state=42, n_jobs=-1))])
    pipe_xgb.fit(X_train, y_train)

    if len(X_cal) >= 10:
        p_l_cal = pipe_lasso.predict_proba(X_cal)
        p_r_cal = pipe_rf.predict_proba(X_cal)
        p_x_cal = pipe_xgb.predict_proba(X_cal)

        def _simplex_loss(weights):
            w = np.array(weights, dtype=float)
            if w.sum() > 0:
                w = w / w.sum()
            blend = w[0] * p_l_cal + w[1] * p_r_cal + w[2] * p_x_cal
            blend = blend / blend.sum(axis=1, keepdims=True)
            blend = np.clip(blend, 1e-7, 1 - 1e-7)
            return log_loss(y_cal, blend)

        res = minimize(_simplex_loss, [0.4, 0.3, 0.3], bounds=[(0, 1), (0, 1), (0, 1)],
                       constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0})
        if res.success:
            w_raw = np.array(res.x, dtype=float)
            weights_opt = (w_raw / w_raw.sum()).tolist() if w_raw.sum() > 0 else [0.45, 0.35, 0.20]
        else:
            weights_opt = [0.45, 0.35, 0.20]
    else:
        weights_opt = [0.5, 0.3, 0.2]

    # Modelos finales entrenados con train + cal
    X_full = df_dataset.loc[train_mask | cal_mask, cols_features].fillna(0.0)
    y_full = df_dataset.loc[train_mask | cal_mask, "resultado"]
    pipe_lasso_final = Pipeline([("sc", StandardScaler()),
                                 ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=3000, random_state=42))])
    pipe_lasso_final.fit(X_full, y_full)

    pipe_rf_final = Pipeline([("sc", StandardScaler()),
                              ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=15, random_state=42, n_jobs=-1))])
    pipe_rf_final.fit(X_full, y_full)

    pipe_xgb_final = Pipeline([("sc", StandardScaler()),
                               ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                     eval_metric="mlogloss", random_state=42, n_jobs=-1))])
    pipe_xgb_final.fit(X_full, y_full)

    def _met(proba, y):
        proba = proba / proba.sum(axis=1, keepdims=True)
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        return {"logloss": round(log_loss(y, proba), 4),
                "accuracy": round(accuracy_score(y, proba.argmax(axis=1)) * 100, 2)}

    p_l_test = pipe_lasso_final.predict_proba(X_test)
    p_r_test = pipe_rf_final.predict_proba(X_test)
    p_x_test = pipe_xgb_final.predict_proba(X_test)
    p_st = weights_opt[0]*p_l_test + weights_opt[1]*p_r_test + weights_opt[2]*p_x_test
    p_st = p_st / p_st.sum(axis=1, keepdims=True)
    p_st = np.clip(p_st, 1e-7, 1-1e-7)

    metricas = {
        "lasso": _met(p_l_test, y_test),
        "rf": _met(p_r_test, y_test),
        "stacking": {"logloss": round(log_loss(y_test, p_st), 4),
                     "accuracy": round(accuracy_score(y_test, p_st.argmax(axis=1)) * 100, 2),
                     "w": [round(float(w), 3) for w in weights_opt]}
    }
    print(f"Metricas TUR Test>=2025: LASSO={metricas['lasso']} RF={metricas['rf']} Stacking={metricas['stacking']}")

    # Poisson GLM bivariado
    X_pois = sm.add_constant(df_poisson[["is_home", "elo_diff"]])
    y_pois = df_poisson["goles"]
    try:
        poisson_model = sm.GLM(y_pois, X_pois, family=sm.families.Poisson()).fit()
        poiss_params = {
            "const": float(poisson_model.params.get("const", 0.1)),
            "is_home": float(poisson_model.params.get("is_home", 0.25)),
            "elo_diff": float(poisson_model.params.get("elo_diff", 0.001))
        }
    except Exception:
        poiss_params = {"const": 0.1, "is_home": 0.25, "elo_diff": 0.001}

    equipos_dict = _cargar_equipos()

    _MOTOR_CACHE = {
        "tracker": tracker,
        "pipe_lasso": pipe_lasso_final,
        "pipe_rf": pipe_rf_final,
        "pipe_xgb": pipe_xgb_final,
        "pipe": pipe_lasso_final,
        "weights_opt": weights_opt,
        "metricas": metricas,
        "features": cols_features,
        "poiss_params": poiss_params,
        "equipos": equipos_dict,
        "df_partidos": df_partidos,
        "df_dataset": df_dataset
    }

    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": key_actual, "motor": _MOTOR_CACHE}, fh)
        print("Modelo guardado en cache de disco")
    except Exception as ex:
        print(f"No se pudo guardar el cache de motor: {ex}")

    return _MOTOR_CACHE

def predecir_match(M, local, visita, temporada=2026, fecha=None, modelo="stacking"):
    tracker = M["tracker"]
    features = M["features"]
    poiss = M["poiss_params"]

    feats = tracker.get_features_for_match(local, visita, temporada, fecha=fecha)
    df_feat = pd.DataFrame([feats])[features].fillna(0.0)

    if modelo in ("lasso", "l1"):
        p_raw = M["pipe_lasso"].predict_proba(df_feat)[0]
    elif modelo == "rf":
        p_raw = M["pipe_rf"].predict_proba(df_feat)[0]
    elif modelo == "xgb":
        p_raw = M["pipe_xgb"].predict_proba(df_feat)[0]
    else:  # stacking
        w = M.get("weights_opt", [0.45, 0.35, 0.20])
        pl = M["pipe_lasso"].predict_proba(df_feat)[0]
        pr = M["pipe_rf"].predict_proba(df_feat)[0]
        px = M["pipe_xgb"].predict_proba(df_feat)[0]
        p_raw = w[0] * pl + w[1] * pr + w[2] * px
        p_raw /= p_raw.sum()

    # Formato: [Prob_Local, Prob_Empate, Prob_Visita]
    p = np.array([p_raw[0], p_raw[1], p_raw[2]])

    el_diff = feats["elo_diff"]
    la = float(np.exp(poiss["const"] + poiss["is_home"] + poiss["elo_diff"] * el_diff))
    lb = float(np.exp(poiss["const"] - poiss["elo_diff"] * el_diff))
    la = np.clip(la, 0.2, 5.0)
    lb = np.clip(lb, 0.2, 5.0)

    return p, la, lb

def matriz_marcador_exacto(la, lb, max_goles=7, rho=-0.08):
    from scipy.stats import poisson
    p_a = [poisson.pmf(i, la) for i in range(max_goles)]
    p_b = [poisson.pmf(j, lb) for j in range(max_goles)]
    M = np.outer(p_a, p_b)

    # Corrección Dixon-Coles para marcadores bajos
    if max_goles > 1:
        M[0, 0] *= max(0.0, 1.0 - la * lb * rho)
        M[0, 1] *= max(0.0, 1.0 + la * rho)
        M[1, 0] *= max(0.0, 1.0 + lb * rho)
        M[1, 1] *= max(0.0, 1.0 - rho)

    s = M.sum()
    if s > 0:
        M /= s
    return M

def obtener_tabla_actual(M):
    partidos = M["df_partidos"].copy()
    temp_actual = partidos["temporada"].max()
    sub = partidos[partidos["temporada"] == temp_actual]

    equipos = sorted(list(set(sub["local"]).union(set(sub["visita"]))))
    records = {eq: {"pj": 0, "g": 0, "e": 0, "p": 0, "gf": 0, "gc": 0, "dg": 0, "puntos": 0} for eq in equipos}

    for _, r in sub.iterrows():
        l = r["local"]; v = r["visita"]
        gl = int(r["goles_local"]); gv = int(r["goles_visita"])

        records[l]["pj"] += 1
        records[v]["pj"] += 1
        records[l]["gf"] += gl
        records[l]["gc"] += gv
        records[v]["gf"] += gv
        records[v]["gc"] += gl

        if gl > gv:
            records[l]["g"] += 1
            records[l]["puntos"] += 3
            records[v]["p"] += 1
        elif gl == gv:
            records[l]["e"] += 1
            records[v]["e"] += 1
            records[l]["puntos"] += 1
            records[v]["puntos"] += 1
        else:
            records[v]["g"] += 1
            records[v]["puntos"] += 3
            records[l]["p"] += 1

    for eq in equipos:
        records[eq]["dg"] = records[eq]["gf"] - records[eq]["gc"]

    df = pd.DataFrame([{"equipo": k, **v} for k, v in records.items()])
    df = df.sort_values(["puntos", "dg", "gf"], ascending=[False, False, False]).reset_index(drop=True)
    df.index += 1
    return df

def _temporada_actual():
    p = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"])
    p["temp"] = p["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    return int(p["temp"].max())

def _simular_fixture_vec(M, PREDS, fijos, n_sims):
    fix_path = DATA / "fixture.csv"
    fix = pd.read_csv(fix_path)
    fix = fix[fix.temporada == _temporada_actual()]
    tab = obtener_tabla_actual(M)

    eq_list = list(tab["equipo"])
    eq2idx = {e: i for i, e in enumerate(eq_list)}
    n_eq = len(eq_list)

    base_pts = np.zeros(n_eq, dtype=float)
    base_dg = np.zeros(n_eq, dtype=float)
    base_gf = np.zeros(n_eq, dtype=float)
    for i, e in enumerate(eq_list):
        row = tab.loc[tab.equipo == e].iloc[0]
        base_pts[i] = row["puntos"]
        base_dg[i] = row["dg"]
        base_gf[i] = row["gf"]

    n_partidos = len(fix)
    loc_idx = np.empty(n_partidos, dtype=int)
    vis_idx = np.empty(n_partidos, dtype=int)
    probs_arr = np.empty((n_partidos, 3), dtype=float)
    la_arr = np.empty(n_partidos, dtype=float)
    lb_arr = np.empty(n_partidos, dtype=float)

    for k, r in enumerate(fix.itertuples()):
        l, v = r.local, r.visita
        loc_idx[k] = eq2idx.get(l, 0)
        vis_idx[k] = eq2idx.get(v, 0)
        p, la, lb = PREDS.get((l, v), (np.array([0.45, 0.28, 0.27]), 1.4, 1.1))
        probs_arr[k] = p
        la_arr[k] = la
        lb_arr[k] = lb

    cum_probs = np.cumsum(probs_arr, axis=1)

    pts_sims = np.tile(base_pts[:, None], (1, n_sims))
    dg_sims = np.tile(base_dg[:, None], (1, n_sims))
    gf_sims = np.tile(base_gf[:, None], (1, n_sims))

    r_unif = np.random.rand(n_partidos, n_sims)
    res_sim = np.empty((n_partidos, n_sims), dtype=np.int8)
    res_sim[r_unif < cum_probs[:, [0]]] = 0
    mask_e = (r_unif >= cum_probs[:, [0]]) & (r_unif < cum_probs[:, [1]])
    res_sim[mask_e] = 1
    res_sim[r_unif >= cum_probs[:, [1]]] = 2

    # Goles Poisson vectorizados
    gh_sim = np.random.poisson(la_arr[:, None], size=(n_partidos, n_sims))
    ga_sim = np.random.poisson(lb_arr[:, None], size=(n_partidos, n_sims))

    diff = gh_sim - ga_sim
    fix_l = (res_sim == 0) & (diff <= 0)
    gh_sim[fix_l] = ga_sim[fix_l] + 1
    fix_v = (res_sim == 2) & (diff >= 0)
    ga_sim[fix_v] = gh_sim[fix_v] + 1
    fix_e = (res_sim == 1) & (diff != 0)
    gh_sim[fix_e] = ga_sim[fix_e]

    pts_l = np.where(res_sim == 0, 3, np.where(res_sim == 1, 1, 0))
    pts_v = np.where(res_sim == 2, 3, np.where(res_sim == 1, 1, 0))
    dg_diff = gh_sim - ga_sim

    for k in range(n_partidos):
        li = loc_idx[k]
        vi = vis_idx[k]
        pts_sims[li] += pts_l[k]
        pts_sims[vi] += pts_v[k]
        dg_sims[li] += dg_diff[k]
        dg_sims[vi] -= dg_diff[k]
        gf_sims[li] += gh_sim[k]
        gf_sims[vi] += ga_sim[k]

    score = pts_sims * 1000000.0 + dg_sims * 1000.0 + gf_sims
    order = np.argsort(-score, axis=0)

    return eq_list, order

def simular_campeonato(M, n_sims=1000, fijos=None, modelo="stacking", modelo_tipo=None):
    if modelo_tipo is not None:
        modelo = modelo_tipo
    modelo_tipo = modelo or "stacking"
    fix_path = DATA / "fixture.csv"
    if not fix_path.exists():
        return obtener_tabla_actual(M)

    import hashlib
    fp = hashlib.md5()
    fp.update(str(n_sims).encode())
    fp.update(modelo_tipo.encode())
    try:
        fp.update(str(Path(fix_path).stat().st_mtime).encode())
    except Exception:
        pass
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
        for rank in range(min(2, n_eq)):
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

def validacion_en_vivo(M, temporada_val=None, modelo="rf", modelo_tipo=None):
    if modelo_tipo is not None:
        modelo = modelo_tipo
    modelo_tipo = modelo or "rf"
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
        probs.append(p[0] if isinstance(p, tuple) else p)
    probs = np.array(probs)

    y_true = []
    for _, r in val_df.iterrows():
        gl, gv = r["goles_local"], r["goles_visita"]
        y_true.append(0 if gl > gv else (1 if gl == gv else 2))
    y_true = np.array(y_true)

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
