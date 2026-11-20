import sys as _sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from comun.motor_comun import (
    _elo_default,
    _none_default,
    cargar_equipos_csv,
    haversine_km,
    PiRatingTracker,
    PiRatingsTracker,
    _TrackerPrePartido,
    _modelo_pre_partido,
)
from comun.motor_comun import hash_modulos_comunes
from comun.motor_comun import seleccionar_c_lasso, excluir_variables_estimadas, entrenar_stacking_oof
from comun.monte_carlo import clave_cache_liga, simular_fixture_desde_tabla
import sys
from datetime import datetime
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
from scipy.stats import poisson
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA = Path(__file__).resolve().parent / "data"
EQUIPOS_PATH = DATA / "equipos.csv"

def _cargar_equipos():
    return cargar_equipos_csv(EQUIPOS_PATH)

DESCIENDEN = 2  # Descienden los 2 últimos clubes (puestos 15 y 16)
CUPOS_COPA = 4  # 1 Champions League, 2-4 Europa / Conference League

SQUAD_VALUES = {
    "Bodo/Glimt": 45.0, "Molde": 30.0, "SK Brann": 25.0, "Rosenborg": 25.0,
    "Viking FK": 22.0, "Lillestrom": 15.0, "Vålerenga": 15.0, "Tromso": 14.0,
    "Fredrikstad": 12.0, "Stromsgodset": 12.0, "Sarpsborg FK": 11.0, "Sandefjord": 10.0,
    "Hamarkameratene": 9.0, "Kristiansund BK": 9.0, "Haugesund": 9.0, "KFUM Oslo": 8.0,
    "Odds BK": 8.0, "Aalesund": 7.0, "Stabaek": 7.0, "Bryne": 6.0,
    "IK Start": 6.0, "Mjondalen IF": 5.0, "FK Jerv": 5.0
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



K_LIGA = 35.0      # ELO K-factor para la Norwegian Eliteserien
HOME_ADV = 50.0    # Ventaja de localía estándar en Noruega

COORDS_NORWAY = {
    "Bodo/Glimt": (67.2764, 14.3986),
    "Molde": (62.7356, 7.1472),
    "SK Brann": (60.3669, 5.3575),
    "Rosenborg": (63.4122, 10.4061),
    "Viking FK": (58.9144, 5.7314),
    "Lillestrom": (59.9625, 11.0664),
    "Vålerenga": (59.9189, 10.7997),
    "Tromso": (69.6583, 18.9372),
    "Fredrikstad": (59.2103, 10.9317),
    "Stromsgodset": (59.7369, 10.1983),
    "Sarpsborg FK": (59.2828, 11.1097),
    "Sandefjord": (59.1389, 10.2181),
    "Hamarkameratene": (60.7967, 11.0858),
    "Kristiansund BK": (63.1067, 7.7497),
    "Haugesund": (59.4175, 5.2750),
    "KFUM Oslo": (59.8833, 10.7917),
    "Odds BK": (59.2086, 9.5936),
    "Aalesund": (62.4703, 6.1878),
    "Stabaek": (59.9194, 10.5828),
    "Bryne": (58.7356, 5.6483),
    "IK Start": (58.1561, 8.0317),
    "Mjondalen IF": (59.7483, 10.0150),
    "FK Jerv": (58.3456, 8.5878)
}

ALTITUDES_NORWAY = {
    "Bodo/Glimt": 12, "Molde": 5, "SK Brann": 30, "Rosenborg": 35,
    "Viking FK": 25, "Lillestrom": 110, "Vålerenga": 70, "Tromso": 45,
    "Fredrikstad": 10, "Stromsgodset": 15, "Sarpsborg FK": 35, "Sandefjord": 20,
    "Hamarkameratene": 140, "Kristiansund BK": 20, "Haugesund": 25, "KFUM Oslo": 90,
    "Odds BK": 40, "Aalesund": 10, "Stabaek": 45, "Bryne": 30,
    "IK Start": 15, "Mjondalen IF": 15, "FK Jerv": 20
}


def get_distance_km(local, visita):
    if local in COORDS_NORWAY and visita in COORDS_NORWAY:
        c1 = COORDS_NORWAY[local]; c2 = COORDS_NORWAY[visita]
        return haversine_km(c1[0], c1[1], c2[0], c2[1])
    return 350.0

def get_altitude_diff(local, visita):
    al = ALTITUDES_NORWAY.get(local, 30)
    av = ALTITUDES_NORWAY.get(visita, 30)
    return float(al - av)

def get_squad_value(equipo, temporada):
    if not DF_SQUAD_VALUES.empty:
        sub = DF_SQUAD_VALUES[(DF_SQUAD_VALUES["equipo"] == equipo) & (DF_SQUAD_VALUES["temporada"] == temporada)]
        if not sub.empty:
            return float(sub.iloc[0]["squad_value"])
    return float(SQUAD_VALUES.get(equipo, 12.0))

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
        "squad_size": 28.0, "avg_age": 25.3, "foreigners": 9.0, "pct_foreigners": 0.30,
        "squad_value": float(SQUAD_VALUES.get(equipo, 12.0)), "stadium_capacity": 10000.0,
        "avg_attendance": 6500.0, "stadium_occupation": 0.65
    }


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

        if stats_dict:
            m_data = {"fecha": f, "local": local, "visita": visita}
            m_data.update(stats_dict)
            self.history[local].append(m_data)
            self.history[visita].append(m_data)
            self.history_home[local].append(m_data)
            self.history_away[visita].append(m_data)

_MOTOR_CACHE = None

def cargar(force_retrain=False):
    global _MOTOR_CACHE
    if _MOTOR_CACHE is not None and not force_retrain:
        return _MOTOR_CACHE

    cache_path = DATA.parent / "modelo_nor.pkl"
    partidos_csv = DATA / "partidos.csv"
    box_csv = DATA / "box_score.csv"

    if not partidos_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de partidos: {partidos_csv}")

    import hashlib
    h = hashlib.md5()
    for p in [partidos_csv, box_csv, ADV_FEATURES_PATH]:
        if p.exists():
            h.update(str(p.stat().st_mtime).encode())
    # El modelo también depende del código del motor: si cambia, se re-entrena
    h.update(hashlib.md5(Path(__file__).read_bytes()).hexdigest().encode())
    # y del código compartido en comun/ (auditoría: antes no invalidaba la caché)
    h.update(hash_modulos_comunes().encode())
    key_actual = h.hexdigest()

    if cache_path.exists() and not force_retrain:
        try:
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("key") == key_actual:
                print("Motor Noruega cargado desde cache de disco")
                _MOTOR_CACHE = saved["motor"]
                return _MOTOR_CACHE
        except Exception as ex:
            print(f"No se pudo cargar cache de modelo noruego: {ex}")

    print("Entrenando motor Noruega (Eliteserien)...")
    df_partidos = pd.read_csv(partidos_csv, parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)
    df_box = pd.read_csv(box_csv) if box_csv.exists() else pd.DataFrame()
    box_dict = {}
    if not df_box.empty and "event_id" in df_box.columns:
        box_dict = df_box.set_index("event_id").to_dict("index")

    tracker = StateTracker()
    filas_dataset = []
    filas_poisson = []

    for _, r in df_partidos.iterrows():
        eid = r.get("event_id")
        l, v = r["local"], r["visita"]
        gl, gv = int(r["goles_local"]), int(r["goles_visita"])
        fecha = r["fecha"]
        temp = int(r["temporada"])

        feats = tracker.get_features_for_match(l, v, temp, fecha=fecha, reset_season=True)

        res = 0 if gl > gv else (1 if gl == gv else 2)
        feats["resultado"] = res
        feats["goles_local"] = gl
        feats["goles_visita"] = gv
        feats["temporada"] = temp
        feats["fecha"] = fecha
        feats["local"] = l
        feats["visita"] = v
        filas_dataset.append(feats)

        filas_poisson.append({
            "goles": gl, "is_home": 1.0,
            "elo_diff": feats["elo_diff"],
            "alt_diff": feats["altitude_diff"],
            "squad_diff": feats["squad_value_diff"]
        })
        filas_poisson.append({
            "goles": gv, "is_home": 0.0,
            "elo_diff": -feats["elo_diff"],
            "alt_diff": -feats["altitude_diff"],
            "squad_diff": -feats["squad_value_diff"]
        })

        st_data = box_dict.get(eid, {}) if eid else {}
        tracker.registrar_partido(l, v, gl, gv, fecha, stats_dict=st_data)

    df_dataset = pd.DataFrame(filas_dataset)

    cols_ignore = {"resultado", "goles_local", "goles_visita", "temporada", "fecha", "local", "visita"}
    cols_features = [c for c in df_dataset.columns if c not in cols_ignore]
    # Fuera variables de plantilla sintéticas/retroactivas (scripts/build_advanced_features.py)
    cols_features = excluir_variables_estimadas(cols_features)

    # Walk-forward splits (temporada anual)
    train_mask = df_dataset["temporada"] <= 2023
    cal_mask   = df_dataset["temporada"] == 2024
    test_mask  = df_dataset["temporada"] >= 2025

    X_train_raw = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    cols_features = [c for c in cols_features if X_train_raw[c].std() > 1e-5]

    X_train = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    y_train = df_dataset.loc[train_mask, "resultado"]
    X_cal   = df_dataset.loc[cal_mask,   cols_features].fillna(0.0)
    y_cal   = df_dataset.loc[cal_mask,   "resultado"]
    X_test  = df_dataset.loc[test_mask,  cols_features].fillna(0.0)
    y_test  = df_dataset.loc[test_mask,  "resultado"]

    # 1. LASSO L1 con optimizador SAGA
    # C de LASSO elegido con CV temporal dentro del train (auditoría: antes se elegía con el test)
    best_c, _ = seleccionar_c_lasso(X_train, y_train, grid=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

    X_full = df_dataset.loc[train_mask | cal_mask, cols_features].fillna(0.0)
    y_full = df_dataset.loc[train_mask | cal_mask, "resultado"]

    # Modelos base regularizados y Stacking Out-Of-Fold temporal (sin fuga in-sample de calibración)
    pipe_lasso, pipe_rf, pipe_xgb, w_opt = entrenar_stacking_oof(
        X_full, y_full, best_c, binario=False, n_splits=4, reg_l2=0.03
    )

    # Métricas
    def _met(proba, y):
        proba = np.clip(proba, 1e-7, 1.0 - 1e-7)
        return {"logloss": round(float(log_loss(y, proba, labels=[0, 1, 2])), 4),
                "accuracy": round(float(accuracy_score(y, proba.argmax(axis=1)) * 100.0), 2)}

    p_l_test = pipe_lasso.predict_proba(X_test)
    p_r_test = pipe_rf.predict_proba(X_test)
    p_x_test = pipe_xgb.predict_proba(X_test)
    p_st_test = w_opt[0] * p_l_test + w_opt[1] * p_r_test + w_opt[2] * p_x_test
    p_st_test = p_st_test / p_st_test.sum(axis=1, keepdims=True)

    metricas = {
        "lasso": _met(p_l_test, y_test),
        "rf": _met(p_r_test, y_test),
        "stacking": {**_met(p_st_test, y_test), "w": [round(float(x), 3) for x in w_opt]}
    }
    print(f"Metricas NOR Test>=2025: LASSO={metricas['lasso']} RF={metricas['rf']} Stacking={metricas['stacking']}")

    # 5. Modelo Poisson con corrección Dixon-Coles
    df_p = pd.DataFrame(filas_poisson)
    df_p["intercept"] = 1.0
    cols_reg = ["intercept", "is_home", "elo_diff", "alt_diff", "squad_diff"]
    try:
        glm = sm.GLM(df_p["goles"], df_p[cols_reg], family=sm.families.Poisson()).fit()
        poisson_params = {
            "const": float(glm.params["intercept"]),
            "is_home": float(glm.params["is_home"]),
            "elo": float(glm.params["elo_diff"]),
            "alt": float(glm.params["alt_diff"]),
            "squad": float(glm.params["squad_diff"])
        }
    except Exception as ex:
        print(f"GLM fallback: {ex}")
        poisson_params = {"const": 0.25, "is_home": 0.25, "elo": 0.0015, "alt": 0.0001, "squad": 0.15}

    salida = {
        "pipe_lasso": pipe_lasso,
        "pipe_rf": pipe_rf,
        "pipe_xgb": pipe_xgb,
        "weights_stacking": w_opt,
        "metricas": metricas,
        "features": cols_features,
        "tracker": tracker,
        "poisson_params": poisson_params,
        "df_partidos": df_partidos,
        "df_dataset": df_dataset,
        "equipos": _cargar_equipos()
    }

    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": key_actual, "motor": salida}, fh)
        print("Modelo guardado en cache de disco")
    except Exception as ex:
        print(f"No se pudo escribir cache de disco: {ex}")

    _MOTOR_CACHE = salida
    return salida


def predecir_match(M, local, visita, temporada=2026, modelo="stacking"):
    tracker = M["tracker"]
    feats = tracker.get_features_for_match(local, visita, temporada)
    X = pd.DataFrame([feats])[M["features"]].fillna(0.0)

    if modelo == "lasso":
        proba = M["pipe_lasso"].predict_proba(X)[0]
    elif modelo == "rf":
        proba = M["pipe_rf"].predict_proba(X)[0]
    elif modelo == "xgb":
        proba = M["pipe_xgb"].predict_proba(X)[0]
    else:  # stacking
        w = M["weights_stacking"]
        pl = M["pipe_lasso"].predict_proba(X)[0]
        pr = M["pipe_rf"].predict_proba(X)[0]
        px = M["pipe_xgb"].predict_proba(X)[0]
        proba = w[0] * pl + w[1] * pr + w[2] * px
        proba = proba / proba.sum()

    p_h = float(proba[0])
    p_d = float(proba[1])
    p_a = float(proba[2])

    pp = M["poisson_params"]
    d_elo = feats["elo_diff"]
    d_alt = feats["altitude_diff"]
    d_sq = feats["squad_value_diff"]

    log_mu_l = pp["const"] + pp["is_home"] + pp["elo"] * d_elo + pp["alt"] * d_alt + pp["squad"] * d_sq
    log_mu_v = pp["const"] - pp["elo"] * d_elo - pp["alt"] * d_alt - pp["squad"] * d_sq

    mu_l = max(0.2, min(5.0, float(np.exp(log_mu_l))))
    mu_v = max(0.2, min(5.0, float(np.exp(log_mu_v))))

    return np.array([p_h, p_d, p_a]), mu_l, mu_v


def tau(x, y, mu_l, mu_v, rho=-0.08):
    if x == 0 and y == 0:
        return 1.0 - mu_l * mu_v * rho
    elif x == 0 and y == 1:
        return 1.0 + mu_l * rho
    elif x == 1 and y == 0:
        return 1.0 + mu_v * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def matriz_marcador_exacto(mu_l, mu_v, max_goles=6, rho=-0.08):
    mat = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            p_base = poisson.pmf(i, mu_l) * poisson.pmf(j, mu_v)
            mat[i, j] = p_base * tau(i, j, mu_l, mu_v, rho=rho)
    mat = np.maximum(mat, 0.0)
    s = mat.sum()
    if s > 0:
        mat /= s
    return mat


def _temporada_actual():
    # En Noruega la liga se juega en año calendario (primavera - otoño)
    return datetime.now().year


def obtener_tabla_actual(M):
    df_p = M["df_partidos"]
    temp = _temporada_actual()
    df_t = df_p[df_p.temporada == temp]

    equipos = sorted(list(set(df_t["local"]).union(set(df_t["visita"]))))
    if not equipos:
        equipos = sorted(list(M["tracker"].elos.keys()))

    stats = {e: {"equipo": e, "pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "dg": 0, "puntos": 0} for e in equipos}
    for _, r in df_t.iterrows():
        l, v, gl, gv = r["local"], r["visita"], r["goles_local"], r["goles_visita"]
        if l in stats and v in stats and pd.notna(gl) and pd.notna(gv):
            stats[l]["pj"] += 1
            stats[v]["pj"] += 1
            stats[l]["gf"] += int(gl)
            stats[v]["gf"] += int(gv)
            stats[l]["gc"] += int(gv)
            stats[v]["gc"] += int(gl)
            if gl > gv:
                stats[l]["pg"] += 1
                stats[l]["puntos"] += 3
                stats[v]["pp"] += 1
            elif gl == gv:
                stats[l]["pe"] += 1
                stats[v]["pe"] += 1
                stats[l]["puntos"] += 1
                stats[v]["puntos"] += 1
            else:
                stats[v]["pg"] += 1
                stats[v]["puntos"] += 3
                stats[l]["pp"] += 1

    for e in equipos:
        stats[e]["dg"] = stats[e]["gf"] - stats[e]["gc"]

    df_res = pd.DataFrame(list(stats.values())).sort_values(by=["puntos", "dg", "gf"], ascending=[False, False, False]).reset_index(drop=True)
    df_res.index = df_res.index + 1
    return df_res


def _simular_fixture_vec(M, preds_dict, fijos, n_sims=1000, modelo_tipo="stacking", seed=None, return_pts=False):
    fix_path = DATA / "fixture.csv"
    fix = pd.read_csv(fix_path)
    fix = fix[fix.temporada == _temporada_actual()].copy()
    tab_act = obtener_tabla_actual(M)
    res = simular_fixture_desde_tabla(
        M, fix, tab_act, preds_dict, fijos=fijos, n_sims=n_sims,
        k_liga=K_LIGA, home_adv=HOME_ADV, mov_mode="std",
        col_dg="dg", col_gf="gf",
        predecir_fallback=lambda l, v: predecir_match(M, l, v, modelo=modelo_tipo),
        liga=Path(__file__).resolve().parent.name, seed=seed,
    )
    if return_pts:
        return res["eqs"], res["order"], res["PTS"].mean(axis=1)
    return res["eqs"], res["order"]


def simular_campeonato(M, n_sims=1000, fijos=None, modelo="stacking", modelo_tipo=None, seed=42):
    if modelo_tipo is not None:
        modelo = modelo_tipo
    modelo_tipo = modelo or "stacking"

    fix_path = DATA / "fixture.csv"
    if not fix_path.exists():
        return obtener_tabla_actual(M)

    cache_key = clave_cache_liga(__file__, DATA, n_sims, modelo_tipo, fijos=fijos)

    cache_path = DATA.parent / "simulacion_mc.pkl"
    resultados = {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("key") == cache_key:
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

    eqs, order, pts_medios = _simular_fixture_vec(
        M, PREDS, fijos, n_sims, modelo_tipo=modelo_tipo, seed=seed, return_pts=True
    )
    pts_esp = dict(zip(eqs, pts_medios))
    n_eq = len(eqs)
    counts_campeon = {e: 0 for e in eqs}
    counts_champ = {e: 0 for e in eqs}
    counts_copas = {e: 0 for e in eqs}
    counts_desc = {e: 0 for e in eqs}

    for sim in range(n_sims):
        c = eqs[order[0, sim]]
        counts_campeon[c] += 1
        for rank in range(min(1, n_eq)):
            counts_champ[eqs[order[rank, sim]]] += 1
        for rank in range(min(CUPOS_COPA, n_eq)):
            counts_copas[eqs[order[rank, sim]]] += 1
        for rank in range(max(0, n_eq - DESCIENDEN), n_eq):
            counts_desc[eqs[order[rank, sim]]] += 1

    res = []
    for eq in eqs:
        res.append({
            "equipo": eq,
            "Selección": eq,
            "Puntos esperados": round(float(pts_esp[eq]), 1),
            "P_campeon": counts_campeon[eq] / n_sims,
            "P_champions": counts_champ[eq] / n_sims,
            "P_copas": counts_copas[eq] / n_sims,
            "P_descenso": counts_desc[eq] / n_sims
        })

    df_res = pd.DataFrame(res).sort_values(by=["P_campeon", "Puntos esperados"], ascending=[False, False]).reset_index(drop=True)
    df_res.index = df_res.index + 1

    resultados[modelo_tipo] = df_res
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": cache_key, "results": resultados}, fh)
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
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year)
    if temporada_val is None:
        temporada_val = partidos["temporada"].max()
    val_df = partidos[partidos.temporada == temporada_val].copy()
    if len(val_df) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    # Cada partido se predice con las variables que existían ANTES de jugarlo, no con el
    # tracker final (que ya incluye su resultado y los de todos los partidos posteriores).
    pre_partido = _modelo_pre_partido(M, "df_dataset")
    val_df = val_df[[pre_partido(r.fecha, r.local, r.visita) is not None
                     for r in val_df.itertuples(index=False)]].copy()
    if len(val_df) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    probs = []
    for _, r in val_df.iterrows():
        p = predecir_match(pre_partido(r["fecha"], r["local"], r["visita"]), r["local"], r["visita"], temporada=temporada_val, modelo=modelo_tipo)
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

if __name__ == "__main__":
    cargar()
