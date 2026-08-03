"""
Motor de Machine Learning e Inferencia Híbrida para la Liga Profesional de Fútbol de Argentina (arg.1).
Modelo calibrado con Pi-Ratings, Fatigue Factor (Días de descanso y Congestión) y Distancia de Viaje (Haversine km).
Stacking óptimo (LASSO L1 + Random Forest) + Matriz Bivariada de Poisson + Simulador Monte Carlo.
"""
import pickle
from pathlib import Path
import warnings
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar, minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings("ignore")

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



COORDS_ARGENTINA = {
    # Buenos Aires & Gran Buenos Aires
    "Boca Juniors": (-34.6356, -58.3647), "River Plate": (-34.5453, -58.4497),
    "Racing Club": (-34.6678, -58.3686), "Independiente": (-34.6703, -58.3711),
    "San Lorenzo": (-34.6517, -58.4403), "Vélez Sarsfield": (-34.6353, -58.5208),
    "Huracán": (-34.6433, -58.3969), "Argentinos Juniors": (-34.6037, -58.3816),
    "Platense": (-34.5425, -58.4772), "Barracas Central": (-34.6469, -58.3853),
    "Deportivo Riestra": (-34.6542, -58.4486), "Tigre": (-34.4372, -58.5528),
    "Lanús": (-34.7083, -58.3917), "Banfield": (-34.7447, -58.3908),
    "Defensa y Justicia": (-34.8219, -58.2778), "Arsenal Sarandí": (-34.6858, -58.3400),
    
    # Rosario & Santa Fe
    "Rosario Central": (-32.9468, -60.6393), "Newell's Old Boys": (-32.9468, -60.6393),
    "Unión de Santa Fe": (-31.6333, -60.7000), "Colón de Santa Fe": (-31.6333, -60.7000),
    
    # La Plata
    "Estudiantes de La Plata": (-34.9214, -57.9545), "Gimnasia La Plata": (-34.9214, -57.9545),
    
    # Córdoba & Río Cuarto
    "Talleres de Córdoba": (-31.4201, -64.1888), "Belgrano": (-31.4201, -64.1888),
    "Instituto": (-31.4201, -64.1888), "Estudiantes de Río Cuarto": (-33.1307, -64.3499),
    
    # Mendoza & San Juan
    "Godoy Cruz": (-32.8895, -68.8458), "Independiente Rivadavia": (-32.8895, -68.8458),
    "Gimnasia Mendoza": (-32.8895, -68.8458), "San Martín San Juan": (-31.5375, -68.5364),
    
    # Tucumán, Santiago del Estero, Junín, Mar del Plata, Paraná
    "Atlético Tucumán": (-26.8083, -65.2176), "Central Córdoba": (-27.7834, -64.2642),
    "Sarmiento": (-34.5833, -60.9500), "Aldosivi": (-38.0000, -57.5500),
    "Patronato": (-31.7333, -60.5333)
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def get_distance_km(local, visita):
    c_loc = COORDS_ARGENTINA.get(local)
    c_vis = COORDS_ARGENTINA.get(visita)
    if not c_loc or not c_vis:
        return 0.0
    return haversine_km(c_vis[0], c_vis[1], c_loc[0], c_loc[1])


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

K_LIGA = 35.0
HOME_ADV = 55.0


def get_advanced_features(team, season):
    if len(DF_ADV_FEATURES) > 0:
        df_eq = DF_ADV_FEATURES[DF_ADV_FEATURES.equipo == team]
        if len(df_eq) > 0:
            row = df_eq[df_eq.temporada == season]
            if len(row) > 0:
                return row.iloc[0].to_dict()
            diffs = (df_eq["temporada"] - season).abs()
            best_idx = diffs.idxmin()
            return df_eq.loc[best_idx].to_dict()
    return {
        "squad_size": 28.0, "avg_age": 26.5, "foreigners": 3.0, "pct_foreigners": 0.10,
        "squad_value": 25.0, "stadium_capacity": 35000.0, "avg_attendance": 22000.0, "stadium_occupation": 0.65
    }


def get_squad_value(team, season):
    adv = get_advanced_features(team, season)
    return float(adv.get("squad_value", 25.0))


def actualizar_elo(r_local, r_visita, goles_local, goles_visita):
    diff = goles_local - goles_visita
    if diff > 0:
        w = 1.0
    elif diff == 0:
        w = 0.5
    else:
        w = 0.0
        
    dr = r_local - r_visita
    we = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
    
    abs_d = abs(diff)
    if abs_d <= 1:
        mult = 1.0
    elif abs_d == 2:
        mult = 1.5
    else:
        mult = (11.0 + abs_d) / 8.0
        
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
        vl = get_squad_value(local, temporada)
        vv = get_squad_value(visita, temporada)
        feats["squad_value_diff"] = np.log(max(vl, 0.1)) - np.log(max(vv, 0.1))
        feats["h2h_diff"] = self.h2h_goles[(local, visita)]

        feat_l = get_advanced_features(local, temporada)
        feat_v = get_advanced_features(visita, temporada)
        feats["avg_age_diff"] = feat_l["avg_age"] - feat_v["avg_age"]
        feats["squad_size_diff"] = feat_l["squad_size"] - feat_v["squad_size"]
        feats["pct_foreigners_diff"] = feat_l["pct_foreigners"] - feat_v["pct_foreigners"]
        feats["stadium_capacity"] = np.log(max(float(feat_l["stadium_capacity"]), 1.0))
        feats["stadium_occupation"] = float(feat_l["stadium_occupation"])
        feats["avg_attendance"] = np.log(max(float(feat_l["avg_attendance"]), 1.0))

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
        self.match_count[local] += 1
        self.match_count[visita] += 1

        pts_l = 3 if ga > gb else (1 if ga == gb else 0)
        pts_v = 3 if gb > ga else (1 if ga == gb else 0)
        self.season_pts[local] += pts_l
        self.season_pts[visita] += pts_v
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
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year)
    box_path = DATA / "box_score.csv"
    box = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame(columns=["event_id"])
    
    box_dict = {}
    for r in box.itertuples(index=False):
        sl = {s: getattr(r, f"local_{s}", None) for s in STATS}
        sv = {s: getattr(r, f"visita_{s}", None) for s in STATS}
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

    df_dataset = pd.DataFrame(filas_X)

    cols_features = [
        "elo_diff", "squad_value_diff", "h2h_diff",
        "avg_age_diff", "squad_size_diff", "pct_foreigners_diff",
        "stadium_capacity", "stadium_occupation", "avg_attendance",
        "form_diff", "gf_diff", "ga_diff", "ppg_diff",
        "rest_days_diff", "congestion_14d_diff", "pi_diff", "pi_overall_diff",
        "travel_dist_log_km"
    ]
    for s in STATS:
        cols_features.append(f"{s}_total_diff")
        cols_features.append(f"{s}_sede_diff")

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

    # C-search con CV temporal sobre train (sin mirar el test)
    best_c = 0.05; best_loss = 999.0
    from sklearn.metrics import log_loss
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        _losses = []
        for tr, va in tscv.split(X_train):
            _pipe = Pipeline([("sc", StandardScaler()),
                              ("lr", LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=3000))])
            _pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
            _losses.append(log_loss(y_train.iloc[va], _pipe.predict_proba(X_train.iloc[va]), labels=[0,1,2]))
        _loss = float(np.mean(_losses))
        if _loss < best_loss:
            best_loss = _loss; best_c = C
    print(f"Mejor C para Lasso (CV temporal en train): {best_c} | Log-Loss CV: {best_loss:.4f}")

    pipe_lasso_base = Pipeline([("sc", StandardScaler()),
                                 ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=3000, random_state=42))])
    pipe_lasso_base.fit(X_train, y_train)
    pipe_rf_base = Pipeline([("sc", StandardScaler()),
                              ("rf", RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=30, random_state=42, n_jobs=-1))])
    pipe_rf_base.fit(X_train, y_train)
    pipe_xgb_base = Pipeline([("sc", StandardScaler()), ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="mlogloss"))])
    pipe_xgb_base.fit(X_train, y_train)

    if len(X_cal) >= 10:
        # La calibración isotónica overfitea brutalmente con pocos datos de fútbol ruidosos.
        # Desactivamos CalibratedClassifierCV y usamos los estimadores base, igual que en el resto de las ligas.
        pipe_lasso_cal = pipe_lasso_base
        pipe_rf_cal = pipe_rf_base
        pipe_xgb_cal = pipe_xgb_base
    else:
        pipe_lasso_cal = pipe_lasso_base
        pipe_rf_cal = pipe_rf_base
        pipe_xgb_cal = pipe_xgb_base

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

    # BUG 2 FIX (correcto, sin data leakage):
    # Los modelos de producción son pipe_lasso_cal / pipe_rf_cal ya construidos arriba:
    #   base entrenado en X_train  →  calibrado en X_cal (que la base nunca vio)
    # Esto evita el leakage que ocurría al re-entrenar en X_full y luego calibrar
    # con el mismo X_cal que ya estaba incluido en X_full.
    pipe_lasso_final = pipe_lasso_cal
    pipe_rf_final    = pipe_rf_cal
    pipe_xgb = pipe_xgb_cal

    pipe_final = pipe_lasso_final  # backward compat

    def _met(proba, y):
        proba = np.clip(proba, 1e-7, 1-1e-7)
        return {"logloss": round(log_loss(y, proba), 4),
                "accuracy": round(accuracy_score(y, proba.argmax(axis=1)) * 100, 2)}

    met_lasso = _met(pipe_lasso_final.predict_proba(X_test), y_test)
    met_rf    = _met(pipe_rf_final.predict_proba(X_test), y_test)
    met_xgb   = _met(pipe_xgb.predict_proba(X_test), y_test)
    w_opt = w_opt / w_opt.sum()
    p_stack   = np.clip(w_opt[0]*pipe_lasso_final.predict_proba(X_test) + w_opt[1]*pipe_rf_final.predict_proba(X_test) + w_opt[2]*pipe_xgb.predict_proba(X_test), 1e-7, 1-1e-7)
    met_stack = {"logloss": round(log_loss(y_test, p_stack), 4),
                 "accuracy": round(accuracy_score(y_test, p_stack.argmax(axis=1)) * 100, 2),
                 "w": [round(float(x), 3) for x in w_opt]}
    metricas = {"lasso": met_lasso, "rf": met_rf, "xgb": met_xgb, "stacking": met_stack}

    # BUG 1 FIX: Modelo Poisson GLM para goles CON variable de localía (is_home).
    # Sin is_home el intercept era la media pooled ~1.054, subestimando goles del local
    # en -0.165 y sobreestimando los del visitante en +0.150 (diferencia real = +0.316 goles).
    # Solo train+cal (temporada <= 2024): los partidos de test no entrenan el modelo.
    df_goles = df_dataset[df_dataset["temporada"] <= 2024]
    df_dataset["d_elo_l"] = df_dataset["elo_diff"]
    df_dataset["d_elo_v"] = -df_dataset["elo_diff"]

    df_p_local  = df_goles[["goles_local",  "elo_diff"]].rename(columns={"goles_local":  "goles", "elo_diff": "d_elo"})
    df_p_visita = df_goles[["goles_visita", "elo_diff"]].rename(columns={"goles_visita": "goles", "elo_diff": "d_elo"})
    df_p_local["is_home"]  = 1
    df_p_visita["is_home"] = 0
    df_poisson = pd.concat([df_p_local, df_p_visita], ignore_index=True)

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    model_p = smf.glm(formula="goles ~ d_elo + is_home", data=df_poisson, family=sm.families.Poisson()).fit()
    poisson_params = {
        "const": model_p.params["Intercept"],
        "d":     model_p.params["d_elo"],
        "home":  model_p.params["is_home"],
    }

    return {
        "pipe": pipe_final,
        "pipe_lasso": pipe_lasso_final,
        "pipe_rf": pipe_rf_final,
        "alpha_stack": alpha_opt,
        "stack_w": w_opt,
        "pipe_xgb": pipe_xgb,
        "metricas": metricas,
        "poisson_params": poisson_params,
        "features": cols_features,
        "tracker": tracker,
        "equipos": _cargar_equipos(),
        "df_dataset": df_dataset,
        "partidos": partidos
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
    # En Argentina la temporada es el año calendario
    return int(pd.Timestamp.now().year)


def predecir_match(M, local, visita, temporada=None, modelo="rf"):
    tracker = M["tracker"]
    features = M["features"]
    poisson_params = M["poisson_params"]
    if temporada is None:
        temporada = _temporada_actual()
    
    feats = tracker.get_features_for_match(local, visita, temporada, reset_season=False)
    df_feat = pd.DataFrame([feats])[features].fillna(0.0)
    
    if modelo in ("lasso", "l1"):
        pipe = M["pipe_lasso"]
        p_raw = pipe.predict_proba(df_feat)[0]
    elif modelo == "stacking":
        w = M.get("stack_w", np.array([M.get("alpha_stack", 0.4), 1-M.get("alpha_stack", 0.4), 0.0]))
        p_l = M["pipe_lasso"].predict_proba(df_feat)[0]
        p_r = M["pipe_rf"].predict_proba(df_feat)[0]
        p_x = M["pipe_xgb"].predict_proba(df_feat)[0]
        p_raw = w[0]*p_l + w[1]*p_r + w[2]*p_x
        p_raw = p_raw / p_raw.sum()
    elif modelo == "xgb":
        p_raw = M["pipe_xgb"].predict_proba(df_feat)[0]
    else:
        pipe = M["pipe_rf"]
        p_raw = pipe.predict_proba(df_feat)[0]
        
    p = np.array([p_raw[2], p_raw[1], p_raw[0]])
    
    d_elo_l = feats["elo_diff"]
    home_coef = poisson_params.get("home", 0.0)  # compatibilidad si falta la clave
    la = float(np.exp(poisson_params["const"] + poisson_params["d"] * d_elo_l + home_coef * 1))
    d_elo_v = -feats["elo_diff"]
    lb = float(np.exp(poisson_params["const"] + poisson_params["d"] * d_elo_v + home_coef * 0))

    return p, la, lb


def grilla_goles(M, local, visita, max_goles=6, modelo="rf"):
    p, la, lb = predecir_match(M, local, visita, modelo=modelo)
    
    px = poisson.pmf(np.arange(max_goles + 1), la)
    py = poisson.pmf(np.arange(max_goles + 1), lb)
    M_pois = np.outer(px, py)
    
    M_pois /= M_pois.sum()
    
    p_win_l = np.tril(M_pois, -1).sum()
    p_draw  = np.diag(M_pois).sum()
    p_win_v = np.triu(M_pois, 1).sum()
    
    p_pois = np.array([p_win_l, p_draw, p_win_v])
    p_pois /= p_pois.sum()
    
    w_ml = 0.65
    w_pois = 0.35
    p_final = w_ml * p + w_pois * p_pois
    p_final /= p_final.sum()
    
    eps = 1e-6
    scale = (p_final + eps) / (p_pois + eps)
    
    M_adj = np.copy(M_pois)
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            if i > j:
                M_adj[i, j] *= scale[0]
            elif i == j:
                M_adj[i, j] *= scale[1]
            else:
                M_adj[i, j] *= scale[2]
                
    M_adj /= M_adj.sum()
    return M_adj, p_final, (la, lb)


def cuota(prob):
    return 1.0 / max(prob, 1e-4)


def mercados(mix):
    mk = {}
    p_local = np.tril(mix, -1).sum()
    p_empate = np.diag(mix).sum()
    p_visita = np.triu(mix, 1).sum()
    
    mk["1X2"] = {"Local": p_local, "Empate": p_empate, "Visita": p_visita}
    mk["Doble Oportunidad"] = {
        "1X": p_local + p_empate,
        "X2": p_empate + p_visita,
        "12": p_local + p_visita
    }
    
    n = mix.shape[0]
    gi, gj = np.indices((n, n))
    for ln in [1.5, 2.5, 3.5]:
        po = mix[gi + gj > ln].sum()
        mk[f"Over {ln}"] = po
        mk[f"Under {ln}"] = 1.0 - po
        
    btts_si = mix[1:, 1:].sum()
    mk["Ambos marcan (BTTS sí)"] = btts_si
    mk["BTTS no"] = 1.0 - btts_si
    
    flat = mix.ravel()
    top_indices = flat.argsort()[::-1][:5]
    top_scores = []
    for ix in top_indices:
        g1, g2 = divmod(ix, mix.shape[1])
        top_scores.append((g1, g2, flat[ix]))
    mk["_top_marcadores"] = top_scores
    return mk


def _torneo_actual(M):
    # En Argentina el año calendario contiene DOS torneos (Copa de la Liga ene-jul,
    # Liga Profesional ago-nov), ambos etiquetados como temporada 2026 en partidos.csv.
    # El torneo vigente es el definido por fixture.csv; los partidos jugados de ese
    # torneo son los que caen dentro de su ventana de fechas.
    fix = pd.read_csv(DATA / "fixture.csv", parse_dates=["fecha"]) if (DATA / "fixture.csv").exists() else pd.DataFrame()
    fix = fix[fix.temporada == _temporada_actual()] if not fix.empty else fix
    if len(fix) == 0:
        return pd.DataFrame(), fix
    inicio_torneo = fix["fecha"].min() - pd.Timedelta(days=1)
    partidos = M["partidos"].copy()
    partidos["fecha"] = pd.to_datetime(partidos["fecha"])
    p_actuales = partidos[(partidos["temporada"] == _temporada_actual()) & (partidos["fecha"] >= inicio_torneo)]
    return p_actuales, fix




def evolucion_tabla(M, temporada=None):
    """Posicion en la tabla de cada equipo a lo largo de la temporada (formato largo).

    Eje X util: pj (partidos jugados). Ordena con criterio puntos, dif_goles, goles_favor,
    consistente en todas las ligas (independiente de ordenar_tabla de cada motor).
    """
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    partidos["temporada"] = partidos["fecha"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    if temporada is None:
        temporada = partidos["temporada"].max()
    pp = partidos[partidos["temporada"] == temporada]
    eqs = sorted(set(pp["local"].unique()).union(set(pp["visita"].unique())))
    pts = {e: 0 for e in eqs}
    gf = {e: 0 for e in eqs}
    gc = {e: 0 for e in eqs}
    pj = {e: 0 for e in eqs}
    filas = []
    for fecha in sorted(pp["fecha"].unique()):
        dia = pp[pp["fecha"] == fecha]
        for r in dia.itertuples(index=False):
            l, v = r.local, r.visita
            if l not in eqs or v not in eqs:
                continue
            pj[l] += 1; pj[v] += 1
            gf[l] += r.goles_local; gf[v] += r.goles_visita
            gc[l] += r.goles_visita; gc[v] += r.goles_local
            if r.goles_local > r.goles_visita:
                pts[l] += 3
            elif r.goles_local == r.goles_visita:
                pts[l] += 1; pts[v] += 1
            else:
                pts[v] += 3
        filas_tab = []
        for e in eqs:
            filas_tab.append({"equipo": e, "pj": pj[e], "puntos": pts[e],
                              "goles_contra": gc[e], "goles_favor": gf[e],
                              "dif_goles": gf[e] - gc[e]})
        tab = pd.DataFrame(filas_tab)
        tab = tab.sort_values(by=["puntos", "dif_goles", "goles_favor"],
                              ascending=False).reset_index(drop=True)
        pos = {e: i + 1 for i, e in enumerate(tab["equipo"])}
        for e in eqs:
            if pj[e] >= 1:
                filas.append({"fecha": fecha.date().isoformat(), "pj": pj[e],
                              "equipo": e, "posicion": pos[e], "puntos": pts[e]})
    return pd.DataFrame(filas)

def simular_fixture_regular(M, PREDS, fijos=None):
    # Solo se siembran partidos jugados del torneo vigente (Liga Profesional 2026).
    # Los resultados de la Copa de la Liga (torneo anterior) no se mezclan.
    p_actuales, fix = _torneo_actual(M)
    
    tabla = defaultdict(lambda: {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0})
    
    todos_activos = set(p_actuales["local"]).union(set(p_actuales["visita"]))
    if not fix.empty:
        todos_activos = todos_activos.union(set(fix["local"])).union(set(fix["visita"]))
    if not todos_activos:
        if len(DF_ADV_FEATURES) > 0:
            todos_activos = set(DF_ADV_FEATURES["equipo"].unique())
        else:
            todos_activos = set(partidos["local"].unique())
        
    for eq in todos_activos:
        tabla[eq] = {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0}
        
    for r in p_actuales.itertuples(index=False):
        l, v, gl, gv = r.local, r.visita, int(r.goles_local), int(r.goles_visita)
        if l not in tabla: tabla[l] = {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0}
        if v not in tabla: tabla[v] = {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0}
        tabla[l]["GF"] += gl
        tabla[l]["GC"] += gv
        tabla[v]["GF"] += gv
        tabla[v]["GC"] += gl
        if gl > gv:
            tabla[l]["PTS"] += 3
            tabla[l]["PG"] += 1
            tabla[v]["PP"] += 1
        elif gl == gv:
            tabla[l]["PTS"] += 1
            tabla[v]["PTS"] += 1
            tabla[l]["PE"] += 1
            tabla[v]["PE"] += 1
        else:
            tabla[v]["PTS"] += 3
            tabla[v]["PG"] += 1
            tabla[l]["PP"] += 1
            
    if not fix.empty:
        for r in fix.itertuples(index=False):
            l, v = r.local, r.visita
            # Partidos ya jugados del torneo actual se siembran desde partidos.csv
            if r.estado != "pre" and pd.notna(r.goles_local) and pd.notna(r.goles_visita):
                continue
            if l not in tabla: tabla[l] = {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0}
            if v not in tabla: tabla[v] = {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0}
            if fijos and (l, v) in fijos:
                gl, gv = fijos[(l, v)]
            else:
                p, la, lb = PREDS.get((l, v), (None, 1.2, 1.2))
                gl = int(np.random.poisson(la))
                gv = int(np.random.poisson(lb))
            tabla[l]["GF"] += gl
            tabla[l]["GC"] += gv
            tabla[v]["GF"] += gv
            tabla[v]["GC"] += gl
            if gl > gv:
                tabla[l]["PTS"] += 3
                tabla[l]["PG"] += 1
                tabla[v]["PP"] += 1
            elif gl == gv:
                tabla[l]["PTS"] += 1
                tabla[v]["PTS"] += 1
                tabla[l]["PE"] += 1
                tabla[v]["PE"] += 1
            else:
                tabla[v]["PTS"] += 3
                tabla[v]["PG"] += 1
                tabla[l]["PP"] += 1
            
    return tabla


def monte_carlo(M, n_sims=4000, fijos=None, modelo="rf", seed=42):
    if seed is not None:
        np.random.seed(seed)
    import hashlib
    fp = hashlib.sha256()
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
                if modelo in resultados:
                    print("Simulación Monte Carlo cargada desde cache de disco")
                    return resultados[modelo]
        except Exception as ex:
            print(f"Cache de simulación inválido ({ex}); re-simulando...")

    fix = pd.read_csv(DATA / "fixture.csv") if (DATA / "fixture.csv").exists() else pd.DataFrame()
    fix = fix[fix.temporada == _temporada_actual()] if not fix.empty else fix

    PREDS = {}
    if not fix.empty:
        for r in fix.itertuples(index=False):
            l, v = r.local, r.visita
            if (l, v) not in PREDS:
                p, la, lb = predecir_match(M, l, v, modelo=modelo)
                PREDS[(l, v)] = (p, la, lb)
            
    partidos = M["partidos"]
    p_actuales = partidos[partidos.temporada == _temporada_actual()]
    todos_activos = set(p_actuales["local"]).union(set(p_actuales["visita"]))
    if not fix.empty:
        todos_activos = todos_activos.union(set(fix["local"])).union(set(fix["visita"]))
    if not todos_activos:
        todos_activos = set(partidos["local"].unique())
        
    resultados_pts = defaultdict(list)
    resultados_campeon = defaultdict(int)
    resultados_copas = defaultdict(int)
    resultados_descenso = defaultdict(int)
    
    for _ in range(n_sims):
        tabla = simular_fixture_regular(M, PREDS, fijos)
        filas = []
        for eq, s in tabla.items():
            filas.append({
                "equipo": eq, "PTS": s["PTS"], "DG": s["GF"] - s["GC"], "GF": s["GF"]
            })
        df_t = pd.DataFrame(filas).sort_values(by=["PTS", "DG", "GF"], ascending=False).reset_index(drop=True)
        
        for idx, row in df_t.iterrows():
            eq = row["equipo"]
            pts = row["PTS"]
            resultados_pts[eq].append(pts)
            if idx == 0:
                resultados_campeon[eq] += 1
            if idx < 4:
                resultados_copas[eq] += 1
            if idx >= len(df_t) - 2:
                resultados_descenso[eq] += 1
                
    filas_res = []
    for eq in todos_activos:
        pts_list = resultados_pts[eq]
        mean_pts = np.mean(pts_list) if pts_list else 0.0
        filas_res.append({
            "Selección": eq,
            "Puntos esperados": round(mean_pts, 1),
            "P_campeon": resultados_campeon[eq] / n_sims,
            "P_copas": resultados_copas[eq] / n_sims,
            "P_descenso": resultados_descenso[eq] / n_sims
        })
        
    df_res = pd.DataFrame(filas_res).sort_values(by="Puntos esperados", ascending=False).reset_index(drop=True)
    try:
        resultados[modelo] = df_res
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": fp.hexdigest(), "results": resultados}, fh)
        print("Simulación Monte Carlo guardada en cache de disco")
    except Exception as ex:
        print(f"No se pudo guardar el cache de simulación: {ex}")
    return df_res


def simular_campeonato(M, n_sims=4000, modelo="rf", seed=42):
    return monte_carlo(M, n_sims=n_sims, modelo=modelo, seed=seed)


def validacion_en_vivo(M, temporada_val=2026):
    df = M["df_dataset"]
    df_val = df[df["temporada"] == temporada_val].copy()
    if len(df_val) == 0:
        return None, None, None
        
    pipe = M["pipe"]
    features = M["features"]
    
    X_val = df_val[features].fillna(0.0)
    y_val = df_val["resultado"]
    
    probs = pipe.predict_proba(X_val)
    preds = pipe.predict(X_val)
    
    ll = log_loss(y_val, probs, labels=[0, 1, 2])
    acc = accuracy_score(y_val, preds)
    
    freqs = y_val.value_counts(normalize=True)
    baseline_probs = np.zeros_like(probs)
    for i, c in enumerate([0, 1, 2]):
        baseline_probs[:, i] = freqs.get(c, 0.33)
    ll_base = log_loss(y_val, baseline_probs, labels=[0, 1, 2])
    
    met = {"n": len(df_val), "acierto": acc, "logloss": ll, "logloss_base": ll_base}
    
    df_val["Prob_Visita"] = probs[:, 0]
    df_val["Prob_Empate"] = probs[:, 1]
    df_val["Prob_Local"] = probs[:, 2]
    df_val["Prediccion"] = preds
    
    df_val = df_val.sort_values("fecha").reset_index(drop=True)
    log_losses = []
    for i in range(1, len(df_val) + 1):
        sub_y = df_val["resultado"].iloc[:i]
        sub_p = probs[:i]
        try:
            curr_ll = log_loss(sub_y, sub_p, labels=[0, 1, 2])
            log_losses.append(curr_ll)
        except Exception:
            log_losses.append(np.nan)
            
    evol = pd.DataFrame({"partido": range(1, len(df_val) + 1), "logloss_acum": log_losses})
    return df_val, met, evol
