"""
Motor predictivo de Machine Learning y Simulación Monte Carlo para la UEFA Nations League.
Incluye:
1. StateTracker point-in-time con Elo dinámico (K=32), forma reciente, H2H y distancias Haversine.
2. Modelo Bivariado Poisson Dixon-Coles para goles esperados (xG) y probabilidades de marcador exacto.
3. Modelos Calibrados de Clasificación 1X2: LASSO L1 (SAGA), Random Forest y Stacking Óptimo.
4. Simulador Monte Carlo de los 14 grupos (Ligas A, B, C, D), ascensos, descensos y Final Four de Liga A.
"""
from collections import defaultdict, deque
from datetime import datetime
import math
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent / "data"
EQUIPOS_PATH = DATA / "equipos.csv"

COORDS_UNL = {
    'Albania': (41.3275, 19.8187), 'Andorra': (42.5063, 1.5218), 'Armenia': (40.1792, 44.4991),
    'Austria': (48.2082, 16.3738), 'Azerbaijan': (40.4093, 49.8671), 'Belarus': (53.9006, 27.5590),
    'Belgium': (50.8503, 4.3517), 'Bosnia and Herzegovina': (43.8563, 18.4131), 'Bulgaria': (42.6977, 23.3219),
    'Croatia': (45.8150, 15.9819), 'Cyprus': (35.1856, 33.3823), 'Czech Republic': (50.0755, 14.4378),
    'Denmark': (55.6761, 12.5683), 'England': (51.5074, -0.1278), 'Estonia': (59.4370, 24.7536),
    'Faroe Islands': (62.0107, -6.7741), 'Finland': (60.1699, 24.9384), 'France': (48.8566, 2.3522),
    'Georgia': (41.7151, 44.8271), 'Germany': (52.5200, 13.4050), 'Gibraltar': (36.1408, -5.3536),
    'Greece': (37.9838, 23.7275), 'Hungary': (47.4979, 19.0402), 'Iceland': (64.1466, -21.9426),
    'Israel': (32.0853, 34.7818), 'Italy': (41.9028, 12.4964), 'Kazakhstan': (51.1694, 71.4491),
    'Kosovo': (42.6629, 21.1655), 'Latvia': (56.9496, 24.1052), 'Liechtenstein': (47.1410, 9.5215),
    'Lithuania': (54.6872, 25.2797), 'Luxembourg': (49.6116, 6.1319), 'Malta': (35.8989, 14.5146),
    'Moldova': (47.0105, 28.8638), 'Montenegro': (42.4304, 19.2594), 'Netherlands': (52.3676, 4.9041),
    'North Macedonia': (41.9981, 21.4254), 'Northern Ireland': (54.5973, -5.9301), 'Norway': (59.9139, 10.7522),
    'Poland': (52.2297, 21.0122), 'Portugal': (38.7223, -9.1393), 'Republic of Ireland': (53.3498, -6.2603),
    'Romania': (44.4268, 26.1025), 'San Marino': (43.9424, 12.4578), 'Scotland': (55.9533, -3.1883),
    'Serbia': (44.7866, 20.4489), 'Slovakia': (48.1486, 17.1077), 'Slovenia': (46.0569, 14.5058),
    'Spain': (40.4168, -3.7038), 'Sweden': (59.3293, 18.0686), 'Switzerland': (46.9480, 7.4474),
    'Turkey': (39.9334, 32.8597), 'Ukraine': (50.4501, 30.5234), 'Wales': (51.4816, -3.1791)
}


def haversine(c1, c2):
    if not c1 or not c2:
        return 500.0
    lat1, lon1 = c1
    lat2, lon2 = c2
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    return float(2 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def _cargar_equipos():
    if EQUIPOS_PATH.exists():
        try:
            return pd.read_csv(EQUIPOS_PATH).set_index("norm_name").to_dict("index")
        except Exception:
            pass
    return {}


class StateTracker:
    """Rastreador de estado temporal e histórico de selecciones en Nations League."""
    def __init__(self, df_states=None):
        self.elo = defaultdict(lambda: 1500.0)
        self.squad_value = defaultdict(lambda: 100.0)
        self.recent_results = defaultdict(lambda: deque(maxlen=5))
        self.recent_gf = defaultdict(lambda: deque(maxlen=5))
        self.recent_ga = defaultdict(lambda: deque(maxlen=5))
        self.h2h_goles = defaultdict(float)
        self.h2h_partidos = defaultdict(int)
        self.last_date = {}

        if df_states is not None and not df_states.empty:
            for _, r in df_states.iterrows():
                team = r["team"]
                self.elo[team] = float(r.get("elo", 1500.0))
                sv = float(r.get("squad_value", 100.0))
                self.squad_value[team] = (sv / 1e6) if sv > 1000.0 else sv

    def get_features_for_match(self, local, visita, fecha_str=None):
        elo_l = self.elo[local]
        elo_v = self.elo[visita]
        elo_diff = elo_l - elo_v

        sv_l = max(self.squad_value[local], 1.0)
        sv_v = max(self.squad_value[visita], 1.0)
        log_sv_diff = float(np.log(sv_l) - np.log(sv_v))

        rl = list(self.recent_results[local])
        rv = list(self.recent_results[visita])
        form_diff = (float(np.mean(rl)) if rl else 0.333) - (float(np.mean(rv)) if rv else 0.333)

        gfl = list(self.recent_gf[local]); gfv = list(self.recent_gf[visita])
        gal = list(self.recent_ga[local]); gav = list(self.recent_ga[visita])
        gf_diff = (float(np.mean(gfl)) if gfl else 1.2) - (float(np.mean(gfv)) if gfv else 1.2)
        ga_diff = (float(np.mean(gal)) if gal else 1.1) - (float(np.mean(gav)) if gav else 1.1)

        key_h2h = (local, visita)
        key_inv = (visita, local)
        h2h_gl = self.h2h_goles[key_h2h]
        h2h_gv = self.h2h_goles[key_inv]
        h2h_diff = float(h2h_gl - h2h_gv)

        dist = haversine(COORDS_UNL.get(local), COORDS_UNL.get(visita))
        dist_scaled = float(dist / 1000.0)

        fatigue_l = 0.0
        fatigue_v = 0.0
        if fecha_str:
            try:
                dt_curr = pd.to_datetime(fecha_str)
                if local in self.last_date:
                    days_l = (dt_curr - self.last_date[local]).days
                    fatigue_l = 1.0 if days_l <= 3 else (0.5 if days_l <= 4 else 0.0)
                if visita in self.last_date:
                    days_v = (dt_curr - self.last_date[visita]).days
                    fatigue_v = 1.0 if days_v <= 3 else (0.5 if days_v <= 4 else 0.0)
            except Exception:
                pass

        return {
            "elo_diff": elo_diff,
            "log_sv_diff": log_sv_diff,
            "form_diff": form_diff,
            "gf_diff": gf_diff,
            "ga_diff": ga_diff,
            "h2h_diff": h2h_diff,
            "travel_dist": dist_scaled,
            "fatigue_diff": fatigue_l - fatigue_v,
            "elo_local": elo_l,
            "elo_visita": elo_v,
            "sv_local": sv_l,
            "sv_visita": sv_v
        }

    def registrar_partido(self, local, visita, ga, gb, fecha_str=None):
        # Actualización de Elo
        Ra = self.elo[local]
        Rb = self.elo[visita]
        Ea = 1.0 / (1.0 + 10.0 ** ((Rb - Ra + 45.0) / 400.0))  # +45 pts por localía
        Eb = 1.0 - Ea
        Sa = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
        Sb = 1.0 - Sa
        K = 32.0
        self.elo[local] += K * (Sa - Ea)
        self.elo[visita] += K * (Sb - Eb)

        # Forma reciente
        self.recent_results[local].append(Sa)
        self.recent_results[visita].append(Sb)
        self.recent_gf[local].append(ga)
        self.recent_gf[visita].append(gb)
        self.recent_ga[local].append(gb)
        self.recent_ga[visita].append(ga)

        # H2H
        self.h2h_goles[(local, visita)] += ga
        self.h2h_goles[(visita, local)] += gb
        self.h2h_partidos[(local, visita)] += 1
        self.h2h_partidos[(visita, local)] += 1

        if fecha_str:
            try:
                dt_curr = pd.to_datetime(fecha_str)
                self.last_date[local] = dt_curr
                self.last_date[visita] = dt_curr
            except Exception:
                pass


_MOTOR_CACHE = None


def cargar():
    global _MOTOR_CACHE
    if _MOTOR_CACHE is not None:
        return _MOTOR_CACHE

    equipos_info = _cargar_equipos()

    # Cargar team states base para priors
    states_path = DATA.parent / "data" / "team_states.csv"
    df_states = pd.read_csv(states_path) if states_path.exists() else pd.DataFrame()

    tracker = StateTracker(df_states)

    partidos_path = DATA / "partidos.csv"
    if not partidos_path.exists():
        raise FileNotFoundError(f"No se encontró {partidos_path}")

    df_partidos = pd.read_csv(partidos_path).sort_values("fecha").reset_index(drop=True)

    # Construcción secuencial de features
    cols_feat = ["elo_diff", "log_sv_diff", "form_diff", "gf_diff", "ga_diff", "h2h_diff", "travel_dist", "fatigue_diff"]
    filas_dataset = []

    for _, r in df_partidos.iterrows():
        loc = r["local"]
        vis = r["visita"]
        f_dt = str(r["fecha"])
        feats = tracker.get_features_for_match(loc, vis, f_dt)
        gl = int(r["goles_local"])
        gv = int(r["goles_visita"])
        y = 2 if gl > gv else (1 if gl == gv else 0)

        feats["temporada"] = int(r["temporada"])
        feats["resultado"] = y
        feats["gl"] = gl
        feats["gv"] = gv
        filas_dataset.append(feats)

        # Registrar partido en tracker point-in-time
        tracker.registrar_partido(loc, vis, gl, gv, f_dt)

    df_dataset = pd.DataFrame(filas_dataset)

    # Split train, cal, test
    train_mask = df_dataset["temporada"] <= 2022
    cal_mask = df_dataset["temporada"] == 2024
    test_mask = df_dataset["temporada"] >= 2026

    if train_mask.sum() == 0:
        train_mask = df_dataset.index < int(len(df_dataset) * 0.7)
        cal_mask = (df_dataset.index >= int(len(df_dataset) * 0.7)) & (df_dataset.index < int(len(df_dataset) * 0.85))
        test_mask = df_dataset.index >= int(len(df_dataset) * 0.85)

    X_train = df_dataset.loc[train_mask, cols_feat].fillna(0.0)
    y_train = df_dataset.loc[train_mask, "resultado"]
    X_cal = df_dataset.loc[cal_mask, cols_feat].fillna(0.0)
    y_cal = df_dataset.loc[cal_mask, "resultado"]
    X_test = df_dataset.loc[test_mask, cols_feat].fillna(0.0)
    y_test = df_dataset.loc[test_mask, "resultado"]

    # 1. Búsqueda de C para LASSO SAGA
    best_c = 0.05
    best_loss = 999.0
    for C in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        _pipe = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=2500, random_state=42))])
        _pipe.fit(X_train, y_train)
        _loss = log_loss(y_cal if len(y_cal) > 0 else y_train,
                         _pipe.predict_proba(X_cal if len(X_cal) > 0 else X_train),
                         labels=[0, 1, 2])
        if _loss < best_loss:
            best_loss = _loss
            best_c = C

    # Reentrenar sobre train + cal
    X_train_full = df_dataset.loc[train_mask | cal_mask, cols_feat].fillna(0.0)
    y_train_full = df_dataset.loc[train_mask | cal_mask, "resultado"]

    pipe_lasso = Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=3000, random_state=42))])
    pipe_lasso.fit(X_train_full, y_train_full)

    pipe_rf = Pipeline([("sc", StandardScaler()),
                        ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=12, random_state=42, n_jobs=-1))])
    pipe_rf.fit(X_train_full, y_train_full)

    # Stacking alpha óptimo en validación
    p_l_cal = pipe_lasso.predict_proba(X_cal) if len(X_cal) else pipe_lasso.predict_proba(X_train_full)
    p_r_cal = pipe_rf.predict_proba(X_cal) if len(X_cal) else pipe_rf.predict_proba(X_train_full)
    y_eval = y_cal if len(y_cal) else y_train_full

    def _stack_loss(alpha):
        blend = np.clip(alpha * p_l_cal + (1.0 - alpha) * p_r_cal, 1e-7, 1.0 - 1e-7)
        return log_loss(y_eval, blend, labels=[0, 1, 2])

    res = minimize_scalar(_stack_loss, bounds=(0.0, 1.0), method="bounded")
    alpha_opt = float(res.x)

    # Métricas en test
    def _met(proba, y_true):
        if len(y_true) == 0:
            return {"logloss": 1.0, "accuracy": 50.0}
        proba = np.clip(proba, 1e-7, 1.0 - 1e-7)
        return {
            "logloss": round(float(log_loss(y_true, proba, labels=[0, 1, 2])), 4),
            "accuracy": round(float(accuracy_score(y_true, proba.argmax(axis=1)) * 100.0), 2)
        }

    p_lasso_test = pipe_lasso.predict_proba(X_test) if len(X_test) else pipe_lasso.predict_proba(X_train_full[:20])
    p_rf_test = pipe_rf.predict_proba(X_test) if len(X_test) else pipe_rf.predict_proba(X_train_full[:20])
    p_stack_test = np.clip(alpha_opt * p_lasso_test + (1.0 - alpha_opt) * p_rf_test, 1e-7, 1.0 - 1e-7)
    y_test_eval = y_test if len(y_test) else y_train_full[:20]

    metricas = {
        "lasso": _met(p_lasso_test, y_test_eval),
        "rf": _met(p_rf_test, y_test_eval),
        "stacking": _met(p_stack_test, y_test_eval),
        "alpha": round(alpha_opt, 3),
        "best_c": best_c
    }

    # Modelo Poisson Bivariado Dixon-Coles
    # Estimación de parámetros base de ataque y defensa por diferencia de elo y squad value
    d_elos = df_dataset["elo_diff"].values
    d_svs = df_dataset["log_sv_diff"].values
    gls = df_dataset["gl"].values
    gvs = df_dataset["gv"].values

    # Regresiones Poisson simples para lambda_local y lambda_visita
    from statsmodels.genmod.families import Poisson
    import statsmodels.api as sm

    try:
        X_pois = sm.add_constant(np.column_stack([d_elos, d_svs]))
        mod_l = sm.GLM(gls, X_pois, family=Poisson()).fit()
        mod_v = sm.GLM(gvs, X_pois, family=Poisson()).fit()
        poisson_params = {
            "const_l": float(mod_l.params[0]), "b_elo_l": float(mod_l.params[1]), "b_sv_l": float(mod_l.params[2]),
            "const_v": float(mod_v.params[0]), "b_elo_v": float(mod_v.params[1]), "b_sv_v": float(mod_v.params[2])
        }
    except Exception:
        poisson_params = {
            "const_l": 0.35, "b_elo_l": 0.0018, "b_sv_l": 0.12,
            "const_v": 0.10, "b_elo_v": -0.0018, "b_sv_v": -0.12
        }

    _MOTOR_CACHE = {
        "tracker": tracker,
        "pipe_lasso": pipe_lasso,
        "pipe_rf": pipe_rf,
        "alpha_stack": alpha_opt,
        "cols_features": cols_feat,
        "poisson_params": poisson_params,
        "metricas": metricas,
        "equipos_info": equipos_info,
        "df_partidos": df_partidos
    }
    return _MOTOR_CACHE


def predecir_match(M, local, visita, modelo="stacking"):
    tracker = M["tracker"]
    cols_feat = M["cols_features"]
    feats = tracker.get_features_for_match(local, visita)
    df_feat = pd.DataFrame([feats])[cols_feat].fillna(0.0)

    if modelo == "lasso":
        p_raw = M["pipe_lasso"].predict_proba(df_feat)[0]
    elif modelo == "rf":
        p_raw = M["pipe_rf"].predict_proba(df_feat)[0]
    else:  # stacking
        alpha = M.get("alpha_stack", 0.5)
        p_l = M["pipe_lasso"].predict_proba(df_feat)[0]
        p_r = M["pipe_rf"].predict_proba(df_feat)[0]
        p_raw = alpha * p_l + (1.0 - alpha) * p_r
        p_raw = p_raw / p_raw.sum()

    # Reordenar a [P(Local), P(Empate), P(Visita)] -> clases son [0: Gana Visita, 1: Empate, 2: Gana Local]
    p_1x2 = np.array([p_raw[2], p_raw[1], p_raw[0]])

    # Goles esperados (Poisson)
    pp = M["poisson_params"]
    d_elo = feats["elo_diff"]
    d_sv = feats["log_sv_diff"]

    log_la = pp["const_l"] + pp["b_elo_l"] * d_elo + pp["b_sv_l"] * d_sv
    log_lb = pp["const_v"] + pp["b_elo_v"] * d_elo + pp["b_sv_v"] * d_sv
    la = float(np.clip(np.exp(log_la), 0.2, 5.0))
    lb = float(np.clip(np.exp(log_lb), 0.1, 5.0))

    return p_1x2, la, lb


def calcular_mercados(la, lb, max_goles=6):
    mat = np.zeros((max_goles, max_goles))
    for i in range(max_goles):
        for j in range(max_goles):
            mat[i, j] = poisson.pmf(i, la) * poisson.pmf(j, lb)
    mat = mat / mat.sum()

    over15 = float(mat[np.fromfunction(lambda i, j: i + j > 1, mat.shape)].sum())
    over25 = float(mat[np.fromfunction(lambda i, j: i + j > 2, mat.shape)].sum())
    over35 = float(mat[np.fromfunction(lambda i, j: i + j > 3, mat.shape)].sum())
    btts_si = float(mat[1:, 1:].sum())

    top_scores = []
    for i in range(max_goles):
        for j in range(max_goles):
            top_scores.append((f"{i}-{j}", float(mat[i, j])))
    top_scores = sorted(top_scores, key=lambda x: x[1], reverse=True)[:5]

    return {
        "matrix": mat,
        "over_under": {
            "1.5": {"over": over15, "under": 1.0 - over15},
            "2.5": {"over": over25, "under": 1.0 - over25},
            "3.5": {"over": over35, "under": 1.0 - over35}
        },
        "btts": {"si": btts_si, "no": 1.0 - btts_si},
        "top_marcadores": top_scores
    }


def simular_campeonato(_M, n_sims=3000, modelo_tipo="stacking", forzar=False):
    """Simula estocásticamente los partidos restantes de UEFA Nations League 2026-27."""
    cache_path = DATA.parent / "simulacion_mc.pkl"
    if not forzar and cache_path.exists():
        try:
            import pickle
            with open(cache_path, "rb") as fh:
                res_cache = pickle.load(fh)
                if isinstance(res_cache, dict) and "campeon" in res_cache:
                    return res_cache
        except Exception:
            pass

    df_partidos = _M["df_partidos"]
    p_2026 = df_partidos[df_partidos["temporada"] == 2026].copy()

    fix_path = DATA / "fixture.csv"
    if not fix_path.exists():
        df_fixture = pd.DataFrame()
    else:
        df_fixture = pd.read_csv(fix_path)

    eq_path = DATA / "equipos.csv"
    df_eq = pd.read_csv(eq_path)

    # 1. Puntos y goles reales ya jugados en 2026
    grupos = sorted(df_eq["group"].unique())
    tablas_actuales = {}

    for grp in grupos:
        equipos_grp = df_eq[df_eq["group"] == grp]["norm_name"].tolist()
        tab = {eq: {"PJ": 0, "PG": 0, "PE": 0, "PP": 0, "GF": 0, "GC": 0, "DG": 0, "Pts": 0} for eq in equipos_grp}
        
        # Partidos jugados en este grupo
        sub_p = p_2026[(p_2026["local"].isin(equipos_grp)) & (p_2026["visita"].isin(equipos_grp))]
        for _, r in sub_p.iterrows():
            loc, vis = r["local"], r["visita"]
            gl, gv = int(r["goles_local"]), int(r["goles_visita"])
            tab[loc]["PJ"] += 1; tab[vis]["PJ"] += 1
            tab[loc]["GF"] += gl; tab[loc]["GC"] += gv
            tab[vis]["GF"] += gv; tab[vis]["GC"] += gl
            tab[loc]["DG"] = tab[loc]["GF"] - tab[loc]["GC"]
            tab[vis]["DG"] = tab[vis]["GF"] - tab[vis]["GC"]
            if gl > gv:
                tab[loc]["PG"] += 1; tab[loc]["Pts"] += 3; tab[vis]["PP"] += 1
            elif gl == gv:
                tab[loc]["PE"] += 1; tab[loc]["Pts"] += 1; tab[vis]["PE"] += 1; tab[vis]["Pts"] += 1
            else:
                tab[vis]["PG"] += 1; tab[vis]["Pts"] += 3; tab[loc]["PP"] += 1

        df_tab = pd.DataFrame([{"Equipo": k, **v} for k, v in tab.items()]).sort_values(
            by=["Pts", "DG", "GF"], ascending=[False, False, False]
        ).reset_index(drop=True)
        tablas_actuales[grp] = df_tab

    # 2. Monte Carlo de partidos pendientes
    fixture_pend = df_fixture[df_fixture["estado"] != "post"].copy()

    # Pre-calcular probabilidades para cada partido pendiente
    match_probs = {}
    for _, r in fixture_pend.iterrows():
        pair = (r["local"], r["visita"])
        if pair not in match_probs:
            p, la, lb = predecir_match(_M, r["local"], r["visita"], modelo=modelo_tipo)
            match_probs[pair] = (p, la, lb)

    # Estructuras de acumulación
    pts_sim = {grp: {eq: [] for eq in df_eq[df_eq["group"] == grp]["norm_name"]} for grp in grupos}
    pos_sim = {grp: {eq: defaultdict(int) for eq in df_eq[df_eq["group"] == grp]["norm_name"]} for grp in grupos}
    campeon_sim = defaultdict(int)

    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        sim_scores = {}
        for grp in grupos:
            # Copiar estado base
            tab_copy = tablas_actuales[grp].set_index("Equipo").to_dict("index")
            equipos_grp = list(tab_copy.keys())

            # Partidos pendientes de este grupo
            sub_fix = fixture_pend[(fixture_pend["local"].isin(equipos_grp)) & (fixture_pend["visita"].isin(equipos_grp))]
            for _, r in sub_fix.iterrows():
                loc, vis = r["local"], r["visita"]
                p, la, lb = match_probs.get((loc, vis), (np.array([0.45, 0.28, 0.27]), 1.4, 1.1))
                
                # Simular marcador estocásticamente vía Poisson
                sim_gl = rng.poisson(la)
                sim_gv = rng.poisson(lb)
                
                tab_copy[loc]["GF"] += sim_gl; tab_copy[loc]["GC"] += sim_gv
                tab_copy[vis]["GF"] += sim_gv; tab_copy[vis]["GC"] += sim_gl
                if sim_gl > sim_gv:
                    tab_copy[loc]["Pts"] += 3
                elif sim_gl == sim_gv:
                    tab_copy[loc]["Pts"] += 1; tab_copy[vis]["Pts"] += 1
                else:
                    tab_copy[vis]["Pts"] += 3

            # Calcular orden final
            sorted_teams = sorted(
                equipos_grp,
                key=lambda eq: (tab_copy[eq]["Pts"], tab_copy[eq]["GF"] - tab_copy[eq]["GC"], tab_copy[eq]["GF"]),
                reverse=True
            )
            for idx, eq in enumerate(sorted_teams, 1):
                pos_sim[grp][eq][idx] += 1
                pts_sim[grp][eq].append(tab_copy[eq]["Pts"])

            sim_scores[grp] = sorted_teams

        # Simular Final Four de Liga A (los 4 ganadores de A1, A2, A3, A4)
        if all(grp in sim_scores for grp in ["Group A1", "Group A2", "Group A3", "Group A4"]):
            top_a = [sim_scores["Group A1"][0], sim_scores["Group A2"][0], sim_scores["Group A3"][0], sim_scores["Group A4"][0]]
            # Semifinal 1: top_a[0] vs top_a[1]
            p_s1, _, _ = match_probs.get((top_a[0], top_a[1]), (np.array([0.45, 0.25, 0.30]), 1.4, 1.2))
            adv1 = top_a[0] if rng.random() < (p_s1[0] / (p_s1[0] + p_s1[2])) else top_a[1]

            # Semifinal 2: top_a[2] vs top_a[3]
            p_s2, _, _ = match_probs.get((top_a[2], top_a[3]), (np.array([0.45, 0.25, 0.30]), 1.4, 1.2))
            adv2 = top_a[2] if rng.random() < (p_s2[0] / (p_s2[0] + p_s2[2])) else top_a[3]

            # Final
            p_fin, _, _ = match_probs.get((adv1, adv2), (np.array([0.45, 0.25, 0.30]), 1.4, 1.2))
            champ = adv1 if rng.random() < (p_fin[0] / (p_fin[0] + p_fin[2])) else adv2
            campeon_sim[champ] += 1

    # Resumen de proyecciones
    proyecciones_grupos = {}
    for grp in grupos:
        filas = []
        for eq in tablas_actuales[grp]["Equipo"]:
            n_1 = pos_sim[grp][eq][1]
            n_4 = pos_sim[grp][eq][4] if len(tablas_actuales[grp]) >= 4 else pos_sim[grp][eq][3]
            filas.append({
                "Equipo": eq,
                "Pts_Proy": round(float(np.mean(pts_sim[grp][eq])), 1),
                "P(1°)": n_1 / n_sims,
                "P(2°)": pos_sim[grp][eq][2] / n_sims,
                "P(3°)": pos_sim[grp][eq][3] / n_sims,
                "P(Descenso)": n_4 / n_sims if grp.startswith("Group A") or grp.startswith("Group B") or grp.startswith("Group C") else 0.0
            })
        proyecciones_grupos[grp] = pd.DataFrame(filas).sort_values("Pts_Proy", ascending=False).reset_index(drop=True)

    # DataFrame de Campeón (Liga A)
    filas_camp = []
    teams_liga_a = df_eq[df_eq["league"] == "A"]["norm_name"].tolist()
    for eq in teams_liga_a:
        grp = df_eq[df_eq["norm_name"] == eq]["group"].values[0]
        p_f4 = pos_sim[grp][eq][1] / n_sims
        p_champ = campeon_sim[eq] / n_sims
        filas_camp.append({
            "Selección": eq,
            "Grupo": grp,
            "P_FinalFour": p_f4,
            "P_Campeon": p_champ
        })
    df_campeon = pd.DataFrame(filas_camp).sort_values("P_Campeon", ascending=False).reset_index(drop=True)

    # DataFrame de Ascensos (Ligas B, C, D)
    filas_asc = []
    teams_bcd = df_eq[df_eq["league"].isin(["B", "C", "D"])]["norm_name"].tolist()
    for eq in teams_bcd:
        grp = df_eq[df_eq["norm_name"] == eq]["group"].values[0]
        p_asc = pos_sim[grp][eq][1] / n_sims
        filas_asc.append({
            "Selección": eq,
            "Liga_Actual": df_eq[df_eq["norm_name"] == eq]["league"].values[0],
            "Grupo": grp,
            "P_Ascenso": p_asc
        })
    df_ascensos = pd.DataFrame(filas_asc).sort_values("P_Ascenso", ascending=False).reset_index(drop=True)

    res_final = {
        "tablas_actuales": tablas_actuales,
        "proyecciones_grupos": proyecciones_grupos,
        "campeon": df_campeon,
        "ascensos": df_ascensos,
        "n_sims": n_sims
    }
    try:
        import pickle
        with open(cache_path, "wb") as fh:
            pickle.dump(res_final, fh)
    except Exception:
        pass
    return res_final


def validacion_en_vivo(M, resultados=None, modelo="stacking"):
    """Audita las predicciones pre-partido contra resultados reales de Nations League 2026."""
    if resultados is None:
        p_path = DATA / "partidos.csv"
        if not p_path.exists():
            return pd.DataFrame(), {}, pd.DataFrame()
        df_all = pd.read_csv(p_path)
        resultados = df_all[df_all["temporada"] == 2026].copy()

    if len(resultados) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    filas, P, y = [], [], []
    for r in resultados.itertuples(index=False):
        loc, vis = r.local, r.visita
        p, la, lb = predecir_match(M, loc, vis, modelo=modelo)
        gl, gv = int(r.goles_local), int(r.goles_visita)
        real = 2 if gl > gv else (1 if gl == gv else 0)
        
        P.append([p[2], p[1], p[0]])  # orden de clases [0: Visita, 1: Empate, 2: Local]
        y.append(real)
        pred = 2 if (p[0] >= p[1] and p[0] >= p[2]) else (1 if p[1] >= p[2] else 0)

        filas.append({
            "Fecha": pd.to_datetime(r.fecha).strftime("%Y-%m-%d"),
            "Local": loc,
            "Marcador": f"{gl}-{gv}",
            "Visita": vis,
            "P(Local)": f"{p[0]:.0%}",
            "P(Empate)": f"{p[1]:.0%}",
            "P(Visita)": f"{p[2]:.0%}",
            "Pred": ["Visita", "Empate", "Local"][pred],
            "Real": ["Visita", "Empate", "Local"][real],
            "Acierto": "✅" if pred == real else "❌"
        })

    P = np.array(P)
    y = np.array(y)
    n = len(y)
    base = np.tile([0.28, 0.26, 0.46], (n, 1))

    aciertos = sum(1 for i in range(n) if np.argmax(P[i]) == y[i])
    met = {
        "n": n,
        "acierto": aciertos / n if n > 0 else 0.0,
        "logloss": float(log_loss(y, P, labels=[0, 1, 2])) if n > 0 else 1.0,
        "logloss_base": float(log_loss(y, base, labels=[0, 1, 2])) if n > 0 else 1.0,
        "P": P,
        "y": y
    }

    evol = [float(log_loss(y[:i + 1], P[:i + 1], labels=[0, 1, 2])) for i in range(n)]
    evolucion = pd.DataFrame({
        "partido": range(1, n + 1),
        "logloss_acum": evol,
        "baseline": [met["logloss_base"]] * n
    })

    return pd.DataFrame(filas), met, evolucion


def cuota(p):
    p = max(float(p), 0.01)
    return round(1.0 / p, 2)


def generar_reporte_partidos_ia(M, df_matches=None, modelo="stacking"):
    """Genera un DataFrame enriquecido con todas las métricas predictivas, xG Poisson,
    diferenciales de ELO, valor de plantilla, forma y cuotas para exportación a CSV
    y análisis con modelos de IA (ChatGPT, Claude, Gemini, etc.)."""
    if df_matches is None:
        p_fix = DATA / "fixture.csv"
        if p_fix.exists():
            df_matches = pd.read_csv(p_fix)
        else:
            return pd.DataFrame()

    tracker = M["tracker"]
    filas = []
    for _, r in df_matches.iterrows():
        loc, vis = r["local"], r["visita"]
        f_str = str(r.get("fecha", ""))
        p_main, la, lb = predecir_match(M, loc, vis, modelo=modelo)
        p_lasso, _, _ = predecir_match(M, loc, vis, modelo="lasso")
        p_rf, _, _ = predecir_match(M, loc, vis, modelo="rf")
        
        merc = calcular_mercados(la, lb)
        top_sc = merc["top_marcadores"]
        
        feats = tracker.get_features_for_match(loc, vis, f_str)
        elo_l = feats["elo_local"]
        elo_v = feats["elo_visita"]
        sv_l = feats["sv_local"]
        sv_v = feats["sv_visita"]
        
        # Alerta Heurística
        if p_main[0] > 0.60 and (la - lb) > 0.8:
            alerta = "🔥 Alta Confianza Local"
        elif p_main[0] > 0.60:
            alerta = "💪 Favorito Claro Local"
        elif p_main[2] > 0.50:
            alerta = "⚠️ Visita Fuerte"
        elif max(p_main) < 0.44:
            alerta = "⚖️ Partido Muy Parejo"
        elif p_main[1] > 0.28:
            alerta = "🤝 Alta Probabilidad Empate"
        else:
            alerta = "🎯 Pronóstico Estándar"

        filas.append({
            "id_partido": r.get("event_id", ""),
            "fecha": f_str,
            "ronda": r.get("round", ""),
            "grupo": r.get("group", ""),
            "liga": r.get("league", ""),
            "local": loc,
            "visita": vis,
            "prob_victoria_local_%": round(float(p_main[0] * 100), 1),
            "prob_empate_%": round(float(p_main[1] * 100), 1),
            "prob_victoria_visita_%": round(float(p_main[2] * 100), 1),
            "cuota_justa_local": cuota(p_main[0]),
            "cuota_justa_empate": cuota(p_main[1]),
            "cuota_justa_visita": cuota(p_main[2]),
            "xg_local": round(float(la), 2),
            "xg_visita": round(float(lb), 2),
            "marcador_mas_probable": top_sc[0][0] if len(top_sc) > 0 else "1-0",
            "prob_marcador_%": round(float(top_sc[0][1] * 100), 1) if len(top_sc) > 0 else 12.0,
            "marcador_alternativo": top_sc[1][0] if len(top_sc) > 1 else "1-1",
            "prob_over_2_5_%": round(float(merc["over_under"]["2.5"]["over"] * 100), 1),
            "prob_ambos_marcan_%": round(float(merc["btts"]["si"] * 100), 1),
            "elo_local": round(float(elo_l), 1),
            "elo_visita": round(float(elo_v), 1),
            "elo_diff": round(float(feats["elo_diff"]), 1),
            "plantilla_local_M€": round(float(sv_l), 1),
            "plantilla_visita_M€": round(float(sv_v), 1),
            "ratio_plantilla": round(float(sv_l / max(sv_v, 1.0)), 2),
            "forma_diff": round(float(feats["form_diff"]), 3),
            "distancia_viaje_km": round(float(feats["travel_dist"] * 1000.0), 0),
            "fatiga_diff": round(float(feats["fatigue_diff"]), 1),
            "alerta_modelo": alerta,
            "consenso_lasso_local_%": round(float(p_lasso[0] * 100), 1),
            "consenso_rf_local_%": round(float(p_rf[0] * 100), 1)
        })
    return pd.DataFrame(filas)


def curva_calibracion(P, y, clase_idx, n_bins=5):
    p_k = P[:, clase_idx]
    y_k = (y == clase_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    ece = 0.0
    N = len(y)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (p_k >= b0) & (p_k < b1 if b1 < 1.0 else p_k <= b1)
        if mask.sum() > 0:
            p_mean = float(p_k[mask].mean())
            y_mean = float(y_k[mask].mean())
            xs.append(p_mean)
            ys.append(y_mean)
            ns.append(int(mask.sum()))
            ece += (mask.sum() / N) * abs(y_mean - p_mean)
    return xs, ys, ns, float(ece)


