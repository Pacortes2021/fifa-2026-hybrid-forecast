import sys
import math
from pathlib import Path
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize_scalar

DATA = Path(__file__).resolve().parent / "data"

# Constantes de LaLiga de España
DESCIENDEN = 3
CUPOS_COPA = 7  # Champions League (1º a 4º) + Europa League (5º y 6º) + Conference (7º)

SQUAD_VALUES = {
    "Real Madrid": 1040.0, "Barcelona": 860.0, "Atlético Madrid": 480.0,
    "Real Sociedad": 380.0, "Athletic Club": 290.0, "Girona": 270.0,
    "Villarreal": 210.0, "Real Betis": 190.0, "Sevilla": 170.0,
    "Valencia": 150.0, "Celta Vigo": 110.0, "Osasuna": 95.0,
    "Getafe": 85.0, "Mallorca": 80.0, "Las Palmas": 75.0,
    "Rayo Vallecano": 70.0, "Alavés": 65.0, "Leganés": 55.0,
    "Valladolid": 50.0, "Espanyol": 55.0, "Granada": 60.0,
    "Cádiz": 50.0, "Almería": 45.0, "Elche": 35.0,
    "Levante": 40.0, "Eibar": 25.0, "Huesca": 20.0
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
K_LIGA = 35.0      # ELO K-factor para LaLiga (las ligas top tienen más volatilidad)
HOME_ADV = 60.0    # Ventaja de local típica de 60 puntos ELO


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
        "squad_size": 25, "avg_age": 25.0, "foreigners": 0, "pct_foreigners": 0.0,
        "stadium_capacity": 35000, "avg_attendance": 20000, "stadium_occupation": 0.5,
        "squad_value": SQUAD_VALUES.get(team, 50.0)
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


class StateTracker:
    def __init__(self):
        self.elos = defaultdict(lambda: ELO_INIT)
        self.history = defaultdict(deque)
        self.home_history = defaultdict(deque)
        self.away_history = defaultdict(deque)
        self.h2h_goles = defaultdict(float)
        self.recent_results = defaultdict(deque)
        self.recent_gf = defaultdict(deque)
        self.recent_ga = defaultdict(deque)
        self.match_count = defaultdict(int)

    def get_features_for_match(self, local, visita, temporada):
        feats = {}
        feats["elo_diff"] = self.elos[local] - self.elos[visita]
        vl = get_squad_value(local, temporada)
        vv = get_squad_value(visita, temporada)
        feats["squad_value_diff"] = np.log(max(vl, 0.1)) - np.log(max(vv, 0.1))
        feats["h2h_diff"] = self.h2h_goles[(local, visita)]

        # Características avanzadas de TM
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

    def registrar_partido(self, local, visita, ga, gb, stats_l=None, stats_v=None):
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
        
        feats = tracker.get_features_for_match(local, visita, r.temporada)
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
        tracker.registrar_partido(local, visita, ga, gb, stats_l, stats_v)
        
    df_features = pd.DataFrame(filas_X)
    
    # Modelo 1: L1 Regularized SAGA logistic regression
    cols_feat = [
        "elo_diff", "squad_value_diff", "h2h_diff",
        "avg_age_diff", "squad_size_diff", "pct_foreigners_diff",
        "stadium_capacity", "stadium_occupation", "avg_attendance",
        "form_diff", "gf_diff", "ga_diff"
    ] + \
                [f"{s}_total_diff" for s in STATS] + [f"{s}_sede_diff" for s in STATS]
                
    train_mask = df_features["temporada"] <= 2023
    cal_mask   = df_features["temporada"] == 2024
    test_mask  = df_features["temporada"] >= 2025

    X_train = df_features.loc[train_mask, cols_feat].fillna(0.0)
    y_train = df_features.loc[train_mask, "resultado"]
    X_cal = df_features.loc[cal_mask, cols_feat].fillna(0.0)
    y_cal = df_features.loc[cal_mask, "resultado"]
    X_test = df_features.loc[test_mask, cols_feat].fillna(0.0)
    y_test = df_features.loc[test_mask, "resultado"]

    pipe_lasso_base = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=0.04, max_iter=4000, random_state=42))])
    pipe_lasso_base.fit(X_train, y_train)
    pipe_rf_base = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(max_depth=5, n_estimators=200, min_samples_split=15, random_state=42, n_jobs=-1))])
    pipe_rf_base.fit(X_train, y_train)

    if len(X_cal) >= 10:
        pipe_lasso_cal = pipe_lasso_base
        pipe_rf_cal = pipe_rf_base
    else:
        pipe_lasso_cal = pipe_lasso_base; pipe_rf_cal = pipe_rf_base

    if len(X_cal) >= 10:
        p_l_cal = pipe_lasso_cal.predict_proba(X_cal)
        p_r_cal = pipe_rf_cal.predict_proba(X_cal)
        def _stkl(alpha):
            return log_loss(y_cal, np.clip(alpha*p_l_cal+(1-alpha)*p_r_cal, 1e-7, 1-1e-7))
        alpha_opt = float(minimize_scalar(_stkl, bounds=(0.0,1.0), method="bounded").x)
    else:
        alpha_opt = 0.4

    X_full = df_features.loc[train_mask | cal_mask, cols_feat].fillna(0.0)
    y_full = df_features.loc[train_mask | cal_mask, "resultado"]
    pipe_lasso = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=0.04, max_iter=4000, random_state=42))])
    pipe_lasso.fit(X_full, y_full)
    pipe_rf = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(max_depth=5, n_estimators=200, min_samples_split=15, random_state=42, n_jobs=-1))])
    pipe_rf.fit(X_full, y_full)

    def _met(proba, y):
        proba = np.clip(proba, 1e-7, 1-1e-7)
        return {"logloss": round(log_loss(y, proba), 4), "accuracy": round(accuracy_score(y, proba.argmax(axis=1))*100, 2)}
    met_lasso = _met(pipe_lasso.predict_proba(X_test), y_test)
    met_rf = _met(pipe_rf.predict_proba(X_test), y_test)
    p_st = np.clip(alpha_opt*pipe_lasso.predict_proba(X_test)+(1-alpha_opt)*pipe_rf.predict_proba(X_test), 1e-7, 1-1e-7)
    met_stack = {"logloss": round(log_loss(y_test,p_st),4), "accuracy": round(accuracy_score(y_test,p_st.argmax(axis=1))*100,2), "alpha": round(alpha_opt,3)}
    metricas = {"lasso": met_lasso, "rf": met_rf, "stacking": met_stack}
    print(f"Metricas ESP Test>=2025: LASSO={met_lasso} RF={met_rf} Stacking={met_stack}")
    
    # Ajuste Poisson para goles esperados
    # Estimamos goles promedio en función del ELO diferencial
    largo = pd.concat([
        pd.DataFrame({"g": df_features["goles_local"].values, "d": df_features["elo_diff"].values}),
        pd.DataFrame({"g": df_features["goles_visita"].values, "d": -df_features["elo_diff"].values})
    ])
    
    gp = sm.GLM(largo["g"], sm.add_constant(largo[["d"]]), family=sm.families.Poisson()).fit()
    g_const, g_d = float(gp.params["const"]), float(gp.params["d"])
    
    return {
        "pipe_lasso": pipe_lasso,
        "pipe_rf": pipe_rf,
        "cols": cols_feat,
        "tracker": tracker,
        "g_const": g_const,
        "g_d": g_d,
        "df_features": df_features,
        "alpha_stack": alpha_opt,
        "metricas": metricas
    }


def cargar():
    # Retorna un diccionario con modelos entrenados y estados finales
    return cargar_y_entrenar()


def predecir_match(M, local, visita, modelo_tipo="rf"):
    # Retorna P(Local), P(Empate), P(Visita) en base al modelo seleccionado
    tracker = M["tracker"]
    cols = M["cols"]
    
    # Obtener features en el estado final del tracker
    feats = tracker.get_features_for_match(local, visita, 2026)
    df_test = pd.DataFrame([feats])[cols]
    
    if modelo_tipo == "stacking":
        alpha = M.get("alpha_stack", 0.4)
        p_l = M["pipe_lasso"].predict_proba(df_test)[0]
        p_r = M["pipe_rf"].predict_proba(df_test)[0]
        p_raw = alpha*p_l + (1-alpha)*p_r; p_raw /= p_raw.sum()
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
    la = np.exp(M["g_const"] + M["g_d"] * elo_diff)
    lb = np.exp(M["g_const"] - M["g_d"] * elo_diff)
    
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
                p = predecir_match(M, local, visita, modelo_tipo=modelo_tipo)
            # Simular resultado
            rnd = np.random.rand()
            if rnd < p[0]:
                gl, gv = 2, 0  # Victoria local
            elif rnd < p[0] + p[1]:
                gl, gv = 1, 1  # Empate
            else:
                gl, gv = 0, 2  # Victoria visita
                
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


def simular_campeonato(M, n_sims=3000, fijos=None, modelo_tipo="rf"):
    # Corre simulación de Monte Carlo para obtener probabilidades de campeón, copas y descenso
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
        res.iloc[-DESCIENDEN:]["P_descenso"] = 1.0
        return res
        
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
        PREDS[(r.local, r.visita)] = predecir_match(M, r.local, r.visita, modelo_tipo=modelo_tipo)
        
    counts_campeon = defaultdict(int)
    counts_copas = defaultdict(int)
    counts_descenso = defaultdict(int)
    
    for _ in range(n_sims):
        tabla_sim = simular_fixture_regular(M, PREDS, fijos, modelo_tipo=modelo_tipo)
        
        # Campeón
        counts_campeon[tabla_sim.iloc[0]["equipo"]] += 1
        
        # Copas
        for eq in tabla_sim.iloc[:CUPOS_COPA]["equipo"]:
            counts_copas[eq] += 1
            
        # Descenso
        for eq in tabla_sim.iloc[-DESCIENDEN:]["equipo"]:
            counts_descenso[eq] += 1
            
    # Armar DataFrame resumen
    res = []
    for eq in sorted(list(eqs)):
        res.append({
            "equipo": eq,
            "P_campeon": counts_campeon[eq] / n_sims,
            "P_copas": counts_copas[eq] / n_sims,
            "P_descenso": counts_descenso[eq] / n_sims
        })
    return pd.DataFrame(res).sort_values("P_campeon", ascending=False).reset_index(drop=True)


def validacion_en_vivo(M, temporada_val=2026, modelo_tipo="rf"):
    # Mismo reporte de validación en vivo para LaLiga
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
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
            
            pipe = M["pipe_rf"] if modelo_tipo == "rf" else M["pipe_lasso"]
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
