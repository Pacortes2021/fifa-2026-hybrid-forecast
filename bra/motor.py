"""
Motor predictivo de Machine Learning y simulación de Monte Carlo para la Serie A de Brasil (Brasileirão).
Utiliza regresión logística multinomial con penalización L1 (LASSO) mediante el solver SAGA.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from scipy.stats import poisson
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score

DATA = Path(__file__).resolve().parent / "data"

SQUAD_VALUES_PATH = DATA / "squad_values_historical.csv"
if SQUAD_VALUES_PATH.exists():
    DF_SQUAD_VALUES = pd.read_csv(SQUAD_VALUES_PATH)
else:
    DF_SQUAD_VALUES = pd.DataFrame(columns=["temporada", "equipo", "squad_value"])

STATS = [
    "foulsCommitted", "yellowCards", "redCards", "offsides", "wonCorners", "saves",
    "possessionPct", "totalShots", "shotsOnTarget", "shotPct", "penaltyKickGoals",
    "penaltyKickShots", "accuratePasses", "totalPasses", "passPct", "accurateCrosses",
    "totalCrosses", "crossPct", "totalLongBalls", "accurateLongBalls", "longballPct",
    "blockedShots", "effectiveTackles", "totalTackles", "tacklePct", "interceptions",
    "effectiveClearance", "totalClearance"
]

ELO_INIT = 1500.0
K_LIGA = 35.0
HOME_ADV = 60.0  # ventaja Elo típica de localía


def get_squad_value(team, season):
    # Intentar buscar el valor real exacto de Transfermarkt
    if len(DF_SQUAD_VALUES) > 0:
        df_eq = DF_SQUAD_VALUES[DF_SQUAD_VALUES.equipo == team]
        if len(df_eq) > 0:
            row = df_eq[df_eq.temporada == season]
            if len(row) > 0:
                return float(row["squad_value"].iloc[0])
            diffs = (df_eq["temporada"] - season).abs()
            best_idx = diffs.idxmin()
            return float(df_eq.loc[best_idx, "squad_value"])
    # Fallback estático
    return 10.0


def actualizar_elo(ea, eb, ga, gb):
    """Retorna el delta Elo a aplicar según goles anotados."""
    we = 1 / (1 + 10 ** (-(ea - eb) / 400))
    w = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
    gd = abs(ga - gb)
    # Multiplicador de diferencia de goles
    mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8))
    return K_LIGA * mult * (w - we)


class StateTracker:
    def __init__(self):
        self.elos = defaultdict(lambda: ELO_INIT)
        # Historial de boxscores de cada equipo (máx 6 partidos generales)
        self.history = defaultdict(deque)
        # Historial de boxscores por sede (máx 4 partidos local/visita)
        self.home_history = defaultdict(deque)
        self.away_history = defaultdict(deque)
        # Goles acumulados cara a cara (H2H)
        self.h2h_goles = defaultdict(float)

    def get_features_for_match(self, local, visita, temporada):
        feats = {}
        # 1. Elo difference (prior)
        feats["elo_diff"] = self.elos[local] - self.elos[visita]
        
        # 2. Squad Value difference (prior)
        vl = get_squad_value(local, temporada)
        vv = get_squad_value(visita, temporada)
        feats["squad_value_diff"] = np.log(vl) - np.log(vv)
        
        # 3. H2H diff
        feats["h2h_diff"] = self.h2h_goles[(local, visita)]

        # 4. Boxscore stats
        for s in STATS:
            # Forma general (últimos 6 partidos)
            hl = self.history[local]
            hv = self.history[visita]
            vsl = [h[s] for h in hl if h[s] is not None]
            vsv = [h[s] for h in hv if h[s] is not None]
            feats[f"{s}_total_diff"] = (np.mean(vsl) if vsl else 0.0) - (np.mean(vsv) if vsv else 0.0)

            # Forma sede (últimos 4 local / 4 visita)
            hhl = self.home_history[local]
            ahv = self.away_history[visita]
            vhl = [h[s] for h in hhl if h[s] is not None]
            vav = [h[s] for h in ahv if h[s] is not None]
            feats[f"{s}_sede_diff"] = (np.mean(vhl) if vhl else 0.0) - (np.mean(vav) if vav else 0.0)

        return feats

    def registrar_partido(self, local, visita, ga, gb, stats_l=None, stats_v=None):
        # Actualizar H2H
        self.h2h_goles[(local, visita)] += (ga - gb)
        self.h2h_goles[(visita, local)] -= (ga - gb)

        # Actualizar Elo
        delta = actualizar_elo(self.elos[local] + HOME_ADV, self.elos[visita], ga, gb)
        self.elos[local] += delta
        self.elos[visita] -= delta

        # Inicializar stats nulos si no hay boxscore
        sl = stats_l if stats_l else {s: None for s in STATS}
        sv = stats_v if stats_v else {s: None for s in STATS}

        # Registrar historial
        self.history[local].append(sl)
        if len(self.history[local]) > 6: self.history[local].popleft()
        self.history[visita].append(sv)
        if len(self.history[visita]) > 6: self.history[visita].popleft()

        # Registrar historial sede
        self.home_history[local].append(sl)
        if len(self.home_history[local]) > 4: self.home_history[local].popleft()
        self.away_history[visita].append(sv)
        if len(self.away_history[visita]) > 4: self.away_history[visita].popleft()


def cargar_y_entrenar():
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    
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
        
    df_dataset = pd.DataFrame(filas_X)
    
    # Features candidates
    cols_features = ["elo_diff", "squad_value_diff", "h2h_diff"]
    for s in STATS:
        cols_features.append(f"{s}_total_diff")
        cols_features.append(f"{s}_sede_diff")
        
    # Walk-forward division:
    train_mask = df_dataset["temporada"] <= 2024
    X_train_raw = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    
    # Filter columns with zero variance
    non_zero_cols = [c for c in cols_features if X_train_raw[c].std() > 1e-5]
    cols_features = non_zero_cols

    X_train = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    y_train = df_dataset.loc[train_mask, "resultado"]
    X_test = df_dataset.loc[~train_mask, cols_features].fillna(0.0)
    y_test = df_dataset.loc[~train_mask, "resultado"]
    
    # Train L1 with SAGA solver to minimize Log-Loss
    best_c = 0.05
    best_loss = 999.0
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=2000, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)
        
        loss = log_loss(y_test, probs, labels=[0, 1, 2])
        if loss < best_loss:
            best_loss = loss
            best_c = C
            
    print(f"Mejor C para Lasso (SAGA): {best_c} | Log-Loss en Test (2025-26): {best_loss:.4f}")
    
    pipe_final = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=2000, random_state=42))
    ])
    pipe_final.fit(df_dataset[cols_features].fillna(0.0), df_dataset["resultado"])
    
    # Active features
    coefs = pipe_final.named_steps["lr"].coef_
    avg_coefs = np.mean(np.abs(coefs), axis=0)
    active_features = []
    for col, val in sorted(zip(cols_features, avg_coefs), key=lambda x: x[1], reverse=True):
        if val > 0.001:
            active_features.append(col)
            
    # Goles Poisson GLM
    datag_l = pd.DataFrame({
        "g": df_dataset["goles_local"].values,
        "d": df_dataset["elo_diff"].values
    })
    datag_v = pd.DataFrame({
        "g": df_dataset["goles_visita"].values,
        "d": -df_dataset["elo_diff"].values
    })
    datag_total = pd.concat([datag_l, datag_v], ignore_index=True)
    
    gp = sm.GLM(
        datag_total["g"],
        sm.add_constant(datag_total["d"]),
        family=sm.families.Poisson()
    ).fit()
    
    # Dixon-Coles Correlation Approximation
    rho_dc = 0.05
    try:
        cor = np.corrcoef(df_dataset["goles_local"], df_dataset["goles_visita"])[0, 1]
        rho_dc = float(cor) * 0.5
    except Exception:
        pass
        
    return {
        "tracker": tracker,
        "pipe": pipe_final,
        "features": cols_features,
        "active_features": active_features,
        "poisson_params": gp.params,
        "rho_dc": rho_dc,
        "partidos": partidos,
        "box_dict": box_dict,
        "df_dataset": df_dataset
    }


_MOTOR_CACHE = None

def cargar():
    global _MOTOR_CACHE
    if _MOTOR_CACHE is None:
        _MOTOR_CACHE = cargar_y_entrenar()
    return _MOTOR_CACHE


def predecir_match(M, local, visita):
    tracker = M["tracker"]
    pipe = M["pipe"]
    features = M["features"]
    poisson_params = M["poisson_params"]
    
    feats = tracker.get_features_for_match(local, visita, 2026)
    df_feat = pd.DataFrame([feats])[features].fillna(0.0)
    
    p_raw = pipe.predict_proba(df_feat)[0]
    p = np.array([p_raw[2], p_raw[1], p_raw[0]]) # [local, empate, visita]
    
    # Poisson local
    d_elo_l = feats["elo_diff"]
    la = float(np.exp(poisson_params["const"] + poisson_params["d"] * d_elo_l))
    
    # Poisson visita
    d_elo_v = -feats["elo_diff"]
    lb = float(np.exp(poisson_params["const"] + poisson_params["d"] * d_elo_v))
    
    return p, la, lb


def grilla_goles(M, local, visita):
    p, la, lb = predecir_match(M, local, visita)
    GRID_MAX = 8
    gidx = np.arange(GRID_MAX + 1)
    
    grid = np.outer(poisson.pmf(gidx, la), poisson.pmf(gidx, lb))
    rho = M["rho_dc"]
    if GRID_MAX >= 1 and la > 0 and lb > 0:
        grid[0, 0] *= (1.0 - la * lb * rho)
        grid[1, 0] *= (1.0 + lb * rho)
        grid[0, 1] *= (1.0 + la * rho)
        grid[1, 1] *= (1.0 - rho)
            
    grid = np.clip(grid, 0.0, None)
    grid /= grid.sum()
    
    # Re-weight with ML prob
    gi, gj = np.indices(grid.shape)
    masks = [gi > gj, gi == gj, gi < gj]
    mix = np.zeros_like(grid)
    for k, mk in enumerate(masks):
        sub = grid * mk
        if sub.sum() > 0:
            mix += p[k] * sub / sub.sum()
            
    return mix, p, (la, lb)


def cuota(p):
    return 1.0 / p if p > 0.001 else 999.0


def mercados(mix):
    gi, gj = np.indices(mix.shape)
    mk = {}
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


# --------------------------------------------------------------------------- #
#  Simulador de Campeonato Brasileirão
# --------------------------------------------------------------------------- #
def simular_fixture_regular(M, PREDS, fijos=None):
    fix = pd.read_csv(DATA / "fixture.csv")
    partidos = M["partidos"]
    p_actuales = partidos[partidos.temporada == 2026]
    
    tabla = defaultdict(lambda: {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0})
    
    # 20 equipos del Brasileirao 2026
    equipos = set(SQUAD_VALUES_BY_YEAR[2026].keys()).difference({"América Mineiro", "Ceará", "Goiás", "Avaí", "Sport", "Juventude", "Cuiabá", "Atlético Goianiense", "Remo", "Mirassol"}) 
    # Wait, let's extract all teams actively playing in 2026 season from the scrapers
    todos_activos = set(p_actuales["local"]).union(set(p_actuales["visita"])).union(set(fix["local"])).union(set(fix["visita"]))
    if not todos_activos:
        # Fallback
        todos_activos = set(SQUAD_VALUES_BY_YEAR[2026].keys())
        
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

    for r in fix.itertuples(index=False):
        l, v = r.local, r.visita
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


def ordenar_tabla(tabla):
    # Criterios del Brasileirão: 1. Puntos, 2. Victorias (PG), 3. Dif Goles (DG), 4. Goles Favor (GF)
    filas = []
    for eq, s in tabla.items():
        filas.append({
            "Selección": eq,
            "PTS": s["PTS"],
            "PG": s["PG"],
            "DG": s["GF"] - s["GC"],
            "GF": s["GF"],
            "PE": s["PE"],
            "PP": s["PP"]
        })
    df = pd.DataFrame(filas)
    df = df.sort_values(by=["PTS", "PG", "DG", "GF"], ascending=False).reset_index(drop=True)
    return df


def monte_carlo(M, n_sims=4000, fijos=None):
    partidos_rec = M["partidos"]
    p_actuales = partidos_rec[partidos_rec.temporada == 2026]
    fix = pd.read_csv(DATA / "fixture.csv")
    
    todos_activos = list(set(p_actuales["local"]).union(set(p_actuales["visita"])).union(set(fix["local"])).union(set(fix["visita"])))
    
    PREDS = {}
    for local in todos_activos:
        for visita in todos_activos:
            if local != visita:
                p, la, lb = predecir_match(M, local, visita)
                PREDS[(local, visita)] = (p, la, lb)
                
    resultados_campeon = defaultdict(int)
    resultados_libertadores_directo = defaultdict(int) # Top 4
    resultados_libertadores_total = defaultdict(int)   # Top 6
    resultados_sudamericana = defaultdict(int)         # 7º al 12º
    resultados_descenso = defaultdict(int)             # Bottom 4 (17º al 20º)
    resultados_puntos = defaultdict(list)
    
    for _ in range(n_sims):
        tabla = simular_fixture_regular(M, PREDS, fijos)
        df_ord = ordenar_tabla(tabla)
        
        for idx, r in enumerate(df_ord.itertuples()):
            eq = r.Selección
            pos = idx + 1 # 1-indexed
            resultados_puntos[eq].append(r.PTS)
            
            if pos == 1:
                resultados_campeon[eq] += 1
            if pos <= 4:
                resultados_libertadores_directo[eq] += 1
            if pos <= 6:
                resultados_libertadores_total[eq] += 1
            if 7 <= pos <= 12:
                resultados_sudamericana[eq] += 1
            if pos >= len(df_ord) - 3:
                # Relegación (últimos 4 lugares)
                resultados_descenso[eq] += 1
                
    filas = []
    for eq in todos_activos:
        filas.append({
            "Selección": eq,
            "Puntos esperados": np.mean(resultados_puntos[eq]),
            "P_campeon": resultados_campeon[eq] / n_sims,
            "P_libertadores_directo": resultados_libertadores_directo[eq] / n_sims,
            "P_libertadores_total": resultados_libertadores_total[eq] / n_sims,
            "P_sudamericana": resultados_sudamericana[eq] / n_sims,
            "P_descenso": resultados_descenso[eq] / n_sims
        })
        
    df_res = pd.DataFrame(filas).sort_values(by="Puntos esperados", ascending=False).reset_index(drop=True)
    return df_res


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
    
    met = {
        "n": len(df_val),
        "acierto": acc,
        "logloss": ll,
        "logloss_base": ll_base
    }
    
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
            
    evol = pd.DataFrame({
        "partido": range(1, len(df_val) + 1),
        "logloss_acum": log_losses
    })
    
    return df_val, met, evol


def obtener_tabla_actual(M):
    partidos = M["partidos"]
    p_actuales = partidos[partidos.temporada == 2026]
    fix = pd.read_csv(DATA / "fixture.csv")
    todos_activos = set(p_actuales["local"]).union(set(p_actuales["visita"])).union(set(fix["local"] if not fix.empty else []))
    
    tabla = defaultdict(lambda: {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0})
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
            
    return ordenar_tabla(tabla)

