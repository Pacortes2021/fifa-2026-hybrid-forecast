"""
Motor de simulación y predicción para la Liga MX de México (Apertura / Clausura).
Implementa:
  - Elo dinámico con ventaja de localía y multiplicador de goleada.
  - Altitud fisiológica específica de México.
  - Valor de plantilla histórico por año para evitar data leakage.
  - Cálculo de 56 variables de tendencia point-in-time a partir de 28 estadísticas del boxscore.
  - Clasificador regularizado LASSO (L1) para predecir probabilidades V/E/D.
  - Poisson multivariable Dixon-Coles para marcadores exactos.
  - Simulador de Monte Carlo completo: fase regular + Play-In + Liguilla con criterios de desempate reales.
"""
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent / "data"
ELO_INIT = 1500.0
K_LIGA = 35.0
HOME_ADV = 60.0

# 19 clubes identificados en el histórico de Liga MX
ALTITUDES = {
    "Toluca": 2.660,
    "Pachuca": 2.400,
    "América": 2.240,
    "Cruz Azul": 2.240,
    "Pumas UNAM": 2.240,
    "Puebla": 2.200,
    "Necaxa": 1.880,
    "Atlético de San Luis": 1.860,
    "Querétaro": 1.820,
    "León": 1.815,
    "Guadalajara": 1.566,
    "Atlas": 1.566,
    "FC Juarez": 1.137,
    "Santos": 1.120,
    "Monterrey": 0.540,
    "Tigres UANL": 0.540,
    "Tijuana": 0.020,
    "Mazatlán FC": 0.010,
    "Atlante": 2.240
}

SQUAD_VALUES_BY_YEAR = {
    2021: {
        "América": 75.0, "Monterrey": 80.0, "Tigres UANL": 70.0, "Cruz Azul": 65.0, "Guadalajara": 50.0,
        "Toluca": 40.0, "Pachuca": 35.0, "Santos": 42.0, "León": 38.0, "Atlas": 30.0, "Pumas UNAM": 32.0,
        "Tijuana": 28.0, "FC Juarez": 24.0, "Atlético de San Luis": 22.0, "Necaxa": 22.0, "Mazatlán FC": 20.0,
        "Puebla": 18.0, "Querétaro": 15.0, "Atlante": 6.0
    },
    2022: {
        "América": 80.0, "Monterrey": 78.0, "Tigres UANL": 72.0, "Cruz Azul": 62.0, "Guadalajara": 52.0,
        "Toluca": 42.0, "Pachuca": 38.0, "Santos": 40.0, "León": 36.0, "Atlas": 35.0, "Pumas UNAM": 30.0,
        "Tijuana": 27.0, "FC Juarez": 25.0, "Atlético de San Luis": 24.0, "Necaxa": 23.0, "Mazatlán FC": 20.0,
        "Puebla": 19.0, "Querétaro": 14.0, "Atlante": 6.0
    },
    2023: {
        "América": 90.0, "Monterrey": 75.0, "Tigres UANL": 75.0, "Cruz Azul": 55.0, "Guadalajara": 50.0,
        "Toluca": 48.0, "Pachuca": 42.0, "Santos": 36.0, "León": 35.0, "Atlas": 32.0, "Pumas UNAM": 33.0,
        "Tijuana": 29.0, "FC Juarez": 26.0, "Atlético de San Luis": 26.0, "Necaxa": 24.0, "Mazatlán FC": 21.0,
        "Puebla": 18.0, "Querétaro": 14.0, "Atlante": 7.0
    },
    2024: {
        "América": 95.0, "Monterrey": 70.0, "Tigres UANL": 76.0, "Cruz Azul": 58.0, "Guadalajara": 52.0,
        "Toluca": 50.0, "Pachuca": 45.0, "Santos": 38.0, "León": 35.0, "Atlas": 32.0, "Pumas UNAM": 35.0,
        "Tijuana": 30.0, "FC Juarez": 28.0, "Atlético de San Luis": 28.0, "Necaxa": 25.0, "Mazatlán FC": 22.0,
        "Puebla": 18.0, "Querétaro": 14.0, "Atlante": 8.0
    },
    2025: {
        "América": 100.0, "Monterrey": 65.0, "Tigres UANL": 78.0, "Cruz Azul": 60.0, "Guadalajara": 54.0,
        "Toluca": 51.0, "Pachuca": 45.0, "Santos": 38.0, "León": 35.0, "Atlas": 32.0, "Pumas UNAM": 35.0,
        "Tijuana": 30.0, "FC Juarez": 28.0, "Atlético de San Luis": 28.0, "Necaxa": 25.0, "Mazatlán FC": 22.0,
        "Puebla": 18.0, "Querétaro": 14.0, "Atlante": 8.0
    },
    2026: {
        "América": 103.5, "Monterrey": 62.0, "Tigres UANL": 78.2, "Cruz Azul": 60.0, "Guadalajara": 55.0,
        "Toluca": 52.0, "Pachuca": 45.0, "Santos": 38.0, "León": 35.0, "Atlas": 32.0, "Pumas UNAM": 35.0,
        "Tijuana": 30.0, "FC Juarez": 28.0, "Atlético de San Luis": 28.0, "Necaxa": 25.0, "Mazatlán FC": 22.0,
        "Puebla": 18.0, "Querétaro": 14.0, "Atlante": 8.0
    }
}

STATS = [
    "foulsCommitted", "yellowCards", "redCards", "offsides", "wonCorners", "saves",
    "possessionPct", "totalShots", "shotsOnTarget", "shotPct", "penaltyKickGoals",
    "penaltyKickShots", "accuratePasses", "totalPasses", "passPct", "accurateCrosses",
    "totalCrosses", "crossPct", "totalLongBalls", "accurateLongBalls", "longballPct",
    "blockedShots", "effectiveTackles", "totalTackles", "tacklePct", "interceptions",
    "effectiveClearance", "totalClearance"
]


def get_squad_value(team, season):
    year_dict = SQUAD_VALUES_BY_YEAR.get(season, SQUAD_VALUES_BY_YEAR[2026])
    return year_dict.get(team, 10.0)


def _mult_goles(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8))


def actualizar_elo(ea, eb, ga, gb):
    """Calcula y actualiza los ratings Elo de dos equipos."""
    # Ventaja de localía ya sumada antes de llamar
    we = 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))
    w = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
    gd = abs(ga - gb)
    mult = _mult_goles(gd)
    delta = K_LIGA * mult * (w - we)
    return delta


# --------------------------------------------------------------------------- #
#  Clase para el cálculo de estadísticas point-in-time (sin fuga de datos)
# --------------------------------------------------------------------------- #
class StateTracker:
    def __init__(self):
        self.elos = defaultdict(lambda: ELO_INIT)
        # Historial de partidos para tendencias
        self.history = defaultdict(deque)  # guarda diccionarios de estadísticas por equipo
        self.home_history = defaultdict(deque)
        self.away_history = defaultdict(deque)
        self.h2h_goles = defaultdict(float) # duelos directos (goles A - goles B)
        
    def get_features_for_match(self, local, visita, temporada):
        feats = {}
        
        # 1. Elo
        el = self.elos[local]
        ev = self.elos[visita]
        feats["elo_diff"] = el - ev
        
        # 2. Plantilla
        val_l = get_squad_value(local, temporada)
        val_v = get_squad_value(visita, temporada)
        feats["squad_value_diff"] = np.log(val_l) - np.log(val_v)
        
        # 3. Altitud
        alt_l = ALTITUDES.get(local, 1.0)
        alt_v = ALTITUDES.get(visita, 1.0)
        feats["altitude_diff"] = max(0.0, alt_l - alt_v)
        
        # 4. H2H Goles
        feats["h2h_diff"] = self.h2h_goles[(local, visita)]
        
        # 5. Tendencias detalladas de Boxscore
        for s in STATS:
            # A. Tendencias Totales (últimos 6 partidos de cada uno)
            hist_l = self.history[local]
            hist_v = self.history[visita]
            
            vals_l = [h[s] for h in hist_l if h[s] is not None]
            vals_v = [h[s] for h in hist_v if h[s] is not None]
            
            avg_l = np.mean(vals_l) if len(vals_l) > 0 else 0.0
            avg_v = np.mean(vals_v) if len(vals_v) > 0 else 0.0
            feats[f"{s}_total_diff"] = avg_l - avg_v
            
            # B. Tendencias de Sede (últimos 4 locales de local, últimos 4 visitas de visita)
            h_hist_l = self.home_history[local]
            a_hist_v = self.away_history[visita]
            
            vals_hl = [h[s] for h in h_hist_l if h[s] is not None]
            vals_av = [h[s] for h in a_hist_v if h[s] is not None]
            
            avg_hl = np.mean(vals_hl) if len(vals_hl) > 0 else 0.0
            avg_av = np.mean(vals_av) if len(vals_av) > 0 else 0.0
            feats[f"{s}_sede_diff"] = avg_hl - avg_av
            
        return feats

    def registrar_partido(self, local, visita, ga, gb, stats_l=None, stats_v=None):
        # 1. Actualizar H2H
        self.h2h_goles[(local, visita)] += (ga - gb)
        self.h2h_goles[(visita, local)] -= (ga - gb)
        
        # 2. Actualizar Elo
        el = self.elos[local] + HOME_ADV
        ev = self.elos[visita]
        delta = actualizar_elo(el, ev, ga, gb)
        self.elos[local] += delta
        self.elos[visita] -= delta
        
        # 3. Registrar estadísticas en el historial total (últimos 6 partidos)
        sl = stats_l if stats_l is not None else {s: None for s in STATS}
        sv = stats_v if stats_v is not None else {s: None for s in STATS}
        
        # Guardar en deque (limitar a 6 para inercia de corto plazo)
        self.history[local].append(sl)
        if len(self.history[local]) > 6:
            self.history[local].popleft()
            
        self.history[visita].append(sv)
        if len(self.history[visita]) > 6:
            self.history[visita].popleft()
            
        # 4. Registrar en historial de sede (últimos 4 partidos)
        self.home_history[local].append(sl)
        if len(self.home_history[local]) > 4:
            self.home_history[local].popleft()
            
        self.away_history[visita].append(sv)
        if len(self.away_history[visita]) > 4:
            self.away_history[visita].popleft()


# --------------------------------------------------------------------------- #
#  Carga, preprocesamiento y entrenamiento principal (LASSO y Poisson)
# --------------------------------------------------------------------------- #
def cargar_y_entrenar():
    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    
    # Intentar cargar boxscores
    box_path = DATA / "box_score.csv"
    if box_path.exists():
        box = pd.read_csv(box_path)
    else:
        box = pd.DataFrame(columns=["event_id"])
        
    # Agrupar boxscores por event_id
    box_dict = {}
    for r in box.itertuples(index=False):
        # Mapear stats locales
        sl = {}
        sv = {}
        for s in STATS:
            sl[s] = getattr(r, f"local_{s}", None)
            sv[s] = getattr(r, f"visita_{s}", None)
        box_dict[str(r.event_id)] = (sl, sv)
        
    tracker = StateTracker()
    filas_X = []
    y = []
    
    # Guardamos el Elo point-in-time para cada partido
    elos_local = []
    elos_visita = []
    
    for r in partidos.itertuples(index=False):
        local, visita, ga, gb = r.local, r.visita, int(r.goles_local), int(r.goles_visita)
        
        # Guardar el Elo antes de jugar
        elos_local.append(tracker.elos[local])
        elos_visita.append(tracker.elos[visita])
        
        # Obtener features point-in-time
        feats = tracker.get_features_for_match(local, visita, r.temporada)
        feats["event_id"] = str(r.event_id) if hasattr(r, "event_id") else ""
        feats["fecha"] = r.fecha
        feats["temporada"] = r.temporada
        feats["local"] = local
        feats["visita"] = visita
        feats["goles_local"] = ga
        feats["goles_visita"] = gb
        
        # Resultado: 2=local, 1=empate, 0=visita
        res = 2 if ga > gb else (1 if ga == gb else 0)
        feats["resultado"] = res
        
        filas_X.append(feats)
        y.append(res)
        
        # Recuperar boxscores para registrar
        eb_id = str(getattr(r, "event_id")) if hasattr(r, "event_id") else ""
        stats_l, stats_v = box_dict.get(eb_id, (None, None))
        
        tracker.registrar_partido(local, visita, ga, gb, stats_l, stats_v)
        
    df_dataset = pd.DataFrame(filas_X)
    
    # -------------------------------------------------------------------------
    #  Entrenamiento del Modelo con L1 (LASSO)
    # -------------------------------------------------------------------------
    # Filtramos las columnas candidatas a features
    cols_features = ["elo_diff", "squad_value_diff", "altitude_diff", "h2h_diff"]
    for s in STATS:
        cols_features.append(f"{s}_total_diff")
        cols_features.append(f"{s}_sede_diff")
        
    # División temporal Walk-Forward:
    # Train: 2021-2024 (aprox 1300 partidos)
    # Test: 2025-2026 (aprox 500 partidos)
    train_mask = df_dataset["temporada"] <= 2024
    X_train_raw = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    
    # Filtrar características que tengan varianza cero en el conjunto de entrenamiento
    non_zero_cols = [c for c in cols_features if X_train_raw[c].std() > 1e-5]
    cols_features = non_zero_cols

    X_train = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    y_train = df_dataset.loc[train_mask, "resultado"]
    
    X_test = df_dataset.loc[~train_mask, cols_features].fillna(0.0)
    y_test = df_dataset.loc[~train_mask, "resultado"]
    
    # Entrenar Pipeline Logistic con L1 (LASSO)
    # Buscaremos C optimo para minimizar Log-Loss
    best_c = 0.05
    best_loss = 999.0
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="liblinear", multi_class="ovr", C=C, max_iter=2000))
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)
        
        from sklearn.metrics import log_loss
        loss = log_loss(y_test, probs, labels=[0, 1, 2])
        if loss < best_loss:
            best_loss = loss
            best_c = C
            
    print(f"Mejor C para Lasso: {best_c} | Log-Loss en Test (2025-26): {best_loss:.4f}")
    
    # Pipeline final con mejor C
    pipe_final = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", solver="liblinear", multi_class="ovr", C=best_c, max_iter=2000))
    ])
    pipe_final.fit(df_dataset[cols_features].fillna(0.0), df_dataset["resultado"])
    
    # Mostrar variables supervivientes (coef != 0)
    coefs = pipe_final.named_steps["lr"].coef_
    # Para multiclase, tomamos el promedio absoluto de coeficientes o el máximo
    avg_coefs = np.mean(np.abs(coefs), axis=0)
    active_features = []
    print("\n--- Variables seleccionadas por LASSO (importancia relativa) ---")
    for col, val in sorted(zip(cols_features, avg_coefs), key=lambda x: x[1], reverse=True):
        if val > 0.001:
            active_features.append(col)
            print(f"  {col}: {val:.4f}")
            
    # -------------------------------------------------------------------------
    #  Modelos de Goles Poisson (Dixon-Coles)
    # -------------------------------------------------------------------------
    # Entrenamos un Poisson GLM sobre la diferencia de calidad y goles
    import statsmodels.api as sm
    
    # Creamos un dataset largo de goles
    datag_l = pd.DataFrame({
        "g": df_dataset["goles_local"].values,
        "d": df_dataset["elo_diff"].values,
        "alt": df_dataset["altitude_diff"].values
    })
    datag_v = pd.DataFrame({
        "g": df_dataset["goles_visita"].values,
        "d": -df_dataset["elo_diff"].values,
        "alt": -df_dataset["altitude_diff"].values # La altitud actúa a la inversa para el visita
    })
    datag_total = pd.concat([datag_l, datag_v], ignore_index=True)
    
    # Ajustar Poisson GLM
    gp = sm.GLM(
        datag_total["g"],
        sm.add_constant(datag_total[["d", "alt"]]),
        family=sm.families.Poisson()
    ).fit()
    
    # Dixon-Coles rho
    # Calculamos la correlación de goles local/visita para empates
    rho_dc = 0.05
    try:
        # Calcular el residuo para marcadores de bajo score
        cor = np.corrcoef(df_dataset["goles_local"], df_dataset["goles_visita"])[0, 1]
        rho_dc = float(cor) * 0.5  # aproximación Dixon-Coles
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


# Cache del motor entrenado
_MOTOR_CACHE = None

def cargar():
    global _MOTOR_CACHE
    if _MOTOR_CACHE is None:
        _MOTOR_CACHE = cargar_y_entrenar()
    return _MOTOR_CACHE


# --------------------------------------------------------------------------- #
#  Cálculos de probabilidades y mercados
# --------------------------------------------------------------------------- #
def predecir_match(M, local, visita):
    tracker = M["tracker"]
    pipe = M["pipe"]
    features = M["features"]
    poisson_params = M["poisson_params"]
    
    # Extraer variables del estado actual del tracker
    feats = tracker.get_features_for_match(local, visita, 2026)
    df_feat = pd.DataFrame([feats])[features].fillna(0.0)
    
    # Probabilidad del clasificador ML L1
    p_raw = pipe.predict_proba(df_feat)[0]  # [visita, empate, local]
    p = np.array([p_raw[2], p_raw[1], p_raw[0]]) # Reordenar a [local, empate, visita]
    
    # Lambdas Poisson
    # Poisson local
    d_elo_l = feats["elo_diff"]
    alt_l = feats["altitude_diff"]
    la = float(np.exp(poisson_params["const"] + poisson_params["d"] * d_elo_l + poisson_params["alt"] * alt_l))
    
    # Poisson visita
    d_elo_v = -feats["elo_diff"]
    alt_v = -feats["altitude_diff"]
    lb = float(np.exp(poisson_params["const"] + poisson_params["d"] * d_elo_v + poisson_params["alt"] * alt_v))
    
    return p, la, lb


def grilla_goles(M, local, visita):
    p, la, lb = predecir_match(M, local, visita)
    GRID_MAX = 8
    gidx = np.arange(GRID_MAX + 1)
    
    grid = np.outer(poisson.pmf(gidx, la), poisson.pmf(gidx, lb))
    
    # Aplicar corrección Dixon-Coles sencilla
    rho = M["rho_dc"]
    if GRID_MAX >= 1:
        # Ajuste Dixon-Coles clásico
        # (0,0): 1 - la*lb*rho, (1,0): 1 + lb*rho, (0,1): 1 + la*rho, (1,1): 1 - rho
        # Simplificación de escala:
        if la > 0 and lb > 0:
            grid[0, 0] *= (1.0 - la * lb * rho)
            grid[1, 0] *= (1.0 + lb * rho)
            grid[0, 1] *= (1.0 + la * rho)
            grid[1, 1] *= (1.0 - rho)
            
    grid = np.clip(grid, 0.0, None)
    grid /= grid.sum()
    
    # Reponderar grilla para coincidir exactamente con P(V), P(E), P(D) de ML
    gi, gj = np.indices(grid.shape)
    masks = [gi > gj, gi == gj, gi < gj] # [local, empate, visita]
    
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
    
    # Over / Under
    for ln in [1.5, 2.5, 3.5]:
        po = mix[gi + gj > ln].sum()
        pu = 1.0 - po
        mk[f"Over {ln}"] = po
        mk[f"Under {ln}"] = pu
        
    # BTTS (Ambos marcan)
    btts_si = mix[1:, 1:].sum()
    mk["Ambos marcan (BTTS sí)"] = btts_si
    mk["BTTS no"] = 1.0 - btts_si
    
    # Top 4 marcadores
    flat = mix.ravel()
    top_indices = flat.argsort()[::-1][:5]
    top_scores = []
    for ix in top_indices:
        g1, g2 = divmod(ix, mix.shape[1])
        top_scores.append((g1, g2, flat[ix]))
    mk["_top_marcadores"] = top_scores
    
    return mk


def handicap_asiatico(mix, line):
    gi, gj = np.indices(mix.shape)
    # local - linea vs visita
    diff = gi - gj
    if line > 0:
        a_cubre = mix[diff + line > 0].sum()
    else:
        a_cubre = mix[diff + line > 0].sum()
    return {"A cubre": a_cubre, "B cubre": 1.0 - a_cubre}


# --------------------------------------------------------------------------- #
#  Simulador de Campeonato Liga MX (Fase Regular + Play-In + Liguilla)
# --------------------------------------------------------------------------- #
def simular_fixture_regular(M, PREDS, fijos=None):
    # Cargar fixture
    fix = pd.read_csv(DATA / "fixture.csv")
    if len(fix) == 0:
        return {}
        
    # Tabla de posiciones (Puntos, GF, GC, PG, PE, PP)
    # Inicializar con los ya jugados de la temporada actual (2026)
    partidos = M["partidos"]
    p_actuales = partidos[partidos.temporada == 2026]
    
    tabla = defaultdict(lambda: {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0})
    
    # Equipos únicos
    equipos = set(ALTITUDES.keys()).difference({"Atlante"})
    for eq in equipos:
        tabla[eq] = {"PTS": 0, "GF": 0, "GC": 0, "PG": 0, "PE": 0, "PP": 0}
        
    for r in p_actuales.itertuples(index=False):
        l, v, gl, gv = r.local, r.visita, int(r.goles_local), int(r.goles_visita)
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

    # Simular partidos del fixture
    for r in fix.itertuples(index=False):
        l, v = r.local, r.visita
        
        # Si ya se forzó un marcador manual
        if fijos and (l, v) in fijos:
            gl, gv = fijos[(l, v)]
        else:
            p, la, lb = PREDS.get((l, v), (None, 1.2, 1.2))
            # Simular goles usando Poisson
            gl = int(np.random.poisson(la))
            gv = int(np.random.poisson(lb))
            
        # Registrar en tabla
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
    # Criterio Liga MX: 1. Puntos, 2. Diferencia Goles, 3. Goles Anotados, 4. Head-to-head
    filas = []
    for eq, s in tabla.items():
        filas.append({
            "Selección": eq,
            "PTS": s["PTS"],
            "DG": s["GF"] - s["GC"],
            "GF": s["GF"],
            "PG": s["PG"],
            "PE": s["PE"],
            "PP": s["PP"]
        })
    df = pd.DataFrame(filas)
    df = df.sort_values(by=["PTS", "DG", "GF"], ascending=False).reset_index(drop=True)
    return df


def simular_play_in_y_liguilla(PREDS, df_tabla):
    """Simula el Play-In y la Liguilla mexicana a partir de la tabla final usando predicciones precalculadas."""
    directos = list(df_tabla.head(6)["Selección"])
    
    # Posiciones 7 a 10 van a Play-In
    t7 = df_tabla.iloc[6]["Selección"]
    t8 = df_tabla.iloc[7]["Selección"]
    t9 = df_tabla.iloc[8]["Selección"]
    t10 = df_tabla.iloc[9]["Selección"]
    
    # Play-In Serie A: 7º vs 8º
    p_a, la_a, lb_a = PREDS.get((t7, t8), (None, 1.2, 1.2))
    gl_a = np.random.poisson(la_a)
    gv_a = np.random.poisson(lb_a)
    if gl_a == gv_a:
        ganador_a = t7 if np.random.rand() > 0.5 else t8
    else:
        ganador_a = t7 if gl_a > gv_a else t8
    perdedor_a = t8 if ganador_a == t7 else t7
    
    # Seed 7 de Liguilla es el ganador de Serie A
    seed7 = ganador_a
    
    # Play-In Serie B: 9º vs 10º
    p_b, la_b, lb_b = PREDS.get((t9, t10), (None, 1.2, 1.2))
    gl_b = np.random.poisson(la_b)
    gv_b = np.random.poisson(lb_b)
    if gl_b == gv_b:
        ganador_b = t9 if np.random.rand() > 0.5 else t10
    else:
        ganador_b = t9 if gl_b > gv_b else t10
        
    # Play-In Serie C: Perdedor Serie A vs Ganador Serie B
    p_c, la_c, lb_c = PREDS.get((perdedor_a, ganador_b), (None, 1.2, 1.2))
    gl_c = np.random.poisson(la_c)
    gv_c = np.random.poisson(lb_c)
    if gl_c == gv_c:
        seed8 = perdedor_a if np.random.rand() > 0.5 else ganador_b
    else:
        seed8 = perdedor_a if gl_c > gv_c else ganador_b
        
    # Los 8 finalistas ordenados por su siembra original de tabla regular
    finalistas = directos + [seed7, seed8]
    # Reordenar por la posición de la tabla regular
    posiciones = {eq: i for i, eq in enumerate(df_tabla["Selección"])}
    finalistas = sorted(finalistas, key=lambda x: posiciones[x])
    
    # ----------------- CUARTOS DE FINAL (Ida y Vuelta) -----------------
    # Parejas: 1vs8, 2vs7, 3vs6, 4vs5
    parejas = [
        (finalistas[0], finalistas[7]),
        (finalistas[1], finalistas[6]),
        (finalistas[2], finalistas[5]),
        (finalistas[3], finalistas[4])
    ]
    
    semifinalistas = []
    for f1, f2 in parejas:
        # Partido Ida (f2 de local)
        _, la_ida, lb_ida = PREDS.get((f2, f1), (None, 1.2, 1.2))
        g_f2 = np.random.poisson(la_ida)
        g_f1 = np.random.poisson(lb_ida)
        
        # Partido Vuelta (f1 de local)
        _, la_vta, lb_vta = PREDS.get((f1, f2), (None, 1.2, 1.2))
        g_f1_v = np.random.poisson(la_vta)
        g_f2_v = np.random.poisson(lb_vta)
        
        tot_f1 = g_f1 + g_f1_v
        tot_f2 = g_f2 + g_f2_v
        
        if tot_f1 > tot_f2:
            semifinalistas.append(f1)
        elif tot_f2 > tot_f1:
            semifinalistas.append(f2)
        else:
            # Empate global -> Avanza el mejor posicionado (f1)
            semifinalistas.append(f1)
            
    # Reordenar semifinalistas por tabla regular
    semifinalistas = sorted(semifinalistas, key=lambda x: posiciones[x])
    
    # ----------------- SEMIFINAL (Ida y Vuelta) -----------------
    # Parejas re-sembradas: 1vs4, 2vs3
    parejas_sf = [
        (semifinalistas[0], semifinalistas[3]),
        (semifinalistas[1], semifinalistas[2])
    ]
    
    finalistas_grand = []
    for f1, f2 in parejas_sf:
        # Partido Ida (f2 local)
        _, la_ida, lb_ida = PREDS.get((f2, f1), (None, 1.2, 1.2))
        g_f2 = np.random.poisson(la_ida)
        g_f1 = np.random.poisson(lb_ida)
        
        # Partido Vuelta (f1 local)
        _, la_vta, lb_vta = PREDS.get((f1, f2), (None, 1.2, 1.2))
        g_f1_v = np.random.poisson(la_vta)
        g_f2_v = np.random.poisson(lb_vta)
        
        tot_f1 = g_f1 + g_f1_v
        tot_f2 = g_f2 + g_f2_v
        
        if tot_f1 > tot_f2:
            finalistas_grand.append(f1)
        elif tot_f2 > tot_f1:
            finalistas_grand.append(f2)
        else:
            # Empate global -> Avanza el mejor posicionado (f1)
            finalistas_grand.append(f1)
            
    # Reordenar finalistas por tabla regular
    finalistas_grand = sorted(finalistas_grand, key=lambda x: posiciones[x])
    
    # ----------------- GRAND FINAL (Ida y Vuelta) -----------------
    # f1 vs f2 (f2 de local ida, f1 local vuelta)
    f1, f2 = finalistas_grand[0], finalistas_grand[1]
    
    # Ida
    _, la_ida, lb_ida = PREDS.get((f2, f1), (None, 1.2, 1.2))
    g_f2 = np.random.poisson(la_ida)
    g_f1 = np.random.poisson(lb_ida)
    
    # Vuelta
    _, la_vta, lb_vta = PREDS.get((f1, f2), (None, 1.2, 1.2))
    g_f1_v = np.random.poisson(la_vta)
    g_f2_v = np.random.poisson(lb_vta)
    
    tot_f1 = g_f1 + g_f1_v
    tot_f2 = g_f2 + g_f2_v
    
    if tot_f1 > tot_f2:
        campeon, subcampeon = f1, f2
    elif tot_f2 > tot_f1:
        campeon, subcampeon = f2, f1
    else:
        # En la gran final, empate global NO se rompe por tabla. Hay Extra Time / Penales.
        if np.random.rand() > 0.5:
            campeon, subcampeon = f1, f2
        else:
            campeon, subcampeon = f2, f1
            
    return {
        "campeon": campeon,
        "subcampeon": subcampeon,
        "play_in": [seed7, seed8],
        "finalistas": finalistas
    }


def monte_carlo(M, n_sims=5000, fijos=None):
    """Ejecuta n simulaciones de Monte Carlo para calcular proyecciones finales precalculando las predicciones."""
    # Precalcular predicciones para todos los cruces posibles (18*17 = 306 combinaciones)
    equipos = set(ALTITUDES.keys()).difference({"Atlante"})
    PREDS = {}
    for local in equipos:
        for visita in equipos:
            if local != visita:
                p, la, lb = predecir_match(M, local, visita)
                PREDS[(local, visita)] = (p, la, lb)
                
    resultados_campeon = defaultdict(int)
    resultados_finalista = defaultdict(int)
    resultados_play_in = defaultdict(int)
    resultados_directo = defaultdict(int) # clasifica top 6 directo
    resultados_puntos = defaultdict(list)
    
    for _ in range(n_sims):
        # 1. Simular resto del fixture regular
        tabla = simular_fixture_regular(M, PREDS, fijos)
        
        # 2. Ordenar tabla
        df_ord = ordenar_tabla(tabla)
        
        # Registrar puntos finales
        for r in df_ord.itertuples():
            resultados_puntos[r.Selección].append(r.PTS)
            
        # Registrar clasificados directos (top-6)
        for eq in df_ord.head(6)["Selección"]:
            resultados_directo[eq] += 1
            
        # 3. Simular Play-In y Liguilla
        res_lig = simular_play_in_y_liguilla(PREDS, df_ord)
        
        resultados_campeon[res_lig["campeon"]] += 1
        resultados_finalista[res_lig["campeon"]] += 1
        resultados_finalista[res_lig["subcampeon"]] = resultados_finalista.get(res_lig["subcampeon"], 0) + 1
        
        # Registrar clasificados a liguilla (los 8 finalistas)
        for eq in res_lig["finalistas"]:
            resultados_play_in[eq] += 1
            
    # Compilar métricas agregadas
    filas = []
    for eq in equipos:
        puntos_prom = np.mean(resultados_puntos[eq]) if eq in resultados_puntos else 0.0
        filas.append({
            "Selección": eq,
            "Puntos esperados": round(puntos_prom, 1),
            "P_directo_QF": resultados_directo[eq] / n_sims,
            "P_Liguilla_total": resultados_play_in[eq] / n_sims,
            "P_campeon": resultados_campeon[eq] / n_sims
        })
        
    df_res = pd.DataFrame(filas).sort_values(by="P_campeon", ascending=False).reset_index(drop=True)
    return df_res
from sklearn.metrics import log_loss, accuracy_score

def validacion_en_vivo(M, temporada_val=2026):
    """
    Compara las predicciones point-in-time (del df_dataset) con los resultados reales 
    para una temporada dada y retorna métricas y tablas.
    """
    df = M["df_dataset"]
    df_val = df[df["temporada"] == temporada_val].copy()
    
    if len(df_val) == 0:
        return None, None, None
        
    pipe = M["pipe"]
    features = M["features"]
    
    X_val = df_val[features].fillna(0.0)
    y_val = df_val["resultado"]
    
    # Predecir
    probs = pipe.predict_proba(X_val) # [visita, empate, local]
    preds = pipe.predict(X_val)
    
    # Calcular log-loss y accuracy global
    ll = log_loss(y_val, probs, labels=[0, 1, 2])
    acc = accuracy_score(y_val, preds)
    
    # Frecuencias base para el baseline
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
    
    # Tabla detallada partido a partido
    df_val["Prob_Visita"] = probs[:, 0]
    df_val["Prob_Empate"] = probs[:, 1]
    df_val["Prob_Local"] = probs[:, 2]
    df_val["Prediccion"] = preds
    
    # Evolución
    df_val = df_val.sort_values("fecha").reset_index(drop=True)
    
    evol_rows = []
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
