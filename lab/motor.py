"""
Motor compartido del laboratorio (app_lab.py). Reutiliza la metodología del proyecto
principal (modelo base elo+h2h+valor con ponderación K-factor, Poisson de goles, simulación
Monte Carlo del bracket oficial) y añade lo necesario para los 4 frentes nuevos:

  1. Mercados de apuestas por partido (Over/Under, BTTS, marcador exacto, hándicap).
  2. Modelo vivo: actualizar Elo + forma con resultados reales y re-simular el torneo restante.
  3. Validación: calibración y backtesting económico (value betting).
  4. Robustez: sensibilidad de partido e intervalos de confianza del Monte Carlo.

No toca ni app.py ni los datos: solo LEE de ../data/.
"""
from pathlib import Path
from itertools import permutations
from collections import Counter
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

DATA = Path(__file__).resolve().parent.parent / "data"

BASE_VARS = ["elo_diff", "h2h_diff", "squad_value_diff"]
HYBRID_VARS = BASE_VARS + ["goles_anotados_diff", "goles_recibidos_diff", "tiros_arco_diff"]
ETIQUETAS = {0: "Derrota", 1: "Empate", 2: "Victoria"}

GRUPOS = {
    "A": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["Australia", "Paraguay", "Turkey", "United States"],
    "E": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "F": ["Japan", "Netherlands", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I": ["France", "Iraq", "Norway", "Senegal"],
    "J": ["Algeria", "Argentina", "Austria", "Jordan"],
    "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L": ["Croatia", "England", "Ghana", "Panama"],
}
MUNDIALISTAS = [t for eqs in GRUPOS.values() for t in eqs]
GRUPO_DE = {t: g for g, eqs in GRUPOS.items() for t in eqs}
ANFITRIONES = {"United States", "Canada", "Mexico"}

R32 = {
    73: ("2A", "2B"), 74: ("1E", ("3", list("ABCDF"))), 75: ("1F", "2C"), 76: ("1C", "2F"),
    77: ("1I", ("3", list("CDFGH"))), 78: ("2E", "2I"), 79: ("1A", ("3", list("CEFHI"))),
    80: ("1L", ("3", list("EHIJK"))), 81: ("1D", ("3", list("BEFIJ"))), 82: ("1G", ("3", list("AEHIJ"))),
    83: ("2K", "2L"), 84: ("1H", "2J"), 85: ("1B", ("3", list("EFGIJ"))), 86: ("1J", "2H"),
    87: ("1K", ("3", list("DEIJL"))), 88: ("2D", "2G"),
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL_M = (101, 102)

K_FACTOR = {"amistoso": 1.0, "clasificatoria": 2.0, "nations_league": 2.0, "continental": 2.5, "mundial": 3.0}
GRID_MAX = 10


def tipo_competicion(c):
    c = str(c).lower()
    if "amistoso" in c: return "amistoso"
    if "clasif" in c: return "clasificatoria"
    if "nations league" in c: return "nations_league"
    if "mundial" in c: return "mundial"
    return "continental"


# --------------------------------------------------------------------------- #
#  Carga y entrenamiento (cacheable desde Streamlit)
# --------------------------------------------------------------------------- #
def cargar():
    df = pd.read_csv(DATA / "modelado_espn.csv", parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)
    states = pd.read_csv(DATA / "team_states.csv").set_index("team")
    hist = pd.read_csv(DATA / "results.csv", parse_dates=["date"]).dropna(subset=["home_score", "away_score"])

    m48 = set(MUNDIALISTAS)
    duelos = hist[hist.home_team.isin(m48) & hist.away_team.isin(m48)]
    H2H = {}
    for r in duelos.itertuples(index=False):
        d = r.home_score - r.away_score
        for a, b, s in ((r.home_team, r.away_team, d), (r.away_team, r.home_team, -d)):
            H2H.setdefault((a, b), []).append(s)
    H2H = {k: float(np.mean(v)) for k, v in H2H.items()}

    peso = df.competicion.map(tipo_competicion).map(K_FACTOR).values
    data_b = df.dropna(subset=BASE_VARS + ["ea_overall_diff"]).reset_index(drop=True)
    data_h = df.dropna(subset=HYBRID_VARS).reset_index(drop=True)
    wb = data_b.competicion.map(tipo_competicion).map(K_FACTOR).values
    wh = data_h.competicion.map(tipo_competicion).map(K_FACTOR).values

    pipe_base = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    pipe_base.fit(data_b[BASE_VARS], data_b["resultado"], m__sample_weight=wb)
    pipe_hyb = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    pipe_hyb.fit(data_h[HYBRID_VARS], data_h["resultado"], m__sample_weight=wh)

    # Poisson Elo -> goles esperados (motor de marcadores)
    espn = pd.read_csv(DATA / "espn_stats.csv", parse_dates=["fecha"])
    espn = espn[(espn.fecha >= "2019-01-01") & espn.goles_local.notna()]
    largo = pd.concat([
        pd.DataFrame({"g": espn.goles_local.values, "d": (espn.elo_local - espn.elo_visita).values}),
        pd.DataFrame({"g": espn.goles_visita.values, "d": (espn.elo_visita - espn.elo_local).values})])
    gp = sm.GLM(largo["g"], sm.add_constant(largo[["d"]]), family=sm.families.Poisson()).fit()
    gb0_, gb1_ = float(gp.params["const"]), float(gp.params["d"])
    # Parámetro rho de Dixon-Coles (corrige la subestimación de empates de bajo score), estimado por
    # máxima verosimilitud del 1X2 sobre los partidos. El Poisson simple subestima los empates; este
    # ajuste sube 0-0 y 1-1 y baja 1-0/0-1. (En ligas con pocos empates, rho≈0 y no cambia nada.)
    rho_dc = _estimar_rho(espn, gb0_, gb1_)

    # Modelo de stats del partido construido RIGUROSAMENTE (selección de variables + validación
    # temporal): dataset point-in-time, medias 'concedidas' por equipo y un modelo por estadística.
    espn_full = pd.read_csv(DATA / "espn_stats.csv", parse_dates=["fecha"]).sort_values("fecha")
    Dstats, conc, glob = _features_stats(espn_full)
    modelos_stats = _entrenar_stats(Dstats)

    return {
        "df": df, "states": states, "hist": hist, "H2H": H2H,
        "pipe_base": pipe_base, "pipe_hyb": pipe_hyb,
        "gb0": gb0_, "gb1": gb1_, "rho_dc": rho_dc,
        "conc": conc, "stat_global": glob, "stats_data": Dstats, "modelos_stats": modelos_stats,
        "CORTE_TEST": "2025-01-01",
    }


def _dc_ajuste(grid, la, lb, rho):
    """Aplica la corrección Dixon-Coles a las 4 celdas de bajo score de una grilla de marcadores."""
    if rho == 0:
        return grid
    g = grid.copy()
    g[0, 0] *= 1 - la * lb * rho; g[0, 1] *= 1 + la * rho
    g[1, 0] *= 1 + lb * rho;       g[1, 1] *= 1 - rho
    return np.clip(g, 1e-12, None)


def _estimar_rho(espn, gb0, gb1):
    """Estima rho (Dixon-Coles) por mínimo log-loss del 1X2 sobre los partidos. Devuelve rho en [-0.25, 0]."""
    from sklearn.metrics import log_loss
    la = np.exp(gb0 + gb1 * (espn.elo_local - espn.elo_visita).values)
    lb = np.exp(gb0 + gb1 * (espn.elo_visita - espn.elo_local).values)
    y = np.where(espn.goles_local.values > espn.goles_visita.values, 2,
                 np.where(espn.goles_local.values == espn.goles_visita.values, 1, 0))
    gmax = 8; g = np.arange(gmax + 1)
    pml = poisson.pmf(g, la[:, None]); pmv = poisson.pmf(g, lb[:, None])   # (n, gmax+1)
    best_rho, best_ll = 0.0, np.inf
    for rho in np.linspace(-0.25, 0.0, 26):
        grids = pml[:, :, None] * pmv[:, None, :]                         # (n, G, G)
        grids[:, 0, 0] *= 1 - la * lb * rho; grids[:, 0, 1] *= 1 + la * rho
        grids[:, 1, 0] *= 1 + lb * rho;       grids[:, 1, 1] *= 1 - rho
        grids = np.clip(grids, 1e-12, None); grids /= grids.sum((1, 2), keepdims=True)
        gi, gj = np.indices((gmax + 1, gmax + 1))
        P = np.stack([(grids * (gi < gj)).sum((1, 2)),          # gana visita
                      np.trace(grids, axis1=1, axis2=2),         # empate (diagonal)
                      (grids * (gi > gj)).sum((1, 2))], axis=1)  # gana local
        ll = log_loss(y, P / P.sum(1, keepdims=True), labels=[0, 1, 2])
        if ll < best_ll:
            best_ll, best_rho = ll, float(rho)
    return best_rho


# --------------------------------------------------------------------------- #
#  Modelo de stats del partido (córners / tiros al arco / faltas / posesión)
#  Construido de cero: features point-in-time + selección de variables + validación temporal.
# --------------------------------------------------------------------------- #
STATS_BASE = ["goles", "tiros", "tiros_arco", "corners", "faltas"]
# Sets de variables elegidos por selección forward con CV temporal (ver análisis):
SETS_STATS = {
    "corners": ["elo_diff", "prop_tiros_f", "riv_tiros_c", "es_local", "prop_corners_f"],
    "tiros_arco": ["elo_diff", "prop_tiros_f", "riv_tiros_c", "es_local"],
    "faltas": ["riv_faltas_c", "prop_faltas_f", "riv_goles_c"],
    "posesion": ["elo_diff", "prop_pos", "riv_tiros_c", "es_local", "prop_tiros_arco_c", "prop_corners_c"],
}


def _features_stats(espn, ventana=8):
    """Construye el dataset point-in-time (2 obs/partido) con promedios móviles walk-forward de las
       stats propias y del rival (favor y concedidas) + Elo. Devuelve (dataset, conc_actual, medias)."""
    from collections import defaultdict, deque
    favor = {s: defaultdict(lambda: deque(maxlen=ventana)) for s in STATS_BASE}
    contra = {s: defaultdict(lambda: deque(maxlen=ventana)) for s in STATS_BASE}
    posq = defaultdict(lambda: deque(maxlen=ventana))
    filas = []

    def perfil(eq):
        d = {f"{s}_f": (np.mean(favor[s][eq]) if favor[s][eq] else np.nan) for s in STATS_BASE}
        d.update({f"{s}_c": (np.mean(contra[s][eq]) if contra[s][eq] else np.nan) for s in STATS_BASE})
        d["pos"] = np.mean(posq[eq]) if posq[eq] else np.nan
        return d

    for r in espn.itertuples(index=False):
        a, b = r.local, r.visita
        vals = {s: (getattr(r, f"{s}_local"), getattr(r, f"{s}_visita")) for s in STATS_BASE}
        pl, pv = r.posesion_local, r.posesion_visita
        if all(len(favor["corners"][x]) >= 4 for x in (a, b)):
            pa, pb = perfil(a), perfil(b)
            for eq, pe, riv, ea, eb, es_loc, idx in (
                    (a, pa, pb, r.elo_local, r.elo_visita, 1, 0),
                    (b, pb, pa, r.elo_visita, r.elo_local, 0, 1)):
                fila = {f"prop_{k}": v for k, v in pe.items()}
                fila.update({f"riv_{k}": v for k, v in riv.items()})
                fila.update({"elo_prop": ea, "elo_riv": eb, "elo_diff": ea - eb, "es_local": es_loc,
                             "fecha": r.fecha, "t_corners": vals["corners"][idx],
                             "t_tiros_arco": vals["tiros_arco"][idx], "t_faltas": vals["faltas"][idx],
                             "t_posesion": (pl if idx == 0 else pv)})
                filas.append(fila)
        for s in STATS_BASE:
            vl, vv = vals[s]
            if not (pd.isna(vl) or pd.isna(vv)):
                favor[s][a].append(vl); contra[s][a].append(vv)
                favor[s][b].append(vv); contra[s][b].append(vl)
        if not (pd.isna(pl) or pd.isna(pv)):
            posq[a].append(pl); posq[b].append(pv)

    D = pd.DataFrame(filas)
    # estado 'concedido' actual de cada equipo (media de lo que concede) + posesión, para predecir
    conc_rows = {}
    for eq in set(list(favor["corners"].keys())):
        row = {f"{s}_c": (np.mean(contra[s][eq]) if contra[s][eq] else np.nan) for s in STATS_BASE}
        row["pos"] = np.mean(posq[eq]) if posq[eq] else np.nan
        conc_rows[eq] = row
    conc = pd.DataFrame(conc_rows).T
    glob = {f"{s}_c": float(np.nanmean(conc[f"{s}_c"])) for s in STATS_BASE}
    conc = conc.fillna(pd.Series(glob))
    return D, conc, glob


def _entrenar_stats(D):
    """Entrena un modelo por estadística con su set de variables: Poisson para conteos
       (córners/tiros al arco/faltas), lineal para posesión. Devuelve {stat: (modelo, cols, tipo)}."""
    modelos = {}
    for stat, cols in SETS_STATS.items():
        tgt = f"t_{stat}"
        d = D.dropna(subset=cols + [tgt])
        if stat == "posesion":
            m = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]).fit(d[cols], d[tgt])
            modelos[stat] = (m, cols, "lineal")
        else:
            m = sm.GLM(d[tgt], sm.add_constant(d[cols], has_constant="add"),
                       family=sm.families.Poisson()).fit()
            modelos[stat] = (m, cols, "poisson")
    return modelos


# --------------------------------------------------------------------------- #
#  Features y probabilidades de un cruce
# --------------------------------------------------------------------------- #
def h2h_par(hist, a, b):
    mm = hist[((hist.home_team == a) & (hist.away_team == b)) | ((hist.home_team == b) & (hist.away_team == a))]
    if len(mm) == 0:
        return 0.0
    d = np.where(mm.home_team == a, mm.home_score - mm.away_score, mm.away_score - mm.home_score)
    return float(d.mean())


def features_cruce(M, a, b, states=None):
    """Las variables del cruce. `states` permite inyectar un estado modificado (sensibilidad / vivo)."""
    st = M["states"] if states is None else states
    sa, sb = st.loc[a], st.loc[b]
    return {
        "elo_diff": sa.elo - sb.elo,
        "squad_value_diff": np.log(sa.squad_value) - np.log(sb.squad_value),
        "h2h_diff": M["H2H"].get((a, b), 0.0),
        "goles_anotados_diff": sa.goles_anotados_avg - sb.goles_anotados_avg,
        "goles_recibidos_diff": sa.goles_recibidos_avg - sb.goles_recibidos_avg,
        "tiros_arco_diff": sa.tiros_arco_avg - sb.tiros_arco_avg,
    }


def prob_partido(M, a, b, cancha="auto", modelo="base", states=None):
    """[P(gana a), P(empate), P(gana b)] con simetrización / localía de anfitrión."""
    pipe = M["pipe_base"] if modelo == "base" else M["pipe_hyb"]
    cols = BASE_VARS if modelo == "base" else HYBRID_VARS
    fa = pd.DataFrame([features_cruce(M, a, b, states)])[cols]
    fb = pd.DataFrame([features_cruce(M, b, a, states)])[cols]
    pa = pipe.predict_proba(fa)[0]; pb = pipe.predict_proba(fb)[0]
    va = np.array([pa[2], pa[1], pa[0]]); vb = np.array([pb[0], pb[1], pb[2]])
    if cancha == "1": return va
    if cancha == "2": return vb
    if cancha == "auto" and a in ANFITRIONES and b not in ANFITRIONES: return va
    if cancha == "auto" and b in ANFITRIONES and a not in ANFITRIONES: return vb
    return (va + vb) / 2


def lambdas(M, a, b, states=None):
    st = M["states"] if states is None else states
    d = st.loc[a, "elo"] - st.loc[b, "elo"]
    return float(np.exp(M["gb0"] + M["gb1"] * d)), float(np.exp(M["gb0"] - M["gb1"] * d))


def cuota(p, margen=0.0):
    """Cuota decimal (odds) de una probabilidad. margen=0 -> cuota justa; >0 -> cuota de bookmaker."""
    p = float(np.clip(p, 1e-6, 1.0))
    return 1.0 / (p * (1.0 + margen))


def ultimos_partidos(M, equipo, n=6, extra=None):
    """Últimos n partidos del equipo (historial + resultados extra ya jugados, p. ej. ESPN del Mundial).
       Devuelve DataFrame: fecha, rival, resultado (texto), gf, gc, signo (G/E/P)."""
    h = M["hist"][["date", "home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"date": "fecha", "home_team": "local", "away_team": "visita",
                 "home_score": "goles_local", "away_score": "goles_visita"})
    if extra is not None and len(extra):
        e = extra[["fecha", "local", "visita", "goles_local", "goles_visita"]].copy()
        e["fecha"] = pd.to_datetime(e["fecha"])
        h = pd.concat([h, e], ignore_index=True)
    h = h[(h.local == equipo) | (h.visita == equipo)].sort_values("fecha").tail(n).iloc[::-1]
    filas = []
    for r in h.itertuples(index=False):
        local = r.local == equipo
        gf = int(r.goles_local if local else r.goles_visita)
        gc = int(r.goles_visita if local else r.goles_local)
        rival = r.visita if local else r.local
        signo = "🟢" if gf > gc else ("⚪" if gf == gc else "🔴")
        filas.append({"fecha": pd.to_datetime(r.fecha).date(), "rival": rival,
                      "marcador": f"{gf}-{gc}", "loc": "vs" if local else "@", "res": signo})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
#  FRENTE 1 · grilla de marcadores y mercados de apuestas
# --------------------------------------------------------------------------- #
def grilla(M, a, b, cancha="auto", modelo="base", states=None):
    """Matriz de probabilidad de cada marcador (reponderada por las prob. V/E/D del clasificador).
       La grilla base de goles lleva la corrección Dixon-Coles (más realista en marcadores de bajo
       score: 0-0, 1-1...). El 1X2 lo sigue fijando el clasificador (que ya lo predice mejor)."""
    p = prob_partido(M, a, b, cancha, modelo, states)
    la, lb = lambdas(M, a, b, states)
    g = np.arange(GRID_MAX + 1)
    grid = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb))
    grid = _dc_ajuste(grid, la, lb, M.get("rho_dc", 0.0)); grid /= grid.sum()
    gi, gj = np.indices(grid.shape)
    masks = (gi > gj, gi == gj, gi < gj)
    mix = sum(p[k] * (grid * mk) / (grid * mk).sum() for k, mk in enumerate(masks))
    return mix, p, (la, lb)


def mercados(mix):
    """Probabilidades de los mercados clásicos a partir de la grilla de marcadores."""
    n = mix.shape[0]
    gi, gj = np.indices((n, n))
    tot = gi + gj
    out = {}
    for linea in (0.5, 1.5, 2.5, 3.5):
        out[f"Over {linea}"] = float(mix[tot > linea].sum())
        out[f"Under {linea}"] = float(mix[tot < linea].sum())
    out["Ambos marcan (BTTS sí)"] = float(mix[(gi >= 1) & (gj >= 1)].sum())
    out["BTTS no"] = 1 - out["Ambos marcan (BTTS sí)"]
    # marcadores más probables
    flat = mix.ravel()
    top = flat.argsort()[::-1][:5]
    out["_top_marcadores"] = [(int(i // n), int(i % n), float(flat[i])) for i in top]
    return out


def handicap_asiatico(mix, linea_a):
    """P de cubrir un hándicap entero para el equipo A (filas). linea_a: +1, -1, etc. (entero)."""
    n = mix.shape[0]
    gi, gj = np.indices((n, n))
    margen = (gi + linea_a) - gj
    return {"A cubre": float(mix[margen > 0].sum()),
            "push": float(mix[margen == 0].sum()),
            "B cubre": float(mix[margen < 0].sum())}


# --------------------------------------------------------------------------- #
#  FRENTE 1b · estadísticas esperadas del partido (córners, tiros al arco, faltas, posesión)
# --------------------------------------------------------------------------- #
def _fila_stats(M, eq, riv, es_local):
    """Reconstruye las features point-in-time de un equipo vs su rival, desde team_states + conc."""
    st, conc, g = M["states"], M["conc"], M["stat_global"]
    cget = lambda team, col: float(conc.loc[team, col]) if team in conc.index else g.get(col, 0.0)
    return {
        "prop_tiros_f": float(st.loc[eq, "tiros_avg"]),
        "prop_corners_f": float(st.loc[eq, "corners_avg"]),
        "prop_faltas_f": float(st.loc[eq, "faltas_avg"]),
        "prop_pos": float(st.loc[eq, "posesion_avg"]),
        "prop_tiros_arco_c": cget(eq, "tiros_arco_c"),
        "prop_corners_c": cget(eq, "corners_c"),
        "riv_tiros_c": cget(riv, "tiros_c"),
        "riv_faltas_c": cget(riv, "faltas_c"),
        "riv_goles_c": cget(riv, "goles_c"),
        "elo_diff": float(st.loc[eq, "elo"] - st.loc[riv, "elo"]),
        "es_local": es_local,
    }


def _predecir_stat(M, stat, fila):
    m, cols, tipo = M["modelos_stats"][stat]
    X = pd.DataFrame([fila])[cols]
    if tipo == "poisson":
        return float(m.predict(sm.add_constant(X, has_constant="add"))[0])
    return float(m.predict(X)[0])


def stats_esperadas(M, a, b, cancha="neutral"):
    """Estadísticas esperadas de cada equipo con los modelos entrenados (selección de variables +
       validación temporal). Posesión renormalizada a 100. Devuelve dict por estadística."""
    if cancha == "1":
        la_loc, lb_loc = 1, 0
    elif cancha == "2":
        la_loc, lb_loc = 0, 1
    else:  # neutral / auto: sin ventaja de localía en las stats
        la_loc, lb_loc = 0.5, 0.5
    fa = _fila_stats(M, a, b, la_loc)
    fb = _fila_stats(M, b, a, lb_loc)
    res = {}
    for stat in ("corners", "tiros_arco", "faltas"):
        res[stat] = (_predecir_stat(M, stat, fa), _predecir_stat(M, stat, fb))
    pa = _predecir_stat(M, "posesion", fa)
    pb = _predecir_stat(M, "posesion", fb)
    res["posesion"] = (pa / (pa + pb) * 100, pb / (pa + pb) * 100)  # renormalizada a 100
    return res


def over_under_total(la, lb, lineas):
    """Over/Under del total (A+B) modelado como Poisson(la+lb). Devuelve {linea: (P_over, P_under)}."""
    lam = la + lb
    out = {}
    for ln in lineas:
        p_under = float(poisson.cdf(int(np.floor(ln)), lam))
        out[ln] = (1 - p_under, p_under)
    return out


def tablas_grupos(resultados=None):
    """Tabla de posiciones de cada grupo a partir de los resultados ya jugados.
       Devuelve dict {letra: DataFrame con Pos, Equipo, PJ, G, E, P, GF, GC, DG, Pts}."""
    res = resultados if resultados is not None else pd.DataFrame(
        columns=["local", "visita", "goles_local", "goles_visita"])
    out = {}
    for g, eqs in GRUPOS.items():
        st = {t: dict(PJ=0, G=0, E=0, P=0, GF=0, GC=0) for t in eqs}
        for r in res.itertuples(index=False):
            if r.local not in eqs or r.visita not in eqs:
                continue
            a, b = r.local, r.visita
            ga, gb = int(r.goles_local), int(r.goles_visita)
            st[a]["PJ"] += 1; st[b]["PJ"] += 1
            st[a]["GF"] += ga; st[a]["GC"] += gb
            st[b]["GF"] += gb; st[b]["GC"] += ga
            if ga > gb: st[a]["G"] += 1; st[b]["P"] += 1
            elif gb > ga: st[b]["G"] += 1; st[a]["P"] += 1
            else: st[a]["E"] += 1; st[b]["E"] += 1
        filas = []
        for t in eqs:
            s = st[t]
            filas.append({"Equipo": t, **s, "DG": s["GF"] - s["GC"], "Pts": 3 * s["G"] + s["E"]})
        d = pd.DataFrame(filas).sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
        d.insert(0, "Pos", d.index + 1)
        out[g] = d
    return out


# --------------------------------------------------------------------------- #
#  FRENTE 2 · modelo vivo: actualizar Elo + forma con resultados reales
# --------------------------------------------------------------------------- #
def _elo_update(ea, eb, ga, gb, k=60.0):
    """Actualización Elo estilo eloratings (cancha neutral, K de Mundial, multiplicador de goleada)."""
    we = 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))
    w = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
    gd = abs(ga - gb)
    mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8.0))
    delta = k * mult * (w - we)
    return ea + delta, eb - delta


def actualizar_estados(M, resultados):
    """Devuelve una copia de team_states con Elo y forma actualizados por los resultados ya jugados.
       resultados: DataFrame con columnas local, visita, goles_local, goles_visita."""
    st = M["states"].copy()
    for r in resultados.itertuples(index=False):
        a, b = r.local, r.visita
        if a not in st.index or b not in st.index:
            continue
        ga, gb = int(r.goles_local), int(r.goles_visita)
        ea, eb = _elo_update(st.loc[a, "elo"], st.loc[b, "elo"], ga, gb)
        st.loc[a, "elo"], st.loc[b, "elo"] = ea, eb
        # forma: media móvil de ventana ~8 (peso 1/8 al nuevo dato)
        st.loc[a, "goles_anotados_avg"] = st.loc[a, "goles_anotados_avg"] * 7 / 8 + ga / 8
        st.loc[a, "goles_recibidos_avg"] = st.loc[a, "goles_recibidos_avg"] * 7 / 8 + gb / 8
        st.loc[b, "goles_anotados_avg"] = st.loc[b, "goles_anotados_avg"] * 7 / 8 + gb / 8
        st.loc[b, "goles_recibidos_avg"] = st.loc[b, "goles_recibidos_avg"] * 7 / 8 + ga / 8
    return st


# --------------------------------------------------------------------------- #
#  Simulación Monte Carlo (con soporte de partidos fijados = ya jugados)
# --------------------------------------------------------------------------- #
def asignar_terceros(grupos_terceros):
    slots = {p: sorted(set(R32[p][1][1]) & grupos_terceros) for p in R32 if isinstance(R32[p][1], tuple)}
    asign, dueno = {}, {}

    def intenta(p, vis):
        for g in slots[p]:
            if g in vis:
                continue
            vis.add(g)
            if g not in dueno or intenta(dueno[g], vis):
                dueno[g] = p; asign[p] = g
                return True
        return False

    for p in sorted(slots):
        intenta(p, set())
    for p in sorted(slots):
        if p not in asign:
            asign[p] = sorted(set(grupos_terceros) - set(asign.values()))[0]
    return asign


def _precalcular(M, states, modelo, rng):
    PROB, SCORE = {}, {}
    for a, b in permutations(MUNDIALISTAS, 2):
        if (b, a) in PROB:
            PROB[(a, b)] = PROB[(b, a)][::-1]
        else:
            PROB[(a, b)] = prob_partido(M, a, b, "auto", modelo, states)
        la, lb = lambdas(M, a, b, states)
        grid = np.outer(poisson.pmf(np.arange(GRID_MAX + 1), la), poisson.pmf(np.arange(GRID_MAX + 1), lb))
        grid /= grid.sum()
        gi, gj = np.indices(grid.shape)
        SCORE[(a, b)] = [np.cumsum((grid * mk).ravel() / (grid * mk).sum()) for mk in (gi > gj, gi == gj, gi < gj)]
    return PROB, SCORE


def monte_carlo(M, n_sims=10000, modelo="base", states=None, fijos=None, seed=42):
    """Simula el Mundial n_sims veces. `fijos`: dict {(a,b): (ga,gb)} de partidos de grupo ya jugados."""
    rng = np.random.default_rng(seed)
    st = M["states"] if states is None else states
    PROB, SCORE = _precalcular(M, st, modelo, rng)
    fijos = fijos or {}

    def jugar_partido(a, b, elim=False):
        if not elim and (a, b) in fijos:
            ga, gb = fijos[(a, b)]
            gan = a if ga > gb else (b if gb > ga else None)
            return ga, gb, gan
        if not elim and (b, a) in fijos:
            gb, ga = fijos[(b, a)]
            gan = a if ga > gb else (b if gb > ga else None)
            return ga, gb, gan
        p = PROB[(a, b)]
        u = rng.random()
        res = 0 if u < p[0] else (1 if u < p[0] + p[1] else 2)
        ga, gb = divmod(int(np.searchsorted(SCORE[(a, b)][res], rng.random())), GRID_MAX + 1)
        if res == 0: return ga, gb, a
        if res == 2: return ga, gb, b
        if elim: return ga, gb, (a if rng.random() < p[0] / (p[0] + p[2]) else b)
        return ga, gb, None

    def jugar_grupo(eqs):
        pts = dict.fromkeys(eqs, 0); gf = dict.fromkeys(eqs, 0); gc = dict.fromkeys(eqs, 0)
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = eqs[i], eqs[j]
                ga, gb, gan = jugar_partido(a, b)
                gf[a] += ga; gc[a] += gb; gf[b] += gb; gc[b] += ga
                if gan == a: pts[a] += 3
                elif gan == b: pts[b] += 3
                else: pts[a] += 1; pts[b] += 1
        return sorted(((t, pts[t], gf[t] - gc[t], gf[t], rng.random()) for t in eqs),
                      key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)

    cont = {t: {"campeon": 0, "final": 0, "semi": 0, "octavos": 0} for t in MUNDIALISTAS}
    for _ in range(n_sims):
        primeros, segundos, terceros = {}, {}, []
        for g, eqs in GRUPOS.items():
            tabla = jugar_grupo(eqs)
            primeros[g], segundos[g] = tabla[0][0], tabla[1][0]
            t = tabla[2]
            terceros.append({"grupo": g, "equipo": t[0], "pts": t[1], "gd": t[2], "gf": t[3], "rnd": t[4]})
        terceros.sort(key=lambda d: (d["pts"], d["gd"], d["gf"], d["rnd"]), reverse=True)
        tg = {d["grupo"]: d["equipo"] for d in terceros[:8]}
        asign = asignar_terceros(set(tg))
        slot = lambda s, n: (tg[asign[n]] if isinstance(s, tuple)
                             else (primeros[s[1]] if s[0] == "1" else segundos[s[1]]))
        W = {}
        for n, (sa, sb) in R32.items():
            _, _, W[n] = jugar_partido(slot(sa, n), slot(sb, n), elim=True)
        for ronda in (R16, QF, SF):
            for n, (p1, p2) in ronda.items():
                _, _, W[n] = jugar_partido(W[p1], W[p2], elim=True)
        fin = (W[FINAL_M[0]], W[FINAL_M[1]])
        _, _, camp = jugar_partido(*fin, elim=True)
        cont[camp]["campeon"] += 1
        for t in set(fin): cont[t]["final"] += 1
        for t in set([W[p] for p in (97, 98, 99, 100)]) | set(fin): cont[t]["semi"] += 1
        for t in set(W[n] for n in R32): cont[t]["octavos"] += 1

    res = pd.DataFrame([{"Selección": t, "grupo": GRUPO_DE[t],
                         "P_campeon": c["campeon"] / n_sims, "P_final": c["final"] / n_sims,
                         "P_semi": c["semi"] / n_sims, "P_octavos": c["octavos"] / n_sims}
                        for t, c in cont.items()]).sort_values("P_campeon", ascending=False).reset_index(drop=True)
    return res


# --------------------------------------------------------------------------- #
#  FRENTE 3 · validación (calibración y backtesting económico)
# --------------------------------------------------------------------------- #
def backtest_test(M, modelo="base"):
    """Evalúa el modelo en el hold-out temporal 2025-26, devuelve P, y, y diagnósticos por tipo."""
    df = M["df"]; corte = M["CORTE_TEST"]
    cols = BASE_VARS if modelo == "base" else HYBRID_VARS
    extra = ["ea_overall_diff"] if modelo == "base" else []
    d = df.dropna(subset=cols + extra).copy()
    d["tipo"] = d.competicion.map(tipo_competicion)
    tr = d[d.fecha < corte]; te = d[d.fecha >= corte].reset_index(drop=True)
    w = tr.competicion.map(tipo_competicion).map(K_FACTOR).values
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    pipe.fit(tr[cols], tr["resultado"], m__sample_weight=w)
    P = pipe.predict_proba(te[cols])
    return P, te["resultado"].values, te


def curva_calibracion(P, y, clase=2, n_bins=8):
    p = P[:, clase]; o = (y == clase).astype(float)
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-9
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(o[m].mean()); ns.append(int(m.sum()))
    ece = sum(n * abs(x - y_) for x, y_, n in zip(xs, ys, ns)) / max(sum(ns), 1)
    return np.array(xs), np.array(ys), np.array(ns), float(ece)


def value_betting(M, modelo="base", margen=0.05, umbral=0.0):
    """Backtest económico honesto: el 'mercado' = baseline solo-Elo con margen; el modelo apuesta
       1 unidad cuando su valor esperado es positivo. Devuelve resumen y curva acumulada de ROI."""
    df = M["df"]; corte = M["CORTE_TEST"]
    d = df.dropna(subset=BASE_VARS + ["ea_overall_diff"]).copy().sort_values("fecha")
    tr = d[d.fecha < corte]; te = d[d.fecha >= corte].reset_index(drop=True)
    wtr = tr.competicion.map(tipo_competicion).map(K_FACTOR).values

    cols_full = BASE_VARS if modelo == "base" else HYBRID_VARS
    d_h = df.dropna(subset=HYBRID_VARS).copy().sort_values("fecha")
    tr_h = d_h[d_h.fecha < corte]; te_h = d_h[d_h.fecha >= corte].reset_index(drop=True)
    # modelo del jugador
    pj = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    if modelo == "base":
        pj.fit(tr[BASE_VARS], tr["resultado"], m__sample_weight=wtr)
        Pj = pj.predict_proba(te[BASE_VARS]); y = te["resultado"].values
    else:
        wh = tr_h.competicion.map(tipo_competicion).map(K_FACTOR).values
        pj.fit(tr_h[HYBRID_VARS], tr_h["resultado"], m__sample_weight=wh)
        Pj = pj.predict_proba(te_h[HYBRID_VARS]); y = te_h["resultado"].values
        te = te_h
    # mercado sintético = baseline solo-Elo, con margen (overround)
    pm = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    base_tr = tr if modelo == "base" else tr_h
    pm.fit(base_tr[["elo_diff"]], base_tr["resultado"],
           m__sample_weight=base_tr.competicion.map(tipo_competicion).map(K_FACTOR).values)
    Pmkt = pm.predict_proba(te[["elo_diff"]])
    cuota = 1.0 / (Pmkt * (1 + margen))   # cuotas con sobre-ronda del bookmaker

    apuestas, retornos, fechas = [], [], []
    for i in range(len(y)):
        ev = Pj[i] * cuota[i] - 1.0          # valor esperado por unidad en cada resultado
        k = int(np.argmax(ev))
        if ev[k] > umbral:
            apuestas.append(1.0)
            retornos.append(cuota[i, k] if y[i] == k else 0.0)
            fechas.append(te.fecha.iloc[i])
    apuestas = np.array(apuestas); retornos = np.array(retornos)
    n = len(apuestas)
    if n == 0:
        return {"n_apuestas": 0, "roi": 0.0, "ganancia": 0.0, "acierto": 0.0}, pd.DataFrame()
    ganancia = retornos.sum() - apuestas.sum()
    curva = pd.DataFrame({"fecha": fechas, "pl": retornos - 1.0})
    curva = curva.sort_values("fecha")
    curva["acumulado"] = curva["pl"].cumsum()
    return {"n_apuestas": n, "roi": ganancia / apuestas.sum(),
            "ganancia": float(ganancia), "acierto": float((retornos > 0).mean())}, curva


# --------------------------------------------------------------------------- #
#  Validación del panel de stats (backtest walk-forward, sin fuga)
# --------------------------------------------------------------------------- #
def backtest_stats(M):
    """Valida el modelo de stats con split temporal: entrena en <2025, mide en 2025-26 (hold-out).
       Compara el modelo (selección de variables) contra el baseline 'promedio propio del equipo'
       y 'media global'. Devuelve un DataFrame de métricas. Usa el dataset point-in-time de cargar()."""
    D = M["stats_data"]
    nombres = {"corners": "Córners", "tiros_arco": "Tiros al arco", "faltas": "Faltas", "posesion": "Posesión %"}
    propio_col = {"corners": "prop_corners_f", "tiros_arco": "prop_tiros_arco_f",
                  "faltas": "prop_faltas_f", "posesion": "prop_pos"}
    filas = []
    for stat, cols in SETS_STATS.items():
        tgt = f"t_{stat}"
        d = D.dropna(subset=cols + [tgt, propio_col[stat]]).sort_values("fecha")
        tr, te = d[d.fecha < "2025-01-01"], d[d.fecha >= "2025-01-01"]
        if len(te) == 0:
            continue
        if stat == "posesion":
            m = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]).fit(tr[cols], tr[tgt])
            pred = m.predict(te[cols])
        else:
            m = sm.GLM(tr[tgt], sm.add_constant(tr[cols], has_constant="add"),
                       family=sm.families.Poisson()).fit()
            pred = m.predict(sm.add_constant(te[cols], has_constant="add"))
        mae_mod = float(np.mean(np.abs(pred - te[tgt])))
        mae_prop = float(np.mean(np.abs(te[propio_col[stat]] - te[tgt])))
        mae_glob = float(np.mean(np.abs(tr[tgt].mean() - te[tgt])))
        filas.append({"Estadística": nombres[stat], "media": round(te[tgt].mean(), 1),
                      "MAE modelo": round(mae_mod, 2), "MAE prom. propio": round(mae_prop, 2),
                      "MAE media global": round(mae_glob, 2)})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
#  Validación EN VIVO: el modelo contra los resultados reales del Mundial
# --------------------------------------------------------------------------- #
def validacion_en_vivo(M, resultados, modelo="base"):
    """Compara la predicción pre-partido del modelo contra el resultado real de cada partido del
       Mundial ya jugado. Devuelve (tabla detalle, métricas, evolución log-loss acumulado)."""
    from sklearn.metrics import log_loss
    st = M["states"]
    filas, P, y = [], [], []
    for r in resultados.itertuples(index=False):
        a, b = r.local, r.visita
        if a not in st.index or b not in st.index:
            continue
        p = prob_partido(M, a, b, "auto", modelo)   # [P(gana local a), empate, P(gana visita b)]
        ga, gb = int(r.goles_local), int(r.goles_visita)
        real = 2 if ga > gb else (1 if ga == gb else 0)   # 2=gana local, 1=empate, 0=gana visita
        P.append([p[2], p[1], p[0]]); y.append(real)      # orden de clases [0,1,2] para log_loss
        pred = 2 if (p[0] >= p[1] and p[0] >= p[2]) else (1 if p[1] >= p[2] else 0)
        filas.append({"Fecha": pd.to_datetime(r.fecha).date(), "Local": a, "Marcador": f"{ga}-{gb}", "Visita": b,
                      "P(local)": f"{p[0]:.0%}", "P(empate)": f"{p[1]:.0%}", "P(visita)": f"{p[2]:.0%}",
                      "Real": ["Gana visita", "Empate", "Gana local"][real],
                      "Acierto": "✅" if pred == real else "❌"})
    P, y = np.array(P), np.array(y)
    n = len(y)
    if n == 0:
        return pd.DataFrame(), {}, pd.DataFrame()
    base = np.tile([0.279, 0.275, 0.446], (n, 1))   # frecuencias históricas [visita, empate, local]
    aciertos = sum(1 for i in range(n) if np.argmax(P[i]) == y[i])
    met = {"n": n, "acierto": aciertos / n,
           "logloss": log_loss(y, P, labels=[0, 1, 2]),
           "logloss_base": log_loss(y, base, labels=[0, 1, 2]),
           "p_empate_pred": float(P[:, 1].mean()), "empates_reales": float((y == 1).mean())}
    # evolución del log-loss acumulado (para ver si mejora conforme avanza el torneo)
    evol = [log_loss(y[:i + 1], P[:i + 1], labels=[0, 1, 2]) for i in range(n)]
    evolucion = pd.DataFrame({"partido": range(1, n + 1), "logloss_acum": evol,
                              "baseline": [met["logloss_base"]] * n})
    return pd.DataFrame(filas), met, evolucion


# --------------------------------------------------------------------------- #
#  FRENTE 4 · robustez (IC del Monte Carlo)
# --------------------------------------------------------------------------- #
def ic_montecarlo(p, n_sims):
    """Error estándar e IC 95% de una probabilidad estimada por Monte Carlo (proporción binomial)."""
    se = np.sqrt(p * (1 - p) / n_sims)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


# --------------------------------------------------------------------------- #
#  Cuadro de eliminatorias REAL (ya definido) — simulación del campeón
# --------------------------------------------------------------------------- #
def prob_eliminatoria(M, a, b, states=None, modelo="base"):
    """P(a avanza sobre b) en un partido de eliminatoria: no hay empate, así que se reparte
       la probabilidad de empate del clasificador proporcional a la fuerza de cada lado."""
    st = M["states"] if states is None else states
    if a is None or b is None:
        return 1.0 if b is None else 0.0
    if a not in st.index or b not in st.index:
        return 1.0 if a in st.index else 0.0
    pa, pe, pb = prob_partido(M, a, b, "auto", modelo, st)
    s = pa + pb
    return float(pa + pe * pa / s) if s > 0 else 0.5


def _feed(g):
    """Traduce un nº de partido FIFA al casillero (ronda, idx) que lo alimenta."""
    if 73 <= g <= 88:
        return ("R32", g - 72)
    if 89 <= g <= 96:
        return ("R16", g - 88)
    if 97 <= g <= 100:
        return ("QF", g - 96)
    return ("SF", g - 100)


def bracket_real(res, r32_espn):
    """Construye el cuadro REAL de eliminatorias combinando dos fuentes fiables:
       - los 16 CRUCES REALES de ESPN (`r32_espn`: lista de {home,away,state,gh,ga}) — hechos;
       - el ÁRBOL OFICIAL FIFA (R32/R16/QF/SF/FINAL_M de este módulo) — fijo y verificado:
         N de ESPN = nº de partido FIFA − 72, así que la plantilla reproduce el árbol exacto.
       Cada cruce de ESPN se coloca en su llave por el equipo 1°/2° YA conocido (de las posiciones
       reales de grupo); el rival (a veces un tercero) viene tal cual de ESPN. Esto evita depender
       de la numeración interna de ESPN (que no es cronológica) y de adivinar la asignación de
       terceros (que la regla FIFA hace por una tabla específica)."""
    grupos = res.iloc[:72]
    tablas = tablas_grupos(grupos)
    primeros = {g: t.iloc[0]["Equipo"] for g, t in tablas.items()}
    segundos = {g: t.iloc[1]["Equipo"] for g, t in tablas.items()}
    matchups = list(r32_espn)

    def conocido(slot):
        if isinstance(slot, tuple):   # ranura de tercero -> equipo aún no determinable por nosotros
            return None
        return primeros[slot[1]] if slot[0] == "1" else segundos[slot[1]]

    R32d, usados = {}, set()
    for g, (sa, sb) in R32.items():
        known = [k for k in (conocido(sa), conocido(sb)) if k is not None]
        idx = next(i for i, m in enumerate(matchups)
                   if i not in usados and all(k in (m["home"], m["away"]) for k in known))
        usados.add(idx)
        m = matchups[idx]
        R32d[g - 72] = {"home": m["home"], "away": m["away"], "state": m["state"],
                        "gh": m["gh"], "ga": m["ga"],
                        "winner": m.get("winner"), "pens": m.get("pens")}
    R16d = {g - 88: {"home": _feed(a), "away": _feed(b)} for g, (a, b) in R16.items()}
    QFd = {g - 96: {"home": _feed(a), "away": _feed(b)} for g, (a, b) in QF.items()}
    SFd = {g - 100: {"home": _feed(a), "away": _feed(b)} for g, (a, b) in SF.items()}
    FINd = {"home": _feed(FINAL_M[0]), "away": _feed(FINAL_M[1])}
    return {"R32": R32d, "R16": R16d, "QF": QFd, "SF": SFd, "FINAL": FINd}


def simular_bracket(M, bracket, states=None, n_sims=15000, modelo="base", seed=42, fijos_ko=None):
    """Monte Carlo de SOLO las eliminatorias, desde el cuadro real ya definido.
       `fijos_ko`: dict {frozenset({a,b}): ganador} de cruces de KO ya jugados (cualquier ronda).
       Devuelve {'tabla': DataFrame por equipo (P de llegar a cada ronda + P_campeon),
                 'reach': {(ronda, num): {equipo: prob de GANAR ese partido}}, 'n_sims'}."""
    st = M["states"] if states is None else states
    rng = np.random.default_rng(seed)
    R32, R16, QF, SF, FINAL = bracket["R32"], bracket["R16"], bracket["QF"], bracket["SF"], bracket["FINAL"]
    fijos_ko = fijos_ko or {}

    cache = {}
    def p_adv(a, b):
        key = (a, b)
        if key not in cache:
            v = prob_eliminatoria(M, a, b, st, modelo)
            cache[key] = v
            cache[(b, a)] = 1.0 - v
        return cache[key]

    def jugar(a, b):
        if a is None or b is None:
            return a if b is None else b
        fij = fijos_ko.get(frozenset({a, b}))   # ¿ya se jugó este cruce?
        if fij is not None:
            return fij
        return a if rng.random() < p_adv(a, b) else b

    win = {("R32", n): Counter() for n in R32}
    for rk, rd in (("R16", R16), ("QF", QF), ("SF", SF)):
        win.update({(rk, n): Counter() for n in rd})
    win[("FINAL", 1)] = Counter()

    for _ in range(n_sims):
        W = {"R32": {}, "R16": {}, "QF": {}, "SF": {}}
        for n, m in R32.items():
            w = jugar(m["home"], m["away"])
            W["R32"][n] = w
            win[("R32", n)][w] += 1
        for rk, rd in (("R16", R16), ("QF", QF), ("SF", SF)):
            for n, m in rd.items():
                a = W[m["home"][0]][m["home"][1]]
                b = W[m["away"][0]][m["away"][1]]
                w = jugar(a, b)
                W[rk][n] = w
                win[(rk, n)][w] += 1
        a = W[FINAL["home"][0]][FINAL["home"][1]]
        b = W[FINAL["away"][0]][FINAL["away"][1]]
        win[("FINAL", 1)][jugar(a, b)] += 1

    reach = {k: {t: c / n_sims for t, c in cnt.items()} for k, cnt in win.items()}
    r32_de = {}
    for n, m in R32.items():
        r32_de[m["home"]] = n
        r32_de[m["away"]] = n

    def suma(rk, rd, t):
        return sum(reach[(rk, n)].get(t, 0.0) for n in rd)

    teams = sorted(r32_de)
    filas = [{"Selección": t,
              "P_R16": reach[("R32", r32_de[t])].get(t, 0.0),
              "P_QF": suma("R16", R16, t), "P_SF": suma("QF", QF, t),
              "P_final": suma("SF", SF, t), "P_campeon": reach[("FINAL", 1)].get(t, 0.0)}
             for t in teams]
    tabla = pd.DataFrame(filas).sort_values("P_campeon", ascending=False).reset_index(drop=True)
    return {"tabla": tabla, "reach": reach, "n_sims": n_sims}
