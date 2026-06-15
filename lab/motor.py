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
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

    return {
        "df": df, "states": states, "hist": hist, "H2H": H2H,
        "pipe_base": pipe_base, "pipe_hyb": pipe_hyb,
        "gb0": float(gp.params["const"]), "gb1": float(gp.params["d"]),
        "CORTE_TEST": "2025-01-01",
    }


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


# --------------------------------------------------------------------------- #
#  FRENTE 1 · grilla de marcadores y mercados de apuestas
# --------------------------------------------------------------------------- #
def grilla(M, a, b, cancha="auto", modelo="base", states=None):
    """Matriz de probabilidad de cada marcador (reponderada por las prob. V/E/D del clasificador)."""
    p = prob_partido(M, a, b, cancha, modelo, states)
    la, lb = lambdas(M, a, b, states)
    g = np.arange(GRID_MAX + 1)
    grid = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb)); grid /= grid.sum()
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
#  FRENTE 4 · robustez (IC del Monte Carlo)
# --------------------------------------------------------------------------- #
def ic_montecarlo(p, n_sims):
    """Error estándar e IC 95% de una probabilidad estimada por Monte Carlo (proporción binomial)."""
    se = np.sqrt(p * (1 - p) / n_sims)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
