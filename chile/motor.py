"""
Motor del predictor de la Primera División de Chile. Misma filosofía que el del Mundial, adaptada
a una liga de clubes (la localía es real, no se simetriza):

  - Elo cronológico con ventaja de localía y multiplicador de goleada.
  - Modelo de resultado (logística multinomial V/E/D) con features point-in-time, validado en el tiempo.
  - Poisson de goles para marcadores.
  - Simulador del campeonato: parte de la tabla actual y simula el fixture restante miles de veces
    -> P(campeón), P(clasificar a copas), P(descenso), posición y puntos esperados.
"""
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA = Path(__file__).resolve().parent / "data"
ELO_INIT, K_LIGA, HOME_ADV = 1500.0, 30.0, 55.0
DESCIENDEN = 2          # nº de equipos que descienden (último y penúltimo)
CUPOS_COPA = 4          # aprox. clasificación a torneos continentales (top-4)


def _mult_goles(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8))


def calcular_elo(part):
    """Elo cronológico. Devuelve dict de Elo final + columnas pre-partido para entrenar."""
    elo = defaultdict(lambda: ELO_INIT)
    pl, pv = [], []
    for r in part.itertuples(index=False):
        el, ev = elo[r.local], elo[r.visita]
        pl.append(el); pv.append(ev)
        we = 1.0 / (1.0 + 10 ** (-((el + HOME_ADV) - ev) / 400.0))
        gl, gv = r.goles_local, r.goles_visita
        w = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        delta = K_LIGA * _mult_goles(gl - gv) * (w - we)
        elo[r.local] = el + delta; elo[r.visita] = ev - delta
    return dict(elo), pl, pv


def cargar():
    part = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)
    fixture = pd.read_csv(DATA / "fixture.csv", parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)
    part["goles_local"] = part.goles_local.astype(int)
    part["goles_visita"] = part.goles_visita.astype(int)

    elo, pre_l, pre_v = calcular_elo(part)
    part = part.copy(); part["elo_local"] = pre_l; part["elo_visita"] = pre_v

    # features point-in-time: forma (ppg últimos 5) y h2h (dif. de gol promedio previa)
    ppg = defaultdict(lambda: deque(maxlen=5)); h2h = defaultdict(list)
    f_ppg_l, f_ppg_v, f_h2h = [], [], []
    for r in part.itertuples(index=False):
        f_ppg_l.append(np.mean(ppg[r.local]) if ppg[r.local] else 1.0)
        f_ppg_v.append(np.mean(ppg[r.visita]) if ppg[r.visita] else 1.0)
        par = tuple(sorted((r.local, r.visita)))
        prev = h2h[par]
        d = np.mean([x if r.local == par[0] else -x for x in prev]) if prev else 0.0
        f_h2h.append(d if r.local == par[0] else -d)
        # actualizar
        gl, gv = r.goles_local, r.goles_visita
        rl, rv = (3, 0) if gl > gv else ((1, 1) if gl == gv else (0, 3))
        ppg[r.local].append(rl); ppg[r.visita].append(rv)
        h2h[par].append(gl - gv)
    part["ppg_local"], part["ppg_visita"], part["h2h_diff"] = f_ppg_l, f_ppg_v, f_h2h
    part["elo_diff"] = part.elo_local + HOME_ADV - part.elo_visita
    part["ppg_diff"] = part.ppg_local - part.ppg_visita
    part["resultado"] = np.where(part.goles_local > part.goles_visita, 2,
                                 np.where(part.goles_local == part.goles_visita, 1, 0))

    FEATS = ["elo_diff", "ppg_diff", "h2h_diff"]
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    pipe.fit(part[FEATS], part.resultado)

    # Poisson de goles (con localía): dos obs por partido
    largo = pd.concat([
        pd.DataFrame({"g": part.goles_local, "d": part.elo_local + HOME_ADV - part.elo_visita, "loc": 1}),
        pd.DataFrame({"g": part.goles_visita, "d": part.elo_visita - HOME_ADV - part.elo_local, "loc": 0})])
    import statsmodels.api as sm
    gp = sm.GLM(largo["g"], sm.add_constant(largo[["d", "loc"]]), family=sm.families.Poisson()).fit()

    # estado actual de cada equipo (Elo + forma + h2h base)
    estado = {}
    for t in elo:
        estado[t] = {"elo": elo[t], "ppg": np.mean(ppg[t]) if ppg[t] else 1.0}
    return {"part": part, "fixture": fixture, "elo": elo, "estado": estado, "h2h": h2h,
            "pipe": pipe, "FEATS": FEATS, "gp_params": dict(gp.params),
            "equipos_2026": sorted(set(part[part.temporada == 2026].local) | set(part[part.temporada == 2026].visita))}


# --------------------------------------------------------------------------- #
#  Predicción de un partido
# --------------------------------------------------------------------------- #
def _h2h(M, a, b):
    par = tuple(sorted((a, b)))
    prev = M["h2h"].get(par, [])
    if not prev:
        return 0.0
    d = np.mean([x if a == par[0] else -x for x in prev])
    return float(d if a == par[0] else -d)


def features(M, local, visita):
    e = M["estado"]
    return {"elo_diff": e[local]["elo"] + HOME_ADV - e[visita]["elo"],
            "ppg_diff": e[local]["ppg"] - e[visita]["ppg"],
            "h2h_diff": _h2h(M, local, visita)}


def prob_partido(M, local, visita):
    """[P(gana local), P(empate), P(gana visita)] con localía real del que juega de local."""
    p = M["pipe"].predict_proba(pd.DataFrame([features(M, local, visita)])[M["FEATS"]])[0]
    return np.array([p[2], p[1], p[0]])


def lambdas(M, local, visita):
    b = M["gp_params"]
    dl = M["estado"][local]["elo"] + HOME_ADV - M["estado"][visita]["elo"]
    dv = M["estado"][visita]["elo"] - HOME_ADV - M["estado"][local]["elo"]
    return float(np.exp(b["const"] + b["d"] * dl + b["loc"])), float(np.exp(b["const"] + b["d"] * dv))


def grilla(M, local, visita, gmax=8):
    p = prob_partido(M, local, visita)
    la, lb = lambdas(M, local, visita)
    g = np.arange(gmax + 1)
    grid = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb)); grid /= grid.sum()
    gi, gj = np.indices(grid.shape)
    mix = sum(p[k] * (grid * mk) / (grid * mk).sum() for k, mk in enumerate((gi > gj, gi == gj, gi < gj)))
    return mix, p, (la, lb)


# --------------------------------------------------------------------------- #
#  Tabla actual y simulación del campeonato
# --------------------------------------------------------------------------- #
def tabla_actual(M):
    """Tabla de posiciones 2026 a partir de los partidos ya jugados."""
    p26 = M["part"][M["part"].temporada == 2026]
    eq = M["equipos_2026"]
    st = {t: dict(PJ=0, G=0, E=0, P=0, GF=0, GC=0) for t in eq}
    for r in p26.itertuples(index=False):
        a, b, gl, gv = r.local, r.visita, r.goles_local, r.goles_visita
        st[a]["PJ"] += 1; st[b]["PJ"] += 1; st[a]["GF"] += gl; st[a]["GC"] += gv
        st[b]["GF"] += gv; st[b]["GC"] += gl
        if gl > gv: st[a]["G"] += 1; st[b]["P"] += 1
        elif gl == gv: st[a]["E"] += 1; st[b]["E"] += 1
        else: st[b]["G"] += 1; st[a]["P"] += 1
    df = pd.DataFrame([{"Equipo": t, **s, "DG": s["GF"] - s["GC"], "Pts": 3 * s["G"] + s["E"]} for t, s in st.items()])
    df = df.sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
    df.insert(0, "Pos", df.index + 1)
    return df


def simular_campeonato(M, n_sims=10000, seed=42):
    """Parte de la tabla actual y simula el fixture restante n_sims veces."""
    rng = np.random.default_rng(seed)
    tabla = tabla_actual(M).set_index("Equipo")
    pts0 = tabla["Pts"].to_dict(); gd0 = tabla["DG"].to_dict()
    eq = list(tabla.index)
    fix = [(r.local, r.visita) for r in M["fixture"].itertuples(index=False)
           if r.local in pts0 and r.visita in pts0]
    # precalcular prob y goles esperados de cada partido del fixture
    P, LAM = {}, {}
    for a, b in set(fix):
        P[(a, b)] = prob_partido(M, a, b); LAM[(a, b)] = lambdas(M, a, b)

    pos_count = {t: np.zeros(len(eq) + 1) for t in eq}     # histograma de posiciones
    campeon = defaultdict(int); copa = defaultdict(int); desc = defaultdict(int)
    pts_final = defaultdict(list)
    for _ in range(n_sims):
        pts = dict(pts0); gd = dict(gd0)
        for a, b in fix:
            p = P[(a, b)]; u = rng.random()
            res = 0 if u < p[0] else (1 if u < p[0] + p[1] else 2)   # 0 gana local
            la, lb = LAM[(a, b)]
            ga, gb = rng.poisson(la), rng.poisson(lb)
            if res == 0 and ga <= gb: ga = gb + 1
            if res == 2 and gb <= ga: gb = ga + 1
            if res == 1: gb = ga
            gd[a] += ga - gb; gd[b] += gb - ga
            if ga > gb: pts[a] += 3
            elif ga == gb: pts[a] += 1; pts[b] += 1
            else: pts[b] += 3
        orden = sorted(eq, key=lambda t: (pts[t], gd[t], rng.random()), reverse=True)
        for i, t in enumerate(orden):
            pos_count[t][i + 1] += 1
            pts_final[t].append(pts[t])
        campeon[orden[0]] += 1
        for t in orden[:CUPOS_COPA]: copa[t] += 1
        for t in orden[-DESCIENDEN:]: desc[t] += 1

    filas = []
    for t in eq:
        filas.append({"Equipo": t, "Pts_actual": pts0[t],
                      "P_campeon": campeon[t] / n_sims, "P_copa": copa[t] / n_sims, "P_descenso": desc[t] / n_sims,
                      "pos_esperada": float(np.average(np.arange(1, len(eq) + 1), weights=pos_count[t][1:])),
                      "pts_proy": float(np.mean(pts_final[t]))})
    return pd.DataFrame(filas).sort_values("P_campeon", ascending=False).reset_index(drop=True)
