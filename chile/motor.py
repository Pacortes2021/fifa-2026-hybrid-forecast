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


SQUAD_VALUES_BY_YEAR = {
    2021: {
        "Colo Colo": 20.0, "Universidad Católica": 19.0, "Universidad de Chile": 14.0,
        "Unión La Calera": 10.5, "Unión Española": 9.0, "Everton CD": 8.0,
        "Audax Italiano": 7.0, "Palestino": 7.5, "O'Higgins": 7.0, "Huachipato": 6.5,
        "Cobresal": 5.5, "Ñublense": 5.0, "Antofagasta": 7.0, "Curicó Unido": 5.5,
        "Melipilla": 5.0, "Santiago Wanderers": 5.5, "La Serena": 6.5, "Coquimbo Unido": 4.0,
        "Deportes Iquique": 4.0, "Cobreloa": 3.5, "Deportes Concepcion": 1.5,
        "Deportes Limache": 2.0, "Universidad de Concepción": 3.5, "Copiapó": 3.0,
        "Magallanes": 2.5, "Unión Wanderers": 3.5
    },
    2022: {
        "Colo Colo": 22.0, "Universidad Católica": 17.5, "Universidad de Chile": 13.0,
        "Unión La Calera": 9.5, "Unión Española": 9.5, "Everton CD": 8.5,
        "Audax Italiano": 7.5, "Palestino": 7.2, "O'Higgins": 7.2, "Huachipato": 7.0,
        "Cobresal": 5.8, "Ñublense": 7.5, "Antofagasta": 6.5, "Curicó Unido": 6.5,
        "Melipilla": 3.5, "Santiago Wanderers": 4.0, "La Serena": 6.0, "Coquimbo Unido": 5.5,
        "Deportes Iquique": 4.2, "Cobreloa": 3.8, "Deportes Concepcion": 2.0,
        "Deportes Limache": 2.2, "Universidad de Concepción": 3.8, "Copiapó": 3.5,
        "Magallanes": 3.2, "Unión Wanderers": 3.5
    },
    2023: {
        "Colo Colo": 24.0, "Universidad Católica": 16.0, "Universidad de Chile": 15.0,
        "Unión La Calera": 8.0, "Unión Española": 10.0, "Everton CD": 9.0,
        "Audax Italiano": 8.0, "Palestino": 8.0, "O'Higgins": 7.5, "Huachipato": 8.5,
        "Cobresal": 6.2, "Ñublense": 8.0, "Antofagasta": 4.5, "Curicó Unido": 5.8,
        "Melipilla": 3.0, "Santiago Wanderers": 3.8, "La Serena": 4.0, "Coquimbo Unido": 6.5,
        "Deportes Iquique": 4.5, "Cobreloa": 4.2, "Deportes Concepcion": 2.5,
        "Deportes Limache": 2.5, "Universidad de Concepción": 4.0, "Copiapó": 5.5,
        "Magallanes": 6.0, "Unión Wanderers": 3.5
    },
    2024: {
        "Colo Colo": 23.0, "Universidad Católica": 15.5, "Universidad de Chile": 16.5,
        "Unión La Calera": 7.0, "Unión Española": 9.0, "Everton CD": 9.5,
        "Audax Italiano": 7.0, "Palestino": 7.8, "O'Higgins": 6.8, "Huachipato": 8.2,
        "Cobresal": 6.5, "Ñublense": 7.0, "Antofagasta": 4.8, "Curicó Unido": 4.2,
        "Melipilla": 3.2, "Santiago Wanderers": 3.5, "La Serena": 4.2, "Coquimbo Unido": 7.5,
        "Deportes Iquique": 6.2, "Cobreloa": 6.0, "Deportes Concepcion": 3.0,
        "Deportes Limache": 3.0, "Universidad de Concepción": 4.2, "Copiapó": 5.2,
        "Magallanes": 4.5, "Unión Wanderers": 3.5
    },
    2025: {
        "Colo Colo": 24.5, "Universidad Católica": 15.0, "Universidad de Chile": 17.5,
        "Unión La Calera": 6.5, "Unión Española": 9.5, "Everton CD": 9.2,
        "Audax Italiano": 7.2, "Palestino": 8.0, "O'Higgins": 7.0, "Huachipato": 8.0,
        "Cobresal": 6.2, "Ñublense": 7.2, "Antofagasta": 5.0, "Curicó Unido": 4.0,
        "Melipilla": 3.5, "Santiago Wanderers": 3.5, "La Serena": 4.5, "Coquimbo Unido": 8.0,
        "Deportes Iquique": 6.5, "Cobreloa": 5.5, "Deportes Concepcion": 3.5,
        "Deportes Limache": 3.5, "Universidad de Concepción": 4.5, "Copiapó": 4.5,
        "Magallanes": 4.5, "Unión Wanderers": 3.5
    },
    2026: {
        "Colo Colo": 25.0, "Universidad Católica": 16.0, "Universidad de Chile": 18.0,
        "Unión La Calera": 6.5, "Unión Española": 10.0, "Everton CD": 9.5,
        "Audax Italiano": 7.5, "Palestino": 8.0, "O'Higgins": 7.0, "Huachipato": 8.0,
        "Cobresal": 6.5, "Ñublense": 7.5, "Antofagasta": 5.5, "Curicó Unido": 4.0,
        "Melipilla": 3.5, "Santiago Wanderers": 3.5, "La Serena": 5.0, "Coquimbo Unido": 8.5,
        "Deportes Iquique": 6.0, "Cobreloa": 4.5, "Deportes Concepcion": 4.5,
        "Deportes Limache": 4.5, "Universidad de Concepción": 4.5, "Copiapó": 4.5,
        "Magallanes": 4.5, "Unión Wanderers": 3.5
    }
}


def get_squad_value(team, season):
    # Obtener el diccionario del año, o del año más cercano disponible
    year_dict = SQUAD_VALUES_BY_YEAR.get(season, SQUAD_VALUES_BY_YEAR[2026])
    return year_dict.get(team, 5.0)


def _mult_goles(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8))


def calcular_elo(part):
    """Elo cronológico con regresión a la media inter-temporada e inicialización
       de ascendidos a 1420."""
    seasons = sorted(part.temporada.unique())
    teams_by_season = {}
    for s in seasons:
        df_s = part[part.temporada == s]
        teams_by_season[s] = set(df_s.local.unique()) | set(df_s.visita.unique())
    
    elo = {}
    first_season = seasons[0]
    for t in teams_by_season[first_season]:
        elo[t] = ELO_INIT

    pl, pv = [], []
    last_season = first_season

    for r in part.itertuples(index=False):
        # Detectar cambio de temporada
        if r.temporada != last_season:
            # 1. Regresión inter-temporada (25% a la media)
            for t in elo:
                elo[t] = 0.75 * elo[t] + 0.25 * ELO_INIT
            
            # 2. Inicializar ascendidos (no jugaron en la temporada anterior) a 1420
            s_idx = seasons.index(r.temporada)
            prev_s = seasons[s_idx - 1]
            for t in teams_by_season[r.temporada]:
                if t not in elo or t not in teams_by_season[prev_s]:
                    elo[t] = 1420.0
            
            last_season = r.temporada

        if r.local not in elo:
            elo[r.local] = 1500.0
        if r.visita not in elo:
            elo[r.visita] = 1500.0

        el, ev = elo[r.local], elo[r.visita]
        pl.append(el); pv.append(ev)
        we = 1.0 / (1.0 + 10 ** (-((el + HOME_ADV) - ev) / 400.0))
        gl, gv = r.goles_local, r.goles_visita
        w = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        delta = K_LIGA * _mult_goles(gl - gv) * (w - we)
        elo[r.local] = el + delta; elo[r.visita] = ev - delta

    return dict(elo), pl, pv


def cargar(en_vivo=True):
    part_path = DATA / "partidos.csv"
    fix_path = DATA / "fixture.csv"
    
    part = pd.read_csv(part_path, parse_dates=["fecha"])
    fixture = pd.read_csv(fix_path, parse_dates=["fecha"])
    
    if en_vivo:
        try:
            import requests
            url = "https://site.api.espn.com/apis/site/v2/sports/soccer/chi.1/scoreboard?dates=20260101-20261231&limit=400"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                eventos = r.json().get("events", [])
                filas_2026 = []
                for e in eventos:
                    try:
                        comp = e["competitions"][0]; cs = comp["competitors"]
                        h = next(x for x in cs if x["homeAway"] == "home")
                        a = next(x for x in cs if x["homeAway"] == "away")
                        estado = e["status"]["type"]["state"]
                        try:
                            gl = int(h["score"]) if h.get("score") not in (None, "") else None
                            gv = int(a["score"]) if a.get("score") not in (None, "") else None
                        except (TypeError, ValueError):
                            gl = gv = None
                        filas_2026.append({
                            "fecha": pd.to_datetime(e["date"]).tz_localize(None),
                            "temporada": 2026, "local": h["team"]["displayName"], "visita": a["team"]["displayName"],
                            "goles_local": gl, "goles_visita": gv, "estado": estado,
                        })
                    except Exception:
                        continue
                if filas_2026:
                    df_2026 = pd.DataFrame(filas_2026).drop_duplicates(subset=["fecha", "local", "visita"])
                    jugados_2026 = df_2026[df_2026.estado == "post"].dropna(subset=["goles_local", "goles_visita"])
                    fixture_2026 = df_2026[df_2026.estado == "pre"]
                    
                    part_hist = part[part.temporada < 2026]
                    part = pd.concat([part_hist, jugados_2026], ignore_index=True)
                    fixture = fixture_2026
        except Exception:
            pass # Usar cache de disco silenciosamente si hay fallas de red
            
    part = part.sort_values("fecha").reset_index(drop=True)
    fixture = fixture.sort_values("fecha").reset_index(drop=True)
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
    part["squad_value_diff"] = part.apply(lambda r: np.log(get_squad_value(r.local, r.temporada)) - np.log(get_squad_value(r.visita, r.temporada)), axis=1)
    part["resultado"] = np.where(part.goles_local > part.goles_visita, 2,
                                 np.where(part.goles_local == part.goles_visita, 1, 0))

    FEATS = ["elo_diff", "ppg_diff", "h2h_diff", "squad_value_diff"]
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    pipe.fit(part[FEATS], part.resultado)

    # Calcular el boost de goles anotados de local vs de visita para cada equipo
    boosts = {}
    for t in set(part.local.unique()) | set(part.visita.unique()):
        h_matches = part[part.local == t]
        a_matches = part[part.visita == t]
        if len(h_matches) > 3 and len(a_matches) > 3:
            boosts[t] = float(h_matches.goles_local.mean() - a_matches.goles_visita.mean())
        else:
            boosts[t] = 0.20  # valor por defecto

    # Determinar la mediana del boost para hacer un split binario óptimo (K=2 Quantiles)
    median_boost = float(np.median(list(boosts.values())))
    levels_dict = {eq: (1 if v >= median_boost else 0) for eq, v in boosts.items()}

    # Poisson de goles (con Niveles Binarios de Localía): dos obs por partido.
    largo = pd.concat([
        pd.DataFrame({
            "g": part.goles_local, "d": part.elo_local - part.elo_visita, "es_local": 1,
            "level": [levels_dict.get(x, 0) for x in part.local]
        }),
        pd.DataFrame({
            "g": part.goles_visita, "d": part.elo_visita - part.elo_local, "es_local": 0,
            "level": 0
        })
    ]).reset_index(drop=True)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    gp = smf.glm("g ~ d + es_local + es_local:level", data=largo, family=sm.families.Poisson()).fit()
    
    gp_params = {
        "const": float(gp.params["Intercept"]),
        "d": float(gp.params["d"]),
        "loc_0": float(gp.params["es_local"]),
        "loc_1": float(gp.params["es_local"] + gp.params["es_local:level"])
    }

    # estado actual de cada equipo (Elo + forma + h2h base)
    estado = {}
    for t in elo:
        estado[t] = {"elo": elo[t], "ppg": np.mean(ppg[t]) if ppg[t] else 1.0}

    # Panel de stats del partido (córners, tiros al arco, faltas, tarjetas, posesión) desde el box score
    stats_modelos, stats_estado = _entrenar_stats_chile(part)

    return {"part": part, "fixture": fixture, "elo": elo, "estado": estado, "h2h": h2h,
            "pipe": pipe, "FEATS": FEATS, "gp_params": gp_params, "levels_dict": levels_dict,
            "stats_modelos": stats_modelos, "stats_estado": stats_estado,
            "equipos_2026": sorted(set(part[part.temporada == 2026].local) | set(part[part.temporada == 2026].visita))}


# Stats del box score a modelar (objetivo -> conteo/posesión)
STATS_OBJ = ["wonCorners", "shotsOnTarget", "foulsCommitted", "yellowCards", "possessionPct", "xg"]
STATS_NOMBRE = {"wonCorners": "Córners", "shotsOnTarget": "Tiros al arco", "foulsCommitted": "Faltas",
                "yellowCards": "Tarjetas amarillas", "possessionPct": "Posesión %", "xg": "Goles esperados (xG Proxy)"}


def _entrenar_stats_chile(part):
    """Construye features point-in-time del box score y entrena un modelo por stat (Poisson para
       conteos, lineal para posesión). Devuelve (modelos, estado actual favor/concedido por equipo)."""
    bs_path = DATA / "box_score.csv"
    if not bs_path.exists():
        return {}, {}
    bs = pd.read_csv(bs_path, parse_dates=["fecha"])
    key = lambda d, a, b: (pd.Timestamp(d).normalize(), a, b)
    bsi = {key(r.fecha, r.local_equipo, r.visita_equipo): r for r in bs.itertuples(index=False)}
    V = 6
    favor = {s: defaultdict(lambda: deque(maxlen=V)) for s in STATS_OBJ}
    contra = {s: defaultdict(lambda: deque(maxlen=V)) for s in STATS_OBJ}
    rows = []
    for r in part.itertuples(index=False):
        a, b = r.local, r.visita
        rec = bsi.get(key(r.fecha, a, b))
        
        # Calcular xG sintético si el registro existe
        xg_l, xg_v = np.nan, np.nan
        if rec is not None:
            sot_l = getattr(rec, "local_shotsOnTarget", np.nan)
            tot_l = getattr(rec, "local_totalShots", np.nan)
            sot_v = getattr(rec, "visita_shotsOnTarget", np.nan)
            tot_v = getattr(rec, "visita_totalShots", np.nan)
            
            if not (pd.isna(sot_l) or pd.isna(tot_l)):
                xg_l = 0.30 * sot_l + 0.05 * max(0.0, tot_l - sot_l)
            if not (pd.isna(sot_v) or pd.isna(tot_v)):
                xg_v = 0.30 * sot_v + 0.05 * max(0.0, tot_v - sot_v)

        if rec is not None:
            fila = {"elo_diff": r.elo_local + HOME_ADV - r.elo_visita}
            for s in STATS_OBJ:
                fa = np.mean(favor[s][a]) if favor[s][a] else np.nan
                cb = np.mean(contra[s][b]) if contra[s][b] else np.nan
                fila[f"prop_{s}_f"], fila[f"riv_{s}_c"] = fa, cb
                
                if s == "xg":
                    vl = xg_l
                else:
                    vl = getattr(rec, f"local_{s}", np.nan)
                fila[f"t_{s}"] = vl
            rows.append(fila)
            
        if rec is not None:
            for s in STATS_OBJ:
                if s == "xg":
                    vl, vv = xg_l, xg_v
                else:
                    vl = getattr(rec, f"local_{s}", np.nan)
                    vv = getattr(rec, f"visita_{s}", np.nan)
                if not (pd.isna(vl) or pd.isna(vv)):
                    favor[s][a].append(vl); contra[s][a].append(vv)
                    favor[s][b].append(vv); contra[s][b].append(vl)
    D = pd.DataFrame(rows)
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    modelos = {}
    for s in STATS_OBJ:
        cols = ["elo_diff", f"prop_{s}_f", f"riv_{s}_c"]; tgt = f"t_{s}"
        d = D.dropna(subset=cols + [tgt])
        if s == "possessionPct":
            m = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]).fit(d[cols], d[tgt])
            modelos[s] = (m, cols, "lineal")
        else:
            m = sm.GLM(d[tgt], sm.add_constant(d[cols], has_constant="add"), family=sm.families.Poisson()).fit()
            modelos[s] = (m, cols, "poisson")
    estado = {}
    for s in STATS_OBJ:
        for t in set(list(favor[s].keys()) + list(contra[s].keys())):
            estado.setdefault(t, {})
            estado[t][f"{s}_f"] = float(np.mean(favor[s][t])) if favor[s][t] else np.nan
            estado[t][f"{s}_c"] = float(np.mean(contra[s][t])) if contra[s][t] else np.nan
    glob = {f"{s}_{d}": float(np.nanmean([estado[t].get(f"{s}_{d}", np.nan) for t in estado]))
            for s in STATS_OBJ for d in ("f", "c")}
    return modelos, {"equipo": estado, "glob": glob}


def stats_esperadas(M, local, visita):
    """Estadísticas esperadas del local y la visita (córners, tiros al arco, faltas, tarjetas, posesión, xG)."""
    import statsmodels.api as sm
    mods, est = M.get("stats_modelos", {}), M.get("stats_estado", {})
    if not mods:
        return {}
    eq, glob = est["equipo"], est["glob"]

    def pred(s, a, b):
        m, cols, tipo = mods[s]
        ea = eq.get(a, {}).get(f"{s}_f", glob[f"{s}_f"])
        cb = eq.get(b, {}).get(f"{s}_c", glob[f"{s}_c"])
        ea = glob[f"{s}_f"] if (ea != ea) else ea; cb = glob[f"{s}_c"] if (cb != cb) else cb
        X = pd.DataFrame([{"elo_diff": M["estado"][a]["elo"] + HOME_ADV - M["estado"][b]["elo"],
                           f"prop_{s}_f": ea, f"riv_{s}_c": cb}])[cols]
        if tipo == "poisson":
            return float(m.predict(sm.add_constant(X, has_constant="add"))[0])
        return float(m.predict(X)[0])

    res = {}
    for s in STATS_OBJ:
        res[s] = (pred(s, local, visita), pred(s, visita, local))
    # posesión renormalizada a 100
    pa, pb = res["possessionPct"]
    res["possessionPct"] = (pa / (pa + pb) * 100, pb / (pa + pb) * 100)
    return res


# --------------------------------------------------------------------------- #
#  Análisis riguroso de variables (VIF + forward + significancia + comparación)
# --------------------------------------------------------------------------- #
def analisis_variables(M, corte="2025-07-01"):
    """Justifica la elección de variables: construye un conjunto amplio de candidatas point-in-time
       y reporta VIF (multicolinealidad), correlación, qué elige el forward y cómo rinden distintos
       modelos en el hold-out temporal. Devuelve (df_candidatas, set_final, df_modelos)."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import log_loss, accuracy_score

    part = M["part"]
    elo = {}
    first_season = sorted(part.temporada.unique())[0]
    # Recrear la misma inicialización que calcular_elo
    seasons = sorted(part.temporada.unique())
    teams_by_season = {}
    for s in seasons:
        df_s = part[part.temporada == s]
        teams_by_season[s] = set(df_s.local.unique()) | set(df_s.visita.unique())
    for t in teams_by_season[first_season]:
        elo[t] = ELO_INIT

    ppg5, gf, gc = defaultdict(lambda: deque(maxlen=5)), defaultdict(lambda: deque(maxlen=5)), defaultdict(lambda: deque(maxlen=5))
    h2h = defaultdict(list); rows = []
    last_season = first_season

    for r in part.itertuples(index=False):
        a, b, gl, gv = r.local, r.visita, r.goles_local, r.goles_visita
        if r.temporada != last_season:
            for t in elo:
                elo[t] = 0.75 * elo[t] + 0.25 * ELO_INIT
            s_idx = seasons.index(r.temporada)
            prev_s = seasons[s_idx - 1]
            for t in teams_by_season[r.temporada]:
                if t not in elo or t not in teams_by_season[prev_s]:
                    elo[t] = 1420.0
            last_season = r.temporada

        if a not in elo: elo[a] = 1500.0
        if b not in elo: elo[b] = 1500.0

        par = tuple(sorted((a, b))); prev = h2h[par]
        hh = (np.mean([x if a == par[0] else -x for x in prev]) if prev else 0.0)
        rows.append({
            "elo_diff": elo[a] + HOME_ADV - elo[b],
            "ppg5_diff": (np.mean(ppg5[a]) if ppg5[a] else 1.0) - (np.mean(ppg5[b]) if ppg5[b] else 1.0),
            "gf_diff": (np.mean(gf[a]) if gf[a] else 1.2) - (np.mean(gf[b]) if gf[b] else 1.2),
            "gc_diff": (np.mean(gc[a]) if gc[a] else 1.2) - (np.mean(gc[b]) if gc[b] else 1.2),
            "h2h_diff": hh if a == par[0] else -hh,
            "squad_value_diff": np.log(get_squad_value(a, r.temporada)) - np.log(get_squad_value(b, r.temporada)),
            "fecha": r.fecha,
            "resultado": 2 if gl > gv else (1 if gl == gv else 0)})
        we = 1 / (1 + 10 ** (-((elo[a] + HOME_ADV) - elo[b]) / 400)); w = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        delta = K_LIGA * _mult_goles(gl - gv) * (w - we); elo[a] += delta; elo[b] -= delta
        rl, rv = (3, 0) if gl > gv else ((1, 1) if gl == gv else (0, 3))
        ppg5[a].append(rl); ppg5[b].append(rv); gf[a].append(gl); gc[a].append(gv); gf[b].append(gv); gc[b].append(gl)
        h2h[par].append(gl - gv)

    D = pd.DataFrame(rows)
    CAND = ["elo_diff", "ppg5_diff", "gf_diff", "gc_diff", "h2h_diff", "squad_value_diff"]
    tr, te = D[D.fecha < corte], D[D.fecha >= corte]; y = tr.resultado; cv = TimeSeriesSplit(5)

    Xv = sm.add_constant(tr[CAND])
    vif = {f: variance_inflation_factor(Xv.values, i + 1) for i, f in enumerate(CAND)}

    def ll(cols):
        p = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
        return -cross_val_score(p, tr[cols], y, cv=cv, scoring="neg_log_loss").mean()
    sel, rem, best = [], CAND[:], 99
    while rem:
        sc = {f: ll(sel + [f]) for f in rem}; bf = min(sc, key=sc.get)
        if sc[bf] < best - 0.001:
            sel.append(bf); rem.remove(bf); best = sc[bf]
        else:
            break

    df_cand = pd.DataFrame([{"variable": f, "corr_resultado": round(tr[[f, "resultado"]].corr().iloc[0, 1], 2),
                             "VIF": round(vif[f], 1), "elegida": "✅" if f in sel else "—"} for f in CAND])

    modelos = [("Logística (completa)", Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]), ["elo_diff", "ppg5_diff", "h2h_diff", "squad_value_diff"]),
               ("Logística (elo+forma+h2h)", Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]), ["elo_diff", "ppg5_diff", "h2h_diff"]),
               ("Logística (solo Elo)", Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]), ["elo_diff"]),
               ("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42), CAND),
               ("Gradient Boosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42), CAND)]
    filas = []
    for nom, mod, cols in modelos:
        mod.fit(tr[cols], y); P = mod.predict_proba(te[cols])
        filas.append({"Modelo": nom, "LogLoss": round(log_loss(te.resultado, P, labels=[0, 1, 2]), 4),
                      "Acierto": f"{accuracy_score(te.resultado, P.argmax(1)):.0%}"})
    base = np.tile(np.bincount(y) / len(y), (len(te), 1))
    filas.append({"Modelo": "Baseline frecuencias", "LogLoss": round(log_loss(te.resultado, base, labels=[0, 1, 2]), 4), "Acierto": "—"})
    return df_cand, sel, pd.DataFrame(filas)


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
            "h2h_diff": _h2h(M, local, visita),
            "squad_value_diff": np.log(get_squad_value(local, 2026)) - np.log(get_squad_value(visita, 2026))}


def prob_partido(M, local, visita):
    """[P(gana local), P(empate), P(gana visita)] con localía real del que juega de local."""
    p = M["pipe"].predict_proba(pd.DataFrame([features(M, local, visita)])[M["FEATS"]])[0]
    return np.array([p[2], p[1], p[0]])


def lambdas(M, local, visita):
    b = M["gp_params"]
    lvls = M.get("levels_dict", {})
    dl = M["estado"][local]["elo"] - M["estado"][visita]["elo"]
    dv = M["estado"][visita]["elo"] - M["estado"][local]["elo"]
    
    lvl = lvls.get(local, 0)
    loc_l = b["loc_1"] if lvl == 1 else b["loc_0"]
        
    la = float(np.exp(b["const"] + b["d"] * dl + loc_l))
    lb = float(np.exp(b["const"] + b["d"] * dv))
    return la, lb


def dixon_coles_adj(la, lb, rho, gmax=8):
    tau = np.ones((gmax + 1, gmax + 1))
    if la > 0 and lb > 0:
        tau[0, 0] = 1.0 - la * lb * rho
        tau[1, 0] = 1.0 + la * rho
        tau[0, 1] = 1.0 + lb * rho
        tau[1, 1] = 1.0 - rho
    return tau


def grilla(M, local, visita, gmax=8):
    p = prob_partido(M, local, visita)
    la, lb = lambdas(M, local, visita)
    g = np.arange(gmax + 1)
    grid = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb))
    # Aplicar ajuste de Dixon-Coles (rho = -0.12 para la liga chilena)
    tau = dixon_coles_adj(la, lb, -0.12, gmax)
    grid *= tau
    grid /= grid.sum()
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
    
    # Precalcular la grilla mixta de goles de cada partido para muestreo consistente
    MIX = {}
    for a, b in set(fix):
        mix_grid, _, _ = grilla(M, a, b)
        MIX[(a, b)] = mix_grid

    pos_count = {t: np.zeros(len(eq) + 1) for t in eq}     # histograma de posiciones
    campeon = defaultdict(int); copa = defaultdict(int); desc = defaultdict(int)
    pts_final = defaultdict(list)
    for _ in range(n_sims):
        pts = dict(pts0); gd = dict(gd0)
        for a, b in fix:
            mix_grid = MIX[(a, b)]
            flat_grid = mix_grid.flatten()
            idx = rng.choice(len(flat_grid), p=flat_grid)
            dim = mix_grid.shape[1]
            ga, gb = idx // dim, idx % dim
            
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
