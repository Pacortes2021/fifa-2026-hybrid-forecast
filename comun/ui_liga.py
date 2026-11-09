"""
Interfaz estándar de las ligas domésticas de fútbol (ver docs/revision/ui_estandar.md).

Cada `<liga>/app_<liga>.py` define un `CONFIG` y llama a `run_liga_app(mo, rec, rec_box, CONFIG)`.
Todas las ligas comparten así las mismas pestañas, el mismo orden de componentes y los mismos
cálculos de mercados; lo propio de cada liga (colores, equipos por defecto, columnas de la
proyección) va en `CONFIG`.

Motores soportados hoy:
  - E1  (eng, esp, bund): predecir_match -> p [L,E,V]; grilla_goles; validación con textos.
  - E1b (ita, fra, ned, por, bel): como E1, pero validación con enteros 0=local/1=empate/2=visita,
    evolución partidos/logloss y simulación con P_champions y Puntos esperados.
  - D   (tur, sco, aut, den, gre, nor, swe): predecir_match -> (p, la, lb) [L,E,V]; sin grilla_goles
    (se usa matriz_marcador_exacto reescalada al 1X2); claves features/df_dataset; tabla con dg/gf.
El adaptador (`_normalizar_*`) deja todo en un formato único para la UI.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

MODELOS = [
    ("stacking", "🔀 Stacking (combinación de modelos)"),
    ("rf", "🌲 Random Forest"),
    ("lasso", "📐 LASSO L1 (Regresión)"),
    ("xgb", "🚀 XGBoost (Gradient Boosting)"),
    ("elo", "📈 Solo Elo (regresión logística)"),
]
NOMBRE_CORTO = {"stacking": "Stacking", "rf": "Random Forest", "lasso": "LASSO L1", "xgb": "XGBoost", "elo": "Solo Elo"}
RESULTADOS = ["Gana local", "Empate", "Gana visita"]


# ─────────────────────────────── estilos y escudos ───────────────────────────────
def inyectar_estilos(acento, acento2):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
.main-title {{ text-align:center; font-size:2.6rem; font-weight:800;
    background:linear-gradient(135deg,{acento},{acento2}); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }}
.main-subtitle {{ text-align:center; font-size:1.05rem; color:#64748b; margin-bottom:1.6rem; }}
.card-title {{ font-size:1.2rem; font-weight:700; color:{acento};
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }}
.sec-title {{ font-size:1.5rem; font-weight:800; color:{acento}; margin:0.8rem 0 0.6rem 0; }}
.vs-text {{ text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    box-shadow:0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -4px rgba(0,0,0,0.05);
    border-radius:16px; border:1px solid #e2e8f0; }}
</style>
""", unsafe_allow_html=True)


def _info_equipo(equipos, team):
    for e in (equipos or {}).values():
        if team in (e.get("norm_name"), e.get("name"), e.get("short_name")):
            return e
    return {}


def logo_url(equipos, team):
    return _info_equipo(equipos, team).get("logo") or None


def logo_html(equipos, team, size=64):
    url = logo_url(equipos, team)
    return f'<img src="{url}" width="{size}" style="border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.15);">' if url else ""


def fmt_opcion(equipos, team):
    abbr = _info_equipo(equipos, team).get("abbreviation")
    return f"{abbr} · {team}" if abbr else team


def label_tabla(equipos, team):
    return team if logo_url(equipos, team) else f"⚽ {team}"


def cuota(p):
    return f"{1.0 / p:.2f}" if p and p > 0 else "—"


# ─────────────────────────────── mercados (desde la grilla) ───────────────────────────────
def mercados_desde_grilla(grid):
    """Over/Under, ambos marcan y marcadores más probables a partir de la matriz de goles
    (filas = goles del local, columnas = goles de la visita)."""
    g = np.asarray(grid, dtype=float)
    g = g / g.sum() if g.sum() > 0 else g
    i, j = np.indices(g.shape)
    mk = {}
    for linea in (1.5, 2.5, 3.5):
        over = float(g[(i + j) > linea].sum())
        mk[f"Over {linea}"] = over
        mk[f"Under {linea}"] = 1.0 - over
    mk["BTTS sí"] = float(g[(i > 0) & (j > 0)].sum())
    mk["BTTS no"] = 1.0 - mk["BTTS sí"]
    orden = np.dstack(np.unravel_index(np.argsort(-g, axis=None), g.shape))[0]
    mk["_top"] = [(int(a), int(b), float(g[a, b])) for a, b in orden[:6]]
    return mk


# ─────────────────────────────── adaptador del motor ───────────────────────────────
def _modelos_disponibles(M):
    return [(k, n) for k, n in MODELOS
            if k == "stacking" or f"pipe_{k}" in M or (k == "elo" and "elo_diff" in _dataset(M).columns)]


def _cols(M):
    return M["cols"] if "cols" in M else M["features"]


def _dataset(M):
    return M["df_features"] if "df_features" in M else M["df_dataset"]


def _ajustar_a_1x2(grid, p):
    """Reescala las zonas local/empate/visita de la grilla Poisson para que sumen el 1X2 del modelo
    (lo mismo que hace grilla_goles en las familias E1/E1b)."""
    g = np.asarray(grid, dtype=float).copy()
    i, j = np.indices(g.shape)
    for zona, objetivo in ((i > j, p[0]), (i == j, p[1]), (i < j, p[2])):
        s_zona = g[zona].sum()
        if s_zona > 0:
            g[zona] *= objetivo / s_zona
    return g / g.sum()


def _predecir(mo, M, a, b, modelo, temporada, elo=None):
    """(p [L,E,V], grilla, xG local, xG visita) con el contrato único de la UI."""
    if modelo == "elo":
        _, grid, la, lb = _predecir(mo, M, a, b, "stacking", temporada)
        p = _proba_elo(elo, [_elo_diff(M, a, b, temporada)])[0]
        return p, _ajustar_a_1x2(grid, p), la, lb
    r = mo.predecir_match(M, a, b, temporada=temporada, modelo=modelo)
    if isinstance(r, tuple):  # familias C y D
        p = np.asarray(r[0], dtype=float)
        la, lb = float(r[1]), float(r[2])
        if hasattr(mo, "grilla_goles"):
            gr = mo.grilla_goles(M, a, b, modelo=modelo)
            grid = gr[0] if isinstance(gr, tuple) else gr
            grid = _ajustar_a_1x2(grid, p)
        elif hasattr(mo, "matriz_marcador_exacto"):
            grid = _ajustar_a_1x2(mo.matriz_marcador_exacto(la, lb, 9), p)
        else:
            grid = np.zeros((6, 6))
        return p, grid, la, lb
    p = np.asarray(r, dtype=float)
    la, lb = _xg(M, a, b)
    return p, np.asarray(mo.grilla_goles(M, a, b, modelo=modelo), dtype=float), la, lb


def _tabla(mo, M):
    t = mo.obtener_tabla_actual(M).copy()
    t = t.rename(columns={
        "Selección": "equipo",
        "PTS": "puntos",
        "DG": "dif_goles",
        "dg": "dif_goles",
        "GF": "goles_favor",
        "gf": "goles_favor",
    })
    if "pj" not in t.columns:
        if {"PG", "PE", "PP"} <= set(t.columns):
            t["pj"] = t["PG"] + t["PE"] + t["PP"]
        elif {"g", "e", "p"} <= set(t.columns):
            t["pj"] = t["g"] + t["e"] + t["p"]
        else:
            t["pj"] = 0
    return t.sort_values(["puntos", "dif_goles", "goles_favor"], ascending=False).reset_index(drop=True)


# ─────────────── modelo "Solo Elo": referencia simple para comparar y aprender ───────────────
def _etiquetas_lev(df):
    """0 = gana local, 1 = empate, 2 = gana visita, a partir de los goles (no depende del motor)."""
    gl, gv = df["goles_local"].to_numpy(dtype=float), df["goles_visita"].to_numpy(dtype=float)
    return np.where(gl > gv, 0, np.where(gl == gv, 1, 2))


def _dataset_con_goles(M):
    """Dataset pre-partido con goles. Algunos motores (den, gre) no guardan los goles en el dataset:
    se toman de df_partidos cruzando por fecha, local y visita."""
    d = _dataset(M)
    if {"goles_local", "goles_visita"} <= set(d.columns) or "df_partidos" not in M:
        return d
    goles = M["df_partidos"][["fecha", "local", "visita", "goles_local", "goles_visita"]].copy()
    d = d.copy()
    d["_fk"], goles["_fk"] = pd.to_datetime(d["fecha"]), pd.to_datetime(goles["fecha"])
    return d.merge(goles.drop(columns="fecha"), on=["_fk", "local", "visita"], how="left").drop(columns="_fk")


@st.cache_resource(show_spinner=False)
def _entrenar_elo(_M, codigo, clave_datos, ultima_train=2024):
    """Regresión logística multinomial con una sola variable (elo_diff), entrenada con las mismas
    temporadas que los modelos finales (<= 2024) y evaluada fuera de muestra (>= 2025)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    df = _dataset_con_goles(_M)
    df = df[df["goles_local"].notna() & df["goles_visita"].notna() & df["elo_diff"].notna()]
    tr, te = df[df["temporada"] <= ultima_train], df[df["temporada"] > ultima_train]
    lr = LogisticRegression(max_iter=2000).fit(tr[["elo_diff"]].to_numpy(), _etiquetas_lev(tr))
    frec = np.bincount(_etiquetas_lev(tr), minlength=3) / len(tr)
    met = {}
    if len(te):
        pt = lr.predict_proba(te[["elo_diff"]].to_numpy())
        y = _etiquetas_lev(te)
        met = {"logloss": round(float(log_loss(y, pt, labels=[0, 1, 2])), 4),
               "accuracy": round(float((pt.argmax(1) == y).mean() * 100), 2)}
    return {"lr": lr, "met": met, "frec_train": frec}


def _elo_diff(M, a, b, temporada):
    """La misma variable elo_diff que usa el motor al entrenar (algunos motores incluyen la localía)."""
    t = M["tracker"]
    try:
        f = t.get_features_for_match(a, b, temporada, reset_season=False)
    except TypeError:
        f = t.get_features_for_match(a, b, temporada)
    return float(f["elo_diff"])


def _proba_elo(elo, x):
    return elo["lr"].predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))  # columnas [L, E, V]


def _validar_elo(M, elo, temporada):
    df = _dataset_con_goles(M)
    df = df[(df["temporada"] == temporada) & df["goles_local"].notna()].sort_values("fecha")
    if not len(df):
        return pd.DataFrame(), {}, pd.DataFrame()
    P = _proba_elo(elo, df["elo_diff"].fillna(0.0))
    y = _etiquetas_lev(df)
    out = pd.DataFrame({"fecha": pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d").to_numpy(),
                        "local": df["local"].to_numpy(), "visita": df["visita"].to_numpy(),
                        "goles_local": df["goles_local"].astype(int).to_numpy(), "goles_visita": df["goles_visita"].astype(int).to_numpy(),
                        "resultado": [RESULTADOS[k] for k in y], "Prediccion": [RESULTADOS[k] for k in P.argmax(1)],
                        "Prob_Local": P[:, 0], "Prob_Empate": P[:, 1], "Prob_Visita": P[:, 2]})
    pc = np.clip(P[np.arange(len(y)), y], 1e-7, 1)
    base = np.clip(elo["frec_train"][y], 1e-7, 1)
    met = {"n": len(y), "acierto": float((P.argmax(1) == y).mean()),
           "logloss": float(-np.log(pc).mean()), "logloss_base": float(-np.log(base).mean())}
    evol = pd.DataFrame({"partido": np.arange(1, len(y) + 1), "logloss_acum": np.cumsum(-np.log(pc)) / np.arange(1, len(y) + 1),
                         "baseline": met["logloss_base"]})
    return out, met, evol


def _metricas(M):
    return {k: v for k, v in (M.get("metricas") or {}).items() if isinstance(v, dict) and "logloss" in v}


def _xg(M, a, b):
    """Goles esperados del Poisson del motor (los mismos que usa grilla_goles: incluyen la localía)."""
    try:
        elo = M["tracker"].elos
        d = elo[a] - elo[b]
        la = float(np.exp(M["g_const"] + M["g_d"] * d + M.get("g_home", 0.0)))
        lb = float(np.exp(M["g_const"] - M["g_d"] * d))
        return la, lb
    except Exception:
        return None, None


def _normalizar_validacion(df_val, met, evol):
    if df_val is None or len(df_val) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()
    df = df_val.copy()
    if {"goles_local", "goles_visita"} <= set(df.columns):
        df["resultado"] = np.where(df["goles_local"] > df["goles_visita"], RESULTADOS[0],
                                   np.where(df["goles_local"] == df["goles_visita"], RESULTADOS[1], RESULTADOS[2]))
    elif "resultado" in df and pd.api.types.is_numeric_dtype(df["resultado"]):
        df["resultado"] = df["resultado"].map(lambda v: RESULTADOS[int(v)] if pd.notna(v) else v)

    if "Prediccion" in df:
        if {"Prob_Local", "Prob_Empate", "Prob_Visita"} <= set(df.columns):
            pl, pe, pv = df["Prob_Local"].to_numpy(dtype=float), df["Prob_Empate"].to_numpy(dtype=float), df["Prob_Visita"].to_numpy(dtype=float)
            pred_idx = np.where(pl >= np.maximum(pe, pv), 0, np.where(pe >= pv, 1, 2))
            df["Prediccion"] = [RESULTADOS[i] for i in pred_idx]
        elif pd.api.types.is_numeric_dtype(df["Prediccion"]):
            df["Prediccion"] = df["Prediccion"].map(lambda v: RESULTADOS[int(v)] if pd.notna(v) else v)

    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    ev = evol.copy() if evol is not None else pd.DataFrame()
    ev = ev.rename(columns={"partidos": "partido", "logloss": "logloss_acum"})
    if len(ev) and "baseline" not in ev and met:
        ev["baseline"] = met.get("logloss_base")
    return df, met or {}, ev


def _detalle_variables(mo, M, a, b, temporada):
    """(etiqueta, valor_a, valor_b, formato, menos_es_mejor) para las barras enfrentadas;
    cada variable se omite si el motor no la expone."""
    t = M["tracker"]
    filas = []

    def agregar(etq, fa, fmt=".1f", invertir=False):
        try:
            va, vb = float(fa(a)), float(fa(b))
            if np.isfinite(va) and np.isfinite(vb):
                filas.append((etq, va, vb, fmt, invertir))
        except Exception:
            pass

    media = lambda dq, n=5, d=0.0: float(np.mean(list(dq)[-n:])) if len(dq) else d
    agregar("⚡ ELO Rating", lambda x: t.elos[x], ".0f")
    agregar("💰 Valor plantilla (M€)", lambda x: mo.get_squad_value(x, temporada))
    agregar("🏆 Pi-Rating (local)", lambda x: t.pi_tracker.r_home.get(x, 0.0), ".3f")
    agregar("📈 Puntos por partido", lambda x: t.season_pts[x] / t.season_matches[x] if t.season_matches[x] else 1.33, ".2f")
    agregar("🔥 Forma últimos 5 (0–1)", lambda x: media(t.recent_results[x], d=0.333), ".3f")
    agregar("⚽ Goles a favor (últ. 5)", lambda x: media(t.recent_gf[x], d=1.0), ".2f")
    agregar("🛡️ Goles en contra (últ. 5)", lambda x: media(t.recent_ga[x], d=1.0), ".2f", True)
    agregar("🎂 Edad promedio", lambda x: mo.get_advanced_features(x, temporada).get("avg_age"), ".1f", True)
    agregar("🏟️ Capacidad del estadio", lambda x: mo.get_advanced_features(x, temporada).get("stadium_capacity"), ".0f")
    return filas


def _barra(va, vb, fmt, invertir):
    total = va + vb if (va + vb) > 0 else 1
    pa, pb = va / total, vb / total
    mejor_a = (pa >= pb) != invertir
    ca, cb = ("#10b981", "#9ca3af") if mejor_a else ("#9ca3af", "#10b981")
    return (f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
            f"<span style='min-width:75px;text-align:right;font-weight:700;color:{ca};font-size:.9rem;'>{va:{fmt}}</span>"
            f"<div style='flex:1;background:#e5e7eb;border-radius:8px;height:10px;overflow:hidden;'>"
            f"<div style='width:{pa*100:.1f}%;background:{ca};height:100%;float:left;'></div>"
            f"<div style='width:{pb*100:.1f}%;background:{cb};height:100%;float:right;'></div></div>"
            f"<span style='min-width:75px;font-weight:700;color:{cb};font-size:.9rem;'>{vb:{fmt}}</span></div>")


def _acelerar_pipeline_rf(pipe_rf):
    """Evita el overhead de joblib.Parallel (~18ms por fila) en inferencia individual sobre Random Forest."""
    if pipe_rf is None or not hasattr(pipe_rf, "steps") or len(pipe_rf.steps) != 2:
        return
    sc, rf = pipe_rf.steps[0][1], pipe_rf.steps[1][1]
    if not hasattr(sc, "mean_") or not hasattr(rf, "estimators_"):
        return
    trees = [est.tree_ for est in rf.estimators_]
    mean = sc.mean_.astype(np.float32)
    scale = sc.scale_.astype(np.float32)
    orig_predict_proba = pipe_rf.predict_proba

    def _fast_predict_proba(X):
        if len(X) <= 8:
            arr = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
            x = (arr - mean) / scale
            out = sum(t.predict(x) for t in trees)
            if out.ndim == 3:
                out = out[:, 0, :]
            return out / out.sum(axis=1, keepdims=True)
        return orig_predict_proba(X)

    pipe_rf.predict_proba = _fast_predict_proba


# ─────────────────────────────── cachés ───────────────────────────────
@st.cache_resource(show_spinner="Cargando el modelo (la primera vez puede tardar un minuto)...")
def _cargar_motor(_mo, codigo):
    M = _mo.cargar()
    # Desactivar paralelismo joblib durante inferencia individual (evita ~30-50 ms de overhead por llamada)
    for k in ("pipe_rf", "pipe_xgb", "pipe_lasso", "pipe", "pipe_val"):
        pipe = M.get(k)
        if pipe is not None:
            est = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
    _acelerar_pipeline_rf(M.get("pipe_rf"))
    return M


@st.cache_data(show_spinner=False)
def _leer_partidos(path_str, mtime):
    return pd.read_csv(path_str, parse_dates=["fecha"])


@st.cache_data(show_spinner=False)
def _leer_fixture(path_str, mtime):
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def _tabla_cached(_mo, _M, codigo, clave_datos):
    return _tabla(_mo, _M)


@st.cache_data(show_spinner="Corriendo la simulación de Monte Carlo...")
def _simular(_mo, _M, codigo, clave_datos, modelo, n_sims):
    try:
        df = _mo.simular_campeonato(_M, n_sims=n_sims, modelo=modelo)
    except TypeError:
        df = _mo.simular_campeonato(_M, n_sims=n_sims)
    if "Selección" in df.columns and "equipo" not in df.columns:
        df = df.rename(columns={"Selección": "equipo"})
    return df


@st.cache_data(show_spinner=False)
def _validar(_mo, _M, codigo, clave_datos, temporada, modelo):
    """Evaluación fuera de muestra 100% vectorizada sobre el dataset pre-partido (~10 ms vs ~15 s fila por fila)."""
    try:
        d = _dataset_con_goles(_M)
        df = d[(d["temporada"] == temporada) & d["goles_local"].notna()].sort_values("fecha")
        if not len(df):
            return pd.DataFrame(), {}, pd.DataFrame()
        cols = _cols(_M)
        X = df[cols].fillna(0.0)
        if modelo == "stacking":
            w = _M.get("stack_w", np.array([_M.get("alpha_stack", 0.5), 1.0 - _M.get("alpha_stack", 0.5), 0.0]))
            P = w[0] * _M["pipe_lasso"].predict_proba(X) + w[1] * _M["pipe_rf"].predict_proba(X)
            if len(w) > 2 and w[2] > 0 and "pipe_xgb" in _M:
                P = P + w[2] * _M["pipe_xgb"].predict_proba(X)
        elif f"pipe_{modelo}" in _M:
            P = _M[f"pipe_{modelo}"].predict_proba(X)
        else:
            P = (_M.get("pipe_val") or _M.get("pipe_lasso") or _M.get("pipe_rf")).predict_proba(X)

        hw = d[d["goles_local"] > d["goles_visita"]]
        if "resultado" in d.columns and len(hw) and hw["resultado"].median() == 2:
            P = P[:, ::-1]  # normalizar a [Local, Empate, Visita]
        P = np.clip(P, 1e-7, 1.0 - 1e-7)
        P = P / P.sum(axis=1, keepdims=True)

        y = _etiquetas_lev(df)
        tr = d[(d["temporada"] <= 2024) & d["goles_local"].notna()]
        if not len(tr):
            tr = df
        y_tr = _etiquetas_lev(tr)
        frec_tr = np.bincount(y_tr, minlength=3) / max(len(y_tr), 1)

        out = pd.DataFrame({
            "fecha": pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "local": df["local"].to_numpy(),
            "visita": df["visita"].to_numpy(),
            "goles_local": df["goles_local"].astype(int).to_numpy(),
            "goles_visita": df["goles_visita"].astype(int).to_numpy(),
            "resultado": [RESULTADOS[k] for k in y],
            "Prediccion": [RESULTADOS[k] for k in P.argmax(1)],
            "Prob_Local": P[:, 0],
            "Prob_Empate": P[:, 1],
            "Prob_Visita": P[:, 2],
        })
        pc = np.clip(P[np.arange(len(y)), y], 1e-7, 1.0)
        base = np.clip(frec_tr[y], 1e-7, 1.0)
        met = {
            "n": len(y),
            "acierto": float((P.argmax(1) == y).mean()),
            "logloss": float(-np.log(pc).mean()),
            "logloss_base": float(-np.log(base).mean()),
        }
        evol = pd.DataFrame({
            "partido": np.arange(1, len(y) + 1),
            "logloss_acum": np.cumsum(-np.log(pc)) / np.arange(1, len(y) + 1),
            "baseline": met["logloss_base"],
        })
        return out, met, evol
    except Exception:
        try:
            res = _mo.validacion_en_vivo(_M, temporada_val=temporada, modelo_tipo=modelo)
        except TypeError:
            res = _mo.validacion_en_vivo(_M, temporada_val=temporada)
        return _normalizar_validacion(*(res[:3] if res else (None, None, None)))


# ─────────────────────────────── página ───────────────────────────────
def run_liga_app(mo, rec, rec_box, cfg):
    cod = cfg["codigo"]
    acento = cfg.get("acento", "#1e3a8a")
    inyectar_estilos(acento, cfg.get("acento2", "#64748b"))

    M = _cargar_motor(mo, cod)
    equipos = M.get("equipos", {})
    p_path = mo.DATA / "partidos.csv"
    partidos = _leer_partidos(str(p_path), p_path.stat().st_mtime if p_path.exists() else 0)
    clave_datos = f"{len(partidos)}-{partidos['fecha'].max()}"
    temporada = mo._temporada_actual()

    # ── Sidebar
    st.sidebar.markdown("### 🛠️ Controles del Modelo")
    disponibles = _modelos_disponibles(M)
    met_all = _metricas(M)
    elo = _entrenar_elo(M, cod, clave_datos) if any(k == "elo" for k, _ in disponibles) else None
    if elo and elo["met"]:
        met_all["elo"] = elo["met"]
    con_met = [k for k, _ in disponibles if k in met_all]
    mejor = min(con_met, key=lambda k: met_all[k]["logloss"]) if con_met else "stacking"
    etiquetas = [n + (" ⭐" if k == mejor else "") for k, n in disponibles]
    sel = st.sidebar.selectbox("🤖 Modelo predictivo:", etiquetas, index=[k for k, _ in disponibles].index(mejor),
                               key=f"modelo_{cod}", help="⭐ = menor log-loss fuera de muestra")
    modelo = disponibles[etiquetas.index(sel)][0]

    with st.sidebar.expander("🔄 Actualización de datos ESPN"):
        if not st.session_state.get("portal_admin", True):
            st.caption("Los datos se actualizan automáticamente todos los días desde ESPN.")
        elif st.button("Descargar partidos y re-entrenar", key=f"refresh_{cod}"):
            with st.spinner("Descargando de ESPN..."):
                rec.recolectar()
                rec_box.recolectar()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    st.sidebar.caption(f"🗓️ Datos al {partidos['fecha'].max():%d/%m/%Y} · {len(partidos)} partidos")
    if met_all:
        st.sidebar.markdown("#### 📊 Métricas fuera de muestra")
        for k, _ in disponibles:
            if k in met_all:
                m = met_all[k]
                w = f" (w={m['w']})" if k == "stacking" and "w" in m else ""
                st.sidebar.caption(f"**{NOMBRE_CORTO[k]}{w}{' ⭐' if k == mejor else ''}** — "
                                   f"Log-loss `{m['logloss']:.4f}` · Acierto `{m['accuracy']:.1f}%`")

    # ── Encabezado
    st.markdown(f'<div class="main-title">{cfg["bandera"]} Portal de Predicción {cfg["nombre"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Modelo activo: <b>{NOMBRE_CORTO[modelo]}</b> — LASSO, Random Forest, '
                f'XGBoost y Stacking · Poisson para goles · Monte Carlo para la temporada</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["⚽ Predicción Versus", "📊 Tabla y Proyecciones",
                                      "🔬 Importancia de Variables", "🎯 Validación vs Realidad"])
    with tab1:
        _tab_versus(mo, M, cfg, modelo, equipos, met_all, disponibles, partidos, temporada, clave_datos, elo)
    with tab2:
        _tab_tabla(mo, M, cfg, modelo, equipos, clave_datos)
    with tab3:
        _tab_importancia(M, modelo, acento, met_all, elo)
    with tab4:
        _tab_validacion(mo, M, cfg, modelo, clave_datos, acento, elo)


@st.fragment
def _tab_versus(mo, M, cfg, modelo, equipos, met_all, disponibles, partidos, temporada, clave_datos, elo=None):
    cod = cfg["codigo"]
    st.markdown('<div class="sec-title">Analizador de Enfrentamientos</div>', unsafe_allow_html=True)

    tabla = _tabla_cached(mo, M, cod, clave_datos)
    opciones = sorted(tabla["equipo"]) if len(tabla) else sorted(_dataset(M)["local"].unique())

    fix_path = mo.DATA / "fixture.csv"
    fix = _leer_fixture(str(fix_path), fix_path.stat().st_mtime) if fix_path.exists() else pd.DataFrame()
    if len(fix):
        fix = fix.assign(fecha_dt=pd.to_datetime(fix["fecha"])).sort_values("fecha_dt").head(25)
        mapa = {f"📅 {r.fecha_dt:%d/%m %H:%M} | {r.local} vs {r.visita}": (r.local, r.visita) for r in fix.itertuples()}

        def _cargar_fixture():
            par = mapa.get(st.session_state.get(f"fix_{cod}"))
            if par and par[0] in opciones and par[1] in opciones:
                st.session_state[f"a_{cod}"], st.session_state[f"b_{cod}"] = par

        st.selectbox("⚡ Cargar partido de la próxima fecha:", ["— Seleccionar partido programado —"] + list(mapa),
                     key=f"fix_{cod}", on_change=_cargar_fixture)

    da, db = cfg.get("equipos_default", (None, None))
    ia = opciones.index(da) if da in opciones else 0
    ib = opciones.index(db) if db in opciones else min(1, len(opciones) - 1)
    c1, cvs, c2 = st.columns([5, 1, 5])
    with c1:
        a = st.selectbox("Equipo local", opciones, index=ia, key=f"a_{cod}", format_func=lambda t: fmt_opcion(equipos, t))
        st.markdown(f'<div style="text-align:center">{logo_html(equipos, a)}</div>', unsafe_allow_html=True)
    with cvs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with c2:
        b = st.selectbox("Equipo visitante", opciones, index=ib, key=f"b_{cod}", format_func=lambda t: fmt_opcion(equipos, t))
        st.markdown(f'<div style="text-align:center">{logo_html(equipos, b)}</div>', unsafe_allow_html=True)
    if a == b:
        st.error("Selecciona dos equipos distintos.")
        return

    p, grid, la, lb = _predecir(mo, M, a, b, modelo, temporada, elo)
    mk = mercados_desde_grilla(grid)

    # Alertas cualitativas (sin porcentajes de acierto fijos)
    if p[0] > 0.55 and la is not None and (la - lb) > 1.0:
        st.success(f"🔥 **Alta confianza:** el modelo (>55%) y el Poisson (>1 gol de diferencia) coinciden en **{a}**.")
    elif p[0] > 0.60:
        st.info(f"💪 **Favorito claro:** más de 60% de probabilidad de victoria para **{a}**.")
    elif p[2] > 0.50:
        st.warning(f"⚠️ **Visita fuerte:** más de 50% de probabilidad de victoria para **{b}**, algo poco frecuente.")

    col_p, col_g = st.columns(2)
    with col_p:
        st.markdown('<div class="card-title">Probabilidades 1X2</div>', unsafe_allow_html=True)
        for etq, prob in ((f"Victoria {a}", p[0]), ("Empate", p[1]), (f"Victoria {b}", p[2])):
            st.markdown(f"**{etq}: {prob:.1%}** · cuota justa `{cuota(prob)}`")
            st.progress(float(np.clip(prob, 0, 1)))
    with col_g:
        st.markdown('<div class="card-title">Goles esperados y marcadores</div>', unsafe_allow_html=True)
        if la is not None:
            g1, g2 = st.columns(2)
            g1.metric(f"xG {a}", f"{la:.2f}")
            g2.metric(f"xG {b}", f"{lb:.2f}")
        st.markdown("🎯 **Marcadores más probables:**")
        for x, y, pr in mk["_top"][:4]:
            st.markdown(f"* `{x} - {y}`: **{pr:.1%}** (cuota `{cuota(pr)}`)")

    with st.expander("🤖 Comparativa de modelos para este partido", expanded=False):
        filas = []
        for k, nombre in disponibles:
            pk = _predecir(mo, M, a, b, k, temporada, elo)[0]
            m = met_all.get(k, {})
            filas.append({"Modelo": nombre, f"Victoria {a}": f"{pk[0]:.1%}", "Empate": f"{pk[1]:.1%}",
                          f"Victoria {b}": f"{pk[2]:.1%}",
                          "Log-loss (fuera de muestra)": f"{m['logloss']:.4f}" if m else "—",
                          "Acierto": f"{m['accuracy']:.1f}%" if m else "—"})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    st.markdown('<div class="sec-title">Mercados</div>', unsafe_allow_html=True)
    filas = [{"Mercado": f"{lado} {ln} goles", "Prob.": f"{mk[f'{lado} {ln}']:.1%}", "Cuota justa": cuota(mk[f"{lado} {ln}"])}
             for ln in (1.5, 2.5, 3.5) for lado in ("Over", "Under")]
    filas += [{"Mercado": "Ambos marcan: Sí", "Prob.": f"{mk['BTTS sí']:.1%}", "Cuota justa": cuota(mk["BTTS sí"])},
              {"Mercado": "Ambos marcan: No", "Prob.": f"{mk['BTTS no']:.1%}", "Cuota justa": cuota(mk["BTTS no"])}]
    dnb = p[0] + p[2]
    for etq, pr in ((f"Doble oportunidad: {a} o empate (1X)", p[0] + p[1]), (f"Doble oportunidad: {b} o empate (X2)", p[2] + p[1]),
                    ("Doble oportunidad: sin empate (12)", p[0] + p[2]),
                    (f"Apuesta sin empate: {a}", p[0] / dnb if dnb else 0.5), (f"Apuesta sin empate: {b}", p[2] / dnb if dnb else 0.5)):
        filas.append({"Mercado": etq, "Prob.": f"{pr:.1%}", "Cuota justa": cuota(pr)})
    m1, m2 = st.columns(2)
    m1.dataframe(pd.DataFrame(filas[:8]), hide_index=True, width="stretch")
    m2.dataframe(pd.DataFrame(filas[8:]), hide_index=True, width="stretch")

    st.markdown('<div class="sec-title">Matriz de goles exactos (Poisson ajustado al 1X2)</div>', unsafe_allow_html=True)
    m6 = np.asarray(grid)[:6, :6]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(m6, cmap=cfg.get("cmap", "Reds"))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel(f"Goles de {b}"); ax.set_ylabel(f"Goles de {a}")
    fig.colorbar(im, ax=ax, label="Probabilidad")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{m6[i, j]:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if m6[i, j] > m6.max() * 0.6 else "black")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("🔍 Detalle de variables del modelo", expanded=False):
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-weight:800;padding:6px 0 8px 0;"
                    f"border-bottom:2px solid #e5e7eb;margin-bottom:8px;'><span>🏠 {a}</span>"
                    f"<span style='color:#6b7280;font-size:.8rem;'>Variable</span><span>✈️ {b}</span></div>", unsafe_allow_html=True)
        for etq, va, vb, fmt, inv in _detalle_variables(mo, M, a, b, temporada):
            c_l, c_b = st.columns([1.6, 3.4])
            c_l.markdown(f"<span style='font-size:.83rem;color:#374151;'>{etq}</span>", unsafe_allow_html=True)
            c_b.markdown(_barra(va, vb, fmt, inv), unsafe_allow_html=True)
        try:
            st.caption(f"📍 Distancia de viaje del visitante: {mo.get_distance_km(a, b):.0f} km")
        except Exception:
            pass

    st.markdown('<div class="sec-title">⚔️ Historial de enfrentamientos directos</div>', unsafe_allow_html=True)
    h = partidos[((partidos.local == a) & (partidos.visita == b)) | ((partidos.local == b) & (partidos.visita == a))]
    if h.empty:
        st.info(f"No hay enfrentamientos entre **{a}** y **{b}** en los datos.")
        return
    h = h.sort_values("fecha", ascending=False)
    ga = np.where(h.local == a, h.goles_local, h.goles_visita)
    gb = np.where(h.local == a, h.goles_visita, h.goles_local)
    va, vb = int((ga > gb).sum()), int((gb > ga).sum())
    n = len(h)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Partidos", n)
    k2.metric(f"Victorias {a}", f"{va} ({va / n:.0%})", f"{int(ga.sum())} goles")
    k3.metric("Empates", f"{n - va - vb} ({(n - va - vb) / n:.0%})")
    k4.metric(f"Victorias {b}", f"{vb} ({vb / n:.0%})", f"{int(gb.sum())} goles")
    st.dataframe(pd.DataFrame({"Fecha": h.fecha.dt.strftime("%d/%m/%Y"), "Local": h.local,
                               "Marcador": h.goles_local.astype(int).astype(str) + " - " + h.goles_visita.astype(int).astype(str),
                               "Visitante": h.visita}).head(8), hide_index=True, width="stretch")


@st.fragment
def _tab_tabla(mo, M, cfg, modelo, equipos, clave_datos):
    st.markdown('<div class="sec-title">Tabla y proyección de la temporada</div>', unsafe_allow_html=True)
    col_t, col_p = st.columns(2)
    tabla = _tabla_cached(mo, M, cfg["codigo"], clave_datos).copy()
    tabla.insert(0, "Pos", range(1, len(tabla) + 1))
    tabla["Escudo"] = tabla["equipo"].map(lambda t: logo_url(equipos, t))
    tabla["Equipo"] = tabla["equipo"].map(lambda t: label_tabla(equipos, t))
    vis = tabla[["Pos", "Escudo", "Equipo", "pj", "puntos", "dif_goles", "goles_favor"]].rename(
        columns={"pj": "PJ", "puntos": "PTS", "dif_goles": "DG", "goles_favor": "GF"})
    with col_t:
        st.markdown('<div class="card-title">Tabla de posiciones actual</div>', unsafe_allow_html=True)
        st.dataframe(vis, hide_index=True, width="stretch", height=560,
                     column_config={"Escudo": st.column_config.ImageColumn("", width="small")})

    n_sims = cfg.get("n_sims", 50000)
    modelo_sim = "stacking" if modelo == "elo" else modelo  # la simulación del motor no conoce "Solo Elo"
    proy = _simular(mo, M, cfg["codigo"], clave_datos, modelo_sim, n_sims).copy()
    cols = [(c, etq, cmap) for c, etq, cmap in cfg["columnas_proy"] if c in proy.columns]
    proy["Escudo"] = proy["equipo"].map(lambda t: logo_url(equipos, t))
    proy["Equipo"] = proy["equipo"].map(lambda t: label_tabla(equipos, t))
    extra = ["Puntos esperados"] if "Puntos esperados" in proy.columns else []
    vis_p = proy[["Escudo", "Equipo"] + extra + [c for c, _, _ in cols]].rename(columns={c: etq for c, etq, _ in cols})
    estilo = vis_p.style.format({etq: "{:.1%}" for _, etq, _ in cols} | ({"Puntos esperados": "{:.1f}"} if extra else {}))
    for _, etq, cmap in cols:
        if cmap:
            estilo = estilo.background_gradient(subset=[etq], cmap=cmap)
    with col_p:
        st.markdown('<div class="card-title">Proyección Monte Carlo</div>', unsafe_allow_html=True)
        st.caption(f"{n_sims:,} simulaciones de los partidos que faltan con el modelo {NOMBRE_CORTO[modelo_sim]}.".replace(",", ".")
                   + (" (Solo Elo no está integrado en la simulación de temporada.)" if modelo == "elo" else ""))
        if cfg.get("aviso_proyeccion"):
            st.caption(cfg["aviso_proyeccion"])
        st.dataframe(estilo, hide_index=True, width="stretch", height=560,
                     column_config={"Escudo": st.column_config.ImageColumn("", width="small")})
    st.info(cfg.get("reglas") or f"Copas internacionales: top {getattr(mo, 'CUPOS_COPA', '—')} · Descenso: últimos {getattr(mo, 'DESCIENDEN', '—')}.")


def _tab_importancia_elo(M, elo, acento):
    st.markdown('<div class="sec-title">Cómo decide el modelo "Solo Elo"</div>', unsafe_allow_html=True)
    st.markdown("Usa **una sola variable**: la diferencia de Elo entre local y visita. Una regresión logística la convierte "
                "en probabilidades de victoria local, empate y victoria visitante. Sirve de referencia: si un modelo con "
                "decenas de variables no le gana, esas variables aportan poco.")
    x = np.linspace(-500, 500, 201)
    P = _proba_elo(elo, x)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (etq, col) in enumerate((("Gana local", acento), ("Empate", "#94a3b8"), ("Gana visita", "#0f172a"))):
        ax.plot(x, P[:, k], label=etq, color=col, lw=2)
    ax.axvline(0, color="#cbd5e1", lw=1)
    ax.set_xlabel("Diferencia de Elo (local − visita)"); ax.set_ylabel("Probabilidad"); ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    p0 = P[100]
    st.caption(f"Con equipos parejos (diferencia 0): local {p0[0]:.0%}, empate {p0[1]:.0%}, visita {p0[2]:.0%}. "
               "La ventaja de ser local aparece aunque el Elo sea igual.")
    coef = pd.DataFrame({"Resultado": RESULTADOS, "Coeficiente de elo_diff": np.round(elo["lr"].coef_[:, 0], 5),
                         "Intercepto": np.round(elo["lr"].intercept_, 3)})
    st.dataframe(coef, hide_index=True, width="stretch")


@st.fragment
def _tab_importancia(M, modelo, acento, met_all, elo=None):
    if modelo == "elo":
        _tab_importancia_elo(M, elo, acento)
        return
    cols = _cols(M)
    base = "rf" if modelo == "stacking" else modelo
    if modelo == "stacking":
        w = met_all.get("stacking", {}).get("w")
        st.info(f"El **Stacking** combina LASSO, Random Forest y XGBoost (pesos {w}). Se muestra la importancia del "
                "Random Forest; elige otro modelo en la barra lateral para ver la suya.")
    pipe = M[f"pipe_{base}"]
    est = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    if hasattr(est, "coef_"):
        valores, etq = np.mean(np.abs(est.coef_), axis=0), "Peso absoluto (LASSO)"
        texto = "LASSO (L1) lleva a cero los coeficientes de las variables poco informativas: solo quedan las que pesan fuera de muestra."
    else:
        valores = est.feature_importances_
        etq = "Importancia (Gini)" if base == "rf" else "Importancia (ganancia XGB)"
        texto = ("Random Forest: reducción promedio de la impureza de Gini que aporta cada variable."
                 if base == "rf" else "XGBoost: mejora promedio de la pérdida cada vez que un árbol divide por la variable.")
    st.markdown(f'<div class="sec-title">Importancia de variables: {NOMBRE_CORTO[base]}</div>', unsafe_allow_html=True)
    st.markdown(texto)
    df = pd.DataFrame({"Variable": cols, etq: np.round(valores, 4)})
    df = df[df[etq] > 1e-4].sort_values(etq, ascending=False).reset_index(drop=True)
    c_t, c_g = st.columns([5, 7])
    c_t.dataframe(df, hide_index=True, width="stretch")
    with c_g:
        fig, ax = plt.subplots(figsize=(6, 5))
        top = df.head(15)
        ax.barh(top["Variable"][::-1], top[etq][::-1], color=acento)
        ax.set_title(f"Top 15 variables ({NOMBRE_CORTO[base]})")
        st.pyplot(fig)
        plt.close(fig)


@st.fragment
def _tab_validacion(mo, M, cfg, modelo, clave_datos, acento, elo=None):
    cod = cfg["codigo"]
    st.markdown('<div class="sec-title">El modelo contra la realidad (fuera de muestra)</div>', unsafe_allow_html=True)
    st.markdown(f"Cada partido se predice con **{NOMBRE_CORTO[modelo]}** usando solo la información disponible antes de jugarlo.")
    temporadas = sorted([t for t in _dataset(M)["temporada"].unique() if t >= cfg.get("primer_test", 2025)], reverse=True)
    if not temporadas:
        st.info("Aún no hay temporadas fuera de muestra para validar.")
        return
    temp = st.selectbox("Temporada a validar:", temporadas, key=f"val_temp_{cod}")
    df, met, evol = (_validar_elo(M, elo, int(temp)) if modelo == "elo"
                     else _validar(mo, M, cod, clave_datos, int(temp), modelo))
    if len(df) == 0:
        st.info("Aún no hay partidos jugados en esta temporada.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Partidos", met["n"])
    k2.metric("Acierto (1X2)", f"{met['acierto']:.1%}")
    k3.metric("Log-loss del modelo", f"{met['logloss']:.3f}", f"{met['logloss'] - met['logloss_base']:+.3f} vs baseline",
              delta_color="inverse")
    k4.metric("Log-loss baseline", f"{met['logloss_base']:.3f}")
    if met["logloss"] < met["logloss_base"]:
        st.success(f"El modelo supera al baseline en {met['n']} partidos reales. 👍")
    else:
        st.warning("El modelo no supera al baseline en esta muestra; con pocos partidos la diferencia puede ser ruido.")

    equipos_val = sorted(set(df.local) | set(df.visita))
    filtro = st.selectbox("Filtrar por equipo:", ["Todos"] + equipos_val, key=f"val_eq_{cod}")
    vis = df if filtro == "Todos" else df[(df.local == filtro) | (df.visita == filtro)]
    show = vis[["fecha", "local", "visita", "goles_local", "goles_visita", "resultado", "Prediccion",
                "Prob_Local", "Prob_Empate", "Prob_Visita"]].copy()
    show["Acierto"] = np.where(show["resultado"] == show["Prediccion"], "✅", "❌")
    for c in ("Prob_Local", "Prob_Empate", "Prob_Visita"):
        show[c] = show[c].map(lambda x: f"{x:.1%}")
    st.dataframe(show.rename(columns={"Prediccion": "Predicción"}), hide_index=True, width="stretch")

    c_plot, c_eq = st.columns([6, 4])
    with c_plot:
        if len(evol) >= 3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(evol["partido"], evol["logloss_acum"], "-", color=acento, label="Modelo (acumulado)")
            ax.axhline(met["logloss_base"], color="#475569", ls="--", lw=1.5, label="Baseline")
            ax.set_xlabel("Partidos (orden cronológico)"); ax.set_ylabel("Log-loss acumulado")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
    with c_eq:
        st.markdown("##### % de acierto por equipo")
        filas = []
        for eq in equipos_val:
            d = df[(df.local == eq) | (df.visita == eq)]
            casa, ruta = d[d.local == eq], d[d.visita == eq]
            ac = lambda x: (x.resultado == x.Prediccion).mean() if len(x) else np.nan
            filas.append({"Equipo": eq, "Partidos": len(d), "% Acierto": ac(d), "Local": ac(casa), "Visita": ac(ruta)})
        tab = pd.DataFrame(filas).sort_values("% Acierto", ascending=False)
        st.dataframe(tab.style.format({"% Acierto": "{:.0%}", "Local": "{:.0%}", "Visita": "{:.0%}"}, na_rep="—"),
                     hide_index=True, width="stretch")
