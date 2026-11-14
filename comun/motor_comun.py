"""
Bloques compartidos entre los motores predictivos (`<liga>/motor.py`) de las 19 ligas domésticas
y torneos internacionales:
  - Constantes y defaults serializables para `defaultdict` en pickle (`_elo_default`, `_none_default`)
  - Cálculo geodésico (`haversine_km`) y carga de `equipos.csv` (`cargar_equipos_csv`)
  - Trackers dinámicos de Pi-Ratings (`PiRatingTracker` lineal y `PiRatingsTracker` exponencial Constantinou)
  - Reconstrucción point-in-time sin fuga de datos para validación (`_TrackerPrePartido`, `_modelo_pre_partido`)
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ELO_INIT = 1500.0


def _elo_default():
    return ELO_INIT


def _none_default():
    return None


def cargar_equipos_csv(equipos_path):
    """Lee data/equipos.csv (id estable de ESPN -> info del club: nombre, abreviatura, colores, logo)."""
    p = Path(equipos_path)
    if p.exists():
        try:
            df = pd.read_csv(p).set_index("id")
            return df.to_dict("index")
        except Exception:
            return {}
    return {}


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia ortodrómica en kilómetros entre dos coordenadas (lat, lon)."""
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


class PiRatingTracker:
    """Tracker de Pi-Ratings con actualización lineal sobre el margen de goles."""

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
            "pi_overall_diff": pi_overall_l - pi_overall_v,
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


class PiRatingsTracker:
    """Tracker de Pi-Ratings (Constantinou & Fenton) con amortiguación exponencial."""

    def __init__(self, lambda_param=0.035, gamma_param=0.55):
        self.lmb = lambda_param
        self.gamma = gamma_param
        self.r_home = defaultdict(float)
        self.r_away = defaultdict(float)

    def get_features(self, home, away):
        rh = self.r_home[home]
        ra = self.r_away[away]
        return {
            "pi_diff": rh - ra,
            "pi_overall_diff": (rh + self.r_away[home]) - (ra + self.r_home[away]),
        }

    def registrar_partido(self, home, away, gh, ga):
        gd = gh - ga
        rh = self.r_home[home]
        ra = self.r_away[away]
        diff = rh - ra
        e_gd = (10.0 ** (abs(diff) / 3.0) - 1.0) * (1.0 if diff >= 0 else -1.0)
        err = gd - e_gd
        self.r_home[home] += err * self.lmb
        self.r_away[home] += err * self.lmb * self.gamma
        self.r_away[away] -= err * self.lmb
        self.r_home[away] -= err * self.lmb * self.gamma


class _TrackerPrePartido:
    """Tracker de solo lectura que entrega las variables calculadas ANTES de un partido
    (las mismas filas usadas en el entrenamiento). La validación lo usa en vez del tracker
    final, que ya incluye el resultado del partido evaluado y el de todos los posteriores."""

    def __init__(self, tracker, feats):
        self._tracker = tracker
        self._feats = feats

    def get_features_for_match(self, *args, **kwargs):
        return dict(self._feats)

    def __getattr__(self, nombre):
        tracker = self.__dict__.get("_tracker")
        if tracker is None or nombre.startswith("__"):
            raise AttributeError(nombre)
        return getattr(tracker, nombre)


def _modelo_pre_partido(M, clave_dataset):
    """Función (fecha, local, visita) -> copia de M con el tracker pre-partido de ese
    encuentro, o None si el partido no está en el dataset de entrenamiento."""
    df = M[clave_dataset]
    filas = df.to_dict("records")
    indice = {(pd.Timestamp(f["fecha"]), f["local"], f["visita"]): i for i, f in enumerate(filas)}

    def obtener(fecha, local, visita):
        i = indice.get((pd.Timestamp(fecha), local, visita))
        if i is None:
            return None
        return {**M, "tracker": _TrackerPrePartido(M["tracker"], filas[i])}

    return obtener


# --------------------------------------------------------------------------- #
#  Higiene de validación (auditoría nov-2026)
# --------------------------------------------------------------------------- #
# Variables de plantilla generadas por scripts/build_advanced_features.py: toman el valor de 2026
# y lo proyectan hacia atrás con un factor fijo, de modo que el ranking de plantillas de 2026 se
# usaría para predecir partidos de 2020 (fuga de información del futuro). Solo se conserva
# `stadium_capacity`, que es prácticamente estática y conocida de antemano.
_PREFIJOS_ESTIMADOS = ("squad_value", "squad_size", "avg_age", "pct_foreigners", "foreigners",
                       "avg_attendance", "stadium_occupation", "attendance", "occupation")


def excluir_variables_estimadas(cols):
    """Quita de la lista de variables del modelo las columnas de plantilla sintéticas/retroactivas."""
    return [c for c in cols if not any(c.startswith(p) for p in _PREFIJOS_ESTIMADOS)]


def seleccionar_c_lasso(X_train, y_train, grid=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                        n_splits=4, labels=(0, 1, 2), max_iter=3000, c_default=0.05):
    """Elige la regularización C de LASSO con validación cruzada TEMPORAL dentro del train.

    No recibe X_test / y_test: la propia firma impide contaminar la evaluación final.
    Cada fold entrena con el pasado y evalúa en el bloque inmediatamente posterior
    (TimeSeriesSplit), respetando el orden cronológico de las filas.
    Devuelve (mejor_C, log_loss_cv_medio)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    labels = list(labels)
    y_arr = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    X_df = pd.DataFrame(X_train).reset_index(drop=True)
    n_splits = int(max(2, min(n_splits, len(X_df) // 50))) if len(X_df) >= 100 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_c, best_loss = c_default, np.inf
    for C in grid:
        losses = []
        for tr, va in tscv.split(X_df):
            if y_arr.iloc[tr].nunique() < 2:
                continue
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(penalty="l1", solver="saga", C=C,
                                                       max_iter=max_iter, random_state=42))])
            pipe.fit(X_df.iloc[tr], y_arr.iloc[tr])
            proba = pipe.predict_proba(X_df.iloc[va])
            # Reordenar a `labels` por si algún fold no vio todas las clases
            P = np.full((len(va), len(labels)), 1e-6)
            for j, cls in enumerate(pipe.classes_):
                if cls in labels:
                    P[:, labels.index(cls)] = proba[:, j]
            P = P / P.sum(axis=1, keepdims=True)
            losses.append(log_loss(y_arr.iloc[va], P, labels=labels))
        if losses and float(np.mean(losses)) < best_loss:
            best_c, best_loss = C, float(np.mean(losses))
    return best_c, best_loss


def hash_modulos_comunes():
    """Hash del código compartido en comun/: se incorpora a la clave de caché de cada motor para
    que un cambio en la lógica común invalide los modelos serializados que dependen de ella."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  Evaluación uniforme e histórico baseline Elo (sin fuga del bloque test)
# --------------------------------------------------------------------------- #
def corte_test_temporada(liga=None, df=None):
    """Devuelve la primera temporada perteneciente al bloque test fuera de muestra."""
    if liga in ("ucl", "unl", "nba"):
        return 2026
    if df is not None and "temporada" in df.columns and int((df["temporada"] < 2025).sum()) < 50:
        return 2026
    return 2025


def extraer_features_partido(M, local, visita, temporada=None, fecha=None):
    """Obtiene el diccionario de features desde M['tracker'] tolerando firmas entre familias."""
    t = M["tracker"]
    for fn in (
        lambda: t.get_features_for_match(local, visita, temporada, fecha=fecha, reset_season=False),
        lambda: t.get_features_for_match(local, visita, temporada, reset_season=False),
        lambda: t.get_features_for_match(local, visita, fecha=fecha, temporada=temporada),
        lambda: t.get_features_for_match(local, visita, temporada),
        lambda: t.get_features_for_match(local, visita, fecha),
        lambda: t.get_features_for_match(local, visita),
    ):
        try:
            return fn()
        except TypeError:
            continue
    return t.get_features_for_match(local, visita)


def etiquetas_y_orden_1x2(M, df=None):
    """Devuelve (y_estandar, home_is_two).
    En competiciones 1X2, `y_estandar` usa 0=Local, 1=Empate, 2=Visita y `home_is_two` indica si
    `resultado` del motor codifica Victoria Local como 2 en vez de 0.
    En competiciones binarias (NBA), devuelve `target_win_local` (0=Visita, 1=Local) y `False`."""
    if df is None:
        df = M["df_features"] if "df_features" in M else M["df_dataset"]
    if "target_win_local" in df.columns:
        return df["target_win_local"].to_numpy(dtype=int), False

    if {"goles_local", "goles_visita"} <= set(df.columns):
        gl = df["goles_local"].to_numpy(dtype=float)
        gv = df["goles_visita"].to_numpy(dtype=float)
    elif {"gl", "gv"} <= set(df.columns):
        gl = df["gl"].to_numpy(dtype=float)
        gv = df["gv"].to_numpy(dtype=float)
    elif "df_partidos" in M and {"fecha", "local", "visita"} <= set(df.columns):
        df_p = M["df_partidos"][["fecha", "local", "visita", "goles_local", "goles_visita"]].drop_duplicates(
            subset=["fecha", "local", "visita"]
        ).copy()
        df_p["_f_key"] = pd.to_datetime(df_p["fecha"]).dt.strftime("%Y-%m-%d")
        aux = pd.DataFrame({
            "_f_key": pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d"),
            "local": df["local"].values,
            "visita": df["visita"].values,
        }).merge(df_p[["_f_key", "local", "visita", "goles_local", "goles_visita"]],
                 on=["_f_key", "local", "visita"], how="left")
        gl = aux["goles_local"].to_numpy(dtype=float)
        gv = aux["goles_visita"].to_numpy(dtype=float)
    else:
        y_raw = df["resultado"].to_numpy(dtype=int)
        return y_raw, False

    valid = np.isfinite(gl) & np.isfinite(gv)
    y_lev = np.where(gl > gv, 0, np.where(gl == gv, 1, 2))
    home_is_two = False
    if "resultado" in df.columns and valid.any():
        hw = valid & (gl > gv)
        if hw.any():
            home_is_two = bool(np.nanmedian(df.loc[hw, "resultado"]) == 2)
    return y_lev, home_is_two


def proba_pipe_1x2(pipe, X, home_is_two=False):
    """Predice probabilidades en orden uniforme [P(Local), P(Empate), P(Visita)]."""
    probs_raw = np.asarray(pipe.predict_proba(X), dtype=float)
    clases = list(getattr(pipe, "classes_", [0, 1, 2]))
    mapa = (2, 1, 0) if home_is_two else (0, 1, 2)
    out = np.full((len(X), 3), 1e-7, dtype=float)
    for col_out, cls_raw in enumerate(mapa):
        if cls_raw in clases:
            out[:, col_out] = probs_raw[:, clases.index(cls_raw)]
    out = np.clip(out, 1e-7, 1.0 - 1e-7)
    return out / out.sum(axis=1, keepdims=True)


def entrenar_baseline_elo(M, liga=None, test_desde=None):
    """Entrena un modelo logístico univariante (`elo_diff`) exclusivamente con filas anteriores al
    corte `test_desde`, replicando la referencia Solo Elo sin fuga del conjunto test."""
    from sklearn.linear_model import LogisticRegression

    df = M["df_features"] if "df_features" in M else M["df_dataset"]
    if test_desde is None:
        test_desde = corte_test_temporada(liga, df)

    tr_mask = (df["temporada"] < test_desde).to_numpy()
    if not tr_mask.any():
        tr_mask = np.arange(len(df)) < max(1, int(len(df) * 0.7))

    if "target_win_local" in df.columns:
        if "pipe_elo" in M:
            return {"tipo": "binario", "lr": M["pipe_elo"], "test_desde": test_desde}
        y_bin = df.loc[tr_mask, "target_win_local"].to_numpy(dtype=int)
        X_tr = df.loc[tr_mask, ["elo_diff"]].fillna(0.0).to_numpy()
        lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_bin)
        return {"tipo": "binario", "lr": lr, "test_desde": test_desde}

    y_lev, _ = etiquetas_y_orden_1x2(M, df)
    valid = df["elo_diff"].notna().to_numpy() & tr_mask
    X_tr = df.loc[valid, ["elo_diff"]].fillna(0.0).to_numpy()
    y_tr = y_lev[valid]
    lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    frec = np.bincount(y_tr, minlength=3) / max(len(y_tr), 1)
    return {"tipo": "1x2", "lr": lr, "test_desde": test_desde, "frec_train": frec}


def predecir_elo(M, elo_model, local, visita, temporada=None, fecha=None):
    """Devuelve [P(Local), P(Empate), P(Visita)] del baseline Solo Elo (binario -> empate 0.0)."""
    feats = extraer_features_partido(M, local, visita, temporada=temporada, fecha=fecha)
    elo_diff = float(feats.get("elo_diff", 0.0))
    if elo_model["tipo"] == "binario":
        lr = elo_model["lr"]
        # M['pipe_elo'] espera DataFrame con columna 'elo_diff'
        X_in = pd.DataFrame([{"elo_diff": elo_diff}]) if hasattr(lr, "feature_names_in_") else np.array([[elo_diff]])
        p_home = float(lr.predict_proba(X_in)[0, 1])
        p_home = float(np.clip(p_home, 1e-6, 1.0 - 1e-6))
        return np.array([p_home, 0.0, 1.0 - p_home], dtype=float)
    p = np.asarray(elo_model["lr"].predict_proba(np.array([[elo_diff]])), dtype=float)[0]
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return p / p.sum()


# --------------------------------------------------------------------------- #
#  Stacking Out-Of-Fold (OOF) temporal y modelos base regularizados
# --------------------------------------------------------------------------- #
def _proba_alineada(model, X, labels):
    """Predice probabilidades alineadas al orden exacto de `labels` y normalizadas."""
    probs_raw = np.asarray(model.predict_proba(X), dtype=float)
    clases = list(getattr(model, "classes_", labels))
    out = np.full((len(X), len(labels)), 1e-7, dtype=float)
    for j, cls in enumerate(labels):
        if cls in clases:
            out[:, j] = probs_raw[:, clases.index(cls)]
    out = np.clip(out, 1e-7, 1.0 - 1e-7)
    return out / out.sum(axis=1, keepdims=True)


def modelos_base_regularizados(best_c, n_samples=1000, binario=False, random_state=42, max_iter_lasso=3500):
    """Devuelve `(pipe_lasso, pipe_rf, pipe_xgb)` con regularización ajustada al tamaño muestral.

    Las ligas chicas y torneos cortos (`n_samples < 1000`) reciben árboles menos profundos y
    mayor penalización en hoja/L2 para evitar sobreajuste en muestras pequeñas.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    eval_metric = "logloss" if binario else "mlogloss"
    pipe_lasso = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l1", solver="saga", C=best_c, max_iter=max_iter_lasso, random_state=random_state
        )),
    ])
    if n_samples < 1000:
        rf_clf = RandomForestClassifier(
            n_estimators=250, max_depth=4, min_samples_split=25, min_samples_leaf=12,
            max_features="sqrt", random_state=random_state, n_jobs=-1,
        )
        xgb_clf = XGBClassifier(
            n_estimators=110, max_depth=2, learning_rate=0.025,
            subsample=0.75, colsample_bytree=0.70, min_child_weight=20,
            reg_lambda=8.0, reg_alpha=0.8, eval_metric=eval_metric,
            random_state=random_state, n_jobs=-1,
        )
    else:
        rf_clf = RandomForestClassifier(
            n_estimators=250, max_depth=4, min_samples_split=25, min_samples_leaf=10,
            max_features="sqrt", random_state=random_state, n_jobs=-1,
        )
        xgb_clf = XGBClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.03,
            subsample=0.80, colsample_bytree=0.75, min_child_weight=15,
            reg_lambda=5.0, reg_alpha=0.5, eval_metric=eval_metric,
            random_state=random_state, n_jobs=-1,
        )
    pipe_rf = Pipeline([("sc", StandardScaler()), ("rf", rf_clf)])
    pipe_xgb = Pipeline([("sc", StandardScaler()), ("xgb", xgb_clf)])
    return pipe_lasso, pipe_rf, pipe_xgb


def optimizar_pesos_simplex(probs_list, y, labels=(0, 1, 2), reg_l2=0.03, prior=None):
    """Minimiza `log_loss + reg_l2 * ||w - prior||^2` en el simplex mediante SLSQP."""
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss

    k = len(probs_list)
    if prior is None:
        prior = np.ones(k, dtype=float) / k
    else:
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()

    y_arr = np.asarray(y)
    if len(y_arr) < 10:
        return prior

    probs_norm = []
    for p in probs_list:
        arr = np.asarray(p, dtype=float)
        if arr.ndim == 1:
            arr = np.column_stack([1.0 - arr, arr])
        arr = np.clip(arr, 1e-7, 1.0 - 1e-7)
        arr = arr / arr.sum(axis=1, keepdims=True)
        probs_norm.append(arr)

    labels = list(labels)

    def _obj(w):
        w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
        s = float(w.sum())
        w = w / s if s > 0 else prior
        blend = sum(wi * pi for wi, pi in zip(w, probs_norm))
        blend = np.clip(blend, 1e-7, 1.0 - 1e-7)
        blend = blend / blend.sum(axis=1, keepdims=True)
        return float(log_loss(y_arr, blend, labels=labels) + reg_l2 * np.sum((w - prior) ** 2))

    res = minimize(
        _obj,
        prior.copy(),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
    )
    if not res.success:
        return prior
    w_opt = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
    s = float(w_opt.sum())
    return (w_opt / s) if s > 0 else prior


def entrenar_stacking_oof(X_full, y_full, best_c, binario=False, n_splits=4, reg_l2=0.03,
                          prior=None, max_iter_lasso=3500):
    """Entrena LASSO, RF y XGB con pesos de stacking calibrados estrictamente Out-Of-Fold.

    1. Genera predicciones OOF cronológicas dentro de `X_full` (`TimeSeriesSplit`) sin que
       ningún modelo base vea el bloque sobre el que se evalúa su peso.
    2. Optimiza los pesos del ensamble en el simplex con SLSQP y regularización L2.
    3. Reentrena los tres clasificadores definitivos sobre todo `(X_full, y_full)`.

    Devuelve `(pipe_lasso, pipe_rf, pipe_xgb, weights_array)`.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X_df = pd.DataFrame(X_full).fillna(0.0).reset_index(drop=True)
    y_arr = pd.Series(y_full).reset_index(drop=True).to_numpy(dtype=int)
    labels = [0, 1] if binario else [0, 1, 2]
    n_samples = len(X_df)

    k_folds = int(max(2, min(n_splits, n_samples // 100))) if n_samples >= 200 else 2
    tscv = TimeSeriesSplit(n_splits=k_folds)

    oof_probs = [[], [], []]
    oof_y = []
    for tr_idx, va_idx in tscv.split(X_df):
        if len(np.unique(y_arr[tr_idx])) < 2 or len(va_idx) < 5:
            continue
        m_l, m_r, m_x = modelos_base_regularizados(
            best_c, n_samples=n_samples, binario=binario, max_iter_lasso=max_iter_lasso
        )
        m_l.fit(X_df.iloc[tr_idx], y_arr[tr_idx])
        m_r.fit(X_df.iloc[tr_idx], y_arr[tr_idx])
        m_x.fit(X_df.iloc[tr_idx], y_arr[tr_idx])
        oof_probs[0].append(_proba_alineada(m_l, X_df.iloc[va_idx], labels))
        oof_probs[1].append(_proba_alineada(m_r, X_df.iloc[va_idx], labels))
        oof_probs[2].append(_proba_alineada(m_x, X_df.iloc[va_idx], labels))
        oof_y.append(y_arr[va_idx])

    if oof_y:
        probs_cat = [np.vstack(p) for p in oof_probs]
        y_cat = np.concatenate(oof_y)
        weights = optimizar_pesos_simplex(probs_cat, y_cat, labels=labels, reg_l2=reg_l2, prior=prior)
    else:
        weights = np.array(prior if prior is not None else [1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=float)
        weights = weights / weights.sum()

    pipe_lasso, pipe_rf, pipe_xgb = modelos_base_regularizados(
        best_c, n_samples=n_samples, binario=binario, max_iter_lasso=max_iter_lasso
    )
    pipe_lasso.fit(X_df, y_arr)
    pipe_rf.fit(X_df, y_arr)
    pipe_xgb.fit(X_df, y_arr)
    return pipe_lasso, pipe_rf, pipe_xgb, weights
