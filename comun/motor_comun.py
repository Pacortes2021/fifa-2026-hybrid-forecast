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
