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
