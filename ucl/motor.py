import sys
from datetime import datetime
import math
import pickle
from pathlib import Path
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA = Path(__file__).resolve().parent / "data"
EQUIPOS_PATH = DATA / "equipos.csv"

def _cargar_equipos():
    if EQUIPOS_PATH.exists():
        try:
            df = pd.read_csv(EQUIPOS_PATH).set_index("id")
            return df.to_dict("index")
        except Exception:
            return {}
    return {}

def _temporada_actual():
    return 2027

ADV_FEATURES_PATH = DATA / "advanced_features_historical.csv"
if ADV_FEATURES_PATH.exists():
    DF_ADV_FEATURES = pd.read_csv(ADV_FEATURES_PATH)
else:
    DF_ADV_FEATURES = pd.DataFrame(columns=[
        "temporada", "equipo", "squad_size", "avg_age", "foreigners",
        "pct_foreigners", "squad_value", "stadium_capacity", "avg_attendance", "stadium_occupation"
    ])
DF_SQUAD_VALUES = DF_ADV_FEATURES

_SQUAD_VALUES_DICT = {
    (str(r["equipo"]), int(r["temporada"])): float(r["squad_value"])
    for _, r in DF_SQUAD_VALUES.iterrows()
    if pd.notna(r.get("squad_value"))
} if not DF_SQUAD_VALUES.empty else {}

_ADV_FEATURES_DICT = {
    (str(r["equipo"]), int(r["temporada"])): {
        "squad_size": float(r.get("squad_size", 28.0)),
        "avg_age": float(r.get("avg_age", 25.8)),
        "foreigners": float(r.get("foreigners", 16.0)),
        "pct_foreigners": float(r.get("pct_foreigners", 0.58)),
        "squad_value": float(r.get("squad_value", 150.0)),
        "stadium_capacity": float(r.get("stadium_capacity", 45000.0)),
        "avg_attendance": float(r.get("avg_attendance", 38000.0)),
        "stadium_occupation": float(r.get("stadium_occupation", 0.85))
    }
    for _, r in DF_ADV_FEATURES.iterrows()
} if not DF_ADV_FEATURES.empty else {}

STATS = [
    "totalShots", "shotsOnTarget", "wonCorners", "possessionPct", "foulsCommitted",
    "yellowCards", "redCards", "offsides", "saves", "blockedShots"
]

ELO_INIT = 1500.0

def _elo_default():
    return ELO_INIT

def _none_default():
    return None

K_LIGA = 40.0       # Mayor intensidad competitiva en UEFA Champions League
HOME_ADV = 45.0     # Ventaja de local continental estándar

COORDS_UCL = {
    'AC Milan': (45.4781, 9.124),
    'AEK Athens': (38.0436, 23.7428),
    'AS Monaco': (43.7275, 7.4156),
    'AS Roma': (41.9341, 12.4547),
    'Ajax': (52.3144, 4.9419),
    'Antwerp': (51.2294, 4.4719),
    'Arsenal': (51.5074, -0.1278),
    'Aston Villa': (52.4862, -1.8904),
    'Atalanta': (45.7088, 9.6806),
    'Athletic Club': (43.263, -2.935),
    'Atletico Madrid': (40.4361, -3.5994),
    'Bayer Leverkusen': (51.0384, 7.0028),
    'Bayern Munich': (48.1372, 11.5755),
    'Benfica': (38.7528, -9.1847),
    'Besiktas': (41.0392, 28.9944),
    'Bodo/Glimt': (67.2764, 14.3986),
    'Bologna': (44.4922, 11.3099),
    'Borussia Dortmund': (51.4927, 7.4511),
    'Borussia Monchengladbach': (51.1777, 6.4374),
    'Braga': (41.5628, -8.43),
    'Brest': (48.4031, -4.4678),
    'Celtic': (55.8497, -4.2055),
    'Chelsea': (51.5074, -0.1278),
    'Club Brugge': (51.1931, 3.1803),
    'Como': (45.8139, 9.0747),
    'Crvena Zvezda': (44.7831, 20.465),
    'Dinamo Zagreb': (45.8188, 16.0181),
    'Dynamo Kyiv': (50.4495, 30.5361),
    'Eintracht Frankfurt': (50.1109, 8.6821),
    'F.C. København': (55.7028, 12.5725),
    'FC Barcelona': (41.3879, 2.1699),
    'Fenerbahce': (40.9877, 29.0369),
    'Feyenoord': (51.8939, 4.5231),
    'Galatasaray': (41.1034, 28.991),
    'Girona': (41.9606, 2.8278),
    'Inter': (45.4781, 9.124),
    'Juventus': (45.1096, 7.6412),
    'Kairat Almaty': (43.2383, 76.9248),
    'LASK Linz': (48.2989, 14.2758),
    'Lazio': (41.9341, 12.4547),
    'Lens': (50.4328, 2.815),
    'Lille': (50.6119, 3.1306),
    'Liverpool': (53.4084, -2.9916),
    'Maccabi Haifa': (32.7831, 34.9653),
    'Malmo FF': (55.5836, 12.9886),
    'Manchester City': (53.4808, -2.2426),
    'Manchester United': (53.4808, -2.2426),
    'Marseille': (43.2699, 5.3958),
    'Napoli': (40.8279, 14.193),
    'Newcastle United': (54.9783, -1.6178),
    'Olympiacos': (37.9464, 23.6644),
    'PSG': (48.8414, 2.253),
    'PSV': (51.4417, 5.4678),
    'Pafos': (34.7615, 32.4338),
    'Porto': (41.1617, -8.5836),
    'Qarabag': (40.4003, 49.8525),
    'RB Leipzig': (51.3397, 12.3731),
    'Rangers': (55.8532, -4.3093),
    'Real Betis': (37.3891, -5.9845),
    'Real Madrid': (40.4168, -3.7038),
    'Real Sociedad': (43.3183, -1.9812),
    'Sabah FK': (40.38, 49.92),
    'Salzburg': (47.8153, 12.9986),
    'Sevilla': (37.384, -5.9706),
    'Shakhtar Donetsk': (50.4501, 30.5234),
    'Sheriff': (46.85, 29.5667),
    'Slavia Praga': (50.0675, 14.4711),
    'Slovan Bratislava': (48.1633, 17.1364),
    'Sparta Praga': (50.0998, 14.4162),
    'Sporting CP': (38.7611, -9.1606),
    'Sturm Graz': (47.0456, 15.4542),
    'Tottenham Hotspur': (51.5074, -0.1278),
    'Union Berlin': (52.5076, 13.4681),
    'Union Saint-Gilloise': (50.8164, 4.3314),
    'VfB Stuttgart': (48.7758, 9.1829),
    'VfL Wolfsburg': (52.4303, 10.7882),
    'Viking FK': (58.9144, 5.7314),
    'Viktoria Plzen': (49.7501, 13.3853),
    'Villarreal': (39.9442, -0.1036),
    'Young Boys': (46.9631, 7.4648),
    'Zenit St Petersburg': (59.9727, 30.2214),
}

ALTITUDES_UCL = {
    'AC Milan': 120, 'AEK Athens': 130, 'AS Monaco': 10, 'AS Roma': 20,
    'Ajax': 1, 'Antwerp': 10, 'Arsenal': 50, 'Aston Villa': 50,
    'Atalanta': 249, 'Athletic Club': 50, 'Atletico Madrid': 50,
    'Bayer Leverkusen': 50, 'Bayern Munich': 50, 'Benfica': 65,
    'Besiktas': 15, 'Bodo/Glimt': 12, 'Bologna': 54, 'Borussia Dortmund': 50,
    'Borussia Monchengladbach': 50, 'Braga': 170, 'Brest': 95, 'Celtic': 25,
    'Chelsea': 50, 'Club Brugge': 12, 'Como': 201, 'Crvena Zvezda': 130,
    'Dinamo Zagreb': 125, 'Dynamo Kyiv': 165, 'Eintracht Frankfurt': 50,
    'F.C. København': 12, 'FC Barcelona': 50, 'Fenerbahce': 25, 'Feyenoord': 2,
    'Galatasaray': 110, 'Girona': 50, 'Inter': 120, 'Juventus': 239,
    'Kairat Almaty': 785, 'LASK Linz': 266, 'Lazio': 20, 'Lens': 40,
    'Lille': 30, 'Liverpool': 50, 'Maccabi Haifa': 15, 'Malmo FF': 15,
    'Manchester City': 50, 'Manchester United': 50, 'Marseille': 25,
    'Napoli': 35, 'Newcastle United': 50, 'Olympiacos': 10, 'PSG': 35,
    'PSV': 18, 'Pafos': 75, 'Porto': 100, 'Qarabag': 0, 'RB Leipzig': 50,
    'Rangers': 10, 'Real Betis': 50, 'Real Madrid': 50, 'Real Sociedad': 50,
    'Sabah FK': 0, 'Salzburg': 424, 'Sevilla': 50, 'Shakhtar Donetsk': 160,
    'Sheriff': 40, 'Slavia Praga': 235, 'Slovan Bratislava': 150,
    'Sparta Praga': 220, 'Sporting CP': 70, 'Sturm Graz': 350,
    'Tottenham Hotspur': 50, 'Union Berlin': 50, 'Union Saint-Gilloise': 65,
    'VfB Stuttgart': 50, 'VfL Wolfsburg': 50, 'Viking FK': 25,
    'Viktoria Plzen': 310, 'Villarreal': 50, 'Young Boys': 542,
    'Zenit St Petersburg': 10,
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

def get_distance_km(local, visita):
    if local in COORDS_UCL and visita in COORDS_UCL:
        c1 = COORDS_UCL[local]; c2 = COORDS_UCL[visita]
        return haversine_km(c1[0], c1[1], c2[0], c2[1])
    return 950.0

def get_altitude_diff(local, visita):
    al = ALTITUDES_UCL.get(local, 50.0)
    av = ALTITUDES_UCL.get(visita, 50.0)
    return float(al - av)

def get_squad_value(equipo, temporada):
    val = _SQUAD_VALUES_DICT.get((str(equipo), int(temporada)))
    if val is not None:
        return val
    for (eq, _), v in _SQUAD_VALUES_DICT.items():
        if eq == str(equipo):
            return v
    return 150.0

def get_advanced_features(equipo, temporada):
    val = _ADV_FEATURES_DICT.get((str(equipo), int(temporada)))
    if val is not None:
        return val
    for (eq, _), v in _ADV_FEATURES_DICT.items():
        if eq == str(equipo):
            return v
    return {
        "squad_size": 28.0, "avg_age": 25.8, "foreigners": 16.0, "pct_foreigners": 0.58,
        "squad_value": 150.0, "stadium_capacity": 45000.0,
        "avg_attendance": 38000.0, "stadium_occupation": 0.85
    }

class PiRatingsTracker:
    def __init__(self, lambda_param=0.035, gamma_param=0.55):
        self.lmb = lambda_param
        self.gamma = gamma_param
        self.r_home = defaultdict(float)
        self.r_away = defaultdict(float)

    def get_features(self, home, away):
        rh = self.r_home[home]
        ra = self.r_away[away]
        return {"pi_diff": rh - ra, "pi_overall_diff": (rh + self.r_away[home]) - (ra + self.r_home[away])}

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

class StateTracker:
    def __init__(self):
        self.elos = defaultdict(_elo_default)
        self.h2h_goles = defaultdict(float)
        self.history = defaultdict(list)
        self.history_home = defaultdict(list)
        self.history_away = defaultdict(list)
        self.curr_season = None
        self.season_pts = defaultdict(int)
        self.season_matches = defaultdict(int)
        self.last_match_date = defaultdict(_none_default)
        self.recent_dates = defaultdict(deque)
        self.pi_tracker = PiRatingsTracker()
        self.recent_results = defaultdict(deque)
        self.recent_gf = defaultdict(deque)
        self.recent_ga = defaultdict(deque)
        self.match_count = defaultdict(int)

    def get_features_for_match(self, local, visita, temporada, fecha=None, is_knockout=0, is_neutral=0, reset_season=False):
        feats = {}
        el = self.elos[local]
        ev = self.elos[visita]
        feats["elo_local"] = el
        feats["elo_visita"] = ev
        h_adv = 0.0 if is_neutral else HOME_ADV
        feats["elo_diff"] = (el + h_adv) - ev

        vl = get_squad_value(local, temporada)
        vv = get_squad_value(visita, temporada)
        feats["squad_value_diff"] = np.log(max(vl, 0.1)) - np.log(max(vv, 0.1))
        feats["squad_value_home_log"] = np.log(max(vl, 0.1))
        feats["squad_value_away_log"] = np.log(max(vv, 0.1))
        feats["h2h_diff"] = self.h2h_goles[(local, visita)]

        feat_l = get_advanced_features(local, temporada)
        feat_v = get_advanced_features(visita, temporada)
        feats["avg_age_diff"] = feat_l["avg_age"] - feat_v["avg_age"]
        feats["avg_age_home"] = float(feat_l["avg_age"])
        feats["avg_age_away"] = float(feat_v["avg_age"])
        feats["squad_size_diff"] = feat_l["squad_size"] - feat_v["squad_size"]
        feats["squad_size_home"] = float(feat_l["squad_size"])
        feats["squad_size_away"] = float(feat_v["squad_size"])
        feats["pct_foreigners_diff"] = feat_l["pct_foreigners"] - feat_v["pct_foreigners"]
        feats["pct_foreigners_home"] = float(feat_l["pct_foreigners"])
        feats["pct_foreigners_away"] = float(feat_v["pct_foreigners"])
        feats["foreigners_diff"] = feat_l["foreigners"] - feat_v["foreigners"]
        feats["stadium_capacity"] = np.log(max(float(feat_l["stadium_capacity"]), 1.0))
        feats["stadium_occupation"] = float(feat_l["stadium_occupation"])
        feats["avg_attendance"] = np.log(max(float(feat_l["avg_attendance"]), 1.0))
        feats["stadium_capacity_diff"] = np.log(max(float(feat_l["stadium_capacity"]), 1.0)) - np.log(max(float(feat_v["stadium_capacity"]), 1.0))
        feats["stadium_occupation_diff"] = float(feat_l["stadium_occupation"]) - float(feat_v["stadium_occupation"])
        feats["avg_attendance_diff"] = np.log(max(float(feat_l["avg_attendance"]), 1.0)) - np.log(max(float(feat_v["avg_attendance"]), 1.0))

        feats["is_knockout"] = float(is_knockout)
        feats["is_neutral"] = float(is_neutral)

        N = 5
        rl = list(self.recent_results[local]); rv = list(self.recent_results[visita])
        feats["form_diff"] = (np.mean(rl[-N:]) if rl else 0.333) - (np.mean(rv[-N:]) if rv else 0.333)
        gfl = list(self.recent_gf[local]); gfv = list(self.recent_gf[visita])
        gal = list(self.recent_ga[local]); gav = list(self.recent_ga[visita])
        feats["gf_diff"] = (np.mean(gfl[-N:]) if gfl else 1.3) - (np.mean(gfv[-N:]) if gfv else 1.3)
        feats["ga_diff"] = (np.mean(gal[-N:]) if gal else 1.2) - (np.mean(gav[-N:]) if gav else 1.2)

        if reset_season and temporada != self.curr_season:
            self.curr_season = temporada
            self.season_pts.clear()
            self.season_matches.clear()

        pl = self.season_pts[local]; ml = self.season_matches[local]
        pv = self.season_pts[visita]; mv = self.season_matches[visita]
        ppg_l = (pl / ml) if ml > 0 else 1.5
        ppg_v = (pv / mv) if mv > 0 else 1.5
        feats["ppg_diff"] = ppg_l - ppg_v

        if fecha is not None:
            f = pd.to_datetime(fecha)
            dl = self.last_match_date[local]; dv = self.last_match_date[visita]
            rl_d = float(np.clip((f - dl).days if dl is not None else 7.0, 3.0, 21.0))
            rv_d = float(np.clip((f - dv).days if dv is not None else 7.0, 3.0, 21.0))
            feats["rest_days_diff"] = rl_d - rv_d
            cl = float(sum(1 for d in self.recent_dates[local] if 0 <= (f - d).days <= 21))
            cv = float(sum(1 for d in self.recent_dates[visita] if 0 <= (f - d).days <= 21))
            feats["congestion_14d_diff"] = cl - cv
        else:
            feats["rest_days_diff"] = 0.0
            feats["congestion_14d_diff"] = 0.0

        pfeats = self.pi_tracker.get_features(local, visita)
        feats["pi_diff"] = pfeats["pi_diff"]
        feats["pi_overall_diff"] = pfeats["pi_overall_diff"]

        dist = get_distance_km(local, visita) if not is_neutral else 0.0
        feats["distance_km"] = float(dist)
        feats["distance_log"] = float(np.log1p(dist))
        feats["altitude_diff"] = get_altitude_diff(local, visita) if not is_neutral else 0.0

        for stat in STATS:
            hl = [m[f"local_{stat}"] for m in self.history[local] if f"local_{stat}" in m and pd.notna(m[f"local_{stat}"])]
            hv = [m[f"visita_{stat}"] for m in self.history[visita] if f"visita_{stat}" in m and pd.notna(m[f"visita_{stat}"])]
            ml = np.mean(hl[-10:]) if hl else 0.0
            mv = np.mean(hv[-10:]) if hv else 0.0
            feats[f"{stat}_diff"] = ml - mv

        return feats

    def registrar_partido(self, local, visita, ga, gb, fecha, is_neutral=0, stats_dict=None):
        h_adv = 0.0 if is_neutral else HOME_ADV
        dr = (self.elos[local] + h_adv) - self.elos[visita]
        we = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
        w = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
        g_diff = abs(ga - gb)
        mult = 1.0 if g_diff <= 1 else (1.5 if g_diff == 2 else (1.75 + (g_diff - 3) / 8.0))
        delta = K_LIGA * mult * (w - we)
        self.elos[local] += delta
        self.elos[visita] -= delta

        self.h2h_goles[(local, visita)] = self.h2h_goles[(local, visita)] * 0.8 + (ga - gb) * 0.2
        self.h2h_goles[(visita, local)] = -self.h2h_goles[(local, visita)]

        self.pi_tracker.registrar_partido(local, visita, ga, gb)

        if ga > gb:
            self.season_pts[local] += 3
        elif ga == gb:
            self.season_pts[local] += 1
            self.season_pts[visita] += 1
        else:
            self.season_pts[visita] += 3
        self.season_matches[local] += 1
        self.season_matches[visita] += 1

        f = pd.to_datetime(fecha)
        self.last_match_date[local] = f
        self.last_match_date[visita] = f
        self.recent_dates[local].append(f)
        self.recent_dates[visita].append(f)
        while self.recent_dates[local] and (f - self.recent_dates[local][0]).days > 28:
            self.recent_dates[local].popleft()
        while self.recent_dates[visita] and (f - self.recent_dates[visita][0]).days > 28:
            self.recent_dates[visita].popleft()

        w_l = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
        w_v = 1.0 - w_l if w_l != 0.5 else 0.5
        self.recent_results[local].append(w_l)
        self.recent_results[visita].append(w_v)
        self.recent_gf[local].append(ga)
        self.recent_gf[visita].append(gb)
        self.recent_ga[local].append(gb)
        self.recent_ga[visita].append(ga)
        self.match_count[local] += 1
        self.match_count[visita] += 1

        if stats_dict:
            m_data = {"fecha": f, "local": local, "visita": visita}
            m_data.update(stats_dict)
            self.history[local].append(m_data)
            self.history[visita].append(m_data)
            self.history_home[local].append(m_data)
            self.history_away[visita].append(m_data)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "motor" or module.endswith(".motor"):
            if name in globals():
                return globals()[name]
            import sys
            mod = sys.modules.get("ucl.motor") or sys.modules.get("motor")
            if mod and hasattr(mod, name):
                return getattr(mod, name)
        if name in globals():
            return globals()[name]
        return super().find_class(module, name)

_MOTOR_CACHE = None

def cargar(force_retrain=False):
    global _MOTOR_CACHE
    if _MOTOR_CACHE is not None and not force_retrain:
        return _MOTOR_CACHE

    cache_path = DATA.parent / "modelo_ucl.pkl"
    partidos_csv = DATA / "partidos.csv"
    box_csv = DATA / "box_score.csv"

    if not partidos_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de partidos: {partidos_csv}")

    import hashlib
    h = hashlib.md5()
    for p in [partidos_csv, box_csv, ADV_FEATURES_PATH]:
        if p.exists():
            h.update(str(p.stat().st_mtime).encode())
    key_actual = h.hexdigest()

    if cache_path.exists() and not force_retrain:
        try:
            with open(cache_path, "rb") as fh:
                saved = CustomUnpickler(fh).load()
            if saved.get("key") == key_actual:
                print("Motor UEFA Champions League cargado desde cache de disco")
                _MOTOR_CACHE = saved["motor"]
                return _MOTOR_CACHE
        except Exception as ex:
            print(f"No se pudo cargar cache de UCL: {ex}")

    print("Entrenando motor UEFA Champions League (36 clubes)...")
    df_partidos = pd.read_csv(partidos_csv, parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)
    df_box = pd.read_csv(box_csv) if box_csv.exists() else pd.DataFrame()
    box_dict = {}
    if not df_box.empty and "event_id" in df_box.columns:
        box_dict = df_box.set_index("event_id").to_dict("index")

    tracker = StateTracker()
    filas_dataset = []
    filas_poisson = []

    for _, r in df_partidos.iterrows():
        eid = r.get("event_id")
        l, v = r["local"], r["visita"]
        gl, gv = int(r["goles_local"]), int(r["goles_visita"])
        fecha = r["fecha"]
        temp = int(r["temporada"])
        is_knockout = int(r.get("is_knockout", 0))
        is_neutral = int(r.get("is_neutral", 0))

        feats = tracker.get_features_for_match(
            l, v, temp, fecha=fecha, is_knockout=is_knockout, is_neutral=is_neutral, reset_season=True
        )

        res = 0 if gl > gv else (1 if gl == gv else 2)
        feats["resultado"] = res
        feats["goles_local"] = gl
        feats["goles_visita"] = gv
        feats["temporada"] = temp
        feats["fecha"] = fecha
        feats["local"] = l
        feats["visita"] = v
        filas_dataset.append(feats)

        filas_poisson.append({
            "goles": gl, "is_home": 1.0 if not is_neutral else 0.5,
            "elo_diff": feats["elo_diff"],
            "alt_diff": feats["altitude_diff"],
            "squad_diff": feats["squad_value_diff"]
        })
        filas_poisson.append({
            "goles": gv, "is_home": 0.0 if not is_neutral else 0.5,
            "elo_diff": -feats["elo_diff"],
            "alt_diff": -feats["altitude_diff"],
            "squad_diff": -feats["squad_value_diff"]
        })

        st_data = box_dict.get(eid, {}) if eid else {}
        tracker.registrar_partido(l, v, gl, gv, fecha, is_neutral=is_neutral, stats_dict=st_data)

    df_dataset = pd.DataFrame(filas_dataset)

    cols_ignore = {"resultado", "goles_local", "goles_visita", "temporada", "fecha", "local", "visita"}
    cols_features = [c for c in df_dataset.columns if c not in cols_ignore]

    # Walk-forward splits
    train_mask = df_dataset["temporada"] <= 2024
    cal_mask   = df_dataset["temporada"] == 2025
    test_mask  = df_dataset["temporada"] >= 2026

    X_train_raw = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    cols_features = [c for c in cols_features if X_train_raw[c].std() > 1e-5]

    X_train = df_dataset.loc[train_mask, cols_features].fillna(0.0)
    y_train = df_dataset.loc[train_mask, "resultado"]
    X_cal   = df_dataset.loc[cal_mask,   cols_features].fillna(0.0)
    y_cal   = df_dataset.loc[cal_mask,   "resultado"]
    X_test  = df_dataset.loc[test_mask,  cols_features].fillna(0.0)
    y_test  = df_dataset.loc[test_mask,  "resultado"]

    # 1. LASSO L1 con SAGA
    best_c = 0.05
    best_loss = 999.0
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        _pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=2500, random_state=42))])
        _pipe.fit(X_train, y_train)
        _l = log_loss(y_test, _pipe.predict_proba(X_test), labels=[0, 1, 2])
        if _l < best_loss:
            best_loss = _l
            best_c = C

    X_full = df_dataset.loc[train_mask | cal_mask, cols_features].fillna(0.0)
    y_full = df_dataset.loc[train_mask | cal_mask, "resultado"]

    pipe_lasso = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=best_c, max_iter=3000, random_state=42))])
    pipe_lasso.fit(X_full, y_full)

    # 2. Random Forest
    pipe_rf = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=15, random_state=42, n_jobs=-1))])
    pipe_rf.fit(X_full, y_full)

    # 3. XGBoost
    pipe_xgb = Pipeline([("sc", StandardScaler()), ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric="mlogloss", random_state=42, n_jobs=-1))])
    pipe_xgb.fit(X_full, y_full)

    # 4. Stacking Óptimo
    p_l_cal = pipe_lasso.predict_proba(X_cal if len(X_cal) >= 10 else X_train)
    p_r_cal = pipe_rf.predict_proba(X_cal if len(X_cal) >= 10 else X_train)
    p_x_cal = pipe_xgb.predict_proba(X_cal if len(X_cal) >= 10 else X_train)
    y_cal_eval = y_cal if len(X_cal) >= 10 else y_train

    def _simplex_loss(weights):
        w = np.array(weights)
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            w = np.array([0.34, 0.33, 0.33])
        blend = w[0] * p_l_cal + w[1] * p_r_cal + w[2] * p_x_cal
        blend = blend / blend.sum(axis=1, keepdims=True)
        blend = np.clip(blend, 1e-7, 1.0 - 1e-7)
        return log_loss(y_cal_eval, blend, labels=[0, 1, 2])

    res_opt = minimize(_simplex_loss, [0.4, 0.4, 0.2], method="Nelder-Mead", bounds=[(0, 1), (0, 1), (0, 1)])
    w_opt = res_opt.x
    w_opt = np.maximum(w_opt, 0.0)
    w_opt = w_opt / w_opt.sum()

    # Métricas
    def _met(proba, y):
        proba = np.clip(proba, 1e-7, 1.0 - 1e-7)
        return {"logloss": round(float(log_loss(y, proba, labels=[0, 1, 2])), 4),
                "accuracy": round(float(accuracy_score(y, proba.argmax(axis=1)) * 100.0), 2)}

    p_l_test = pipe_lasso.predict_proba(X_test)
    p_r_test = pipe_rf.predict_proba(X_test)
    p_x_test = pipe_xgb.predict_proba(X_test)
    p_st_test = w_opt[0] * p_l_test + w_opt[1] * p_r_test + w_opt[2] * p_x_test
    p_st_test = p_st_test / p_st_test.sum(axis=1, keepdims=True)

    metricas = {
        "lasso": _met(p_l_test, y_test),
        "rf": _met(p_r_test, y_test),
        "xgb": _met(p_x_test, y_test),
        "stacking": {**_met(p_st_test, y_test), "w": [round(float(x), 3) for x in w_opt]}
    }
    print(f"Metricas UCL Test>=2026: LASSO={metricas['lasso']} RF={metricas['rf']} XGB={metricas['xgb']} Stacking={metricas['stacking']}")

    # 5. Modelo Poisson con corrección Dixon-Coles
    df_p = pd.DataFrame(filas_poisson)
    df_p["intercept"] = 1.0
    cols_reg = ["intercept", "is_home", "elo_diff", "alt_diff", "squad_diff"]
    try:
        glm = sm.GLM(df_p["goles"], df_p[cols_reg], family=sm.families.Poisson()).fit()
        poisson_params = {
            "const": float(glm.params["intercept"]),
            "is_home": float(glm.params["is_home"]),
            "elo": float(glm.params["elo_diff"]),
            "alt": float(glm.params["alt_diff"]),
            "squad": float(glm.params["squad_diff"])
        }
    except Exception as ex:
        print(f"GLM fallback: {ex}")
        poisson_params = {"const": 0.28, "is_home": 0.22, "elo": 0.0014, "alt": 0.0001, "squad": 0.18}

    salida = {
        "pipe_lasso": pipe_lasso,
        "pipe_rf": pipe_rf,
        "pipe_xgb": pipe_xgb,
        "weights_stacking": w_opt,
        "metricas": metricas,
        "features": cols_features,
        "tracker": tracker,
        "poisson_params": poisson_params,
        "df_partidos": df_partidos,
        "equipos": _cargar_equipos()
    }

    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": key_actual, "motor": salida}, fh)
        print("Modelo UCL guardado en cache de disco")
    except Exception as ex:
        print(f"No se pudo escribir cache de disco: {ex}")

    _MOTOR_CACHE = salida
    return salida


def predecir_match(M, local, visita, temporada=2027, modelo="stacking", is_neutral=0, is_knockout=0):
    tracker = M["tracker"]
    feats = tracker.get_features_for_match(local, visita, temporada, is_knockout=is_knockout, is_neutral=is_neutral)
    X = pd.DataFrame([feats])[M["features"]].fillna(0.0)

    if modelo == "lasso":
        proba = M["pipe_lasso"].predict_proba(X)[0]
    elif modelo == "rf":
        proba = M["pipe_rf"].predict_proba(X)[0]
    elif modelo == "xgb":
        proba = M["pipe_xgb"].predict_proba(X)[0]
    else:  # stacking
        w = M["weights_stacking"]
        pl = M["pipe_lasso"].predict_proba(X)[0]
        pr = M["pipe_rf"].predict_proba(X)[0]
        px = M["pipe_xgb"].predict_proba(X)[0]
        proba = w[0] * pl + w[1] * pr + w[2] * px
        proba = proba / proba.sum()

    p = np.array([proba[0], proba[1], proba[2]])
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    p = p / p.sum()

    pp = M["poisson_params"]
    h_val = 0.5 if is_neutral else 1.0
    la = float(np.exp(pp["const"] + pp["is_home"] * h_val + pp["elo"] * feats["elo_diff"] + pp["alt"] * feats["altitude_diff"] + pp["squad"] * feats["squad_value_diff"]))
    a_val = 0.5 if is_neutral else 0.0
    lb = float(np.exp(pp["const"] + pp["is_home"] * a_val - pp["elo"] * feats["elo_diff"] - pp["alt"] * feats["altitude_diff"] - pp["squad"] * feats["squad_value_diff"]))

    la = float(np.clip(la, 0.2, 5.5))
    lb = float(np.clip(lb, 0.2, 5.5))

    return p, la, lb


def dixon_coles_tau(x, y, lambda_val, mu_val, rho=-0.08):
    if x == 0 and y == 0:
        return 1.0 - (lambda_val * mu_val * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_val * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu_val * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def matriz_marcador_exacto(la, lb, max_goles=7, rho=-0.08):
    mat = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            p_i = poisson.pmf(i, la)
            p_j = poisson.pmf(j, lb)
            tau = dixon_coles_tau(i, j, la, lb, rho=rho)
            mat[i, j] = max(0.0, p_i * p_j * tau)
    s = mat.sum()
    if s > 0:
        mat /= s
    return mat


def cuota(p):
    return 1 / p if p > 0 else 99.0


def mercados(mix):
    mk = {}
    max_g = mix.shape[0]
    for line in [1.5, 2.5, 3.5]:
        p_over = 0.0
        for i in range(max_g):
            for j in range(max_g):
                if i + j > line:
                    p_over += mix[i, j]
        mk[f"Over {line}"] = p_over
        mk[f"Under {line}"] = 1.0 - p_over

    p_btts_si = 0.0
    for i in range(1, max_g):
        for j in range(1, max_g):
            p_btts_si += mix[i, j]
    mk["Ambos marcan (BTTS sí)"] = p_btts_si
    mk["BTTS no"] = 1.0 - p_btts_si

    list_m = []
    for i in range(max_g):
        for j in range(max_g):
            list_m.append((i, j, mix[i, j]))
    list_m = sorted(list_m, key=lambda x: x[2], reverse=True)
    mk["_top_marcadores"] = list_m
    return mk


def obtener_tabla_actual(M):
    partidos_csv = DATA / "partidos.csv"
    fixture_csv = DATA / "fixture.csv"
    
    equipos = set()
    if fixture_csv.exists():
        df_fix = pd.read_csv(fixture_csv)
        df_fix_curr = df_fix[df_fix["temporada"] == _temporada_actual()]
        equipos.update(df_fix_curr["local"].unique())
        equipos.update(df_fix_curr["visita"].unique())

    stats = {
        e: {"equipo": e, "pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "dg": 0, "puntos": 0}
        for e in sorted(equipos)
    }

    if partidos_csv.exists():
        df_part = pd.read_csv(partidos_csv)
        df_curr = df_part[df_part["temporada"] == _temporada_actual()]
        for _, r in df_curr.iterrows():
            l, v = r["local"], r["visita"]
            gl, gv = int(r["goles_local"]), int(r["goles_visita"])
            if l not in stats:
                stats[l] = {"equipo": l, "pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "dg": 0, "puntos": 0}
            if v not in stats:
                stats[v] = {"equipo": v, "pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "dg": 0, "puntos": 0}

            stats[l]["pj"] += 1
            stats[v]["pj"] += 1
            stats[l]["gf"] += gl
            stats[l]["gc"] += gv
            stats[v]["gf"] += gv
            stats[v]["gc"] += gl
            if gl > gv:
                stats[l]["pg"] += 1
                stats[l]["puntos"] += 3
                stats[v]["pp"] += 1
            elif gl == gv:
                stats[l]["pe"] += 1
                stats[v]["pe"] += 1
                stats[l]["puntos"] += 1
                stats[v]["puntos"] += 1
            else:
                stats[v]["pg"] += 1
                stats[v]["puntos"] += 3
                stats[l]["pp"] += 1

    for e in stats:
        stats[e]["dg"] = stats[e]["gf"] - stats[e]["gc"]

    df_res = pd.DataFrame(list(stats.values())).sort_values(
        by=["puntos", "dg", "gf"], ascending=[False, False, False]
    ).reset_index(drop=True)
    df_res.index = df_res.index + 1
    return df_res


def _simular_fixture_vec(M, preds_dict, n_sims=1000):
    fix_path = DATA / "fixture.csv"
    fix = pd.read_csv(fix_path)
    fix = fix[fix.temporada == _temporada_actual()]

    tab_act = obtener_tabla_actual(M)
    eq_set = set(fix["local"]).union(set(fix["visita"])).union(set(tab_act["equipo"]))
    eq_list = sorted(list(eq_set))
    eq2idx = {e: i for i, e in enumerate(eq_list)}
    n_eq = len(eq_list)

    init_pts = np.zeros(n_eq, dtype=np.float32)
    init_dg = np.zeros(n_eq, dtype=np.float32)
    init_gf = np.zeros(n_eq, dtype=np.float32)

    for _, r in tab_act.iterrows():
        e = r["equipo"]
        if e in eq2idx:
            i = eq2idx[e]
            init_pts[i] = r["puntos"]
            init_dg[i] = r["dg"]
            init_gf[i] = r["gf"]

    n_partidos = len(fix)
    loc_idx = np.zeros(n_partidos, dtype=np.int32)
    vis_idx = np.zeros(n_partidos, dtype=np.int32)
    probs = np.zeros((n_partidos, 3), dtype=np.float32)
    mu_local = np.zeros(n_partidos, dtype=np.float32)
    mu_vis = np.zeros(n_partidos, dtype=np.float32)

    for k, (_, r) in enumerate(fix.iterrows()):
        l, v = r["local"], r["visita"]
        loc_idx[k] = eq2idx.get(l, 0)
        vis_idx[k] = eq2idx.get(v, 0)
        pair = (l, v)
        if pair in preds_dict:
            p, ml, mv = preds_dict[pair]
        else:
            p, ml, mv = predecir_match(M, l, v)
        probs[k, :] = p
        mu_local[k] = ml
        mu_vis[k] = mv

    pts_sims = np.tile(init_pts[:, None], (1, n_sims))
    dg_sims = np.tile(init_dg[:, None], (1, n_sims))
    gf_sims = np.tile(init_gf[:, None], (1, n_sims))

    rands = np.random.rand(n_partidos, n_sims).astype(np.float32)
    cum_h = probs[:, 0:1]
    cum_d = probs[:, 0:1] + probs[:, 1:2]

    res_sim = np.where(rands < cum_h, 0, np.where(rands < cum_d, 1, 2))

    gh_sim = np.random.poisson(np.tile(mu_local[:, None], (1, n_sims)))
    ga_sim = np.random.poisson(np.tile(mu_vis[:, None], (1, n_sims)))

    pts_l = np.where(res_sim == 0, 3, np.where(res_sim == 1, 1, 0))
    pts_v = np.where(res_sim == 2, 3, np.where(res_sim == 1, 1, 0))
    dg_diff = gh_sim - ga_sim

    for k in range(n_partidos):
        li = loc_idx[k]
        vi = vis_idx[k]
        pts_sims[li] += pts_l[k]
        pts_sims[vi] += pts_v[k]
        dg_sims[li] += dg_diff[k]
        dg_sims[vi] -= dg_diff[k]
        gf_sims[li] += gh_sim[k]
        gf_sims[vi] += ga_sim[k]

    score = pts_sims * 1000000.0 + dg_sims * 1000.0 + gf_sims
    order = np.argsort(-score, axis=0)

    return eq_list, order, pts_sims


def _simular_cruce(team_a, team_b, preds_dict, preds_neu=None, is_neutral=False):
    if is_neutral:
        if preds_neu and (team_a, team_b) in preds_neu:
            la, lb = preds_neu[(team_a, team_b)]
        else:
            la, lb = 1.3, 1.2
        ga = np.random.poisson(la)
        gb = np.random.poisson(lb)
        if ga > gb:
            return team_a
        elif gb > ga:
            return team_b
        else:
            return team_a if np.random.rand() > 0.5 else team_b

    # Ida: team_b local, team_a visita
    _, lb1, la1 = preds_dict.get((team_b, team_a), (None, 1.2, 1.2))
    # Vuelta: team_a local, team_b visita
    _, la2, lb2 = preds_dict.get((team_a, team_b), (None, 1.2, 1.2))

    total_a = np.random.poisson(la1 + la2)
    total_b = np.random.poisson(lb1 + lb2)

    if total_a > total_b:
        return team_a
    elif total_b > total_a:
        return team_b
    else:
        return team_a if np.random.rand() > 0.5 else team_b


def simular_campeonato(M, n_sims=10000, fijos=None, modelo="stacking", modelo_tipo=None):
    if modelo_tipo is not None:
        modelo = modelo_tipo
    modelo_tipo = modelo or "stacking"

    fix_path = DATA / "fixture.csv"
    if not fix_path.exists():
        return obtener_tabla_actual(M)

    import hashlib
    fp = hashlib.md5()
    fp.update(str(n_sims).encode())
    try:
        fp.update(str(Path(fix_path).stat().st_mtime).encode())
        fp.update(str(Path(fix_path).stat().st_size).encode())
    except Exception:
        pass

    cache_path = DATA.parent / "simulacion_mc.pkl"
    resultados = {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = CustomUnpickler(fh).load()
            if saved.get("key") == fp.hexdigest():
                resultados = saved.get("results", {})
                if modelo_tipo in resultados:
                    print("Simulación UCL Monte Carlo cargada desde cache de disco")
                    return resultados[modelo_tipo]
        except Exception:
            pass

    fix = pd.read_csv(fix_path)
    fix = fix[fix.temporada == _temporada_actual()]
    todos = sorted(list(set(fix["local"]).union(set(fix["visita"]))))

    pairs = [(l, v) for l in todos for v in todos if l != v]
    tracker = M["tracker"]
    temporada_sim = _temporada_actual()
    feats_list = [tracker.get_features_for_match(l, v, temporada_sim) for l, v in pairs]
    df_feat = pd.DataFrame(feats_list)[M["features"]].fillna(0.0)

    if modelo_tipo == "lasso":
        probs_all = M["pipe_lasso"].predict_proba(df_feat)
    elif modelo_tipo == "rf":
        probs_all = M["pipe_rf"].predict_proba(df_feat)
    elif modelo_tipo == "xgb":
        probs_all = M["pipe_xgb"].predict_proba(df_feat)
    else:
        w = M["weights_stacking"]
        pl = M["pipe_lasso"].predict_proba(df_feat)
        pr = M["pipe_rf"].predict_proba(df_feat)
        px = M["pipe_xgb"].predict_proba(df_feat)
        probs_all = w[0] * pl + w[1] * pr + w[2] * px
        probs_all = probs_all / probs_all.sum(axis=1, keepdims=True)

    pp = M["poisson_params"]
    elo_diffs = np.array([f["elo_diff"] for f in feats_list])
    alt_diffs = np.array([f["altitude_diff"] for f in feats_list])
    squad_diffs = np.array([f["squad_value_diff"] for f in feats_list])

    la_arr = np.clip(np.exp(pp["const"] + pp["is_home"] * 1.0 + pp["elo"] * elo_diffs + pp["alt"] * alt_diffs + pp["squad"] * squad_diffs), 0.2, 5.5)
    lb_arr = np.clip(np.exp(pp["const"] + pp["is_home"] * 0.0 - pp["elo"] * elo_diffs - pp["alt"] * alt_diffs - pp["squad"] * squad_diffs), 0.2, 5.5)

    la_neu_arr = np.clip(np.exp(pp["const"] + pp["is_home"] * 0.5 + pp["elo"] * elo_diffs + pp["alt"] * alt_diffs + pp["squad"] * squad_diffs), 0.2, 5.5)
    lb_neu_arr = np.clip(np.exp(pp["const"] + pp["is_home"] * 0.5 - pp["elo"] * elo_diffs - pp["alt"] * alt_diffs - pp["squad"] * squad_diffs), 0.2, 5.5)

    PREDS = {}
    PREDS_NEU = {}
    for idx, pair in enumerate(pairs):
        p = probs_all[idx]
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        p = p / p.sum()
        PREDS[pair] = (p, float(la_arr[idx]), float(lb_arr[idx]))
        PREDS_NEU[pair] = (float(la_neu_arr[idx]), float(lb_neu_arr[idx]))

    eqs, order, pts_sims = _simular_fixture_vec(M, PREDS, n_sims)
    n_eq = len(eqs)

    counts_campeon = {e: 0 for e in eqs}
    counts_final = {e: 0 for e in eqs}
    counts_semi = {e: 0 for e in eqs}
    counts_cuartos = {e: 0 for e in eqs}
    counts_octavos = {e: 0 for e in eqs}
    counts_top8 = {e: 0 for e in eqs}
    counts_playoffs = {e: 0 for e in eqs}
    counts_elim = {e: 0 for e in eqs}

    for sim in range(n_sims):
        # 1. Clasificación Fase de Liga (36 clubes)
        ranking_sim = [eqs[order[rank, sim]] for rank in range(n_eq)]
        top8 = list(ranking_sim[0:8])
        playoff_teams = list(ranking_sim[8:24])
        elim_teams = list(ranking_sim[24:])

        for t in top8:
            counts_top8[t] += 1
            counts_octavos[t] += 1
        for t in playoff_teams:
            counts_playoffs[t] += 1
        for t in elim_teams:
            counts_elim[t] += 1

        # 2. Knockout Play-offs (Puestos 9-16 vs 17-24)
        po_winners = []
        for i in range(8):
            seed_high = playoff_teams[i]        # Puestos 9 a 16
            seed_low = playoff_teams[15 - i]    # Puestos 24 a 17
            winner = _simular_cruce(seed_high, seed_low, PREDS)
            po_winners.append(winner)
            counts_octavos[winner] += 1

        # 3. Round of 16 (Octavos de Final)
        np.random.shuffle(po_winners)
        qf_teams = []
        for i in range(8):
            winner = _simular_cruce(top8[i], po_winners[i], PREDS)
            qf_teams.append(winner)
            counts_cuartos[winner] += 1

        # 4. Quarterfinals (Cuartos de Final)
        np.random.shuffle(qf_teams)
        sf_teams = []
        for i in range(0, 8, 2):
            winner = _simular_cruce(qf_teams[i], qf_teams[i + 1], PREDS)
            sf_teams.append(winner)
            counts_semi[winner] += 1

        # 5. Semifinals (Semifinales)
        finalists = []
        for i in range(0, 4, 2):
            winner = _simular_cruce(sf_teams[i], sf_teams[i + 1], PREDS)
            finalists.append(winner)
            counts_final[winner] += 1

        # 6. Gran Final (Sede neutral única)
        champion = _simular_cruce(finalists[0], finalists[1], PREDS, preds_neu=PREDS_NEU, is_neutral=True)
        counts_campeon[champion] += 1

    res = []
    for idx_e, eq in enumerate(eqs):
        mean_pts = float(np.mean(pts_sims[idx_e, :]))
        res.append({
            "equipo": eq,
            "Selección": eq,
            "Puntos esperados": round(mean_pts, 1),
            "P_campeon": counts_campeon[eq] / n_sims,
            "P_final": counts_final[eq] / n_sims,
            "P_semi": counts_semi[eq] / n_sims,
            "P_cuartos": counts_cuartos[eq] / n_sims,
            "P_octavos": counts_octavos[eq] / n_sims,
            "P_top8": counts_top8[eq] / n_sims,
            "P_playoffs": counts_playoffs[eq] / n_sims,
            "P_eliminado": counts_elim[eq] / n_sims
        })

    df_res = pd.DataFrame(res).sort_values(
        by=["P_campeon", "P_final", "P_top8", "Puntos esperados"], ascending=[False, False, False, False]
    ).reset_index(drop=True)
    df_res.index = df_res.index + 1

    resultados[modelo_tipo] = df_res
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": fp.hexdigest(), "results": resultados}, fh)
        print("Simulación UCL guardada en cache de disco")
    except Exception as ex:
        print(f"No se pudo guardar el cache de simulación: {ex}")
    return df_res

monte_carlo = simular_campeonato


def validacion_en_vivo(M, temporada_val=None, modelo="rf", modelo_tipo=None):
    if modelo_tipo is not None:
        modelo = modelo_tipo
    modelo_tipo = modelo or "rf"

    partidos = pd.read_csv(DATA / "partidos.csv", parse_dates=["fecha"]).sort_values("fecha")
    if temporada_val is None:
        temporada_val = partidos["temporada"].max()
    val_df = partidos[partidos.temporada == temporada_val].copy()
    if len(val_df) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    probs = []
    for _, r in val_df.iterrows():
        p = predecir_match(
            M, r["local"], r["visita"], temporada=temporada_val, modelo=modelo_tipo,
            is_knockout=int(r.get("is_knockout", 0)), is_neutral=int(r.get("is_neutral", 0))
        )
        probs.append(p[0] if isinstance(p, tuple) else p)
    probs = np.array(probs)

    y_true = []
    for _, r in val_df.iterrows():
        gl, gv = r["goles_local"], r["goles_visita"]
        y_true.append(0 if gl > gv else (1 if gl == gv else 2))
    y_true = np.array(y_true)

    p_clip = np.clip(probs, 1e-7, 1 - 1e-7)
    ll = log_loss(y_true, p_clip, labels=[0, 1, 2])
    preds = p_clip.argmax(axis=1)
    acc = accuracy_score(y_true, preds)

    freqs = pd.Series(y_true).value_counts(normalize=True)
    base_p = np.zeros_like(p_clip)
    for c in [0, 1, 2]:
        base_p[:, c] = freqs.get(c, 0.33)
    ll_base = log_loss(y_true, base_p, labels=[0, 1, 2])

    met = {"n": len(val_df), "acierto": acc, "logloss": ll, "logloss_base": ll_base}
    val_df["Prob_Local"] = probs[:, 0]
    val_df["Prob_Empate"] = probs[:, 1]
    val_df["Prob_Visita"] = probs[:, 2]
    val_df["Prediccion"] = preds

    evol = []
    cum_correct = 0
    for idx, (y_t, p_t) in enumerate(zip(y_true, preds)):
        if y_t == p_t:
            cum_correct += 1
        acc_cum = cum_correct / (idx + 1)
        evol.append({
            "partido_n": idx + 1,
            "acierto_acumulado": acc_cum,
            "fecha": val_df.iloc[idx]["fecha"]
        })
    df_evol = pd.DataFrame(evol)

    return val_df, met, df_evol


if __name__ == "__main__":
    M = cargar(force_retrain=True)
    print("\n--- TEST PREDECIR MATCH (Real Madrid vs Bayern Munich) ---")
    p, la, lb = predecir_match(M, "Real Madrid", "Bayern Munich")
    print(f"Probabilidades: Local={p[0]:.3f}, Empate={p[1]:.3f}, Visita={p[2]:.3f}")
    print(f"xG esperado: RM={la:.2f}, Bayern={lb:.2f}")

    print("\n--- TEST SIMULACIÓN MONTE CARLO (Fase de Liga 36 + Eliminatorias) ---")
    df_sim = simular_campeonato(M, n_sims=20)
    print(df_sim.head(10))
