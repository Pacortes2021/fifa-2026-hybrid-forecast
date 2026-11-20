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

import sys as _sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from comun.motor_comun import seleccionar_c_lasso, excluir_variables_estimadas, entrenar_stacking_oof
from comun.motor_comun import hash_modulos_comunes
from comun.monte_carlo import (
    clave_cache_liga,
    simular_fixture_desde_tabla,
    simular_partido_vec_estado,
)

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
    """En UEFA Champions League la temporada se identifica por el año de la final (2026/27 -> 2027)."""
    hoy = pd.Timestamp.now()
    return int(hoy.year) + 1 if hoy.month >= 7 else int(hoy.year)

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

MAPPING_UCL_A_LOCAL = {
    'Athletic Club': ('esp', 'Athletic Club'), 'Atletico Madrid': ('esp', 'Atlético de Madrid'),
    'FC Barcelona': ('esp', 'FC Barcelona'), 'Girona': ('esp', 'Girona FC'),
    'Real Betis': ('esp', 'Real Betis'), 'Real Madrid': ('esp', 'Real Madrid'),
    'Real Sociedad': ('esp', 'Real Sociedad'), 'Sevilla': ('esp', 'Sevilla FC'),
    'Villarreal': ('esp', 'Villarreal CF'), 'Arsenal': ('eng', 'Arsenal'),
    'Aston Villa': ('eng', 'Aston Villa'), 'Chelsea': ('eng', 'Chelsea'),
    'Liverpool': ('eng', 'Liverpool'), 'Manchester City': ('eng', 'Manchester City'),
    'Manchester United': ('eng', 'Manchester United'), 'Newcastle United': ('eng', 'Newcastle United'),
    'Tottenham Hotspur': ('eng', 'Tottenham Hotspur'), 'Bayer Leverkusen': ('bund', 'Bayer Leverkusen'),
    'Bayern Munich': ('bund', 'Bayern Munich'), 'Borussia Dortmund': ('bund', 'Borussia Dortmund'),
    'Borussia Monchengladbach': ('bund', 'Borussia Mönchengladbach'), 'Eintracht Frankfurt': ('bund', 'Eintracht Frankfurt'),
    'RB Leipzig': ('bund', 'RB Leipzig'), 'Union Berlin': ('bund', '1. FC Union Berlin'),
    'VfB Stuttgart': ('bund', 'VfB Stuttgart'), 'VfL Wolfsburg': ('bund', 'VfL Wolfsburg'),
    'AC Milan': ('ita', 'AC Milan'), 'AS Roma': ('ita', 'AS Roma'),
    'Atalanta': ('ita', 'Atalanta'), 'Bologna': ('ita', 'Bologna'),
    'Como': ('ita', 'Como'), 'Inter': ('ita', 'Inter'),
    'Juventus': ('ita', 'Juventus'), 'Lazio': ('ita', 'Lazio'),
    'Napoli': ('ita', 'Napoli'), 'AS Monaco': ('fra', 'Monaco'),
    'Brest': ('fra', 'Brest'), 'Lens': ('fra', 'Lens'),
    'Lille': ('fra', 'Lille'), 'Marseille': ('fra', 'Marseille'),
    'PSG': ('fra', 'Paris Saint-Germain'), 'Benfica': ('por', 'Benfica'),
    'Braga': ('por', 'Braga'), 'Porto': ('por', 'Porto'),
    'Sporting CP': ('por', 'Sporting CP'), 'Ajax': ('ned', 'Ajax'),
    'Feyenoord': ('ned', 'Feyenoord'), 'PSV': ('ned', 'PSV'),
    'Celtic': ('sco', 'Celtic'), 'Rangers': ('sco', 'Rangers'),
    'LASK Linz': ('aut', 'LASK Linz'), 'Salzburg': ('aut', 'RB Salzburg'),
    'Sturm Graz': ('aut', 'SK Sturm Graz'), 'Antwerp': ('bel', 'Antwerp'),
    'Club Brugge': ('bel', 'Club Brugge'), 'Union Saint-Gilloise': ('bel', 'Union Saint-Gilloise'),
    'Besiktas': ('tur', 'Besiktas'), 'Fenerbahce': ('tur', 'Fenerbahce'),
    'Galatasaray': ('tur', 'Galatasaray'), 'AEK Athens': ('gre', 'AEK Athens'),
    'Olympiacos': ('gre', 'Olympiacos'), 'F.C. København': ('den', 'F.C. København'),
    'Bodo/Glimt': ('nor', 'Bodo/Glimt'), 'Viking FK': ('nor', 'Viking FK'),
    'Malmo FF': ('swe', 'Malmo FF'),
}

TIER_LIGA = {
    'eng': 1.00, 'esp': 0.95, 'ita': 0.94, 'bund': 0.94, 'fra': 0.85,
    'ned': 0.82, 'por': 0.80, 'bel': 0.75, 'tur': 0.70, 'sco': 0.65,
    'aut': 0.64, 'den': 0.63, 'gre': 0.62, 'nor': 0.60, 'swe': 0.55,
}

class DomesticLeagueTracker:
    def __init__(self, k=32.0, home_adv=50.0):
        self.k = k
        self.home_adv = home_adv
        self.elos = defaultdict(_elo_default)
        self.recent_results = defaultdict(deque)
        self.recent_gf = defaultdict(deque)
        self.recent_ga = defaultdict(deque)
        self.season_pts = defaultdict(int)
        self.season_matches = defaultdict(int)
        self.season_gf = defaultdict(int)
        self.season_ga = defaultdict(int)
        self.last_match_date = defaultdict(_none_default)
        self.recent_dates = defaultdict(deque)
        self.curr_season = None

    def get_state(self, team):
        el = self.elos[team]
        rr = list(self.recent_results[team])
        form = np.mean(rr[-5:]) if rr else 0.5
        gfl = list(self.recent_gf[team])
        gf = np.mean(gfl[-5:]) if gfl else 1.3
        gal = list(self.recent_ga[team])
        ga = np.mean(gal[-5:]) if gal else 1.1
        pts = self.season_pts[team]
        m = self.season_matches[team]
        ppg = (pts / m) if m > 0 else 1.5
        gf_tot = self.season_gf[team]
        ga_tot = self.season_ga[team]
        gd_pm = ((gf_tot - ga_tot) / m) if m > 0 else 0.0
        return {
            'elo': el, 'form': form, 'gf': gf, 'ga': ga, 'ppg': ppg,
            'gd_pm': gd_pm, 'matches': m,
            'last_date': self.last_match_date[team],
            'recent_dates': list(self.recent_dates[team])
        }

    def registrar(self, local, visita, gl, gv, season, fecha):
        if season != self.curr_season:
            self.curr_season = season
            self.season_pts.clear()
            self.season_matches.clear()
            self.season_gf.clear()
            self.season_ga.clear()

        dr = (self.elos[local] + self.home_adv) - self.elos[visita]
        we = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
        w = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        g_diff = abs(gl - gv)
        mult = 1.0 if g_diff <= 1 else (1.5 if g_diff == 2 else (1.75 + (g_diff - 3) / 8.0))
        delta = self.k * mult * (w - we)
        self.elos[local] += delta
        self.elos[visita] -= delta

        w_l = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        w_v = 1.0 - w_l if w_l != 0.5 else 0.5
        self.recent_results[local].append(w_l)
        self.recent_results[visita].append(w_v)
        self.recent_gf[local].append(gl)
        self.recent_gf[visita].append(gv)
        self.recent_ga[local].append(gv)
        self.recent_ga[visita].append(gl)

        self.season_matches[local] += 1
        self.season_matches[visita] += 1
        self.season_gf[local] += gl
        self.season_gf[visita] += gv
        self.season_ga[local] += gv
        self.season_ga[visita] += gl

        f = pd.to_datetime(fecha)
        self.last_match_date[local] = f
        self.last_match_date[visita] = f
        self.recent_dates[local].append(f)
        self.recent_dates[visita].append(f)
        while self.recent_dates[local] and (f - self.recent_dates[local][0]).days > 21:
            self.recent_dates[local].popleft()
        while self.recent_dates[visita] and (f - self.recent_dates[visita][0]).days > 21:
            self.recent_dates[visita].popleft()

        if gl > gv: self.season_pts[local] += 3
        elif gl == gv: self.season_pts[local] += 1; self.season_pts[visita] += 1
        else: self.season_pts[visita] += 3

class MultiLeagueSystem:
    def __init__(self):
        self.trackers = {l: DomesticLeagueTracker() for l in TIER_LIGA.keys()}
        self.league_matches = {}
        self.ptrs = {l: 0 for l in TIER_LIGA.keys()}
        base_dir = DATA.parent.parent

        for l in TIER_LIGA.keys():
            p = base_dir / l / "data" / "partidos.csv"
            if p.exists():
                df = pd.read_csv(p, parse_dates=['fecha']).sort_values('fecha').reset_index(drop=True)
                self.league_matches[l] = df
            else:
                self.league_matches[l] = pd.DataFrame()

    def advance_to(self, target_date):
        for l, df in self.league_matches.items():
            if df.empty: continue
            idx = self.ptrs[l]
            n = len(df)
            tracker = self.trackers[l]
            while idx < n:
                row = df.iloc[idx]
                if row['fecha'] >= target_date: break
                tracker.registrar(
                    row['local'], row['visita'],
                    int(row['goles_local']), int(row['goles_visita']),
                    int(row['temporada']), row['fecha']
                )
                idx += 1
            self.ptrs[l] = idx

    def get_team_domestic_features(self, ucl_team_name, ucl_match_date=None):
        f = pd.to_datetime(ucl_match_date) if ucl_match_date is not None else None
        if ucl_team_name in MAPPING_UCL_A_LOCAL:
            league, local_name = MAPPING_UCL_A_LOCAL[ucl_team_name]
            tier = TIER_LIGA.get(league, 0.60)
            st = self.trackers[league].get_state(local_name)
            last_d = st['last_date']
            if f is not None and last_d is not None:
                rest_d = float(np.clip((f - last_d).days, 2.0, 21.0))
                cong_14d = float(sum(1 for d in st['recent_dates'] if 0 <= (f - d).days <= 14))
            else:
                rest_d = 6.0
                cong_14d = 2.0
            return {
                'dom_elo': st['elo'],
                'dom_elo_adj': st['elo'] * tier,
                'dom_form': st['form'],
                'dom_gf': st['gf'],
                'dom_ga': st['ga'],
                'dom_ppg': st['ppg'],
                'dom_gd': st['gd_pm'],
                'dom_tier': tier,
                'real_rest_days': rest_d,
                'real_cong_14d': cong_14d
            }
        else:
            return {
                'dom_elo': 1650.0, 'dom_elo_adj': 1650.0 * 0.58, 'dom_form': 0.70,
                'dom_gf': 2.1, 'dom_ga': 0.8, 'dom_ppg': 2.2, 'dom_gd': 1.3, 'dom_tier': 0.58,
                'real_rest_days': 6.0, 'real_cong_14d': 2.0
            }

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
        self.multi_leagues = MultiLeagueSystem()

    def get_features_for_match(self, local, visita, temporada, fecha=None, is_knockout=0, is_neutral=0, reset_season=False, leg2_lead_local=0.0, is_leg2=0):
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

        # Concentración de plantilla: valor medio por jugador
        sz_l = max(float(feat_l.get("squad_size", 28.0)), 15.0)
        sz_v = max(float(feat_v.get("squad_size", 28.0)), 15.0)
        feats["squad_val_per_player_diff"] = np.log(max(vl / sz_l, 0.01)) - np.log(max(vv / sz_v, 0.01))

        feats["is_knockout"] = float(is_knockout)
        feats["is_neutral"] = float(is_neutral)
        feats["is_leg2"] = float(is_leg2)
        feats["leg2_agg_lead"] = float(leg2_lead_local)

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

        # Variables domésticas y de descanso/congestión real
        dom_l = self.multi_leagues.get_team_domestic_features(local, fecha)
        dom_v = self.multi_leagues.get_team_domestic_features(visita, fecha)
        feats["dom_elo_adj_diff"] = dom_l["dom_elo_adj"] - dom_v["dom_elo_adj"]
        feats["dom_tier_diff"] = dom_l["dom_tier"] - dom_v["dom_tier"]
        feats["dom_form_diff"] = dom_l["dom_form"] - dom_v["dom_form"]
        feats["dom_gd_diff"] = dom_l["dom_gd"] - dom_v["dom_gd"]
        feats["real_rest_days_diff"] = dom_l["real_rest_days"] - dom_v["real_rest_days"]
        feats["real_congestion_14d_diff"] = dom_l["real_cong_14d"] - dom_v["real_cong_14d"]

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
    # El modelo también depende del código del motor: si cambia, se re-entrena
    h.update(hashlib.md5(Path(__file__).read_bytes()).hexdigest().encode())
    # y del código compartido en comun/ (auditoría: antes no invalidaba la caché)
    h.update(hash_modulos_comunes().encode())
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

    ko_stages = {'round-of-16', 'quarterfinals', 'knockout-round-playoffs', 'semifinals'}
    leg1_matches = {}

    for _, r in df_partidos.iterrows():
        eid = r.get("event_id")
        l, v = r["local"], r["visita"]
        gl, gv = int(r["goles_local"]), int(r["goles_visita"])
        fecha = r["fecha"]
        temp = int(r["temporada"])
        st = r.get("stage", "group-stage")
        is_knockout = int(r.get("is_knockout", 0))
        is_neutral = int(r.get("is_neutral", 0))

        # Avanzar ligas locales estrictamente hasta antes del cotejo UCL
        tracker.multi_leagues.advance_to(fecha)

        is_leg2 = 0
        leg2_lead_local = 0.0
        if is_knockout and st in ko_stages:
            tie_key = (temp, st, frozenset([l, v]))
            if tie_key not in leg1_matches:
                leg1_matches[tie_key] = (l, v, gl, gv, fecha)
            else:
                p_loc, p_vis, p_gl, p_gv, _ = leg1_matches[tie_key]
                is_leg2 = 1
                if l == p_vis and v == p_loc:
                    leg2_lead_local = float(p_gv - p_gl)

        feats = tracker.get_features_for_match(
            l, v, temp, fecha=fecha, is_knockout=is_knockout, is_neutral=is_neutral, reset_season=True,
            leg2_lead_local=leg2_lead_local, is_leg2=is_leg2
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
    # Fuera variables de plantilla sintéticas/retroactivas (scripts/build_advanced_features.py)
    cols_features = excluir_variables_estimadas(cols_features)

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
    # C de LASSO elegido con CV temporal dentro del train (auditoría: antes se elegía con el test)
    best_c, _ = seleccionar_c_lasso(X_train, y_train, grid=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

    X_full = df_dataset.loc[train_mask | cal_mask, cols_features].fillna(0.0)
    y_full = df_dataset.loc[train_mask | cal_mask, "resultado"]

    # 1-4. Modelos base regularizados y Stacking Out-Of-Fold temporal sin fuga
    pipe_lasso, pipe_rf, pipe_xgb, w_opt = entrenar_stacking_oof(
        X_full, y_full, best_c, binario=False, n_splits=4, reg_l2=0.03
    )

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
        "df_dataset": df_dataset,
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


def predecir_match(M, local, visita, temporada=None, modelo="stacking", is_neutral=0, is_knockout=0, leg2_lead_local=0.0, is_leg2=0, fecha=None):
    if temporada is None:
        temporada = _temporada_actual()
    tracker = M["tracker"]
    feats = tracker.get_features_for_match(
        local, visita, temporada, fecha=fecha, is_knockout=is_knockout, is_neutral=is_neutral,
        leg2_lead_local=leg2_lead_local, is_leg2=is_leg2
    )
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


def _predecir_lote_ucl(M, feats_list, modelo_tipo, is_neutral=False):
    """Calcula probabilidades 1X2 y lambdas Poisson para un lote de features de cruces UCL."""
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

    probs_all = np.clip(probs_all, 1e-6, 1.0 - 1e-6)
    probs_all = probs_all / probs_all.sum(axis=1, keepdims=True)

    pp = M["poisson_params"]
    elo_diffs = np.array([f["elo_diff"] for f in feats_list], dtype=np.float64)
    alt_diffs = np.array([f["altitude_diff"] for f in feats_list], dtype=np.float64)
    squad_diffs = np.array([f["squad_value_diff"] for f in feats_list], dtype=np.float64)
    h_val = 0.5 if is_neutral else 1.0
    a_val = 0.5 if is_neutral else 0.0
    eta_delta = pp["elo"] * elo_diffs + pp["alt"] * alt_diffs + pp["squad"] * squad_diffs
    la_arr = np.clip(np.exp(pp["const"] + pp["is_home"] * h_val + eta_delta), 0.2, 5.5)
    lb_arr = np.clip(np.exp(pp["const"] + pp["is_home"] * a_val - eta_delta), 0.2, 5.5)
    return probs_all, la_arr, lb_arr


def simular_campeonato(M, n_sims=10000, fijos=None, modelo="stacking", modelo_tipo=None, seed=42):
    if modelo_tipo is not None:
        modelo = modelo_tipo
    modelo_tipo = modelo or "stacking"

    fix_path = DATA / "fixture.csv"
    if not fix_path.exists():
        return obtener_tabla_actual(M)

    cache_path = DATA.parent / "simulacion_mc.pkl"
    cache_key = clave_cache_liga(__file__, DATA, n_sims, f"ucl_s{seed}", fijos=fijos)
    resultados = {}
    if not fijos and cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                saved = CustomUnpickler(fh).load()
            if saved.get("key") == cache_key:
                resultados = saved.get("results", {})
                if modelo_tipo in resultados:
                    print("Simulación UCL Monte Carlo cargada desde cache de disco")
                    return resultados[modelo_tipo]
        except Exception:
            pass

    fix = pd.read_csv(fix_path)
    fix = fix[fix.temporada == _temporada_actual()]
    tab_act = obtener_tabla_actual(M)
    eqs_all = sorted(set(fix["local"]).union(set(fix["visita"])).union(set(tab_act["equipo"])))

    pairs = [(l, v) for l in eqs_all for v in eqs_all if l != v]
    tracker = M["tracker"]
    temporada_sim = _temporada_actual()

    feats_liga = [tracker.get_features_for_match(l, v, temporada_sim) for l, v in pairs]
    probs_liga, la_liga, lb_liga = _predecir_lote_ucl(M, feats_liga, modelo_tipo, is_neutral=False)

    feats_ko = [
        tracker.get_features_for_match(l, v, temporada_sim, is_knockout=1, is_neutral=0)
        for l, v in pairs
    ]
    probs_ko, la_ko, lb_ko = _predecir_lote_ucl(M, feats_ko, modelo_tipo, is_neutral=False)

    feats_neu = [
        tracker.get_features_for_match(l, v, temporada_sim, is_knockout=1, is_neutral=1)
        for l, v in pairs
    ]
    probs_neu, la_neu, lb_neu = _predecir_lote_ucl(M, feats_neu, modelo_tipo, is_neutral=True)

    PREDS = {
        pair: (probs_liga[idx], float(la_liga[idx]), float(lb_liga[idx]))
        for idx, pair in enumerate(pairs)
    }

    # Fase de Liga dinámica: evoluciona Elo y DFORM cronológicamente y preserva estado terminal para KO
    sim_out = simular_fixture_desde_tabla(
        M,
        fix,
        tab_act,
        PREDS,
        fijos=fijos,
        n_sims=n_sims,
        k_liga=K_LIGA,
        home_adv=HOME_ADV,
        mov_mode="std",
        col_pts="puntos",
        col_dg="dg",
        col_gf="gf",
        peso_pts=1e6,
        peso_dg=1e3,
        peso_gf=1.0,
        liga="ucl",
        seed=seed,
        track_terminal_states=True,
    )

    eqs = sim_out["eqs"]
    ix = sim_out["ix"]
    order = sim_out["order"]
    pts_sims = sim_out["PTS"]
    ELO = sim_out["ELO"]
    DFORM = sim_out["DFORM"]
    elo0 = sim_out["elo0"]
    sens = sim_out["sens"]
    n_eq = len(eqs)

    # Tensores densos O(1) para todas las eliminatorias con estado dinámico
    P0_KO = np.full((n_eq, n_eq, 3), np.array([0.42, 0.27, 0.31], dtype=np.float32), dtype=np.float32)
    LA_KO = np.full((n_eq, n_eq), 1.35, dtype=np.float32)
    LB_KO = np.full((n_eq, n_eq), 1.15, dtype=np.float32)
    P0_NEU = np.full((n_eq, n_eq, 3), np.array([0.36, 0.28, 0.36], dtype=np.float32), dtype=np.float32)
    LA_NEU = np.full((n_eq, n_eq), 1.25, dtype=np.float32)
    LB_NEU = np.full((n_eq, n_eq), 1.25, dtype=np.float32)

    for idx, (l, v) in enumerate(pairs):
        il, iv = ix[l], ix[v]
        P0_KO[il, iv] = probs_ko[idx]
        LA_KO[il, iv] = la_ko[idx]
        LB_KO[il, iv] = lb_ko[idx]
        P0_NEU[il, iv] = probs_neu[idx]
        LA_NEU[il, iv] = la_neu[idx]
        LB_NEU[il, iv] = lb_neu[idx]

    def _preds_ko(ia, ib):
        return P0_KO[ia, ib], LA_KO[ia, ib], LB_KO[ia, ib]

    def _preds_final_neu(ia, ib):
        return P0_NEU[ia, ib], LA_NEU[ia, ib], LB_NEU[ia, ib]

    rng = np.random.default_rng(None if seed is None else int(seed) + 101)
    sims = np.arange(n_sims)

    counts_campeon = np.zeros(n_eq, dtype=np.int64)
    counts_final = np.zeros(n_eq, dtype=np.int64)
    counts_semi = np.zeros(n_eq, dtype=np.int64)
    counts_cuartos = np.zeros(n_eq, dtype=np.int64)
    counts_octavos = np.zeros(n_eq, dtype=np.int64)
    counts_top8 = np.zeros(n_eq, dtype=np.int64)
    counts_playoffs = np.zeros(n_eq, dtype=np.int64)
    counts_elim = np.zeros(n_eq, dtype=np.int64)

    # 1. Clasificación Fase de Liga (top 8 directo, 9-24 playoffs, 25+ eliminados)
    top8_n = min(8, n_eq)
    po_end = min(24, n_eq)
    if top8_n > 0:
        bc_top8 = np.bincount(order[:top8_n, :].ravel(), minlength=n_eq)
        counts_top8 += bc_top8
        counts_octavos += bc_top8
    if po_end > top8_n:
        counts_playoffs += np.bincount(order[top8_n:po_end, :].ravel(), minlength=n_eq)
    if n_eq > po_end:
        counts_elim += np.bincount(order[po_end:, :].ravel(), minlength=n_eq)

    if n_eq >= 24:
        # Permutación inversa vectorizada: rank_of[team_idx, sim] = ranking en Fase de Liga (0..n_eq-1)
        rank_of = np.argsort(order, axis=0)

        def _cruce_vec(t1, t2):
            t1_mejor = rank_of[t1, sims] < rank_of[t2, sims]
            a = np.where(t1_mejor, t1, t2)
            b = np.where(t1_mejor, t2, t1)
            # Ida: peor clasificado (b) juega en casa; Vuelta: mejor clasificado (a) juega en casa
            gb1, ga1, _, _, _ = simular_partido_vec_estado(
                rng, ELO, DFORM, elo0, b, a, _preds_ko, sens,
                k_liga=K_LIGA, home_adv=HOME_ADV, mov_mode="std",
            )
            ga2, gb2, _, _, _ = simular_partido_vec_estado(
                rng, ELO, DFORM, elo0, a, b, _preds_ko, sens,
                k_liga=K_LIGA, home_adv=HOME_ADV, mov_mode="std",
            )
            tot_a = ga1 + ga2
            tot_b = gb1 + gb2
            tie_a = rng.random(n_sims) > 0.5
            gana_a = (tot_a > tot_b) | ((tot_a == tot_b) & tie_a)
            return np.where(gana_a, a, b)

        pos = lambda k: order[k - 1, :]

        def _po_pareja_vec(altos, bajos):
            swap = rng.random(n_sims) < 0.5
            b0 = np.where(swap, bajos[1], bajos[0])
            b1 = np.where(swap, bajos[0], bajos[1])
            return [_cruce_vec(altos[0], b0), _cruce_vec(altos[1], b1)]

        po_por_par = {
            (1, 2): _po_pareja_vec((pos(15), pos(16)), (pos(17), pos(18))),
            (3, 4): _po_pareja_vec((pos(13), pos(14)), (pos(19), pos(20))),
            (5, 6): _po_pareja_vec((pos(11), pos(12)), (pos(21), pos(22))),
            (7, 8): _po_pareja_vec((pos(9), pos(10)), (pos(23), pos(24))),
        }
        for ganadores in po_por_par.values():
            for w in ganadores:
                counts_octavos += np.bincount(w, minlength=n_eq)

        mitad_A, mitad_B = {}, {}
        for (s1, s2), ganadores in po_por_par.items():
            t0, t1 = pos(s1), pos(s2)
            swap_t = rng.random(n_sims) < 0.5
            top_a = np.where(swap_t, t1, t0)
            top_b = np.where(swap_t, t0, t1)
            swap_r = rng.random(n_sims) < 0.5
            riv_a = np.where(swap_r, ganadores[1], ganadores[0])
            riv_b = np.where(swap_r, ganadores[0], ganadores[1])
            mitad_A[(s1, s2)] = (top_a, riv_a)
            mitad_B[(s1, s2)] = (top_b, riv_b)

        finalists = []
        for mitad in (mitad_A, mitad_B):
            octavos = {par: _cruce_vec(*duelo) for par, duelo in mitad.items()}
            for w in octavos.values():
                counts_cuartos += np.bincount(w, minlength=n_eq)
            semi_1 = _cruce_vec(octavos[(1, 2)], octavos[(7, 8)])
            semi_2 = _cruce_vec(octavos[(3, 4)], octavos[(5, 6)])
            counts_semi += np.bincount(semi_1, minlength=n_eq) + np.bincount(semi_2, minlength=n_eq)
            finalista = _cruce_vec(semi_1, semi_2)
            counts_final += np.bincount(finalista, minlength=n_eq)
            finalists.append(finalista)

        # Gran Final en sede neutral (home_adv=0.0, P0_NEU/LA_NEU/LB_NEU)
        ga_f, gb_f, _, _, _ = simular_partido_vec_estado(
            rng, ELO, DFORM, elo0, finalists[0], finalists[1], _preds_final_neu, sens,
            k_liga=K_LIGA, home_adv=0.0, mov_mode="std",
        )
        tie_f = rng.random(n_sims) > 0.5
        champion = np.where((ga_f > gb_f) | ((ga_f == gb_f) & tie_f), finalists[0], finalists[1])
        counts_campeon += np.bincount(champion, minlength=n_eq)

    res = []
    for idx_e, eq in enumerate(eqs):
        mean_pts = float(np.mean(pts_sims[idx_e, :]))
        res.append({
            "equipo": eq,
            "Selección": eq,
            "Puntos esperados": round(mean_pts, 1),
            "P_campeon": float(counts_campeon[idx_e]) / n_sims,
            "P_final": float(counts_final[idx_e]) / n_sims,
            "P_semi": float(counts_semi[idx_e]) / n_sims,
            "P_cuartos": float(counts_cuartos[idx_e]) / n_sims,
            "P_octavos": float(counts_octavos[idx_e]) / n_sims,
            "P_top8": float(counts_top8[idx_e]) / n_sims,
            "P_playoffs": float(counts_playoffs[idx_e]) / n_sims,
            "P_eliminado": float(counts_elim[idx_e]) / n_sims,
        })

    df_res = pd.DataFrame(res).sort_values(
        by=["P_campeon", "P_final", "P_top8", "Puntos esperados"], ascending=[False, False, False, False]
    ).reset_index(drop=True)
    df_res.index = df_res.index + 1

    if not fijos:
        resultados[modelo_tipo] = df_res
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump({"key": cache_key, "results": resultados}, fh)
            print("Simulación UCL guardada en cache de disco")
        except Exception as ex:
            print(f"No se pudo guardar el cache de simulación: {ex}")
    return df_res

monte_carlo = simular_campeonato


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

    # Cada partido se predice con las variables que existían ANTES de jugarlo, no con el
    # tracker final (que ya incluye su resultado y los de todos los partidos posteriores).
    pre_partido = _modelo_pre_partido(M, "df_dataset")
    val_df = val_df[[pre_partido(r.fecha, r.local, r.visita) is not None
                     for r in val_df.itertuples(index=False)]].copy()
    if len(val_df) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    probs = []
    for _, r in val_df.iterrows():
        p = predecir_match(
            pre_partido(r["fecha"], r["local"], r["visita"]), r["local"], r["visita"], temporada=temporada_val, modelo=modelo_tipo,
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


def generar_reporte_partidos_ia(M, df_matches=None, modelo="stacking"):
    """
    Genera un DataFrame enriquecido con todas las métricas predictivas, xG Poisson,
    diferenciales de ELO, valor de plantilla, forma y descanso para exportación
    a CSV y análisis con modelos de IA (ChatGPT, Claude, Gemini, etc.).
    """
    if df_matches is None:
        p_fix = DATA / "fixture.csv"
        if p_fix.exists():
            df_matches = pd.read_csv(p_fix, parse_dates=["fecha"])
            df_matches = df_matches[df_matches["temporada"] == _temporada_actual()]
        else:
            return pd.DataFrame()

    tracker = M["tracker"]
    temp_actual = _temporada_actual()
    filas = []

    for _, r in df_matches.iterrows():
        l, v = r["local"], r["visita"]
        f = pd.to_datetime(r["fecha"]) if pd.notna(r.get("fecha")) else None
        st_name = r.get("stage", "league-phase")
        is_ko = int(r.get("is_knockout", 0))
        is_neu = int(r.get("is_neutral", 0))

        p_main, la, lb = predecir_match(M, l, v, temporada=temp_actual, modelo=modelo, is_neutral=is_neu, is_knockout=is_ko, fecha=f)
        p_lasso = predecir_match(M, l, v, temporada=temp_actual, modelo="lasso", is_neutral=is_neu, is_knockout=is_ko, fecha=f)[0]
        p_rf    = predecir_match(M, l, v, temporada=temp_actual, modelo="rf", is_neutral=is_neu, is_knockout=is_ko, fecha=f)[0]
        p_xgb   = predecir_match(M, l, v, temporada=temp_actual, modelo="xgb", is_neutral=is_neu, is_knockout=is_ko, fecha=f)[0]
        p_stk   = predecir_match(M, l, v, temporada=temp_actual, modelo="stacking", is_neutral=is_neu, is_knockout=is_ko, fecha=f)[0]

        mat = matriz_marcador_exacto(la, lb)
        mk = mercados(mat)
        top_sc = mk.get("_top_marcadores", [])

        elo_l = tracker.elos[l]
        elo_v = tracker.elos[v]
        val_l = get_squad_value(l, temp_actual)
        val_v = get_squad_value(v, temp_actual)

        dom_l = tracker.multi_leagues.get_team_domestic_features(l, f)
        dom_v = tracker.multi_leagues.get_team_domestic_features(v, f)

        # Alerta Heurística
        if p_main[0] > 0.55 and (la - lb) > 0.8:
            alerta = "🔥 Alta Confianza Local"
        elif p_main[0] > 0.60:
            alerta = "💪 Favorito Claro Local"
        elif p_main[2] > 0.50:
            alerta = "⚠️ Visita Fuerte"
        elif max(p_main) < 0.44:
            alerta = "⚖️ Partido Muy Parejo"
        elif p_main[1] > 0.28:
            alerta = "🤝 Alta Probabilidad Empate"
        else:
            alerta = "🎯 Pronóstico Estándar"

        filas.append({
            "id_partido": r.get("event_id", ""),
            "fecha": f.strftime("%Y-%m-%d %H:%M") if f is not None else "",
            "fase": st_name,
            "local": l,
            "visita": v,
            "prob_victoria_local_%": round(float(p_main[0] * 100), 1),
            "prob_empate_%": round(float(p_main[1] * 100), 1),
            "prob_victoria_visita_%": round(float(p_main[2] * 100), 1),
            "cuota_justa_local": round(float(cuota(p_main[0])), 2),
            "cuota_justa_empate": round(float(cuota(p_main[1])), 2),
            "cuota_justa_visita": round(float(cuota(p_main[2])), 2),
            "xg_local": round(float(la), 2),
            "xg_visita": round(float(lb), 2),
            "marcador_mas_probable": f"{top_sc[0][0]}-{top_sc[0][1]} ({top_sc[0][2]:.1%})" if len(top_sc) > 0 else "",
            "marcador_alternativo": f"{top_sc[1][0]}-{top_sc[1][1]} ({top_sc[1][2]:.1%})" if len(top_sc) > 1 else "",
            "prob_over_2_5_%": round(float(mk.get("Over 2.5", 0.5) * 100), 1),
            "prob_ambos_marcan_%": round(float(mk.get("Ambos marcan (BTTS sí)", 0.5) * 100), 1),
            "elo_local": round(float(elo_l), 1),
            "elo_visita": round(float(elo_v), 1),
            "elo_diff_con_localia": round(float(elo_l + (0.0 if is_neu else HOME_ADV) - elo_v), 1),
            "plantilla_local_M€": round(float(val_l), 1),
            "plantilla_visita_M€": round(float(val_v), 1),
            "ratio_plantilla_local_vs_visita": round(float(val_l / max(val_v, 0.1)), 2),
            "liga_local": MAPPING_UCL_A_LOCAL.get(l, ("Otro", ""))[0].upper(),
            "liga_visita": MAPPING_UCL_A_LOCAL.get(v, ("Otro", ""))[0].upper(),
            "elo_domestico_adj_local": round(float(dom_l["dom_elo_adj"]), 1),
            "elo_domestico_adj_visita": round(float(dom_v["dom_elo_adj"]), 1),
            "forma_liga_5p_local_%": round(float(dom_l["dom_form"] * 100), 1),
            "forma_liga_5p_visita_%": round(float(dom_v["dom_form"] * 100), 1),
            "ppg_liga_local": round(float(dom_l["dom_ppg"]), 2),
            "ppg_liga_visita": round(float(dom_v["dom_ppg"]), 2),
            "descanso_dias_local": int(dom_l["real_rest_days"]),
            "descanso_dias_visita": int(dom_v["real_rest_days"]),
            "ventaja_descanso_dias": int(dom_l["real_rest_days"] - dom_v["real_rest_days"]),
            "partidos_ultimos_14d_local": int(dom_l["real_cong_14d"]),
            "partidos_ultimos_14d_visita": int(dom_v["real_cong_14d"]),
            "consenso_lasso_L_D_V": f"{p_lasso[0]:.1%}/{p_lasso[1]:.1%}/{p_lasso[2]:.1%}",
            "consenso_rf_L_D_V": f"{p_rf[0]:.1%}/{p_rf[1]:.1%}/{p_rf[2]:.1%}",
            "consenso_xgb_L_D_V": f"{p_xgb[0]:.1%}/{p_xgb[1]:.1%}/{p_xgb[2]:.1%}",
            "consenso_stacking_L_D_V": f"{p_stk[0]:.1%}/{p_stk[1]:.1%}/{p_stk[2]:.1%}",
            "alerta_modelo": alerta
        })

    return pd.DataFrame(filas)


if __name__ == "__main__":
    M = cargar(force_retrain=True)
    print("\n--- TEST PREDECIR MATCH (Real Madrid vs Bayern Munich) ---")
    p, la, lb = predecir_match(M, "Real Madrid", "Bayern Munich")
    print(f"Probabilidades: Local={p[0]:.3f}, Empate={p[1]:.3f}, Visita={p[2]:.3f}")
    print(f"xG esperado: RM={la:.2f}, Bayern={lb:.2f}")

    print("\n--- TEST SIMULACIÓN MONTE CARLO (Fase de Liga 36 + Eliminatorias) ---")
    df_sim = simular_campeonato(M, n_sims=20)
    print(df_sim.head(10))
