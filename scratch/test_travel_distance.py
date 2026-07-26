"""
Experiment: Testing Travel Distance (Haversine distance in km from Away team city to Home team stadium)
across Chile, Mexico, Brazil, and Spain.
"""
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chile.motor as m_chi
import mex.motor as m_mex
import bra.motor as m_bra
import esp.motor as m_esp

# Coordinates (lat, lon) for cities/teams
COORDS = {
    # 🇨🇱 Chile
    "Colo Colo": (-33.4489, -70.6693), "Universidad de Chile": (-33.4489, -70.6693),
    "Universidad Católica": (-33.4489, -70.6693), "Palestino": (-33.4489, -70.6693),
    "Audax Italiano": (-33.4489, -70.6693), "Unión Española": (-33.4489, -70.6693),
    "Magallanes": (-33.4489, -70.6693), "Barnechea": (-33.4489, -70.6693),
    "Cobreloa": (-22.4544, -68.9292), "Deportes Iquique": (-20.2307, -70.1357),
    "Coquimbo Unido": (-29.9533, -71.3436), "Deportes La Serena": (-29.9027, -71.2520),
    "Everton": (-33.0245, -71.5518), "Santiago Wanderers": (-33.0472, -71.6127),
    "Huachipato": (-36.8270, -73.0503), "Deportes Concepción": (-36.8270, -73.0503),
    "Universidad de Concepción": (-36.8270, -73.0503), "Ñublense": (-36.6067, -72.1034),
    "Cobresal": (-26.2464, -69.6258), "O'Higgins": (-34.1701, -70.7444),
    "Curicó Unido": (-34.9854, -71.2394), "Rangers": (-35.4264, -71.6554),
    "Deportes Antofagasta": (-23.6509, -70.3975), "Deportes Copiapó": (-27.3668, -70.3323),
    "Unión La Calera": (-32.7882, -71.1896), "San Luis": (-32.8804, -71.2483),
    "Deportes Puerto Montt": (-41.4693, -72.9424),

    # 🇲🇽 México
    "América": (19.4326, -99.1332), "Cruz Azul": (19.4326, -99.1332), "Pumas UNAM": (19.4326, -99.1332),
    "Guadalajara": (20.6597, -103.3496), "Atlas": (20.6597, -103.3496),
    "Tigres UANL": (25.6866, -100.3161), "Monterrey": (25.6866, -100.3161),
    "Toluca": (19.2826, -99.6557), "Santos Laguna": (25.5428, -103.4068),
    "Pachuca": (20.1011, -98.7591), "León": (21.1236, -101.6837),
    "Tijuana": (32.5149, -117.0382), "FC Juárez": (31.6904, -106.4245),
    "Puebla": (19.0414, -98.2063), "Necaxa": (21.8853, -102.2916),
    "Querétaro": (20.5888, -100.3899), "Mazatlán FC": (23.2494, -106.4111),
    "Atlético San Luis": (22.1565, -100.9855), "Morelia": (19.7060, -101.1950),

    # 🇧🇷 Brasil
    "Flamengo": (-22.9068, -43.1729), "Fluminense": (-22.9068, -43.1729),
    "Botafogo": (-22.9068, -43.1729), "Vasco da Gama": (-22.9068, -43.1729),
    "Palmeiras": (-23.5505, -46.6333), "São Paulo": (-23.5505, -46.6333),
    "Corinthians": (-23.5505, -46.6333), "Santos": (-23.9608, -46.3339),
    "Red Bull Bragantino": (-22.9525, -46.5419), "Grêmio": (-30.0346, -51.2177),
    "Internacional": (-30.0346, -51.2177), "Atlético Mineiro": (-19.9167, -43.9345),
    "Cruzeiro": (-19.9167, -43.9345), "Athletico Paranaense": (-25.4284, -49.2733),
    "Coritiba": (-25.4284, -49.2733), "Bahia": (-12.9777, -38.5016),
    "Vitória": (-12.9777, -38.5016), "Fortaleza": (-3.7319, -38.5267),
    "Ceará": (-3.7319, -38.5267), "Cuiabá": (-15.6010, -56.0979),
    "Goiás": (-16.6869, -49.2648), "Atlético Goianiense": (-16.6869, -49.2648),
    "Juventude": (-29.1678, -51.1794), "Chapecoense": (-27.1004, -52.6152),
    "América Mineiro": (-19.9167, -43.9345), "Sport Recife": (-8.0476, -34.8770),

    # 🇪🇸 España
    "Real Madrid": (40.4168, -3.7038), "Atlético de Madrid": (40.4168, -3.7038),
    "Rayo Vallecano": (40.4168, -3.7038), "Getafe": (40.3047, -3.7327),
    "Leganés": (40.3282, -3.7635), "FC Barcelona": (41.3879, 2.1699),
    "Espanyol": (41.3879, 2.1699), "Sevilla FC": (37.3891, -5.9845),
    "Real Betis": (37.3891, -5.9845), "Athletic Club": (43.2630, -2.9350),
    "Real Sociedad": (43.3183, -1.9812), "Deportivo Alavés": (42.8467, -2.6716),
    "Valencia CF": (39.4699, -0.3763), "Villarreal CF": (39.9378, -0.1006),
    "Celta de Vigo": (42.2406, -8.7207), "RCD Mallorca": (39.5696, 2.6502),
    "UD Las Palmas": (28.1235, -15.4363), "CA Osasuna": (42.8125, -1.6458),
    "Girona FC": (41.9794, 2.8214), "Cádiz CF": (36.5271, -6.2886),
    "Granada CF": (37.1773, -3.5986), "UD Almería": (36.8340, -2.4637),
    "Elche CF": (38.2669, -0.6983), "SD Eibar": (43.1843, -2.4716)
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def get_distance_km(local, visita):
    c_loc = COORDS.get(local)
    c_vis = COORDS.get(visita)
    if not c_loc or not c_vis:
        return 0.0
    return haversine_km(c_vis[0], c_vis[1], c_loc[0], c_loc[1])

def benchmark_travel(league_name, motor_mod):
    print(f"\n==================================================")
    print(f" 🏆 Evaluando Distancia de Viaje en: {league_name}")
    print(f"==================================================")

    M = motor_mod.cargar()
    df = M.get("df_dataset", M.get("df_features")).copy()
    df = df.sort_values("fecha").reset_index(drop=True)

    dist_kms = []
    for r in df.itertuples():
        loc = getattr(r, "local")
        vis = getattr(r, "visita")
        d = get_distance_km(loc, vis)
        # Log scale distance in km
        dist_kms.append(np.log1p(d))

    df["travel_dist_log_km"] = dist_kms

    feat_key = "features" if "features" in M else "cols"
    base_cols = [c for c in M[feat_key] if c != "travel_dist_log_km"]
    with_travel_cols = list(base_cols) + ["travel_dist_log_km"]

    train_mask = df["temporada"] <= 2023
    cal_mask   = df["temporada"] == 2024
    test_mask  = df["temporada"] >= 2025

    def eval_cols(cols, label):
        X_tr = df.loc[train_mask, cols].fillna(0.0)
        y_tr = df.loc[train_mask, "resultado"]
        X_cal = df.loc[cal_mask, cols].fillna(0.0)
        y_cal = df.loc[cal_mask, "resultado"]
        X_te = df.loc[test_mask, cols].fillna(0.0)
        y_te = df.loc[test_mask, "resultado"]

        p_l = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=3000, random_state=42))])
        p_l.fit(X_tr, y_tr)

        p_r = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=15, random_state=42, n_jobs=-1))])
        p_r.fit(X_tr, y_tr)

        p_l_cal = p_l.predict_proba(X_cal)
        p_r_cal = p_r.predict_proba(X_cal)
        def _stk(a): return log_loss(y_cal, np.clip(a*p_l_cal + (1-a)*p_r_cal, 1e-7, 1-1e-7))
        alpha = float(minimize_scalar(_stk, bounds=(0.0, 1.0), method="bounded").x)

        p_l_te = p_l.predict_proba(X_te)
        p_r_te = p_r.predict_proba(X_te)
        p_st = np.clip(alpha*p_l_te + (1-alpha)*p_r_te, 1e-7, 1-1e-7)

        ll_l = log_loss(y_te, p_l_te)
        acc_l = accuracy_score(y_te, p_l_te.argmax(axis=1))*100
        ll_r = log_loss(y_te, p_r_te)
        acc_r = accuracy_score(y_te, p_r_te.argmax(axis=1))*100
        ll_st = log_loss(y_te, p_st)
        acc_st = accuracy_score(y_te, p_st.argmax(axis=1))*100

        print(f"  [{label:<28}] LASSO LL={ll_l:.4f} Acc={acc_l:.2f}% | RF LL={ll_r:.4f} Acc={acc_r:.2f}% | Stacking LL={ll_st:.4f} Acc={acc_st:.2f}% (α={alpha:.3f})")

    eval_cols(base_cols, "1. BASELINE (Sin Viaje)")
    eval_cols(with_travel_cols, "2. CON DISTANCIA VIAJE (km)")

if __name__ == "__main__":
    benchmark_travel("Liga Chilena", m_chi)
    benchmark_travel("Liga MX", m_mex)
    benchmark_travel("Brasileirão", m_bra)
    benchmark_travel("LaLiga España", m_esp)
