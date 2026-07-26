"""
Experiment: Evaluate adding `ppg_diff` (points per game difference in current season)
across all 4 prediction engines (MEX, BRA, CHI, ESP).
Compares baseline metrics (without ppg_diff) vs advanced metrics (with ppg_diff).
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

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mex.motor as m_mex
import bra.motor as m_bra
import chile.motor as m_chi
import esp.motor as m_esp


def run_benchmark_for_league(league_name, motor_mod):
    print(f"\n==================================================")
    print(f" 🏆 Evaluando Liga: {league_name}")
    print(f"==================================================")

    # 1. cargar datos
    M = motor_mod.cargar()
    df_raw = M.get("df_dataset", M.get("df_features")).copy()

    # Reconstruir StateTracker con ppg_diff
    # Vamos a simular las features base y agregar ppg_diff
    df_raw = df_raw.sort_values("fecha").reset_index(drop=True)

    # Calcular ppg_diff acumulado en el tiempo partido a partido por temporada
    pts_season = {}
    matches_season = {}
    curr_season = None

    ppg_diffs = []
    for r in df_raw.itertuples():
        temp = getattr(r, "temporada")
        if temp != curr_season:
            curr_season = temp
            pts_season = {}
            matches_season = {}

        loc = getattr(r, "local")
        vis = getattr(r, "visita")

        pl = pts_season.get(loc, 0)
        ml = matches_season.get(loc, 0)
        pv = pts_season.get(vis, 0)
        mv = matches_season.get(vis, 0)

        ppg_l = (pl / ml) if ml > 0 else 1.33 # Promedio por defecto
        ppg_v = (pv / mv) if mv > 0 else 1.33

        ppg_diffs.append(ppg_l - ppg_v)

        # Actualizar puntos post partido
        ga = int(getattr(r, "goles_local"))
        gb = int(getattr(r, "goles_visita"))
        if ga > gb:
            pts_season[loc] = pl + 3
            pts_season[vis] = pv
        elif ga == gb:
            pts_season[loc] = pl + 1
            pts_season[vis] = pv + 1
        else:
            pts_season[loc] = pl
            pts_season[vis] = pv + 3
        matches_season[loc] = ml + 1
        matches_season[vis] = mv + 1

    df_raw["ppg_diff"] = ppg_diffs

    features_key = "features" if "features" in M else "cols"
    base_cols = [c for c in M[features_key] if c != "ppg_diff"]

    with_ppg_cols = list(base_cols) + ["ppg_diff"]

    train_mask = df_raw["temporada"] <= 2023
    cal_mask   = df_raw["temporada"] == 2024
    test_mask  = df_raw["temporada"] >= 2025

    def fit_eval(cols):
        X_tr = df_raw.loc[train_mask, cols].fillna(0.0)
        y_tr = df_raw.loc[train_mask, "resultado"]
        X_cal = df_raw.loc[cal_mask, cols].fillna(0.0)
        y_cal = df_raw.loc[cal_mask, "resultado"]
        X_te = df_raw.loc[test_mask, cols].fillna(0.0)
        y_te = df_raw.loc[test_mask, "resultado"]

        # Lasso
        p_l = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=3000, random_state=42))])
        p_l.fit(X_tr, y_tr)

        # RF
        p_r = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=15, random_state=42, n_jobs=-1))])
        p_r.fit(X_tr, y_tr)

        # Stacking alpha on cal
        p_l_cal = p_l.predict_proba(X_cal)
        p_r_cal = p_r.predict_proba(X_cal)
        def _stk_loss(a):
            b = np.clip(a*p_l_cal + (1-a)*p_r_cal, 1e-7, 1-1e-7)
            return log_loss(y_cal, b)
        alpha_opt = float(minimize_scalar(_stk_loss, bounds=(0.0, 1.0), method="bounded").x)

        # Evaluate on Test (>=2025)
        p_l_te = p_l.predict_proba(X_te)
        p_r_te = p_r.predict_proba(X_te)
        p_st_te = np.clip(alpha_opt*p_l_te + (1-alpha_opt)*p_r_te, 1e-7, 1-1e-7)

        ll_l = log_loss(y_te, p_l_te)
        acc_l = accuracy_score(y_te, p_l_te.argmax(axis=1)) * 100

        ll_r = log_loss(y_te, p_r_te)
        acc_r = accuracy_score(y_te, p_r_te.argmax(axis=1)) * 100

        ll_st = log_loss(y_te, p_st_te)
        acc_st = accuracy_score(y_te, p_st_te.argmax(axis=1)) * 100

        # Importancia relativa ppg_diff en LASSO vs elo_diff
        coefs = p_l.named_steps["lr"].coef_
        avg_coefs = np.mean(np.abs(coefs), axis=0)
        coef_dict = dict(zip(cols, avg_coefs))

        return {
            "lasso": (ll_l, acc_l),
            "rf": (ll_r, acc_r),
            "stacking": (ll_st, acc_st, alpha_opt),
            "coefs": coef_dict
        }

    res_base = fit_eval(base_cols)
    res_ppg  = fit_eval(with_ppg_cols)

    print(f"\n--- 1. BASELINE (Sin ppg_diff) ---")
    print(f"  LASSO    : Log-Loss = {res_base['lasso'][0]:.4f} | Acc = {res_base['lasso'][1]:.2f}%")
    print(f"  RF       : Log-Loss = {res_base['rf'][0]:.4f} | Acc = {res_base['rf'][1]:.2f}%")
    print(f"  Stacking : Log-Loss = {res_base['stacking'][0]:.4f} | Acc = {res_base['stacking'][1]:.2f}% (α={res_base['stacking'][2]:.3f})")

    print(f"\n--- 2. CON ppg_diff (Puntos Por Partido en Torneo Actual) ---")
    print(f"  LASSO    : Log-Loss = {res_ppg['lasso'][0]:.4f} | Acc = {res_ppg['lasso'][1]:.2f}%")
    print(f"  RF       : Log-Loss = {res_ppg['rf'][0]:.4f} | Acc = {res_ppg['rf'][1]:.2f}%")
    print(f"  Stacking : Log-Loss = {res_ppg['stacking'][0]:.4f} | Acc = {res_ppg['stacking'][1]:.2f}% (α={res_ppg['stacking'][2]:.3f})")

    # Comparación directa
    diff_ll = res_ppg["stacking"][0] - res_base["stacking"][0]
    diff_acc = res_ppg["stacking"][1] - res_base["stacking"][1]
    print(f"\n--- 📊 IMPACTO EN STACKING ---")
    print(f"  Δ Log-Loss : {diff_ll:+.4f} ({'✅ MEJORA' if diff_ll < 0 else '❌ EMPEORA/IGUAL'})")
    print(f"  Δ Accuracy : {diff_acc:+.2f}% ({'✅ MEJORA' if diff_acc > 0 else '❌ EMPEORA/IGUAL'})")

    if "ppg_diff" in res_ppg["coefs"]:
        c_ppg = res_ppg["coefs"]["ppg_diff"]
        c_elo = res_ppg["coefs"].get("elo_diff", 0.0)
        print(f"  Peso LASSO `ppg_diff`: {c_ppg:.4f} vs `elo_diff`: {c_elo:.4f}")

if __name__ == "__main__":
    run_benchmark_for_league("Liga MX", m_mex)
    run_benchmark_for_league("Brasileirão", m_bra)
    run_benchmark_for_league("Liga Chilena", m_chi)
    run_benchmark_for_league("LaLiga España", m_esp)
