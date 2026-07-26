"""
Experiment: Testing Fatigue Factor features (Rest Days & 14-day Congestion)
out-of-sample (>=2025) across all 4 leagues.
"""
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, deque
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


def benchmark_fatigue(league_name, motor_mod):
    print(f"\n==================================================")
    print(f" 🏆 Evaluando Fatigue Factor en: {league_name}")
    print(f"==================================================")

    M = motor_mod.cargar()
    df = M.get("df_dataset", M.get("df_features")).copy()
    df = df.sort_values("fecha").reset_index(drop=True)

    # Reconstruir días de descanso y congestión
    last_date = defaultdict(lambda: None)
    recent_dates = defaultdict(list)

    rest_diffs = []
    congest_diffs = []

    for r in df.itertuples():
        f = pd.to_datetime(getattr(r, "fecha"))
        loc = getattr(r, "local")
        vis = getattr(r, "visita")

        # Rest days
        d_loc = last_date[loc]
        d_vis = last_date[vis]

        rest_l = (f - d_loc).days if d_loc is not None else 7.0
        rest_v = (f - d_vis).days if d_vis is not None else 7.0

        # Cap between 3 and 14
        rest_l = float(np.clip(rest_l, 3.0, 14.0))
        rest_v = float(np.clip(rest_v, 3.0, 14.0))
        rest_diffs.append(rest_l - rest_v)

        # 14-day congestion
        c_l = sum(1 for d in recent_dates[loc] if (f - d).days <= 14)
        c_v = sum(1 for d in recent_dates[vis] if (f - d).days <= 14)
        congest_diffs.append(float(c_l - c_v))

        # Update post match
        last_date[loc] = f
        last_date[vis] = f
        recent_dates[loc].append(f)
        recent_dates[vis].append(f)

    df["rest_days_diff"] = rest_diffs
    df["congestion_14d_diff"] = congest_diffs

    feat_key = "features" if "features" in M else "cols"
    base_cols = [c for c in M[feat_key] if c not in ("rest_days_diff", "congestion_14d_diff")]
    with_fatigue_cols = list(base_cols) + ["rest_days_diff", "congestion_14d_diff"]

    train_mask = df["temporada"] <= 2023
    cal_mask   = df["temporada"] == 2024
    test_mask  = df["temporada"] >= 2025

    def eval_model(cols):
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

        return (ll_l, acc_l), (ll_r, acc_r), (ll_st, acc_st, alpha)

    res_b = eval_model(base_cols)
    res_f = eval_model(with_fatigue_cols)

    print(f"\n--- BASELINE (Sin Fatigue Factor) ---")
    print(f"  LASSO    : Log-Loss = {res_b[0][0]:.4f} | Acc = {res_b[0][1]:.2f}%")
    print(f"  RF       : Log-Loss = {res_b[1][0]:.4f} | Acc = {res_b[1][1]:.2f}%")
    print(f"  Stacking : Log-Loss = {res_b[2][0]:.4f} | Acc = {res_b[2][1]:.2f}% (α={res_b[2][2]:.3f})")

    print(f"\n--- CON FATIGUE FACTOR (rest_days_diff & congestion_14d_diff) ---")
    print(f"  LASSO    : Log-Loss = {res_f[0][0]:.4f} | Acc = {res_f[0][1]:.2f}%")
    print(f"  RF       : Log-Loss = {res_f[1][0]:.4f} | Acc = {res_f[1][1]:.2f}%")
    print(f"  Stacking : Log-Loss = {res_f[2][0]:.4f} | Acc = {res_f[2][1]:.2f}% (α={res_f[2][2]:.3f})")

    diff_ll = res_f[2][0] - res_b[2][0]
    diff_acc = res_f[2][1] - res_b[2][1]
    print(f"\n--- 📊 IMPACTO EN STACKING ---")
    print(f"  Δ Log-Loss : {diff_ll:+.4f} ({'✅ MEJORA' if diff_ll < 0 else '❌ EMPEORA/IGUAL'})")
    print(f"  Δ Accuracy : {diff_acc:+.2f}% ({'✅ MEJORA' if diff_acc > 0 else '❌ EMPEORA/IGUAL'})")

if __name__ == "__main__":
    benchmark_fatigue("Liga Chilena", m_chi)
    benchmark_fatigue("Liga MX", m_mex)
    benchmark_fatigue("Brasileirão", m_bra)
    benchmark_fatigue("LaLiga España", m_esp)
