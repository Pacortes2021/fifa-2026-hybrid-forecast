"""
Experiment: Testing Pi-Ratings (Constantinou & Fenton 2013)
Replaces or complements ELO in match outcome prediction.
Measures out-of-sample (>=2025) Log-Loss & Accuracy across all 4 leagues.
"""
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
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


class PiRatingTracker:
    def __init__(self, lambda_param=0.035, gamma_param=0.70):
        self.r_home = defaultdict(float) # R_H(i), init 0.0
        self.r_away = defaultdict(float) # R_A(i), init 0.0
        self.lmb = lambda_param
        self.gamma = gamma_param

    def predict_margin(self, home, away):
        rh = self.r_home[home]
        ra = self.r_away[away]
        diff = rh - ra
        # Constantinou expected goal difference formula:
        # e_hat = sign(diff) * (10^(|diff|/3) - 1)
        # simplified linear/sigmoid scaling for stability:
        return diff

    def get_features(self, home, away):
        rh = self.r_home[home]
        ra = self.r_away[away]
        pi_diff = rh - ra
        pi_overall_l = (self.r_home[home] + self.r_away[home]) / 2.0
        pi_overall_v = (self.r_home[away] + self.r_away[away]) / 2.0
        return {
            "pi_diff": pi_diff,
            "pi_home_rating": rh,
            "pi_away_rating": ra,
            "pi_overall_diff": pi_overall_l - pi_overall_v
        }

    def update(self, home, away, goals_home, goals_away):
        e_actual = goals_home - goals_away
        rh = self.r_home[home]
        ra = self.r_away[away]
        e_hat = rh - ra

        error = e_actual - e_hat

        # Update Home team: R_H(H) and R_A(H) via cross effect gamma
        self.r_home[home] += self.lmb * error
        self.r_away[home] += self.gamma * self.lmb * error

        # Update Away team: R_A(A) and R_H(A) via cross effect gamma
        self.r_away[away] -= self.lmb * error
        self.r_home[away] -= self.gamma * self.lmb * error


def benchmark_pi_ratings(league_name, motor_mod):
    print(f"\n==================================================")
    print(f" 🏆 Evaluando Pi-Ratings en: {league_name}")
    print(f"==================================================")

    M = motor_mod.cargar()
    df = M.get("df_dataset", M.get("df_features")).copy()
    df = df.sort_values("fecha").reset_index(drop=True)

    pi_tracker = PiRatingTracker(lambda_param=0.035, gamma_param=0.70)
    pi_diffs = []
    pi_home_r = []
    pi_away_r = []
    pi_overall_d = []

    for r in df.itertuples():
        loc = getattr(r, "local")
        vis = getattr(r, "visita")
        ga  = int(getattr(r, "goles_local"))
        gb  = int(getattr(r, "goles_visita"))

        pfeats = pi_tracker.get_features(loc, vis)
        pi_diffs.append(pfeats["pi_diff"])
        pi_home_r.append(pfeats["pi_home_rating"])
        pi_away_r.append(pfeats["pi_away_rating"])
        pi_overall_d.append(pfeats["pi_overall_diff"])

        pi_tracker.update(loc, vis, ga, gb)

    df["pi_diff"] = pi_diffs
    df["pi_home_rating"] = pi_home_r
    df["pi_away_rating"] = pi_away_r
    df["pi_overall_diff"] = pi_overall_d

    feat_key = "features" if "features" in M else "cols"
    base_cols = [c for c in M[feat_key] if c not in ("pi_diff", "pi_home_rating", "pi_away_rating", "pi_overall_diff")]
    
    # 1. Base (ELO)
    # 2. Con Pi-Ratings agregado
    # 3. Pi-Ratings reemplazando ELO
    with_pi_cols = list(base_cols) + ["pi_diff", "pi_overall_diff"]
    replace_elo_cols = [c for c in base_cols if c != "elo_diff"] + ["pi_diff", "pi_overall_diff"]

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

        print(f"  [{label:<30}] LASSO LL={ll_l:.4f} Acc={acc_l:.2f}% | RF LL={ll_r:.4f} Acc={acc_r:.2f}% | Stacking LL={ll_st:.4f} Acc={acc_st:.2f}% (α={alpha:.3f})")

    eval_cols(base_cols, "1. BASELINE (Solo ELO)")
    eval_cols(with_pi_cols, "2. HÍBRIDO (ELO + Pi-Ratings)")
    eval_cols(replace_elo_cols, "3. SOLO Pi-Ratings (Sin ELO)")

if __name__ == "__main__":
    benchmark_pi_ratings("Liga Chilena", m_chi)
    benchmark_pi_ratings("Liga MX", m_mex)
    benchmark_pi_ratings("Brasileirão", m_bra)
    benchmark_pi_ratings("LaLiga España", m_esp)
