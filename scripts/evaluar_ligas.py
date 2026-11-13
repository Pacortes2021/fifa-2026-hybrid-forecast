"""
Evaluador universal de motores: carga cada competición y vuelca sus métricas de test
fuera de muestra (`logloss`, `brier`, `accuracy`, `logloss_elo`, `delta_vs_elo`, IC 95%
por bootstrap y `n_test`) a un CSV comparable entre versiones del código.

Uso:
    python3 scripts/evaluar_ligas.py --salida outputs/evaluacion_ligas.csv
    python3 scripts/evaluar_ligas.py --salida outputs/evaluacion_despues.csv fra ucl
    python3 scripts/evaluar_ligas.py --comparar outputs/evaluacion_antes.csv outputs/evaluacion_despues.csv

Las métricas evalúan el bloque de TEST temporal fuera de muestra y se contrastan pareadamente
contra el baseline Solo Elo entrenado estrictamente sin fuga del bloque test.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from comun.motor_comun import (
    corte_test_temporada,
    entrenar_baseline_elo,
    etiquetas_y_orden_1x2,
    proba_pipe_1x2,
)

LIGAS = [
    "mex", "arg", "bra", "chile", "esp", "eng", "bund", "ita", "fra", "ned", "por",
    "bel", "tur", "sco", "aut", "den", "gre", "nor", "swe", "ucl", "unl", "nba",
]


def _importar_motor(liga):
    for m in [k for k in list(sys.modules) if k == "motor" or k.startswith("motor.")]:
        del sys.modules[m]
    sys.path = [str(REPO / liga), str(REPO)] + [p for p in sys.path if p not in [str(REPO / l) for l in LIGAS]]
    return __import__("motor")


def _boot_ic95(vals, rng, n_boot=500, signed=False):
    n = len(vals)
    if n < 5:
        return "n < 5"
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(vals[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"[{lo:+.4f}, {hi:+.4f}]" if signed else f"[{lo:.4f}, {hi:.4f}]"


def _normalizar_prob(P):
    P = np.clip(np.asarray(P, dtype=float), 1e-7, 1.0 - 1e-7)
    return P / P.sum(axis=1, keepdims=True)


def _stack_weights(M):
    w = M.get("stack_w")
    if w is None:
        w = M.get("weights_opt")
    if w is None:
        w = M.get("weights_stacking")
    if w is None and "alpha_stack" in M:
        a = float(M["alpha_stack"])
        return np.array([0.0, 1.0 - a, a], dtype=float)
    if w is None:
        return np.array([0.4, 0.4, 0.2], dtype=float)
    w_arr = np.asarray(w, dtype=float)
    s = float(w_arr.sum())
    return w_arr / s if s > 0 else np.array([0.4, 0.4, 0.2], dtype=float)


def _evaluar_liga_uniforme(liga, M, n_boot=500):
    df = M["df_features"] if "df_features" in M else M["df_dataset"]
    cols = M.get("features") or M.get("cols_features") or M.get("cols")
    if cols is None and hasattr(M.get("pipe_lasso"), "feature_names_in_"):
        cols = list(M["pipe_lasso"].feature_names_in_)
    test_desde = corte_test_temporada(liga, df)
    te_mask = (df["temporada"] >= test_desde).to_numpy()
    if not te_mask.any():
        return []

    df_test = df.loc[te_mask].copy()
    X_test = df_test[cols].fillna(0.0)
    n_test = len(df_test)
    rng = np.random.default_rng(42)

    elo_m = entrenar_baseline_elo(M, liga=liga, test_desde=test_desde)
    es_binario = (liga == "nba") or ("target_win_local" in df_test.columns)

    modelos_probs = {}
    if es_binario:
        y_test = df_test["target_win_local"].to_numpy(dtype=int)
        X_elo = df_test[["elo_diff"]].fillna(0.0)
        P_elo = _normalizar_prob(elo_m["lr"].predict_proba(X_elo))

        def _p_bin(pipe):
            return _normalizar_prob(pipe.predict_proba(X_test))

        P_l = _p_bin(M["pipe_lasso"])
        P_r = _p_bin(M["pipe_rf"])
        P_x = _p_bin(M["pipe_xgb"])
        w = _stack_weights(M)
        P_stk = _normalizar_prob(w[0] * P_l + w[1] * P_r + w[2] * P_x)
        modelos_probs = {"lasso": P_l, "rf": P_r, "xgb": P_x, "stacking": P_stk, "elo": P_elo}
    else:
        y_test, home_is_two = etiquetas_y_orden_1x2(M, df_test)
        X_elo = df_test[["elo_diff"]].fillna(0.0).to_numpy()
        cls_e = list(elo_m["lr"].classes_)
        raw_e = np.asarray(elo_m["lr"].predict_proba(X_elo), dtype=float)
        P_elo = np.full((n_test, 3), 1e-7, dtype=float)
        for j in range(3):
            if j in cls_e:
                P_elo[:, j] = raw_e[:, cls_e.index(j)]
        P_elo = _normalizar_prob(P_elo)

        P_l = proba_pipe_1x2(M["pipe_lasso"], X_test, home_is_two=home_is_two)
        P_r = proba_pipe_1x2(M["pipe_rf"], X_test, home_is_two=home_is_two)
        P_x = proba_pipe_1x2(M["pipe_xgb"], X_test, home_is_two=home_is_two) if "pipe_xgb" in M else None
        modelos_probs["lasso"] = P_l
        modelos_probs["rf"] = P_r
        if P_x is not None:
            modelos_probs["xgb"] = P_x
        if "pipe_svm" in M:
            modelos_probs["svm"] = proba_pipe_1x2(M["pipe_svm"], X_test, home_is_two=home_is_two)

        w = _stack_weights(M)
        P_stk = w[0] * P_l + w[1] * P_r + (w[2] * P_x if len(w) > 2 and P_x is not None else 0.0)
        modelos_probs["stacking"] = _normalizar_prob(P_stk)
        modelos_probs["elo"] = P_elo

    def _per_match_metrics(P):
        ll_v = -np.log(P[np.arange(n_test), y_test])
        if es_binario:
            brier_v = (P[:, 1] - y_test) ** 2
        else:
            brier_v = np.sum((P - np.eye(3)[y_test]) ** 2, axis=1)
        acc_pct = float(np.mean(np.argmax(P, axis=1) == y_test) * 100.0)
        return ll_v, brier_v, acc_pct

    ll_elo_vec, _, _ = _per_match_metrics(modelos_probs["elo"])
    ll_elo_mean = float(np.mean(ll_elo_vec))

    filas = []
    for mod, P in modelos_probs.items():
        ll_vec, brier_vec, acc_pct = _per_match_metrics(P)
        ll_m = float(np.mean(ll_vec))
        diff_elo = ll_vec - ll_elo_vec
        filas.append({
            "liga": liga,
            "modelo": mod,
            "test_desde": int(test_desde),
            "n_test": n_test,
            "logloss": round(ll_m, 4),
            "ic95_logloss": _boot_ic95(ll_vec, rng, n_boot=n_boot, signed=False),
            "brier": round(float(np.mean(brier_vec)), 4),
            "accuracy": round(acc_pct, 2),
            "logloss_elo": round(ll_elo_mean, 4),
            "delta_vs_elo": 0.0 if mod == "elo" else round(float(np.mean(diff_elo)), 4),
            "ic95_delta_elo": "—" if mod == "elo" else _boot_ic95(diff_elo, rng, n_boot=n_boot, signed=True),
        })
    return filas


def evaluar(ligas, n_boot=500):
    filas = []
    for liga in ligas:
        print(f"== {liga} ==", flush=True)
        try:
            mo = _importar_motor(liga)
            M = mo.cargar()
            f = _evaluar_liga_uniforme(liga, M, n_boot=n_boot)
            if not f:
                print("  (sin filas en bloque test)")
            filas.extend(f)
            for r in f:
                print(
                    f"  {r['modelo']:<10} n={r['n_test']:<4} LL={r['logloss']:.4f} "
                    f"Brier={r['brier']:.4f} Acc={r['accuracy']:.1f}% ΔElo={r['delta_vs_elo']:+.4f}",
                    flush=True,
                )
        except Exception as ex:
            print(f"  FALLO: {type(ex).__name__}: {ex}", flush=True)
            filas.append({"liga": liga, "modelo": "ERROR", "n_test": 0, "logloss": np.nan})
    return pd.DataFrame(filas)


def comparar(antes, despues):
    a = pd.read_csv(antes)
    d = pd.read_csv(despues)
    m = a.merge(d, on=["liga", "modelo"], suffixes=("_antes", "_despues"))
    m["delta_logloss"] = m["logloss_despues"] - m["logloss_antes"]
    cols = ["liga", "modelo", "logloss_antes", "logloss_despues", "delta_logloss"]
    print(m[cols].round(4).to_string(index=False))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ligas", nargs="*")
    ap.add_argument("--salida", default=str(REPO / "outputs" / "evaluacion_ligas.csv"))
    ap.add_argument("--comparar", nargs=2, metavar=("ANTES", "DESPUES"))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    if args.comparar:
        comparar(*args.comparar)
        return
    df = evaluar(args.ligas or LIGAS, n_boot=args.n_boot)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False)
    print(f"\nGuardado en {args.salida} ({len(df)} filas)")


if __name__ == "__main__":
    main()
