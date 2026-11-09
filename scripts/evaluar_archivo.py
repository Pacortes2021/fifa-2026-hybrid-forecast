"""
Evaluador de predicciones prospectivas archivadas (`outputs/predicciones_log.csv`).

Cruza únicamente los pronósticos fechados ANTES del inicio de cada encuentro (`generado_utc < fecha_partido`)
con los resultados reales observados posteriormente en `<liga>/data/partidos.csv`, calculando:
  - Log-loss prospectivo fuera de muestra
  - Brier Score multiclase / binario
  - Accuracy
  - Intervalo de confianza al 95% por bootstrap sobre el Log-loss

Uso:
    python3 scripts/evaluar_archivo.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from comun.registro_predicciones import LOG_PATH, cargar_registro


def evaluar_registro(log_path=LOG_PATH, n_boot=500):
    df_log = cargar_registro(log_path)
    if df_log.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_log["gen_dt"] = pd.to_datetime(df_log["generado_utc"], utc=True, errors="coerce")
    df_log["part_dt"] = pd.to_datetime(df_log["fecha_partido"], utc=True, errors="coerce")
    # Verificar invariante de causalidad: solo evaluar pronósticos emitidos antes del partido
    df_log = df_log[df_log["gen_dt"] < df_log["part_dt"]].copy()
    if df_log.empty:
        return pd.DataFrame(), pd.DataFrame()

    res_filas = []
    for comp, grp in df_log.groupby("competicion"):
        p_path = REPO / comp / "data" / "partidos.csv"
        if not p_path.exists():
            continue
        df_p = pd.read_csv(p_path)
        if df_p.empty:
            continue
        df_p["fecha_d"] = pd.to_datetime(df_p["fecha"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        grp = grp.copy()
        grp["fecha_d"] = grp["part_dt"].dt.strftime("%Y-%m-%d")

        merged = grp.merge(df_p, on=["local", "visita", "fecha_d"], how="inner")
        if merged.empty:
            continue

        for r in merged.itertuples(index=False):
            gl = float(getattr(r, "goles_local", getattr(r, "pts_local", np.nan)))
            gv = float(getattr(r, "goles_visita", getattr(r, "pts_visita", np.nan)))
            if pd.isna(gl) or pd.isna(gv):
                continue
            y_real = 0 if gl > gv else (1 if gl == gv else 2)
            res_filas.append({
                "competicion": comp,
                "modelo": r.modelo,
                "fecha_partido": r.fecha_d,
                "local": r.local,
                "visita": r.visita,
                "p_local": float(r.p_local),
                "p_empate": float(r.p_empate),
                "p_visita": float(r.p_visita),
                "y_real": y_real,
            })

    df_det = pd.DataFrame(res_filas)
    if df_det.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(42)
    resumen = []
    for (comp, mod), g in df_det.groupby(["competicion", "modelo"]):
        P = g[["p_local", "p_empate", "p_visita"]].to_numpy(dtype=float)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)
        P = P / P.sum(axis=1, keepdims=True)
        y = g["y_real"].to_numpy(dtype=int)
        ll = float(log_loss(y, P, labels=[0, 1, 2]))
        Y_oh = np.eye(3)[y]
        brier = float(np.mean(np.sum((P - Y_oh) ** 2, axis=1)))
        acc = float(np.mean(np.argmax(P, axis=1) == y) * 100.0)

        # IC 95% por bootstrap
        boots = []
        if len(y) >= 5:
            for _ in range(n_boot):
                idx = rng.integers(0, len(y), size=len(y))
                boots.append(log_loss(y[idx], P[idx], labels=[0, 1, 2]))
            ic_low, ic_high = np.percentile(boots, [2.5, 97.5])
            ic_str = f"[{ic_low:.4f}, {ic_high:.4f}]"
        else:
            ic_str = "n < 5"

        resumen.append({
            "competicion": comp,
            "modelo": mod,
            "n_evaluados": len(y),
            "logloss": round(ll, 4),
            "ic95_logloss": ic_str,
            "brier": round(brier, 4),
            "accuracy": round(acc, 1),
        })

    return pd.DataFrame(resumen), df_det


def main():
    df_res, _ = evaluar_registro()
    if df_res.empty:
        print("Aún no hay partidos finalizados que crucen con predicciones archivadas en outputs/predicciones_log.csv.")
        return
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
