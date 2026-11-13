"""
Evaluador de predicciones prospectivas archivadas (`outputs/predicciones_log.csv`).

Cruza únicamente los pronósticos fechados ANTES del inicio de cada encuentro (`generado_utc < fecha_partido`)
con los resultados reales observados posteriormente en `<liga>/data/partidos.csv`, calculando:
  - Log-loss prospectivo fuera de muestra
  - Brier Score multiclase / binario
  - Accuracy
  - Intervalo de confianza al 95% por bootstrap sobre el Log-loss
  - Diferencia pareada contra el baseline Solo Elo (`delta_elo_logloss` + IC 95% pareado)

Uso:
    python3 scripts/evaluar_archivo.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from comun.registro_predicciones import LOG_PATH, cargar_registro


def _valor_marcador(row, *cols):
    for c in cols:
        if hasattr(row, c):
            v = getattr(row, c)
            if pd.notna(v):
                return float(v)
    return np.nan


def _emparejar_resultados(grp, df_p):
    """Empareja pronósticos archivados contra partidos.csv por (local, visita, fecha_d) y event_id."""
    df_p = df_p.copy()
    df_p["fecha_d"] = pd.to_datetime(df_p["fecha"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    grp = grp.copy()
    grp["fecha_d"] = grp["part_dt"].dt.strftime("%Y-%m-%d")
    grp["_row_idx"] = np.arange(len(grp))

    m1 = grp.merge(df_p, on=["local", "visita", "fecha_d"], how="inner")
    matched_idx = set(m1["_row_idx"].tolist()) if not m1.empty else set()

    if "event_id" in df_p.columns and len(matched_idx) < len(grp):
        rem = grp[~grp["_row_idx"].isin(matched_idx)].copy()
        rem["event_id_key"] = rem["partido_id"].astype(str)
        df_p_id = df_p.copy()
        df_p_id["event_id_key"] = df_p_id["event_id"].astype(str)
        m2 = rem.merge(df_p_id, on=["event_id_key", "local", "visita"], how="inner", suffixes=("", "_p"))
        if not m2.empty:
            m1 = pd.concat([m1, m2], ignore_index=True)

    return m1


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

        merged = _emparejar_resultados(grp, df_p)
        if merged.empty:
            continue

        for r in merged.itertuples(index=False):
            gl = _valor_marcador(r, "goles_local", "puntos_local", "pts_local")
            gv = _valor_marcador(r, "goles_visita", "puntos_visita", "pts_visita")
            if pd.isna(gl) or pd.isna(gv):
                continue

            y_real = 0 if gl > gv else (1 if gl == gv else 2)
            p_loc = float(r.p_local)
            p_emp = float(r.p_empate)
            p_vis = float(r.p_visita)

            if comp == "nba":
                p2 = np.clip(np.array([p_loc, p_vis], dtype=float), 1e-6, 1.0 - 1e-6)
                p2 = p2 / p2.sum()
                idx2 = 0 if y_real == 0 else 1
                ll_i = float(-np.log(p2[idx2]))
                brier_i = float(np.sum((p2 - np.eye(2)[idx2]) ** 2))
                pred_cls = 0 if p2[0] >= p2[1] else 2
            else:
                p3 = np.clip(np.array([p_loc, p_emp, p_vis], dtype=float), 1e-6, 1.0 - 1e-6)
                p3 = p3 / p3.sum()
                ll_i = float(-np.log(p3[y_real]))
                brier_i = float(np.sum((p3 - np.eye(3)[y_real]) ** 2))
                pred_cls = int(np.argmax(p3))

            res_filas.append({
                "competicion": comp,
                "partido_id": str(r.partido_id),
                "modelo": str(r.modelo),
                "generado_utc": str(r.generado_utc),
                "fecha_partido": str(r.fecha_d),
                "local": str(r.local),
                "visita": str(r.visita),
                "marcador": f"{int(round(gl))} - {int(round(gv))}",
                "p_local": p_loc,
                "p_empate": p_emp,
                "p_visita": p_vis,
                "y_real": y_real,
                "pred_cls": pred_cls,
                "acierto": bool(pred_cls == y_real),
                "ll_match": ll_i,
                "brier_match": brier_i,
            })

    df_det = pd.DataFrame(res_filas)
    if df_det.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_det = df_det.drop_duplicates(subset=["competicion", "modelo", "fecha_partido", "local", "visita"]).reset_index(drop=True)

    rng = np.random.default_rng(42)
    resumen = []
    for (comp, mod), g in df_det.groupby(["competicion", "modelo"]):
        lls = g["ll_match"].to_numpy(dtype=float)
        ll = float(np.mean(lls))
        brier = float(g["brier_match"].mean())
        acc = float(g["acierto"].mean() * 100.0)

        if len(lls) >= 5:
            boots = [float(np.mean(lls[rng.integers(0, len(lls), size=len(lls))])) for _ in range(n_boot)]
            ic_low, ic_high = np.percentile(boots, [2.5, 97.5])
            ic_str = f"[{ic_low:.4f}, {ic_high:.4f}]"
        else:
            ic_str = "n < 5"

        elo_ll = np.nan
        delta_elo = np.nan
        ic_elo_str = "—" if mod == "elo" else "sin par elo"
        if mod != "elo":
            g_elo = df_det[(df_det["competicion"] == comp) & (df_det["modelo"] == "elo")]
            if not g_elo.empty:
                pares = g.merge(
                    g_elo[["fecha_partido", "local", "visita", "ll_match"]],
                    on=["fecha_partido", "local", "visita"],
                    how="inner",
                    suffixes=("", "_elo"),
                )
                if not pares.empty:
                    elo_ll = round(float(pares["ll_match_elo"].mean()), 4)
                    diffs = (pares["ll_match"] - pares["ll_match_elo"]).to_numpy(dtype=float)
                    delta_elo = round(float(np.mean(diffs)), 4)
                    if len(diffs) >= 5:
                        b_d = [float(np.mean(diffs[rng.integers(0, len(diffs), size=len(diffs))])) for _ in range(n_boot)]
                        d_lo, d_hi = np.percentile(b_d, [2.5, 97.5])
                        ic_elo_str = f"[{d_lo:+.4f}, {d_hi:+.4f}]"
                    else:
                        ic_elo_str = "n < 5"

        resumen.append({
            "competicion": comp,
            "modelo": mod,
            "n_evaluados": len(g),
            "logloss": round(ll, 4),
            "ic95_logloss": ic_str,
            "elo_logloss": elo_ll,
            "delta_elo_logloss": delta_elo,
            "ic95_delta_elo": ic_elo_str,
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
