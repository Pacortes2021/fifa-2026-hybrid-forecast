"""
Vista compartida de Auditoría Prospectiva Inmutable (`outputs/predicciones_log.csv`).

Muestra únicamente predicciones congeladas ANTES del inicio de cada partido (`generado_utc < fecha_partido`),
cruzadas con resultados reales observados posteriormente y comparadas en pares contra Solo Elo.
"""
import numpy as np
import pandas as pd
import streamlit as st

from comun.registro_predicciones import LOG_PATH, cargar_registro
from scripts.evaluar_archivo import evaluar_registro


@st.cache_data(ttl=120)
def _cargar_evaluacion_prospectiva(_mtime):
    """Carga evaluación prospectiva invalidando caché automáticamente al actualizar el log CSV."""
    df_log = cargar_registro()
    df_res, df_det = evaluar_registro()
    return df_log, df_res, df_det


def render_archivo_prospectivo(competicion=None, key_prefix="arch"):
    """Renderiza el panel de evaluación prospectiva ciega global o filtrada por competición."""
    mtime = LOG_PATH.stat().st_mtime if LOG_PATH.exists() else 0.0
    df_log, df_res, df_det = _cargar_evaluacion_prospectiva(mtime)

    if df_log.empty:
        st.info(
            "Aún no hay predicciones archivadas en `outputs/predicciones_log.csv`. "
            "Se registran automáticamente antes de cada jornada con `scripts/archivar_predicciones.py`."
        )
        return

    if competicion is not None:
        df_log = df_log[df_log["competicion"] == competicion].copy()
        if not df_res.empty:
            df_res = df_res[df_res["competicion"] == competicion].copy()
        if not df_det.empty:
            df_det = df_det[df_det["competicion"] == competicion].copy()

    if df_log.empty:
        st.info("Todavía no hay predicciones pre-partido archivadas para esta competición.")
        return

    n_total_reg = df_log[["competicion", "partido_id"]].drop_duplicates().shape[0]
    n_eval_stk = int(df_det[df_det["modelo"] == "stacking"].shape[0]) if not df_det.empty else 0

    st.caption(
        "Solo se evalúan pronósticos sellados estrictamente antes del inicio (`generado_utc < fecha_partido`) "
        "y contrastados contra resultados reales cerrados posteriormente."
    )

    if not df_res.empty:
        df_stk = df_det[df_det["modelo"] == "stacking"]
        df_elo = df_det[df_det["modelo"] == "elo"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidos Sellados / Evaluados", f"{n_total_reg:,} / {n_eval_stk:,}")

        if not df_stk.empty:
            ll_s = float(df_stk["ll_match"].mean())
            acc_s = float(df_stk["acierto"].mean() * 100.0)
            pares = pd.DataFrame()
            if not df_elo.empty:
                pares = df_stk.merge(
                    df_elo[["competicion", "fecha_partido", "local", "visita", "ll_match", "acierto"]],
                    on=["competicion", "fecha_partido", "local", "visita"],
                    how="inner",
                    suffixes=("", "_elo"),
                )
            delta_str = None
            if not pares.empty:
                d_elo = float(np.mean(pares["ll_match"] - pares["ll_match_elo"]))
                delta_str = f"{d_elo:+.4f} vs Solo Elo"
            c2.metric("Log-Loss Stacking (Ciego)", f"{ll_s:.4f}", delta=delta_str, delta_color="inverse")
            c3.metric("Acierto Stacking", f"{acc_s:.1f}%")
            if not pares.empty:
                ll_e = float(pares["ll_match_elo"].mean())
                acc_e = float(pares["acierto_elo"].mean() * 100.0)
                c4.metric("Solo Elo (Mismos Partidos)", f"{ll_e:.4f}", delta=f"Acc {acc_e:.1f}%", delta_color="off")
            else:
                c4.metric("Brier Stacking", f"{float(df_stk['brier_match'].mean()):.4f}")

        st.markdown("##### 📊 Resumen Prospectivo por Competición y Modelo")
        show_res = df_res.rename(columns={
            "competicion": "Liga",
            "modelo": "Modelo",
            "n_evaluados": "N",
            "logloss": "Log-Loss",
            "ic95_logloss": "IC 95% Log-Loss",
            "elo_logloss": "Elo Log-Loss",
            "delta_elo_logloss": "Δ vs Elo",
            "ic95_delta_elo": "IC 95% Δ vs Elo",
            "brier": "Brier",
            "accuracy": "Acierto (%)",
        })
        st.dataframe(show_res, hide_index=True, width="stretch")

        st.markdown("##### 🔍 Detalle Partido a Partido (Evaluados)")
        modelos_det = sorted(df_det["modelo"].unique().tolist())
        mod_sel = st.selectbox(
            "Filtrar modelo:",
            modelos_det,
            index=modelos_det.index("stacking") if "stacking" in modelos_det else 0,
            key=f"{key_prefix}_mod_det",
        )
        vis_det = df_det[df_det["modelo"] == mod_sel].sort_values("fecha_partido", ascending=False).copy()
        vis_det["Resultado OK"] = np.where(vis_det["acierto"], "✅", "❌")
        for c in ("p_local", "p_empate", "p_visita"):
            vis_det[c] = vis_det[c].map(lambda x: f"{x:.1%}")
        vis_det["ll_match"] = vis_det["ll_match"].map(lambda x: round(float(x), 4))
        st.dataframe(
            vis_det[[
                "competicion", "fecha_partido", "local", "visita", "marcador",
                "p_local", "p_empate", "p_visita", "ll_match", "Resultado OK", "generado_utc"
            ]].rename(columns={
                "competicion": "Liga",
                "fecha_partido": "Fecha",
                "local": "Local",
                "visita": "Visita",
                "marcador": "Marcador",
                "p_local": "P(Local)",
                "p_empate": "P(Empate)",
                "p_visita": "P(Visita)",
                "ll_match": "Log-Loss",
                "generado_utc": "Sellado UTC",
            }),
            hide_index=True,
            width="stretch",
        )
    else:
        st.metric("Partidos Pre-Partido Sellados (A la espera de resultado)", f"{n_total_reg:,}")
        st.info("Los partidos archivados aún están programados o pendientes de disputarse; sus métricas aparecerán al finalizar.")

    # Cola de pronósticos futuros sellados pendientes de cierre
    eval_keys = set()
    if not df_det.empty:
        eval_keys = set(zip(df_det["competicion"], df_det["fecha_partido"], df_det["local"], df_det["visita"]))
    df_pend = df_log[df_log["modelo"] == "stacking"].copy()
    df_pend["fecha_d"] = pd.to_datetime(df_pend["fecha_partido"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if eval_keys:
        mask_p = [
            (r.competicion, r.fecha_d, r.local, r.visita) not in eval_keys
            for r in df_pend.itertuples(index=False)
        ]
        df_pend = df_pend[mask_p]

    if not df_pend.empty:
        with st.expander(f"🕒 Pronósticos pre-partido sellados pendientes de resultado ({len(df_pend)})", expanded=False):
            show_p = df_pend.sort_values("fecha_partido").copy()
            for c in ("p_local", "p_empate", "p_visita"):
                show_p[c] = show_p[c].map(lambda x: f"{x:.1%}")
            st.dataframe(
                show_p[[
                    "competicion", "fecha_partido", "local", "visita",
                    "p_local", "p_empate", "p_visita", "version_codigo", "generado_utc"
                ]].rename(columns={
                    "competicion": "Liga",
                    "fecha_partido": "Inicio UTC",
                    "local": "Local",
                    "visita": "Visita",
                    "p_local": "P(Local)",
                    "p_empate": "P(Empate)",
                    "p_visita": "P(Visita)",
                    "version_codigo": "Commit",
                    "generado_utc": "Sellado UTC",
                }),
                hide_index=True,
                width="stretch",
            )
