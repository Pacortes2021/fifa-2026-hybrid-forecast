"""
Aplicación Streamlit para la simulación y predicciones de la UEFA Champions League (UCL).
Modelo Híbrido de 36 clubes con Fase de Liga (Formato Suizo) y Cuadro Eliminatorio.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Insertar el directorio ucl en el path para asegurar la importación del motor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor as mo
import recolectar as rec
import recolectar_boxscore as rec_box

# Estilizado CSS Premium UEFA Champions League (Azul Noche #001438, Cyan Neón, Oro Estelar)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main-title { text-align:center; font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg,#001438,#00E5FF,#FFD700); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#001438;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#001438; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 20, 56, 0.05), 0 4px 6px -4px rgba(0, 20, 56, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


def logo_url(equipos, team):
    """URL del escudo del club (ID estable de ESPN) o None si no existe."""
    if not equipos:
        return None
    for e in equipos.values():
        if e.get("norm_name") == team and e.get("logo"):
            return e["logo"]
    return None


def logo_html(equipos, team, size=64):
    """Devuelve el <img> del escudo oficial de Champions League desde data/equipos.csv."""
    url = logo_url(equipos, team)
    if not url:
        return ""
    return f'<img src="{url}" width="{size}" style="border-radius:10px; box-shadow:0 2px 8px rgba(0,20,56,.15);">'


def fmt_opcion(equipos, team):
    """Etiqueta de opción del selectbox: abreviatura del club + nombre."""
    if equipos:
        for e in equipos.values():
            if e.get("norm_name") == team and e.get("abbreviation"):
                return f"{e['abbreviation']} · {team}"
    return team


def label_tabla(equipos, team):
    """Nombre en tablas: fallback estético limpio."""
    return team


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Corriendo simulaciones Monte Carlo del formato suizo UCL (10.000 iteraciones)...")
def simular_ucl(_M, key, modelo_tipo):
    return mo.simular_campeonato(_M, n_sims=10000, modelo_tipo=modelo_tipo)


def run_app():
    st.sidebar.markdown("### 🛠️ Controles del Modelo (UCL)")

    modelo_sel = st.sidebar.selectbox(
        "🤖 Modelo Predictivo:",
        [
            "✨ Stacking Óptimo (Ensemble Recomendado)",
            "🌲 Random Forest (Árboles)",
            "🎯 LASSO L1 (Regularización SAGA)",
            "🚀 XGBoost (Gradient Boosting)"
        ],
        index=0
    )
    if "LASSO" in modelo_sel:
        modelo_tipo = "lasso"
    elif "Random" in modelo_sel:
        modelo_tipo = "rf"
    elif "XGB" in modelo_sel:
        modelo_tipo = "xgb"
    else:
        modelo_tipo = "stacking"

    with st.sidebar.expander("🔄 Actualización de Datos ESPN"):
        if st.button("Descargar últimos partidos UCL"):
            with st.spinner("Actualizando cotejos desde la API de ESPN..."):
                rec.recolectar()
                rec_box.recolectar()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Datos de Champions actualizados. Recarga la app.")

    M = get_motor()
    if M is None:
        st.error("No se encontraron datos históricos de la UEFA Champions League.")
        return

    tracker = M["tracker"]
    equipos_data = M.get("equipos", {})

    try:
        _partidos_df = pd.read_csv(mo.DATA / "partidos.csv", parse_dates=["fecha"])
        _ult_fecha = pd.to_datetime(_partidos_df["fecha"].max()).date().strftime("%d/%m/%Y")
        st.sidebar.caption(f"🗓️ Datos actualizados: {_ult_fecha} · {len(_partidos_df)} partidos")
    except Exception:
        pass

    # ── Métricas por modelo (sidebar)
    met_all = M.get("metricas", {})
    if met_all:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📊 Métricas Out-of-Sample (2026+)")
        valid_lls = [met_all[k]["logloss"] for k in met_all if isinstance(met_all[k], dict) and "logloss" in met_all[k]]
        mejor_ll = min(valid_lls) if valid_lls else None
        for nombre, clave in [("LASSO", "lasso"), ("RF", "rf"), ("XGB", "xgb"), ("Stacking", "stacking")]:
            if clave not in met_all:
                continue
            m = met_all[clave]
            star = " ⭐" if mejor_ll is not None and m.get("logloss") == mejor_ll else ""
            w_str = f" (w={m['w']})" if clave == "stacking" and "w" in m else ""
            st.sidebar.caption(f"**{nombre}{w_str}{star}** — LL: `{m['logloss']:.4f}` | Acc: `{m['accuracy']:.1f}%`")

    # Encabezado
    nombre_modelo = "✨ Stacking Óptimo" if modelo_tipo == "stacking" else ("🌲 Random Forest" if modelo_tipo == "rf" else ("🚀 XGBoost" if modelo_tipo == "xgb" else "🎯 LASSO L1"))
    st.markdown('<div class="main-title">⭐ UEFA Champions League Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Modelo activo: <b>{nombre_modelo}</b> — Fase de Liga de 36 Clubes · Play-offs y Cuadro Eliminatorio · Poisson Dixon-Coles & ML</div>', unsafe_allow_html=True)

    # 5 Pestañas Canónicas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚽ Predicción Versus",
        "📊 Tabla y Proyecciones",
        "📥 Exportar para IA (CSV)",
        "🔬 Importancia de Variables",
        "🎯 Validación vs Realidad"
    ])

    # ============================================================================
    # TAB 1: Predicción Versus
    # ============================================================================
    with tab1:
        st.markdown('<div class="sec-title">Analizador de Enfrentamientos</div>', unsafe_allow_html=True)

        fix_path = mo.DATA / "fixture.csv"
        fixture_disp = pd.DataFrame()
        equipos_activos = []
        if fix_path.exists():
            fix_raw = pd.read_csv(fix_path)
            fixture_disp = fix_raw[fix_raw.temporada == mo._temporada_actual()].copy()
            equipos_activos = sorted(list(set(fixture_disp["local"]).union(set(fixture_disp["visita"]))))

        if not equipos_activos:
            equipos_activos = sorted(list(tracker.elos.keys()))

        # Cargar partido programado de fixture
        if not fixture_disp.empty:
            fixture_disp["fecha_dt"] = pd.to_datetime(fixture_disp["fecha"])
            proximos = fixture_disp.sort_values("fecha_dt").head(30)
            fix_map = {}
            opciones_fixture = ["— Seleccionar partido del calendario oficial UCL —"]
            for _, r in proximos.iterrows():
                f_str = r["fecha_dt"].strftime("%d/%m %H:%M") if pd.notna(r["fecha_dt"]) else "Fecha TBD"
                lbl = f"📅 {f_str} | {r['local']} vs {r['visita']}"
                opciones_fixture.append(lbl)
                fix_map[lbl] = (r["local"], r["visita"])

            def _on_fixture_change_ucl():
                sel = st.session_state.get("sel_fix_ucl")
                if sel and sel in fix_map:
                    l_sel, v_sel = fix_map[sel]
                    if l_sel in equipos_activos and v_sel in equipos_activos:
                        st.session_state["sel_a_ucl"] = l_sel
                        st.session_state["sel_b_ucl"] = v_sel

            st.selectbox(
                "⚡ Cargar Partido Programado de Fase de Liga:",
                opciones_fixture,
                key="sel_fix_ucl",
                on_change=_on_fixture_change_ucl,
                help="Elige un cotejo del fixture oficial UCL para autocompletar ambos clubes."
            )

        # Configuración de condiciones de partido
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            es_knockout = st.checkbox("⚔️ Partido de Eliminatoria Directa (Knockout / Play-offs)", value=False)
        with col_c2:
            es_neutral = st.checkbox("🏟️ Sede Neutral (Ej. Gran Final)", value=False)

        es_vuelta = False
        ventaja_local = 0
        if es_knockout and not es_neutral:
            col_k1, col_k2 = st.columns([1, 1])
            with col_k1:
                es_vuelta = st.checkbox("🔄 Partido de Vuelta (2nd Leg)", value=False, help="Activa el contexto táctico de la serie a doble partido.")
            with col_k2:
                if es_vuelta:
                    ventaja_local = st.number_input(
                        "Ventaja global local (Ida):", min_value=-10, max_value=10, value=0, step=1,
                        help="Diferencia de goles a favor del local en la ida (ej: +2 si ganó 2-0 afuera, -1 si cayó 1-2 afuera)."
                    )

        # Selección de Equipos
        c1, cvs, c2 = st.columns([5, 1, 5])
        with c1:
            def_a = "Real Madrid" if "Real Madrid" in equipos_activos else equipos_activos[0]
            local = st.selectbox(
                "Club Local", equipos_activos,
                index=equipos_activos.index(def_a) if def_a in equipos_activos else 0,
                key="sel_a_ucl",
                format_func=lambda t: fmt_opcion(equipos_data, t)
            )
            st.markdown(f'<div style="text-align:center;margin-top:0.2rem;">{logo_html(equipos_data, local, 64)}</div>', unsafe_allow_html=True)
            elo_l = tracker.elos[local]
            val_l = mo.get_squad_value(local, mo._temporada_actual())
            st.caption(f"ELO: **{elo_l:.0f} pts** · Plantilla: **€{val_l:.1f}M**")

        with cvs:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

        with c2:
            def_b = "Bayern Munich" if "Bayern Munich" in equipos_activos else (equipos_activos[1] if len(equipos_activos) > 1 else equipos_activos[0])
            visita = st.selectbox(
                "Club Visitante", equipos_activos,
                index=equipos_activos.index(def_b) if def_b in equipos_activos else (1 if len(equipos_activos) > 1 else 0),
                key="sel_b_ucl",
                format_func=lambda t: fmt_opcion(equipos_data, t)
            )
            st.markdown(f'<div style="text-align:center;margin-top:0.2rem;">{logo_html(equipos_data, visita, 64)}</div>', unsafe_allow_html=True)
            elo_v = tracker.elos[visita]
            val_v = mo.get_squad_value(visita, mo._temporada_actual())
            st.caption(f"ELO: **{elo_v:.0f} pts** · Plantilla: **€{val_v:.1f}M**")

        if local == visita:
            st.warning("⚠️ Selecciona dos clubes distintos para pronosticar el partido.")
        else:
            p, la, lb = mo.predecir_match(
                M, local, visita, temporada=mo._temporada_actual(),
                modelo=modelo_tipo, is_neutral=1 if es_neutral else 0, is_knockout=1 if es_knockout else 0,
                leg2_lead_local=float(ventaja_local) if (es_knockout and es_vuelta) else 0.0,
                is_leg2=1 if (es_knockout and es_vuelta) else 0
            )
            mat_dc = mo.matriz_marcador_exacto(la, lb)

            # Alertas de Heurísticas de Alta Efectividad
            if p[0] > 0.55 and (la - lb) > 0.8:
                st.success(f"🔥 **ALERTA DE ALTA CONFIANZA:** Consenso entre Machine Learning ({p[0]:.1%}) y modelo Poisson ({la:.2f} vs {lb:.2f} xG) a favor de **{local}**.")
            elif p[0] > 0.60:
                st.info(f"💪 **FAVORITO CLARO:** El modelo asigna más del 60% ({p[0]:.1%}) de probabilidad de victoria a **{local}**.")
            elif p[2] > 0.50:
                st.success(f"⚠️ **VISITA FUERTE:** Probabilidad superior al 50% ({p[2]:.1%}) para el club visitante (**{visita}**).")

            # ── Probabilidades y Goles Esperados ─────────────────────────────
            col_probs, col_stats = st.columns(2)

            with col_probs:
                st.markdown('<div class="card-title">Probabilidades de Victoria</div>', unsafe_allow_html=True)
                for label, prob in [(f"Victoria {local}", p[0]), ("Empate", p[1]), (f"Victoria {visita}", p[2])]:
                    st.markdown(f"**{label}: {prob:.1%}** (Cuota Justa: `{mo.cuota(prob):.2f}`)")
                    st.progress(float(prob))

                st.markdown("---")
                p_avanza_l = p[0] + p[1] * 0.5
                st.caption(f"Expectativa de pase en eliminatoria: **{local} {p_avanza_l:.1%}** / {visita} {1-p_avanza_l:.1%}")

            with col_stats:
                st.markdown('<div class="card-title">Goles Esperados y Marcadores</div>', unsafe_allow_html=True)
                st.markdown(f"📈 **Goles esperados (Dixon-Coles):**")
                st.markdown(f"*   {local}: `{la:.2f}` xG")
                st.markdown(f"*   {visita}: `{lb:.2f}` xG")

                st.markdown("🎯 **Marcadores más probables:**")
                mk = mo.mercados(mat_dc)
                for g1, g2, pr in mk["_top_marcadores"][:4]:
                    st.markdown(f"*   `{g1} - {g2}`: **{pr:.1%}** (Cuota: `{mo.cuota(pr):.2f}`)")

            # ── Comparativa Directa de los 4 Modelos ────────────────────────
            st.markdown("---")
            st.markdown('<div class="sec-title">🤖 Comparativa Directa entre Modelos para este Partido</div>', unsafe_allow_html=True)
            p_lasso = mo.predecir_match(M, local, visita, temporada=mo._temporada_actual(), modelo="lasso", is_neutral=1 if es_neutral else 0, is_knockout=1 if es_knockout else 0, leg2_lead_local=float(ventaja_local) if (es_knockout and es_vuelta) else 0.0, is_leg2=1 if (es_knockout and es_vuelta) else 0)[0]
            p_rf    = mo.predecir_match(M, local, visita, temporada=mo._temporada_actual(), modelo="rf", is_neutral=1 if es_neutral else 0, is_knockout=1 if es_knockout else 0, leg2_lead_local=float(ventaja_local) if (es_knockout and es_vuelta) else 0.0, is_leg2=1 if (es_knockout and es_vuelta) else 0)[0]
            p_xgb   = mo.predecir_match(M, local, visita, temporada=mo._temporada_actual(), modelo="xgb", is_neutral=1 if es_neutral else 0, is_knockout=1 if es_knockout else 0, leg2_lead_local=float(ventaja_local) if (es_knockout and es_vuelta) else 0.0, is_leg2=1 if (es_knockout and es_vuelta) else 0)[0]
            p_stk   = mo.predecir_match(M, local, visita, temporada=mo._temporada_actual(), modelo="stacking", is_neutral=1 if es_neutral else 0, is_knockout=1 if es_knockout else 0, leg2_lead_local=float(ventaja_local) if (es_knockout and es_vuelta) else 0.0, is_leg2=1 if (es_knockout and es_vuelta) else 0)[0]

            df_comp_mod = pd.DataFrame([
                {
                    "Modelo Predictivo": "🎯 LASSO L1 (Regresión SAGA)",
                    f"Victoria {local}": f"{p_lasso[0]:.1%}",
                    "Empate": f"{p_lasso[1]:.1%}",
                    f"Victoria {visita}": f"{p_lasso[2]:.1%}",
                    "Log-Loss Test (2026+)": f"{met_all.get('lasso',{}).get('logloss','-'):.4f}" if 'lasso' in met_all else "-",
                    "Accuracy Test": f"{met_all.get('lasso',{}).get('accuracy','-'):.1f}%" if 'lasso' in met_all else "-"
                },
                {
                    "Modelo Predictivo": "🌲 Random Forest",
                    f"Victoria {local}": f"{p_rf[0]:.1%}",
                    "Empate": f"{p_rf[1]:.1%}",
                    f"Victoria {visita}": f"{p_rf[2]:.1%}",
                    "Log-Loss Test (2026+)": f"{met_all.get('rf',{}).get('logloss','-'):.4f}" if 'rf' in met_all else "-",
                    "Accuracy Test": f"{met_all.get('rf',{}).get('accuracy','-'):.1f}%" if 'rf' in met_all else "-"
                },
                {
                    "Modelo Predictivo": "🚀 XGBoost",
                    f"Victoria {local}": f"{p_xgb[0]:.1%}",
                    "Empate": f"{p_xgb[1]:.1%}",
                    f"Victoria {visita}": f"{p_xgb[2]:.1%}",
                    "Log-Loss Test (2026+)": f"{met_all.get('xgb',{}).get('logloss','-'):.4f}" if 'xgb' in met_all else "-",
                    "Accuracy Test": f"{met_all.get('xgb',{}).get('accuracy','-'):.1f}%" if 'xgb' in met_all else "-"
                },
                {
                    "Modelo Predictivo": f"✨ Stacking (w={met_all.get('stacking',{}).get('w', [0.0, 0.0, 1.0])})" if 'w' in met_all.get('stacking', {}) else "✨ Stacking Óptimo",
                    f"Victoria {local}": f"{p_stk[0]:.1%}",
                    "Empate": f"{p_stk[1]:.1%}",
                    f"Victoria {visita}": f"{p_stk[2]:.1%}",
                    "Log-Loss Test (2026+)": f"{met_all.get('stacking',{}).get('logloss','-'):.4f}" if 'stacking' in met_all else "-",
                    "Accuracy Test": f"{met_all.get('stacking',{}).get('accuracy','-'):.1f}%" if 'stacking' in met_all else "-"
                }
            ])
            st.dataframe(df_comp_mod, hide_index=True, width='stretch')

            # ── Mercados de Apuestas Derivados ───────────────────────────────
            st.markdown("---")
            st.markdown('<div class="sec-title">💰 Mercados de Apuestas Probabilísticas</div>', unsafe_allow_html=True)
            filas_m = []
            for ln in (1.5, 2.5, 3.5):
                for lado in ("Over", "Under"):
                    pr = mk[f"{lado} {ln}"]
                    filas_m.append({"Mercado": f"{lado} {ln} goles", "Prob.": f"{pr:.1%}", "Cuota justa": f"{mo.cuota(pr):.2f}"})
            for et, key in (("Ambos marcan: Sí", "Ambos marcan (BTTS sí)"), ("Ambos marcan: No", "BTTS no")):
                filas_m.append({"Mercado": et, "Prob.": f"{mk[key]:.1%}", "Cuota justa": f"{mo.cuota(mk[key]):.2f}"})

            # Doble Oportunidad
            p_1x = p[0] + p[1]
            p_x2 = p[2] + p[1]
            p_12 = p[0] + p[2]
            filas_m.append({"Mercado": f"Doble Oportunidad: {local} o Empate (1X)", "Prob.": f"{p_1x:.1%}", "Cuota justa": f"{mo.cuota(p_1x):.2f}"})
            filas_m.append({"Mercado": f"Doble Oportunidad: {visita} o Empate (X2)", "Prob.": f"{p_x2:.1%}", "Cuota justa": f"{mo.cuota(p_x2):.2f}"})
            filas_m.append({"Mercado": f"Doble Oportunidad: {local} o {visita} (12)", "Prob.": f"{p_12:.1%}", "Cuota justa": f"{mo.cuota(p_12):.2f}"})

            # Sin Empate (DNB)
            denom = p[0] + p[2]
            p_dnb1 = p[0] / denom if denom > 0 else 0.5
            p_dnb2 = p[2] / denom if denom > 0 else 0.5
            filas_m.append({"Mercado": f"Sin Empate: {local} (DNB 1)", "Prob.": f"{p_dnb1:.1%}", "Cuota justa": f"{mo.cuota(p_dnb1):.2f}"})
            filas_m.append({"Mercado": f"Sin Empate: {visita} (DNB 2)", "Prob.": f"{p_dnb2:.1%}", "Cuota justa": f"{mo.cuota(p_dnb2):.2f}"})

            mc1, mc2 = st.columns(2)
            mc1.dataframe(pd.DataFrame(filas_m[:7]), hide_index=True, width='stretch')
            mc2.dataframe(pd.DataFrame(filas_m[7:]), hide_index=True, width='stretch')

            # ── Gráfica de la matriz Dixon-Coles ─────────────────────────────
            st.markdown('<div class="sec-title">Matriz de Goles Exactos (Dixon-Coles)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            m6 = mat_dc[:6, :6]
            im = ax.imshow(m6, cmap="Blues")
            ax.set_xticks(range(6)); ax.set_xticklabels(range(6))
            ax.set_yticks(range(6)); ax.set_yticklabels(range(6))
            ax.set_xlabel(f"Goles de {visita}", fontsize=9)
            ax.set_ylabel(f"Goles de {local}", fontsize=9)
            fig.colorbar(im, ax=ax, label="Probabilidad")

            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{m6[i, j]:.1%}", ha="center", va="center",
                            color="white" if m6[i, j] > m6.max() * 0.6 else "black", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── HISTORIAL HEAD-TO-HEAD (ENFRENTAMIENTOS DIRECTOS) ─────────
            st.markdown("---")
            st.markdown('<div class="sec-title">⚔️ Historial Head-to-Head (Enfrentamientos Directos)</div>', unsafe_allow_html=True)

            partidos_df_ucl = pd.read_csv(mo.DATA / "partidos.csv")
            mask_h2h = ((partidos_df_ucl["local"] == local) & (partidos_df_ucl["visita"] == visita)) |                        ((partidos_df_ucl["local"] == visita) & (partidos_df_ucl["visita"] == local))
            df_h2h = partidos_df_ucl[mask_h2h].copy()

            if df_h2h.empty:
                st.info(f"ℹ️ No se registran duelos directos oficiales entre **{local}** y **{visita}** en el dataset reciente de Champions League.")
            else:
                df_h2h["fecha_dt"] = pd.to_datetime(df_h2h["fecha"])
                df_h2h = df_h2h.sort_values("fecha_dt", ascending=False)

                vic_l = sum(1 for _, r in df_h2h.iterrows() if (r["local"] == local and r["goles_local"] > r["goles_visita"]) or (r["visita"] == local and r["goles_visita"] > r["goles_local"]))
                vic_v = sum(1 for _, r in df_h2h.iterrows() if (r["local"] == visita and r["goles_local"] > r["goles_visita"]) or (r["visita"] == visita and r["goles_visita"] > r["goles_local"]))
                emp = len(df_h2h) - vic_l - vic_v

                gol_l = sum(int(r["goles_local"] if r["local"] == local else r["goles_visita"]) for _, r in df_h2h.iterrows())
                gol_v = sum(int(r["goles_local"] if r["local"] == visita else r["goles_visita"]) for _, r in df_h2h.iterrows())

                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                with col_h1:
                    st.metric("Total Duelos", len(df_h2h))
                with col_h2:
                    pct_l = (vic_l / len(df_h2h)) * 100
                    st.metric(f"Victorias {local}", f"{vic_l} ({pct_l:.0f}%)", f"{gol_l} goles")
                with col_h3:
                    pct_e = (emp / len(df_h2h)) * 100
                    st.metric("Empates", f"{emp} ({pct_e:.0f}%)")
                with col_h4:
                    pct_v = (vic_v / len(df_h2h)) * 100
                    st.metric(f"Victorias {visita}", f"{vic_v} ({pct_v:.0f}%)", f"{gol_v} goles")

                filas_hist = []
                for _, r in df_h2h.head(8).iterrows():
                    gl, gv = int(r["goles_local"]), int(r["goles_visita"])
                    if gl > gv:
                        ganador = f"🟢 Gana {r['local']}"
                    elif gv > gl:
                        ganador = f"🟢 Gana {r['visita']}"
                    else:
                        ganador = "⚪ Empate"
                    filas_hist.append({
                        "Fecha": r["fecha_dt"].strftime("%d/%m/%Y"),
                        "Temporada": r["temporada"],
                        "Local": r["local"],
                        "Marcador": f"{gl} - {gv}",
                        "Visitante": r["visita"],
                        "Resultado": ganador
                    })
                st.dataframe(pd.DataFrame(filas_hist), hide_index=True, width='stretch')

            # Variables clave del duelo
            with st.expander("🔍 Métricas Avanzadas del Enfrentamiento"):
                feats = tracker.get_features_for_match(local, visita, mo._temporada_actual(), is_knockout=1 if es_knockout else 0, is_neutral=1 if es_neutral else 0)
                col_f1, col_f2, col_f3 = st.columns(3)
                col_f1.metric("Distancia de Viaje", f"{feats['distance_km']:.0f} km")
                col_f2.metric("Diferencia de Altitud", f"{feats['altitude_diff']:+.0f} m")
                col_f3.metric("Ratio Valor de Plantilla", f"{np.exp(feats['squad_value_diff']):.2f}x")

                col_f4, col_f5, col_f6 = st.columns(3)
                col_f4.metric("Diferencial de Forma (5 PJ)", f"{feats['form_diff']:+.2f}")
                col_f5.metric("Diferencial Pi-Rating", f"{feats['pi_diff']:+.2f}")
                col_f6.metric("Historial H2H (Goles Netos)", f"{feats['h2h_diff']:+.2f}")

    # ============================================================================
    # TAB 2: TABLA & MONTE CARLO
    # ============================================================================
    with tab2:
        st.markdown('<div class="sec-title">Fase de Liga (36 Clubes) y Proyecciones Monte Carlo</div>', unsafe_allow_html=True)
        col_act, col_proj = st.columns(2)

        # 1. Tabla Actual
        df_actual = mo.obtener_tabla_actual(M)
        df_actual_vis = df_actual.copy()
        df_actual_vis["Escudo"] = df_actual_vis["equipo"].apply(lambda t: logo_url(equipos_data, t))
        df_actual_vis["Equipo"] = df_actual_vis["equipo"].apply(lambda t: label_tabla(equipos_data, t))
        df_actual_vis = df_actual_vis[["Escudo", "Equipo", "pj", "puntos", "dg", "gf"]]
        df_actual_vis = df_actual_vis.rename(columns={"pj": "PJ", "puntos": "PTS", "dg": "DG", "gf": "GF"})

        with col_act:
            st.markdown('<div class="card-title">Tabla de Posiciones Actual (Real)</div>', unsafe_allow_html=True)
            st.dataframe(
                df_actual_vis, hide_index=True, width='stretch', height=520,
                column_config={"Escudo": st.column_config.ImageColumn("", width="small")}
            )

        # 2. Proyecciones Monte Carlo
        df_proy = simular_ucl(M, "sim_key_ucl", modelo_tipo)
        df_proy_visual = df_proy.copy()
        df_proy_visual["Escudo"] = df_proy_visual["equipo"].apply(lambda t: logo_url(equipos_data, t))
        df_proy_visual["Equipo"] = df_proy_visual["equipo"].apply(lambda t: label_tabla(equipos_data, t))
        df_proy_visual = df_proy_visual[["Escudo", "Equipo", "Puntos esperados", "P_campeon", "P_final", "P_top8", "P_playoffs", "P_eliminado"]]
        df_proy_visual = df_proy_visual.rename(columns={
            "Puntos esperados": "PTS Proy",
            "P_campeon": "🏆 P(Campeón)",
            "P_final": "🥈 P(Final)",
            "P_top8": "⭐ P(Top 8)",
            "P_playoffs": "⚔️ P(Playoffs)",
            "P_eliminado": "🔻 P(Eliminado)"
        })

        with col_proj:
            st.markdown('<div class="card-title">Proyección de la Temporada en Curso (Monte Carlo)</div>', unsafe_allow_html=True)
            st.caption("10.000 simulaciones completas del formato suizo + playoffs y cuadro eliminatorio.")
            st.dataframe(
                df_proy_visual.style.format({
                    "PTS Proy": "{:.1f}",
                    "🏆 P(Campeón)": "{:.1%}",
                    "🥈 P(Final)": "{:.1%}",
                    "⭐ P(Top 8)": "{:.1%}",
                    "⚔️ P(Playoffs)": "{:.1%}",
                    "🔻 P(Eliminado)": "{:.1%}"
                }).background_gradient(subset=["🏆 P(Campeón)"], cmap="YlOrRd")
                  .background_gradient(subset=["⭐ P(Top 8)"], cmap="Greens")
                  .background_gradient(subset=["🔻 P(Eliminado)"], cmap="Blues"),
                hide_index=True, width='stretch', height=520,
                column_config={"Escudo": st.column_config.ImageColumn("", width="small")}
            )

        st.info("ℹ️ **Nuevo Formato UEFA Champions League**: Los 36 clubes disputan 8 jornadas en una tabla única (4 de local y 4 de visitante). "
                "Los puestos **1º al 8º** clasifican directamente a los **Octavos de Final**. Los puestos **9º al 24º** disputan los **Knockout Play-offs** a ida y vuelta para definir a los otros 8 clasificados. "
                "Los clasificados del **25º al 36º** quedan eliminados definitivamente sin paso a Europa League.")

    # ============================================================================
    # TAB 3: EXPORTAR PARA IA (CSV)
    # ============================================================================
    with tab3:
        st.markdown('<div class="sec-title">📥 Exportación Cuantitativa para Análisis con IA</div>', unsafe_allow_html=True)
        st.markdown(
            "Genera y descarga un dataset enriquecido con **41 variables analíticas** por partido "
            "(probabilidades 1X2, xG Poisson, cuotas justas, diferenciales de ELO y plantillas, descanso y congestión, "
            "forma en ligas locales y consenso multi-modelo). Ideal para alimentar a **ChatGPT, Claude o Gemini** "
            "y obtener análisis cualitativos, tácticos y detección de sorpresas de alto valor."
        )

        fix_path = mo.DATA / "fixture.csv"
        if not fix_path.exists():
            st.warning("⚠️ No se encontró el archivo de fixture oficial de la UEFA Champions League.")
        else:
            df_fix_full = pd.read_csv(fix_path)
            temp_ucl = mo._temporada_actual()
            df_fix_act = df_fix_full[df_fix_full["temporada"] == temp_ucl].copy()
            df_fix_act["fecha_dt"] = pd.to_datetime(df_fix_act["fecha"])
            df_fix_act["dia"] = df_fix_act["fecha_dt"].dt.strftime("%Y-%m-%d")

            # Agrupar fechas en jornadas oficiales
            dias_unicos = sorted(df_fix_act["dia"].dropna().unique())
            jornadas_map = {}
            if dias_unicos:
                j_idx = 1
                curr_j = [dias_unicos[0]]
                for d in dias_unicos[1:]:
                    diff = (pd.to_datetime(d) - pd.to_datetime(curr_j[-1])).days
                    if diff <= 2:
                        curr_j.append(d)
                    else:
                        lbl = f"Jornada {j_idx} ({pd.to_datetime(curr_j[0]).strftime('%d/%m')} - {pd.to_datetime(curr_j[-1]).strftime('%d/%m/%Y')}) · {len(df_fix_act[df_fix_act['dia'].isin(curr_j)])} partidos"
                        jornadas_map[lbl] = curr_j
                        j_idx += 1
                        curr_j = [d]
                lbl = f"Jornada {j_idx} ({pd.to_datetime(curr_j[0]).strftime('%d/%m/%Y')}) · {len(df_fix_act[df_fix_act['dia'].isin(curr_j)])} partidos"
                jornadas_map[lbl] = curr_j

            # Filtros interactivos
            col_f1, col_f2, col_f3 = st.columns([4, 4, 3])

            with col_f1:
                filtro_modo = st.radio(
                    "🎯 Seleccionar Grupo de Partidos:",
                    [
                        "⚡ Próximos Partidos (Restantes de Hoy + Mañana)",
                        "📅 Por Jornada Oficial UCL",
                        "📆 Por Fecha Específica",
                        "🌐 Fixture Completo (36 Clubes)"
                    ],
                    index=0
                )

            with col_f2:
                df_filtrado = pd.DataFrame()
                sufijo_archivo = "proximos"

                if "Próximos" in filtro_modo:
                    df_pend = df_fix_act[df_fix_act["estado"] != "post"]
                    if df_pend.empty:
                        df_pend = df_fix_act
                    primer_dia = df_pend["dia"].min()
                    dias_prox = [primer_dia]
                    dia_sig = (pd.to_datetime(primer_dia) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    if dia_sig in dias_unicos:
                        dias_prox.append(dia_sig)
                    df_filtrado = df_fix_act[df_fix_act["dia"].isin(dias_prox)].copy()
                    st.info(f"📅 Partidos seleccionados: **{', '.join(dias_prox)}** ({len(df_filtrado)} cotejos)")
                    sufijo_archivo = f"proximos_{primer_dia}"

                elif "Jornada" in filtro_modo:
                    lista_jornadas = list(jornadas_map.keys())
                    sel_jornada = st.selectbox("Elegir Jornada:", lista_jornadas, index=0)
                    dias_sel = jornadas_map.get(sel_jornada, [])
                    df_filtrado = df_fix_act[df_fix_act["dia"].isin(dias_sel)].copy()
                    j_num = sel_jornada.split()[1] if len(sel_jornada.split()) > 1 else "ucl"
                    sufijo_archivo = f"jornada_{j_num}"

                elif "Fecha" in filtro_modo:
                    sel_dia = st.selectbox("Elegir Fecha:", dias_unicos, index=0)
                    df_filtrado = df_fix_act[df_fix_act["dia"] == sel_dia].copy()
                    sufijo_archivo = f"fecha_{sel_dia}"

                else:
                    df_filtrado = df_fix_act.copy()
                    st.info(f"🌐 Incluye los **{len(df_filtrado)}** partidos de la Fase de Liga.")
                    sufijo_archivo = "fixture_completo"

            with col_f3:
                modelo_exp = st.selectbox(
                    "🤖 Modelo para Probabilidades:",
                    [
                        "✨ Stacking Óptimo",
                        "🌲 Random Forest",
                        "🚀 XGBoost",
                        "🎯 LASSO L1"
                    ],
                    index=0 if modelo_tipo == "stacking" else (1 if modelo_tipo == "rf" else (2 if modelo_tipo == "xgb" else 3))
                )
                if "Stacking" in modelo_exp:
                    mod_code = "stacking"
                elif "Random" in modelo_exp:
                    mod_code = "rf"
                elif "XGB" in modelo_exp:
                    mod_code = "xgb"
                else:
                    mod_code = "lasso"

                solo_pendientes = st.checkbox("Excluir partidos ya finalizados", value=False)
                if solo_pendientes:
                    df_filtrado = df_filtrado[df_filtrado["estado"] != "post"]

            # Generación del Reporte Cuantitativo
            if df_filtrado.empty:
                st.warning("No hay partidos que coincidan con los filtros seleccionados.")
            else:
                with st.spinner("Calculando 41 métricas avanzadas (xG, descanso real, ELO, plantillas y consenso)..."):
                    df_reporte = mo.generar_reporte_partidos_ia(M, df_matches=df_filtrado, modelo=mod_code)

                # Métricas Resumen
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Partidos Seleccionados", f"{len(df_reporte)}")
                xg_prom = (df_reporte["xg_local"] + df_reporte["xg_visita"]).mean() if not df_reporte.empty else 0.0
                m_c2.metric("Promedio Goles Esperados (xG)", f"{xg_prom:.2f}")
                favoritos_locales = (df_reporte["prob_victoria_local_%"] > 50).sum()
                m_c3.metric("Favoritos Locales (>50%)", f"{favoritos_locales}")
                alertas_destacadas = (df_reporte["alerta_modelo"] != "🎯 Pronóstico Estándar").sum()
                m_c4.metric("Alertas Especiales", f"{alertas_destacadas}")

                # Botón de Descarga prominente
                csv_bytes = df_reporte.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Descargar CSV ({len(df_reporte)} partidos · 41 columnas) para IA",
                    data=csv_bytes,
                    file_name=f"pronosticos_ucl_{sufijo_archivo}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Vista Previa Interactiva
                st.markdown("##### 🔍 Vista Previa del Dataset Enriquecido")
                cols_preview = [
                    "fecha", "local", "visita", "prob_victoria_local_%", "prob_empate_%", "prob_victoria_visita_%",
                    "cuota_justa_local", "cuota_justa_empate", "cuota_justa_visita", "xg_local", "xg_visita",
                    "marcador_mas_probable", "prob_over_2_5_%", "elo_local", "elo_visita",
                    "plantilla_local_M€", "plantilla_visita_M€", "ventaja_descanso_dias", "alerta_modelo"
                ]
                cols_existentes = [c for c in cols_preview if c in df_reporte.columns]

                fmt_dict = {
                    "prob_victoria_local_%": "{:.1f}%",
                    "prob_empate_%": "{:.1f}%",
                    "prob_victoria_visita_%": "{:.1f}%",
                    "cuota_justa_local": "{:.2f}",
                    "cuota_justa_empate": "{:.2f}",
                    "cuota_justa_visita": "{:.2f}",
                    "xg_local": "{:.2f}",
                    "xg_visita": "{:.2f}",
                    "prob_over_2_5_%": "{:.1f}%",
                    "elo_local": "{:.0f}",
                    "elo_visita": "{:.0f}",
                    "plantilla_local_M€": "{:.1f}M€",
                    "plantilla_visita_M€": "{:.1f}M€",
                    "ventaja_descanso_dias": "{:+d}d"
                }
                fmt_apply = {k: v for k, v in fmt_dict.items() if k in cols_existentes}

                st.dataframe(
                    df_reporte[cols_existentes].style.format(fmt_apply),
                    hide_index=True,
                    width='stretch',
                    height=360
                )

                # Prompt estructurado listo para copiar
                with st.expander("🤖 Prompt Maestro de Inteligencia Externa & Análisis Periodístico (Para ChatGPT / Claude / Gemini / Perplexity)", expanded=True):
                    st.markdown(
                        "Copia el siguiente prompt y adjunta el archivo CSV descargado en tu modelo de lenguaje preferido "
                        "(**ChatGPT Plus con Web Search**, **Claude 3.5 Sonnet**, **Gemini 1.5 Pro** o **Perplexity**). "
                        "El prompt incluye **instrucciones estrictas para que la IA NO repita el CSV** y busque activamente información de periodistas, "
                        "partes médicos, ruedas de prensa UEFA, historial y claves tácticas de pizarra:"
                    )
                    prompt_texto = f"""Actúa como un **Jefe de Inteligencia Deportiva y Analista Táctico Senior de Fútbol Europeo**, combinando la analítica cuantitativa avanzada con la cobertura periodística de investigación de élite (estilo The Athletic, L'Équipe, Sky Sports, Kicker, Marca, Fabrizio Romano).

Te adjunto un archivo CSV que contiene las predicciones cuantitativas y métricas probabilísticas de {len(df_reporte)} partidos de la UEFA Champions League, generadas por un modelo híbrido calibrado (Poisson Dixon-Coles bivariado + Machine Learning Stacking con Random Forest, LASSO L1 y XGBoost).

⚠️ REGLAS ESTRICTAS DE RESPUESTA (PROHIBIDO REPETIR EL CSV):
1. ⛔ NO REPETIR NI RESUMIR LOS NÚMEROS DEL CSV: Ya dispongo de las 41 columnas y los porcentajes en mi pantalla. Queda estrictamente prohibido redactar respuestas como 'El equipo A tiene 65% de probabilidad y xG de 1.8'. Usa las probabilidades y los xG únicamente como un ancla cuantitativa silenciosa de fondo.
2. 🔎 INVESTIGACIÓN EXTERNA Y PERIODÍSTICA OBLIGATORIA (Web Search / Tiempo Real / Prensa Especializada):
   - Si tienes capacidad de navegación web o búsqueda en vivo (Google Search, Browsing), REALIZA BÚSQUEDAS de las últimas 24-48 horas sobre cada partido:
     * Ruedas de prensa oficiales UEFA de ambos entrenadores (declaraciones de intenciones, quejas de calendario, rotaciones anunciadas).
     * Partes médicos actualizados: bajas confirmadas por lesión o sanción, titulares indiscutidos entre algodones o que descansan para el torneo local.
     * Reportes de periodistas confiables y enviados especiales sobre el probable XI inicial y el clima de vestuario.
   - Si operas en modo offline o sin navegación, recurre a tu conocimiento táctico exhaustivo: modelos de juego de los técnicos, historial de enfrentamientos directos en Europa (H2H), debilidades estructurales (ej. vulnerabilidad a la contra, balón parado) y jerarquía histórica en Champions.
3. 🥊 CONTRASTE 'MODELO MATEMÁTICO VS REALIDAD PERIODÍSTICA': Tu mayor valor agregado es detectar las contradicciones entre las matemáticas y el contexto humano. Si el modelo marca favorito a un equipo pero la prensa revela que el vestuario está dividido o que el DT reservará a sus figuras, ¡debes destacarlo como una alerta crítica!

---

ESTRUCTURA OBLIGATORIA DE TU INFORME:

1. 🚨 RADAR DE PARTIDOS TRAMPA Y POSIBLES SORPRESAS:
   - Identifica los 2 o 3 partidos con mayor riesgo de romper el pronóstico del modelo y explica por qué las noticias periodísticas o el contexto del vestuario contradicen los números fríos.

2. 📰 BOLETÍN DE PRENSA Y NOVEDADES DE ÚLTIMA HORA:
   - Bajas confirmadas, dudas de última hora y rotaciones clave informadas por la prensa especializada para los cotejos de mayor cartel.
   - Contexto motivacional: ¿Quién se juega la vida en la tabla suiza y quién puede especular con el resultado?

3. ⚔️ RADIOGRAFÍA TÁCTICA PARTIDO POR PARTIDO:
   Para cada partido analizado:
   - Choque de Estilos & Pizarra: ¿Cómo atacará el local y cómo contrarrestará la visita? (ej. presión alta tras pérdida vs bloque bajo y transiciones rápidas; explotación de bandas vs sobrecarga interior).
   - Historial y Factor Psicológico (H2H): Precedentes recientes entre ambos clubes en competiciones europeas, peso de la localía y antecedentes con el árbitro o en estadios hostiles.
   - Duelo Individual Clave: El emparejamiento 1 vs 1 en el terreno que desequilibrará el trámite (ej. extremo desequilibrante vs lateral de perfil bajo, o mediocentro destructor vs mediapunta creativo).
   - Veredicto Cualitativo del Analista: Tu lectura definitiva del desarrollo del juego y resultado más verosímil considerando los xG del modelo junto al factor humano.

4. 💎 DETECCIÓN DE VALOR EN EL MERCADO (Value Insights):
   - ¿Qué cuotas del mercado o percepciones de los aficionados parecen mal calibradas frente a la realidad táctica y médica?
   - Oportunidades interesantes en mercados alternativos (Over/Under 2.5, Ambos Marcan, tarjetas o goles en el primer tiempo)."""
                    st.code(prompt_texto, language="markdown")

    # ============================================================================
    # TAB 4: IMPORTANCIA DE VARIABLES
    # ============================================================================
    with tab4:
        st.markdown('<div class="sec-title">Arquitectura Analítica e Importancia de Variables</div>', unsafe_allow_html=True)

        met = M.get("metricas", {})
        if met:
            st.markdown("#### Rendimiento Out-of-Sample (Test $\ge$ 2026)")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            if "lasso" in met:
                col_m1.metric("LASSO (L1)", f"Acc: {met['lasso']['accuracy']}%", f"LogLoss: {met['lasso']['logloss']}")
            if "rf" in met:
                col_m2.metric("Random Forest", f"Acc: {met['rf']['accuracy']}%", f"LogLoss: {met['rf']['logloss']}")
            if "xgb" in met:
                col_m3.metric("XGBoost", f"Acc: {met['xgb']['accuracy']}%", f"LogLoss: {met['xgb']['logloss']}")
            stk = met.get("stacking", {})
            if stk:
                col_m4.metric("Stacking Óptimo", f"Acc: {stk.get('accuracy', 0)}%", f"LogLoss: {stk.get('logloss', 0)}")
            if "w" in stk:
                st.caption(f"Pesos de ensemble: LASSO={stk['w'][0]} | RF={stk['w'][1]} | XGB={stk['w'][2]}")

        # Explicabilidad según el modelo activo
        importancia = []
        col_val_name = "Importancia Relativa"
        title_graph = "Top 15 Características Predictoras (UEFA Champions League)"

        if modelo_tipo == "lasso":
            pipe = M["pipe_lasso"]
            lr = pipe.named_steps["lr"]
            coefs = lr.coef_
            avg_coef = np.mean(np.abs(coefs), axis=0)
            col_val_name = "Peso Absoluto Promedio"
            title_graph = "Top 15 Predictores (LASSO L1 SAGA)"
            for feat, val in zip(M["features"], avg_coef):
                if val > 1e-4:
                    importancia.append({"Variable": feat, col_val_name: round(float(val), 4)})
        elif modelo_tipo == "xgb":
            pipe = M["pipe_xgb"]
            xgb_model = pipe.named_steps.get("xgb")
            col_val_name = "Importancia (Gain/Weight)"
            title_graph = "Top 15 Predictores (XGBoost)"
            if xgb_model and hasattr(xgb_model, "feature_importances_"):
                for feat, val in zip(M["features"], xgb_model.feature_importances_):
                    importancia.append({"Variable": feat, col_val_name: round(float(val), 4)})
        else:
            pipe = M["pipe_rf"]
            rf_model = pipe.named_steps.get("rf")
            col_val_name = "Importancia (Gini)"
            title_graph = "Top 15 Predictores (Random Forest)"
            if rf_model and hasattr(rf_model, "feature_importances_"):
                for feat, val in zip(M["features"], rf_model.feature_importances_):
                    importancia.append({"Variable": feat, col_val_name: round(float(val), 4)})

        if importancia:
            df_imp = pd.DataFrame(importancia).sort_values(by=col_val_name, ascending=False).reset_index(drop=True)
            col_t, col_g = st.columns([5, 7])
            with col_t:
                st.dataframe(df_imp, hide_index=True, width='stretch')
            with col_g:
                fig, ax = plt.subplots(figsize=(6, 5))
                top_n = df_imp.head(15)
                ax.barh(top_n["Variable"][::-1], top_n[col_val_name][::-1], color="#001438")
                ax.set_title(title_graph)
                ax.set_xlabel(col_val_name)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ============================================================================
    # TAB 5: VALIDACIÓN VS REALIDAD
    # ============================================================================
    with tab5:
        st.markdown('<div class="sec-title">El Modelo contra la Realidad (Out-of-sample)</div>', unsafe_allow_html=True)
        st.caption("Comparación de la predicción pre-partido del modelo contra el resultado real para cotejos jugados.")

        df_val, met_val, df_evol = mo.validacion_en_vivo(M, modelo_tipo=modelo_tipo)
        if df_val is None or df_val.empty:
            st.info("ℹ️ Aún no hay partidos finalizados en la temporada actual para validar.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Partidos Evaluados", f"{met_val['n']}")
            c2.metric("Acierto (1X2)", f"{met_val['acierto']:.1%}")
            c3.metric("Log-Loss Modelo", f"{met_val['logloss']:.3f}",
                      f"{met_val['logloss'] - met_val['logloss_base']:+.3f} vs baseline", delta_color="inverse")
            c4.metric("Log-Loss Baseline", f"{met_val['logloss_base']:.3f}")

            if met_val["logloss"] < met_val["logloss_base"]:
                st.success(f"El modelo supera al baseline histórico en los {met_val['n']} cotejos reales evaluados. 👍")
            else:
                st.warning("⚠️ El modelo se mantiene cercano al baseline histórico.")

            st.markdown("##### Historial Detallado de Predicciones")
            df_val_show = df_val[["fecha", "local", "visita", "goles_local", "goles_visita", "Prob_Local", "Prob_Empate", "Prob_Visita", "Prediccion"]].copy()
            map_res = {0: "Local", 1: "Empate", 2: "Visita"}
            df_val_show["Resultado Real"] = df_val_show.apply(
                lambda r: "Local" if r["goles_local"] > r["goles_visita"] else ("Empate" if r["goles_local"] == r["goles_visita"] else "Visita"), axis=1
            )
            df_val_show["Predicción"] = df_val_show["Prediccion"].map(map_res)
            df_val_show["Acierto"] = (df_val_show["Resultado Real"] == df_val_show["Predicción"]).replace({True: "✅", False: "❌"})
            df_val_show["Marcador"] = df_val_show.apply(lambda r: f"{int(r['goles_local'])} - {int(r['goles_visita'])}", axis=1)

            st.dataframe(
                df_val_show[["fecha", "local", "Marcador", "visita", "Resultado Real", "Predicción", "Acierto", "Prob_Local", "Prob_Empate", "Prob_Visita"]].style.format({
                    "Prob_Local": "{:.1%}",
                    "Prob_Empate": "{:.1%}",
                    "Prob_Visita": "{:.1%}"
                }),
                hide_index=True, width='stretch'
            )

            if not df_evol.empty and len(df_evol) >= 3:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(df_evol["partido_n"], df_evol["acierto_acumulado"], "o-", color="#001438", label="Acierto Acumulado")
                ax.set_xlabel("Partidos Jugados (Cronológico)")
                ax.set_ylabel("Tasa de Acierto")
                ax.set_title("Evolución de Tasa de Acierto en Champions League")
                ax.set_ylim(0, 1.0)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
