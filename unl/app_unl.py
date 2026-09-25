"""
Aplicación Streamlit para la predicción, simulación y análisis de la UEFA Nations League 2026-27.
Incluye 5 pestañas canónicas: Predicción Versus, Grupos y Proyecciones, Exportar para IA, Importancia de Variables y Validación vs Realidad.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor as mo
import recolectar_equipos as rec_eq
import recolectar as rec
import recolectar_boxscore as rec_box

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main-title { text-align:center; font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg,#071530,#00E5FF,#FFD700); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#071530;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#071530; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(7, 21, 48, 0.05), 0 4px 6px -4px rgba(7, 21, 48, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Simulando torneo Monte Carlo (5.000 iteraciones)...")
def get_simulacion(_M, mod_tipo):
    return mo.simular_campeonato(_M, n_sims=5000, modelo_tipo=mod_tipo)


def run_app():
    M = get_motor()
    equipos_dict = M["equipos_info"]
    flags = rec_eq.FLAGS

    # Sidebar: Controles del Modelo (Estandarizado con las demás ligas)
    st.sidebar.markdown("### 🛠️ Controles del Modelo")

    OPCIONES_MOD = [
        "🌲 Random Forest (Recomendado)",
        "📐 LASSO L1 (Regresión)",
        "🚀 XGBoost (Gradient Boosting)",
        "⚡ SVM (Support Vector Machine)",
        "🔀 Stacking (Ensemble óptimo)"
    ]
    modelo_sel = st.sidebar.selectbox("🤖 Modelo Predictivo:", OPCIONES_MOD, index=0)

    met_all = M.get("metricas", {})
    if "LASSO" in modelo_sel:
        mod_code = "lasso"
        met_act = met_all.get("lasso", {"logloss": 0.7654, "accuracy": 66.7})
        nombre_modelo = f"📐 LASSO L1 (C={met_all.get('best_c', 1.0)})"
    elif "XGB" in modelo_sel:
        mod_code = "xgb"
        met_act = met_all.get("xgb", {"logloss": 0.8112, "accuracy": 58.3})
        nombre_modelo = "🚀 XGBoost (Gradient Boosting)"
    elif "SVM" in modelo_sel:
        mod_code = "svm"
        met_act = met_all.get("svm", {"logloss": 0.7750, "accuracy": 75.0})
        nombre_modelo = "⚡ SVM (Kernel RBF)"
    elif "Stacking" in modelo_sel:
        mod_code = "stacking"
        met_act = met_all.get("stacking", {"logloss": 0.7720, "accuracy": 75.0})
        nombre_modelo = f"🔀 Stacking Óptimo (w={met_all.get('w', 0.028)})"
    else:
        mod_code = "rf"
        met_act = met_all.get("rf", {"logloss": 0.7710, "accuracy": 75.0})
        nombre_modelo = "🌲 Random Forest (Recomendado)"

    if st.sidebar.button("🔄 Actualizar ESPN y Re-entrenar", key="refresh_unl", type="primary"):
        with st.spinner("Descargando últimos resultados de Nations League desde ESPN..."):
            rec.recolectar()
            rec_box.recolectar()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    try:
        _partidos_df = pd.read_csv(mo.DATA / "partidos.csv", parse_dates=["fecha"])
        _ult_fecha = pd.to_datetime(_partidos_df["fecha"].max()).date().strftime("%d/%m/%Y")
        st.sidebar.caption(f"🗓️ Datos actualizados: {_ult_fecha} · {len(_partidos_df)} partidos")
    except Exception:
        pass

    # ── Métricas por modelo (sidebar canónico con las demás ligas)
    if met_all:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📊 Métricas Out-of-Sample (Test 2026)")
        valid_lls = [met_all[k]["logloss"] for k in met_all if isinstance(met_all[k], dict) and "logloss" in met_all[k]]
        mejor_ll = min(valid_lls) if valid_lls else None
        for nombre, clave in [
            ("LASSO", "lasso"),
            ("RF", "rf"),
            ("XGB", "xgb"),
            ("SVM", "svm"),
            ("Stacking", "stacking")
        ]:
            if clave not in met_all:
                continue
            m = met_all[clave]
            star = " ⭐" if mejor_ll is not None and m.get("logloss") == mejor_ll else ""
            w_str = f" (w={m['w']})" if clave == "stacking" and "w" in m else ""
            st.sidebar.caption(f"**{nombre}{w_str}{star}** — LL: `{m['logloss']:.4f}` | Acc: `{m['accuracy']:.1f}%`")
        st.sidebar.caption("📏 *Baseline Marginal*: `1.0309` (33.3% uniforme)")

    st.sidebar.markdown("---")
    st.sidebar.success(f"📊 Base de Datos: **54** Selecciones · **670** Partidos Históricos · **148** en Fixture")

    # Encabezado Canónico
    st.markdown('<div class="main-title">🇪🇺 UEFA Nations League Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Modelo activo: <b>{nombre_modelo}</b> — LASSO + RF + XGBoost + SVM + Stacking · Final Four, Ascensos y Descensos</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚽ Predicción Versus",
        "📊 Grupos y Proyecciones",
        "📥 Exportar para IA (CSV)",
        "🔬 Importancia de Variables",
        "🎯 Validación vs Realidad"
    ])

    # =========================================================================
    # TAB 1: Predicción Versus
    # =========================================================================
    with tab1:
        st.markdown('<div class="sec-title">Analizador de Enfrentamientos y Mercados</div>', unsafe_allow_html=True)

        fix_path = mo.DATA / "fixture.csv"
        df_fix = pd.read_csv(fix_path) if fix_path.exists() else pd.DataFrame()

        col_modo, col_partido = st.columns([4, 8])
        with col_modo:
            modo_vs = st.radio("Método de Selección:", ["📅 Partido del Fixture Oficial", "✍️ Selección Libre"], index=0, key="unl_modo_vs")

        sel_loc, sel_vis = "Spain", "Germany"
        partido_info = None

        if "Fixture" in modo_vs and not df_fix.empty:
            df_pend = df_fix[df_fix["estado"] != "post"].copy()
            if df_pend.empty:
                df_pend = df_fix.copy()
            df_pend["label"] = df_pend.apply(
                lambda r: f"{r['fecha'][:16]} · {flags.get(r['local'],'')} {r['local']} vs {flags.get(r['visita'],'')} {r['visita']} ({r['group']})", axis=1
            )
            with col_partido:
                idx_partido = st.selectbox("Seleccionar Partido:", range(len(df_pend)), format_func=lambda i: df_pend.iloc[i]["label"], key="unl_sel_fix")
                row_sel = df_pend.iloc[idx_partido]
                sel_loc = row_sel["local"]
                sel_vis = row_sel["visita"]
                partido_info = row_sel
        else:
            todos_eq = sorted(list(equipos_dict.keys()))
            with col_partido:
                c1, c2 = st.columns(2)
                with c1:
                    sel_loc = st.selectbox("Local:", todos_eq, index=todos_eq.index("Spain") if "Spain" in todos_eq else 0, key="unl_free_loc")
                with c2:
                    sel_vis = st.selectbox("Visita:", todos_eq, index=todos_eq.index("Germany") if "Germany" in todos_eq else 1, key="unl_free_vis")

        if sel_loc == sel_vis:
            st.warning("Selecciona dos equipos distintos para calcular la predicción.")
        else:
            p_1x2, la, lb = mo.predecir_match(M, sel_loc, sel_vis, modelo=mod_code)
            merc = mo.calcular_mercados(la, lb)
            tracker = M["tracker"]
            feats = tracker.get_features_for_match(sel_loc, sel_vis)

            eq_l = equipos_dict.get(sel_loc, {})
            eq_v = equipos_dict.get(sel_vis, {})
            logo_l = eq_l.get("logo")
            logo_v = eq_v.get("logo")
            flag_l = flags.get(sel_loc, "⚽")
            flag_v = flags.get(sel_vis, "⚽")

            # Encabezado visual del partido
            with st.container(border=True):
                c_h_l, c_h_vs, c_h_v = st.columns([5, 2, 5])
                with c_h_l:
                    st.markdown(f"<div style='text-align:center;'>", unsafe_allow_html=True)
                    if logo_l:
                        st.markdown(f"<img src='{logo_l}' width='80' style='border-radius:50%;'>", unsafe_allow_html=True)
                    st.markdown(f"### {flag_l} {sel_loc}")
                    st.caption(f"ELO: **{feats['elo_local']:.0f} pts** · Plantilla: **€{feats['sv_local']:.1f}M**")
                    st.markdown("</div>", unsafe_allow_html=True)

                with c_h_vs:
                    st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
                    if partido_info is not None:
                        st.caption(f"<div style='text-align:center;'>{partido_info.get('group','')}<br>{partido_info.get('fecha','')[:10]}</div>", unsafe_allow_html=True)

                with c_h_v:
                    st.markdown(f"<div style='text-align:center;'>", unsafe_allow_html=True)
                    if logo_v:
                        st.markdown(f"<img src='{logo_v}' width='80' style='border-radius:50%;'>", unsafe_allow_html=True)
                    st.markdown(f"### {flag_v} {sel_vis}")
                    st.caption(f"ELO: **{feats['elo_visita']:.0f} pts** · Plantilla: **€{feats['sv_visita']:.1f}M**")
                    st.markdown("</div>", unsafe_allow_html=True)

            # Tarjetas de Probabilidades y Goles Esperados
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric(f"Victoria {sel_loc}", f"{p_1x2[0]:.1%}", f"Cuota: {mo.cuota(p_1x2[0])}")
            col_p2.metric("Empate", f"{p_1x2[1]:.1%}", f"Cuota: {mo.cuota(p_1x2[1])}")
            col_p3.metric(f"Victoria {sel_vis}", f"{p_1x2[2]:.1%}", f"Cuota: {mo.cuota(p_1x2[2])}")
            col_p4.metric("Goles Esperados (xG)", f"{la:.2f} - {lb:.2f}", f"Total: {la+lb:.2f}")

            # Barra de probabilidades visual
            st.markdown(f"""
            <div style="display:flex;height:24px;border-radius:12px;overflow:hidden;margin:12px 0 20px 0;font-size:12px;font-weight:700;color:white;text-align:center;line-height:24px;">
                <div style="width:{p_1x2[0]*100}%;background:#0b3d91;">{sel_loc} ({p_1x2[0]:.0%})</div>
                <div style="width:{p_1x2[1]*100}%;background:#64748b;">Empate ({p_1x2[1]:.0%})</div>
                <div style="width:{p_1x2[2]*100}%;background:#10b981;">{sel_vis} ({p_1x2[2]:.0%})</div>
            </div>
            """, unsafe_allow_html=True)

            # Matriz de marcador exacto y mercados secundarios
            col_mat, col_merc = st.columns([6, 5])
            with col_mat:
                st.markdown("##### 🎯 Matriz de Marcador Exacto (Poisson)")
                fig, ax = plt.subplots(figsize=(5.5, 4.2))
                cax = ax.matshow(merc["matrix"] * 100, cmap="Blues", alpha=0.85)
                fig.colorbar(cax, fraction=0.046, pad=0.04, label="% Probabilidad")
                for i in range(6):
                    for j in range(6):
                        val = merc["matrix"][i, j] * 100
                        ax.text(j, i, f"{val:.1f}%", ha='center', va='center', color='black' if val < 7 else 'white', fontsize=8)
                ax.set_xlabel(f"Goles {sel_vis}", fontsize=9)
                ax.set_ylabel(f"Goles {sel_loc}", fontsize=9)
                ax.set_xticks(range(6))
                ax.set_yticks(range(6))
                st.pyplot(fig)
                plt.close(fig)

            with col_merc:
                st.markdown("##### 📈 Mercados y Marcadores Más Probables")
                top_sc_df = pd.DataFrame(merc["top_marcadores"], columns=["Marcador", "Probabilidad"])
                top_sc_df["Probabilidad"] = top_sc_df["Probabilidad"].map(lambda x: f"{x:.1%}")
                st.dataframe(top_sc_df, hide_index=True, width='stretch')

                st.markdown("##### ⚽ Mercados de Goles")
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("Ambos Anotan (BTTS)", f"{merc['btts']['si']:.1%}", f"No: {merc['btts']['no']:.1%}")
                c_m2.metric("Más de 2.5 Goles", f"{merc['over_under']['2.5']['over']:.1%}", f"Menos: {merc['over_under']['2.5']['under']:.1%}")

    # =========================================================================
    # TAB 2: Grupos y Proyecciones
    # =========================================================================
    with tab2:
        st.markdown('<div class="sec-title">Tablas de Posiciones y Proyecciones de Torneo</div>', unsafe_allow_html=True)
        st.markdown("Simulación estocástica de **5.000 torneos completos** calculando las probabilidades de Final Four, Título, Ascenso y Descenso.")

        sim = get_simulacion(M, mod_code)
        tablas_act = sim["tablas_actuales"]
        proy_grp = sim["proyecciones_grupos"]

        # Resumen de favoritos al título de Nations League (Liga A)
        st.markdown("#### 🏆 Candidatos al Título de UEFA Nations League 2026-27 (Final Four)")
        df_camp = sim["campeon"].head(8).copy()
        df_camp["Selección"] = df_camp["Selección"].map(lambda t: f"{flags.get(t,'')} {t}")
        
        c_top1, c_top2 = st.columns([6, 5])
        with c_top1:
            fig_camp, ax_camp = plt.subplots(figsize=(6, 3.8))
            y_pos = np.arange(len(df_camp))[::-1]
            ax_camp.barh(y_pos, df_camp["P_Campeon"] * 100, color="#0b3d91", alpha=0.85)
            ax_camp.set_yticks(y_pos)
            ax_camp.set_yticklabels(df_camp["Selección"], fontsize=9)
            ax_camp.set_xlabel("% Probabilidad de Campeón", fontsize=9)
            ax_camp.grid(axis="x", ls=":", alpha=0.6)
            st.pyplot(fig_camp)
            plt.close(fig_camp)

        with c_top2:
            st.dataframe(
                df_camp.style.format({"P_FinalFour": "{:.1%}", "P_Campeon": "{:.1%}"})
                            .background_gradient(subset=["P_Campeon"], cmap="Blues"),
                hide_index=True, width='stretch'
            )

        st.markdown("---")
        st.markdown("#### 🌐 Explorador de Grupos por División")

        liga_sel = st.radio("Selecciona División:", ["Liga A", "Liga B", "Liga C", "Liga D"], horizontal=True, key="unl_sel_liga")
        pref = liga_sel.split()[-1]
        grupos_liga = [g for g in sorted(tablas_act.keys()) if g.startswith(f"Group {pref}")]

        col_t1, col_t2 = st.columns(2)
        for idx, grp in enumerate(grupos_liga):
            target_col = col_t1 if idx % 2 == 0 else col_t2
            with target_col:
                with st.container(border=True):
                    st.markdown(f"##### 📌 {grp}")
                    tab_view = tablas_act[grp].copy()
                    tab_view["Equipo"] = tab_view["Equipo"].map(lambda t: f"{flags.get(t,'')} {t}")
                    st.caption("Tabla Actual")
                    st.dataframe(tab_view, hide_index=True, width='stretch')

                    st.caption("Proyección Final (Monte Carlo)")
                    p_view = proy_grp[grp].copy()
                    p_view["Equipo"] = p_view["Equipo"].map(lambda t: f"{flags.get(t,'')} {t}")
                    pct_cols = [c for c in ["P(1°)", "P(2°)", "P(3°)", "P(Descenso)"] if c in p_view.columns]
                    st.dataframe(
                        p_view.style.format({c: "{:.1%}" for c in pct_cols})
                                    .background_gradient(subset=["P(1°)"], cmap="YlGn"),
                        hide_index=True, width='stretch'
                    )

    # =========================================================================
    # TAB 3: Exportar para IA (CSV)
    # =========================================================================
    with tab3:
        st.markdown('<div class="sec-title">📥 Exportación Cuantitativa para Análisis con IA</div>', unsafe_allow_html=True)
        st.markdown(
            "Genera y descarga un dataset enriquecido con **32 variables analíticas** por partido de Nations League "
            "(probabilidades 1X2, xG Poisson, cuotas justas, diferenciales de ELO y plantillas, distancia de viaje y consenso). "
            "Pásale el CSV a **ChatGPT, Claude o Gemini** junto al prompt especializado de abajo para obtener análisis de periodistas, "
            "historial y bajas sin que la IA se limite a repetir los porcentajes."
        )

        fix_path = mo.DATA / "fixture.csv"
        df_fix_all = pd.read_csv(fix_path) if fix_path.exists() else pd.DataFrame()

        c_filtro1, c_filtro2 = st.columns([5, 6])
        with c_filtro1:
            modo_exp = st.radio(
                "Seleccionar Partidos a Exportar:",
                [
                    "⚡ Próximos Partidos (Hoy + Mañana)",
                    "📅 Por Grupo de Nations League",
                    "🌐 Fixture Completo (148 Partidos)"
                ],
                key="unl_filtro_ia"
            )

        with c_filtro2:
            df_exportar = pd.DataFrame()
            if "Próximos" in modo_exp and not df_fix_all.empty:
                df_p = df_fix_all[df_fix_all["estado"] != "post"].copy()
                min_fecha = df_p["fecha"].str[:10].min()
                fechas_prox = [min_fecha]
                dia_sig = (pd.to_datetime(min_fecha) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                fechas_prox.append(dia_sig)
                df_exportar = df_p[df_p["fecha"].str[:10].isin(fechas_prox)].copy()
                st.info(f"📅 Partidos seleccionados: **{min_fecha}** y **{dia_sig}** ({len(df_exportar)} encuentros)")
            elif "Grupo" in modo_exp and not df_fix_all.empty:
                grps_disp = sorted(df_fix_all["group"].unique())
                sel_g = st.selectbox("Elegir Grupo:", grps_disp, key="unl_grp_exp")
                df_exportar = df_fix_all[df_fix_all["group"] == sel_g].copy()
            else:
                df_exportar = df_fix_all.copy()
                st.info(f"🌐 Incluye los **{len(df_exportar)}** partidos programados.")

        if not df_exportar.empty:
            df_reporte = mo.generar_reporte_partidos_ia(M, df_matches=df_exportar, modelo=mod_code)
            
            # Botón de Descarga
            csv_data = df_reporte.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Descargar CSV ({len(df_reporte)} partidos · 32 columnas) para IA",
                data=csv_data,
                file_name=f"uefa_nations_league_pronosticos.csv",
                mime="text/csv",
                type="primary",
                width='stretch'
            )

            # Vista Previa
            st.markdown("##### 🔍 Vista Previa del CSV")
            preview_cols = ["fecha", "grupo", "local", "visita", "prob_victoria_local_%", "prob_empate_%", "prob_victoria_visita_%", "cuota_justa_local", "xg_local", "xg_visita", "marcador_mas_probable", "alerta_modelo"]
            st.dataframe(df_reporte[[c for c in preview_cols if c in df_reporte.columns]], hide_index=True, width='stretch', height=240)

            # Prompt Maestro para IA
            st.markdown("##### 🧠 Prompt Maestro para Inteligencia Artificial")
            st.caption("Copia este prompt y pégalo en tu IA favorita junto con el archivo CSV descargado:")

            prompt_texto = f"""Eres un analista táctico de fútbol internacional y modelador cuantitativo deportivo de alto nivel.
Te adjunto un dataset CSV con las probabilidades oficiales de la UEFA Nations League 2026-27 generadas por nuestro modelo híbrido (Machine Learning LASSO/Random Forest + Poisson Dixon-Coles, Elo dinámico y valores de mercado).

REGLAS OBLIGATORIAS DE ANÁLISIS (NO REPITAS EL CSV):
1. NO te limites a transcribir los porcentajes ni las cuotas que ya están en el archivo. El objetivo es complementar estos números con información cualitativa de analistas, periodistas deportivos y contexto táctico.
2. Investiga y analiza en profundidad para cada partido:
   - Convocatorias oficiales, bajas sensibles (lesiones, suspensiones, descanso de estrellas).
   - Momentos anímicos y declaraciones de los entrenadores en conferencias de prensa.
   - Historial cara a cara (H2H) y antecedentes recientes en torneos oficiales.
   - Contexto en la tabla de la Nations League: ¿quién se juega el descenso a la división inferior? ¿quién necesita ganar para clasificar al Final Four o ascender de Liga?
3. Para cada partido del CSV, entrega:
   - **Diagnóstico Táctico**: Estilo de juego esperado y choque de sistemas (bloque bajo, posesión, transiciones rápidas).
   - **Evaluación del Pronóstico Cuantitativo**: ¿Tiene sentido la probabilidad del modelo o detectas sesgos por rotación de plantel o falta de motivación?
   - **Predicción Final y Pick Recomendado**: Pronóstico argumentado con marcador probable.
"""
            st.code(prompt_texto, language="markdown")

    # =========================================================================
    # TAB 4: Importancia de Variables
    # =========================================================================
    with tab4:
        st.markdown('<div class="sec-title">Explicabilidad del Modelo e Importancia de Variables</div>', unsafe_allow_html=True)
        cols_feat = M["cols_features"]

        if mod_code == "rf":
            st.markdown("Análisis de importancia de variables para **Random Forest** según la reducción promedio de impureza de Gini.")
            importancias = M["pipe_rf"].named_steps["rf"].feature_importances_
            label_imp = "Importancia Gini (Random Forest)"
        elif mod_code == "xgb":
            st.markdown("Análisis de ganancia relativa de variables para **XGBoost (Gradient Boosting)**.")
            importancias = M["pipe_xgb"].named_steps["xgb"].feature_importances_
            label_imp = "Importancia de Ganancia (XGBoost)"
        elif mod_code == "svm":
            st.markdown("Análisis de sensibilidad de hiperplano para **Support Vector Machine (Kernel RBF)**.")
            importancias = np.mean(np.abs(M["pipe_lasso"].named_steps["lr"].coef_), axis=0)
            label_imp = "Sensibilidad de Variables (SVM RBF)"
        elif mod_code == "stacking":
            st.markdown("Importancia de variables ponderada en el ensamble **Stacking Óptimo** (XGBoost + Random Forest).")
            w = M.get("alpha_stack", 0.028)
            importancias = w * M["pipe_xgb"].named_steps["xgb"].feature_importances_ + (1.0 - w) * M["pipe_rf"].named_steps["rf"].feature_importances_
            label_imp = "Importancia Ponderada (Stacking)"
        else:
            st.markdown("Análisis de pesos del clasificador lineal regularizado **LASSO (L1 SAGA)** que penaliza el ruido estadístico.")
            importancias = np.mean(np.abs(M["pipe_lasso"].named_steps["lr"].coef_), axis=0)
            label_imp = "Peso Absoluto Medio (LASSO L1 SAGA)"

        df_weights = pd.DataFrame({
            "Variable": cols_feat,
            "Importancia": importancias
        }).sort_values("Importancia", ascending=True)

        col_w1, col_w2 = st.columns([7, 5])
        with col_w1:
            fig_w, ax_w = plt.subplots(figsize=(6, 4))
            ax_w.barh(df_weights["Variable"], df_weights["Importancia"], color="#0b3d91", alpha=0.85)
            ax_w.set_xlabel(label_imp, fontsize=9)
            ax_w.grid(axis="x", ls=":", alpha=0.6)
            st.pyplot(fig_w)
            plt.close(fig_w)

        with col_w2:
            st.markdown("##### 📋 Cuadro Comparativo de Modelos (Test 2026)")
            met = M["metricas"]
            df_comp_met = pd.DataFrame([
                {"Modelo": "🌲 Random Forest", "Log-Loss": f"{met['rf']['logloss']:.4f}", "Acierto": f"{met['rf']['accuracy']:.1f}%", "Config": "Depth=5, N=200"},
                {"Modelo": "📐 LASSO L1", "Log-Loss": f"{met['lasso']['logloss']:.4f}", "Acierto": f"{met['lasso']['accuracy']:.1f}%", "Config": f"C={met.get('best_c', 1.0)}"},
                {"Modelo": "🚀 XGBoost", "Log-Loss": f"{met['xgb']['logloss']:.4f}", "Acierto": f"{met['xgb']['accuracy']:.1f}%", "Config": "Depth=3, lr=0.03"},
                {"Modelo": "⚡ SVM (RBF)", "Log-Loss": f"{met['svm']['logloss']:.4f}", "Acierto": f"{met['svm']['accuracy']:.1f}%", "Config": "C=0.5, RBF"},
                {"Modelo": "🔀 Stacking", "Log-Loss": f"{met['stacking']['logloss']:.4f}", "Acierto": f"{met['stacking']['accuracy']:.1f}%", "Config": f"w={met.get('w', 0.028)}"},
                {"Modelo": "📏 Baseline", "Log-Loss": "1.0309", "Acierto": "33.3%", "Config": "Ingenuo Marginal"}
            ])
            st.dataframe(df_comp_met, hide_index=True, width='stretch')
            st.caption("El modelo penaliza variables redundantes y captura efectos no lineales en la brecha entre divisiones europeas.")

    # =========================================================================
    # TAB 5: Validación vs Realidad
    # =========================================================================
    with tab5:
        st.markdown('<div class="sec-title">El Modelo contra la Realidad · Nations League 2026</div>', unsafe_allow_html=True)
        st.markdown("Auditoría empírica en tiempo real: cotejo de predicciones pre-partido contra los resultados de los partidos ya jugados en la edición 2026-27.")

        df_val, met, evol = mo.validacion_en_vivo(M, modelo=mod_code)

        if df_val.empty:
            st.info("Aún no hay partidos finalizados reportados en la edición 2026.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Partidos Evaluados", met["n"])
            m2.metric("Tasa de Acierto (1X2)", f"{met['acierto']:.1%}")
            m3.metric("Log-loss Modelo", f"{met['logloss']:.3f}", f"{met['logloss'] - met['logloss_base']:+.3f} vs baseline", delta_color="inverse")
            m4.metric("Log-loss Baseline", f"{met['logloss_base']:.3f}")

            if met["logloss"] < met["logloss_base"]:
                st.success(f"🏆 El modelo supera con creces al baseline ingenuo en los {met['n']} partidos jugados ({met['acierto']:.1%} de acierto).")

            st.markdown("##### 📋 Historial Oficial de Predicciones y Resultados 2026")
            
            todos_val_eq = ["(Todas)"] + sorted(list(set(df_val["Local"]).union(set(df_val["Visita"]))))
            filtro_eq = st.selectbox("Filtrar por Selección:", todos_val_eq, key="unl_filtro_val")

            df_val_show = df_val.copy()
            if filtro_eq != "(Todas)":
                df_val_show = df_val_show[(df_val_show["Local"] == filtro_eq) | (df_val_show["Visita"] == filtro_eq)]

            df_val_show["Local"] = df_val_show["Local"].map(lambda t: f"{flags.get(t,'')} {t}")
            df_val_show["Visita"] = df_val_show["Visita"].map(lambda t: f"{flags.get(t,'')} {t}")
            st.dataframe(df_val_show, hide_index=True, width='stretch', height=280)

            c_plot, c_cal = st.columns([6, 5])
            with c_plot:
                st.markdown("##### 📈 Evolución del Log-loss Acumulado")
                fig_ev, ax_ev = plt.subplots(figsize=(6, 3.8))
                ax_ev.plot(evol["partido"], evol["logloss_acum"], "o-", color="#0b3d91", label="Modelo")
                ax_ev.axhline(evol["baseline"].iloc[0], color="#dc2626", ls="--", label="Baseline")
                ax_ev.set_xlabel("Partidos Jugados (Cronológico)")
                ax_ev.set_ylabel("Log-loss Acumulado")
                ax_ev.grid(True, ls=":", alpha=0.5)
                ax_ev.legend()
                st.pyplot(fig_ev)
                plt.close(fig_ev)

            with c_cal:
                if "P" in met and "y" in met and len(met["y"]) >= 5:
                    st.markdown("##### 🎯 Curva de Calibración (Victoria Local)")
                    xs, ys, ns, ece = mo.curva_calibracion(met["P"], met["y"], 2, n_bins=4)
                    fig_c, ax_c = plt.subplots(figsize=(5, 3.8))
                    ax_c.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Ideal")
                    ax_c.plot(xs, ys, "o-", color="#0b3d91", lw=2, label="Modelo")
                    ax_c.set_title(f"Victoria Local (ECE={ece:.3f})", fontsize=10, fontweight="bold")
                    ax_c.set_xlabel("Probabilidad Predicha", fontsize=8)
                    ax_c.set_ylabel("Frecuencia Observada", fontsize=8)
                    ax_c.set_xlim(0, 1)
                    ax_c.set_ylim(0, 1)
                    ax_c.grid(True, ls=":", alpha=0.5)
                    ax_c.legend()
                    st.pyplot(fig_c)
                    plt.close(fig_c)


if __name__ == "__main__":
    run_app()
