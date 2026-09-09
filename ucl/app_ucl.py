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
.main-title { text-align:center; font-size:2.9rem; font-weight:900;
    background:linear-gradient(135deg,#001438,#00E5FF,#FFD700); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.15rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#001438;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#001438; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#00E5FF; margin-top:1.6rem; text-shadow:0 0 10px rgba(0,229,255,0.4); }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 20, 56, 0.08), 0 4px 6px -4px rgba(0, 20, 56, 0.05);
    border-radius:16px;
    border: 1px solid #cbd5e1;
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
    return f'<img src="{url}" width="{size}" style="border-radius:10px; box-shadow:0 3px 10px rgba(0,20,56,0.25);">'


def fmt_opcion(equipos, team):
    """Etiqueta de opción del selectbox: abreviatura del club + nombre."""
    if equipos:
        for e in equipos.values():
            if e.get("norm_name") == team and e.get("abbreviation"):
                return f"{e['abbreviation']} · {team}"
    return team


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Corriendo simulaciones Monte Carlo del formato suizo UCL (10.000 iteraciones)...")
def simular_ucl(_M, key, modelo_tipo):
    return mo.simular_campeonato(_M, n_sims=10000, modelo_tipo=modelo_tipo)


def run_app():
    st.sidebar.markdown("### ⭐ Controles del Modelo UCL")

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

    st.markdown('<div class="main-title">⭐ UEFA Champions League Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Fase de Liga de 36 Clubes · Play-offs y Cuadro Eliminatorio · Poisson Dixon-Coles & ML</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚔️ Match Predictor (Versus)",
        "🏆 Tabla 36 & Proyección Monte Carlo",
        "🧠 Importancia de Variables",
        "📈 Validación en Vivo"
    ])

    # -------------------------------------------------------------
    # TAB 1: MATCH PREDICTOR
    # -------------------------------------------------------------
    with tab1:
        # Los 36 clubes activos de la Fase de Liga 2026-2027
        fix_path = mo.DATA / "fixture.csv"
        fixture_disp = pd.DataFrame()
        equipos_activos = []
        if fix_path.exists():
            fix_raw = pd.read_csv(fix_path)
            fixture_disp = fix_raw[fix_raw.temporada == mo._temporada_actual()].copy()
            equipos_activos = sorted(list(set(fixture_disp["local"]).union(set(fixture_disp["visita"]))))

        if not equipos_activos:
            equipos_activos = sorted(list(tracker.elos.keys()))

        def_local_idx = 0
        def_visita_idx = 1 if len(equipos_activos) > 1 else 0

        # Cargar partido programado desde el Fixture oficial de 136 cotejos
        if not fixture_disp.empty:
            st.markdown("#### ⚡ Próximos Partidos Oficiales de Fase de Liga")
            opciones_fixture = ["-- Seleccionar del calendario oficial UCL --"] + [
                f"{r.local} vs {r.visita} ({pd.to_datetime(r.fecha).strftime('%d/%m %H:%M') if pd.notna(r.fecha) else 'Fecha TBD'})"
                for _, r in fixture_disp.head(25).iterrows()
            ]
            partido_elegido = st.selectbox("Cargar cotejo programado:", opciones_fixture, index=0)
            if partido_elegido != "-- Seleccionar del calendario oficial UCL --":
                l_nom = partido_elegido.split(" vs ")[0]
                v_nom = partido_elegido.split(" vs ")[1].split(" (")[0]
                if l_nom in equipos_activos:
                    def_local_idx = equipos_activos.index(l_nom)
                if v_nom in equipos_activos:
                    def_visita_idx = equipos_activos.index(v_nom)

        st.markdown('<div class="sec-title">Configuración del Encuentro</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            es_knockout = st.checkbox("⚔️ Partido de Eliminatoria Directa (Knockout / Play-offs)", value=False)
        with col_c2:
            es_neutral = st.checkbox("🏟️ Sede Neutral (Ej. Gran Final)", value=False)

        col_loc, col_mid, col_vis = st.columns([1.2, 0.4, 1.2])

        with col_loc:
            st.markdown('<div class="card-title">🏠 Club Local</div>', unsafe_allow_html=True)
            local = st.selectbox("Selecciona Local:", equipos_activos, index=def_local_idx, format_func=lambda t: fmt_opcion(equipos_data, t), key="sb_loc")
            escudo_l = logo_html(equipos_data, local, size=75)
            if escudo_l:
                st.markdown(f"<div style='text-align:center;margin-top:5px;'>{escudo_l}</div>", unsafe_allow_html=True)
            elo_l = tracker.elos[local]
            val_l = mo.get_squad_value(local, mo._temporada_actual())
            st.metric("ELO Rating", f"{elo_l:.0f} pts")
            st.metric("Valor Plantilla", f"€{val_l:.1f}M")

        with col_mid:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

        with col_vis:
            st.markdown('<div class="card-title">✈️ Club Visitante</div>', unsafe_allow_html=True)
            visita = st.selectbox("Selecciona Visitante:", equipos_activos, index=def_visita_idx, format_func=lambda t: fmt_opcion(equipos_data, t), key="sb_vis")
            escudo_v = logo_html(equipos_data, visita, size=75)
            if escudo_v:
                st.markdown(f"<div style='text-align:center;margin-top:5px;'>{escudo_v}</div>", unsafe_allow_html=True)
            elo_v = tracker.elos[visita]
            val_v = mo.get_squad_value(visita, mo._temporada_actual())
            st.metric("ELO Rating", f"{elo_v:.0f} pts")
            st.metric("Valor Plantilla", f"€{val_v:.1f}M")

        if local == visita:
            st.warning("⚠️ Selecciona dos clubes distintos para pronosticar el partido.")
        else:
            p, la, lb = mo.predecir_match(
                M, local, visita, temporada=mo._temporada_actual(),
                modelo=modelo_tipo, is_neutral=1 if es_neutral else 0, is_knockout=1 if es_knockout else 0
            )

            st.markdown("---")
            st.markdown('<div class="sec-title">🎯 Probabilidades del Partido</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(f"Victoria {local}", f"{p[0]:.1%}")
                st.progress(float(p[0]))
            with c2:
                st.metric("Empate", f"{p[1]:.1%}")
                st.progress(float(p[1]))
            with c3:
                st.metric(f"Victoria {visita}", f"{p[2]:.1%}")
                st.progress(float(p[2]))

            st.markdown("#### ⚽ Goles Esperados (xG Bivariado Poisson Dixon-Coles)")
            cx1, cx2 = st.columns(2)
            cx1.metric(f"xG {local}", f"{la:.2f}")
            cx2.metric(f"xG {visita}", f"{lb:.2f}")

            # Cuotas implícitas y mercados
            st.markdown("#### 💰 Mercados de Apuestas Probabilísticas")
            cm1, cm2, cm3, cm4 = st.columns(4)
            mat_dc = mo.matriz_marcador_exacto(la, lb)
            p_over25 = 1.0 - sum(mat_dc[i, j] for i in range(7) for j in range(7) if i + j <= 2)
            p_btts = sum(mat_dc[i, j] for i in range(1, 7) for j in range(1, 7))

            cm1.metric("Cuota 1X2 (Local)", f"{1/p[0]:.2f}" if p[0] > 0.01 else ">100")
            cm2.metric("Cuota 1X2 (Empate)", f"{1/p[1]:.2f}" if p[1] > 0.01 else ">100")
            cm3.metric("Más de 2.5 Goles", f"{p_over25:.1%}")
            cm4.metric("Ambos Anotan (BTTS)", f"{p_btts:.1%}")

            # Gráfica de la matriz Dixon-Coles
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

            # Variables clave del duelo
            with st.expander("🔍 Métricas y Variables del Enfrentamiento"):
                feats = tracker.get_features_for_match(local, visita, mo._temporada_actual(), is_knockout=1 if es_knockout else 0, is_neutral=1 if es_neutral else 0)
                col_f1, col_f2, col_f3 = st.columns(3)
                col_f1.metric("Distancia de Viaje", f"{feats['distance_km']:.0f} km")
                col_f2.metric("Diferencia de Altitud", f"{feats['altitude_diff']:+.0f} m")
                col_f3.metric("Ratio Valor de Plantilla", f"{np.exp(feats['squad_value_diff']):.2f}x")

                col_f4, col_f5, col_f6 = st.columns(3)
                col_f4.metric("Diferencial de Forma (5 PJ)", f"{feats['form_diff']:+.2f}")
                col_f5.metric("Diferencial Pi-Rating", f"{feats['pi_diff']:+.2f}")
                col_f6.metric("Historial H2H (Goles Netos)", f"{feats['h2h_diff']:+.2f}")

    # -------------------------------------------------------------
    # TAB 2: TABLA & MONTE CARLO
    # -------------------------------------------------------------
    with tab2:
        st.markdown('<div class="sec-title">🏆 Fase de Liga (36 Clubes) y Proyecciones Monte Carlo</div>', unsafe_allow_html=True)
        tab_actual = mo.obtener_tabla_actual(M)

        col_t1, col_t2 = st.columns([1, 1.3])
        with col_t1:
            st.markdown("#### Tabla Actual de Fase de Liga")
            st.dataframe(tab_actual[["equipo", "pj", "puntos", "dg", "gf", "gc"]], hide_index=False, width='stretch')

        with col_t2:
            st.markdown("#### Proyección Monte Carlo (Liga + Play-offs + Cuadro)")
            df_mc = simular_ucl(M, "sim_key_ucl", modelo_tipo)
            df_mc_disp = df_mc[["equipo", "Puntos esperados", "P_campeon", "P_final", "P_top8", "P_playoffs", "P_eliminado"]].copy()

            st.dataframe(
                df_mc_disp.style.format({
                    "Puntos esperados": "{:.1f}",
                    "P_campeon": "{:.1%}",
                    "P_final": "{:.1%}",
                    "P_top8": "{:.1%}",
                    "P_playoffs": "{:.1%}",
                    "P_eliminado": "{:.1%}"
                }).background_gradient(subset=["P_campeon"], cmap="YlOrRd")
                  .background_gradient(subset=["P_top8"], cmap="Greens")
                  .background_gradient(subset=["P_eliminado"], cmap="Blues"),
                hide_index=True, width='stretch'
            )

        st.info("ℹ️ **Nuevo Formato UEFA Champions League**: Los 36 clubes disputan 8 jornadas en una tabla única (4 cotejos de local y 4 de visitante). "
                "Los puestos **1º al 8º** clasifican directamente a los **Octavos de Final**. Los puestos **9º al 24º** disputan la ronda de **Knockout Play-offs** a ida y vuelta para definir a los otros 8 clasificados. "
                "Los clasificados del **25º al 36º** quedan eliminados definitivamente sin paso a Europa League.")

    # -------------------------------------------------------------
    # TAB 3: IMPORTANCIA DE VARIABLES
    # -------------------------------------------------------------
    with tab3:
        st.markdown('<div class="sec-title">🧠 Arquitectura Analítica e Importancia de Variables</div>', unsafe_allow_html=True)

        met = M.get("metricas", {})
        if met:
            st.markdown("#### Rendimiento Out-of-Sample (Test $\ge$ 2026)")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("LASSO (L1)", f"Acc: {met['lasso']['accuracy']}%", f"LogLoss: {met['lasso']['logloss']}")
            col_m2.metric("Random Forest", f"Acc: {met['rf']['accuracy']}%", f"LogLoss: {met['rf']['logloss']}")
            stk = met.get("stacking", {})
            col_m3.metric("Stacking Óptimo", f"Acc: {stk.get('accuracy', 0)}%", f"LogLoss: {stk.get('logloss', 0)}")
            if "w" in stk:
                st.caption(f"Pesos de ensemble: LASSO={stk['w'][0]} | RF={stk['w'][1]} | XGB={stk['w'][2]}")

        rf_step = M["pipe_rf"].named_steps.get("rf")
        feats_list = M["features"]
        if rf_step and hasattr(rf_step, "feature_importances_"):
            importances = rf_step.feature_importances_
            df_imp = pd.DataFrame({"Variable": feats_list, "Importancia": importances}).sort_values("Importancia", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(df_imp["Variable"][::-1], df_imp["Importancia"][::-1], color="#001438")
            ax.set_title("Top 15 Variables Predictivas Más Influyentes (UEFA Champions League)")
            ax.set_xlabel("Importancia Relativa")
            plt.tight_layout()
            st.pyplot(fig)

    # -------------------------------------------------------------
    # TAB 4: VALIDACIÓN EN VIVO
    # -------------------------------------------------------------
    with tab4:
        st.markdown('<div class="sec-title">📈 Backtesting y Validación en Vivo</div>', unsafe_allow_html=True)
        st.caption("Evaluación de calibración y acierto match por match en cotejos disputados de la temporada actual.")

        df_val, met_val, df_evol = mo.validacion_en_vivo(M, modelo_tipo=modelo_tipo)
        if df_val.empty:
            st.warning("No hay suficientes partidos jugados en la temporada actual para validar.")
        else:
            c_v1, c_v2, c_v3 = st.columns(3)
            c_v1.metric("Partidos Evaluados", f"{met_val['n']}")
            c_v2.metric("Tasa de Acierto (Accuracy)", f"{met_val['acierto']:.1%}")
            c_v3.metric("Log-Loss del Modelo", f"{met_val['logloss']:.3f}", f"Baseline: {met_val['logloss_base']:.3f}", delta_color="inverse")

            if not df_evol.empty:
                st.markdown("#### Evolución de Rendimiento Acumulado")
                st.line_chart(df_evol.set_index("partido_n")[["acierto_acumulado"]])

            with st.expander("📋 Ver detalle de predicciones partido a partido"):
                df_val_show = df_val[["fecha", "local", "visita", "goles_local", "goles_visita", "Prob_Local", "Prob_Empate", "Prob_Visita", "Prediccion"]].copy()
                map_res = {0: "Local", 1: "Empate", 2: "Visita"}
                df_val_show["Resultado Real"] = df_val_show.apply(
                    lambda r: "Local" if r["goles_local"] > r["goles_visita"] else ("Empate" if r["goles_local"] == r["goles_visita"] else "Visita"), axis=1
                )
                df_val_show["Predicción"] = df_val_show["Prediccion"].map(map_res)
                df_val_show["Acierto"] = df_val_show["Resultado Real"] == df_val_show["Predicción"]

                st.dataframe(
                    df_val_show[["fecha", "local", "visita", "goles_local", "goles_visita", "Resultado Real", "Predicción", "Acierto", "Prob_Local", "Prob_Empate", "Prob_Visita"]].style.format({
                        "Prob_Local": "{:.1%}",
                        "Prob_Empate": "{:.1%}",
                        "Prob_Visita": "{:.1%}"
                    }),
                    hide_index=True, width='stretch'
                )


if __name__ == "__main__":
    run_app()
