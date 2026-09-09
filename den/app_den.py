"""
Aplicación Streamlit para la simulación y predicciones de la Danish Superliga (Dinamarca).
Visualizaciones premium de Versus, Mercados, Tabla de Posiciones y Proyecciones de Monte Carlo.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Insertar el directorio den en el path para asegurar la importación del motor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor as mo
import recolectar as rec
import recolectar_boxscore as rec_box

# Estilizado CSS Premium
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main-title { text-align:center; font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg,#C60C30,#8B0000,#1A1A1A); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#C60C30;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#1a1a1a; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Emojis e identidades visuales de los clubes daneses
TEAM_DETAILS = {
    "F.C. København": {"flag": "🔵⚪", "color": "#002f6c"},
    "FC Midtjylland": {"flag": "⚫🔴", "color": "#000000"},
    "Brøndby IF": {"flag": "🟡🔵", "color": "#f8d428"},
    "FC Nordsjælland": {"flag": "🔴🟡", "color": "#d52b1e"},
    "AGF": {"flag": "⚪🔵", "color": "#ffffff"},
    "Silkeborg IF": {"flag": "🔴⚪", "color": "#dc002e"},
    "Randers FC": {"flag": "🔵⚪", "color": "#009fe3"},
    "Viborg FF": {"flag": "🟢⚪", "color": "#00843d"},
    "AaB": {"flag": "🔴⚪", "color": "#c60c30"},
    "Odense Boldklub": {"flag": "🔵⚪", "color": "#003399"},
    "Vejle Boldklub": {"flag": "🔴⚪", "color": "#e30613"},
    "Lyngby Boldklub": {"flag": "🔵⚪", "color": "#174389"},
    "Sønderjyske Fodbold": {"flag": "🔵⚪", "color": "#00457c"},
    "AC Horsens": {"flag": "🟡⚫", "color": "#ffcc00"},
    "Hvidovre IF": {"flag": "🔴🔵", "color": "#c8102e"},
    "FC Fredericia": {"flag": "🔴⚫", "color": "#c8102e"}
}

def logo_url(equipos, team):
    """URL del escudo del club (ID estable de ESPN) o None si no existe."""
    if not equipos:
        return None
    for e in equipos.values():
        if e.get("norm_name") == team and e.get("logo"):
            return e["logo"]
    return None

def logo_html(equipos, team, size=64):
    """Devuelve el <img> del escudo del club desde data/equipos.csv."""
    url = logo_url(equipos, team)
    if not url:
        return ""
    return f'<img src="{url}" width="{size}" style="border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.15);">'

def fmt_opcion(equipos, team):
    """Etiqueta de opción del selectbox: abreviatura del club + nombre."""
    if equipos:
        for e in equipos.values():
            if e.get("norm_name") == team and e.get("abbreviation"):
                return f"{e['abbreviation']} · {team}"
    return team

def label_tabla(equipos, team):
    """Nombre en tablas: emoji solo como fallback si el club no tiene escudo."""
    if logo_url(equipos, team):
        return team
    return get_label(team)

def get_label(team):
    info = TEAM_DETAILS.get(team, {"flag": "⚽"})
    flag = info.get("flag", "⚽")
    return f"{flag} {team}"

@st.cache_resource
def get_motor():
    return mo.cargar()

@st.cache_data(show_spinner="Corriendo simulaciones de Monte Carlo (50.000 iteraciones)...")
def simular_liga(_M, key, modelo_tipo):
    return mo.simular_campeonato(_M, n_sims=50000, modelo_tipo=modelo_tipo)

def run_app():
    st.sidebar.markdown("### 🛠️ Controles del Modelo")

    # Selector de modelo activo
    modelo_sel = st.sidebar.selectbox(
        "🤖 Modelo Predictivo:",
        [
            "🔀 Stacking (Ensemble óptimo)",
            "🌲 Random Forest (Ensemble)",
            "📐 LASSO L1 (Regresión)",
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
        if st.button("Descargar últimos partidos"):
            with st.spinner("Actualizando partidos desde ESPN..."):
                rec.recolectar()
                rec_box.recolectar()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Datos actualizados. Recarga la app.")

    M = get_motor()
    if M is None:
        st.error("No se encontraron datos históricos de la Danish Superliga. Ejecuta la recolección primero.")
        return

    tracker = M["tracker"]
    equipos_data = M.get("equipos", {})

    st.markdown('<div class="main-title">🇩🇰 Danish Superliga Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Predicción Probabilística Híbrida · Poisson Bivariado Dixon-Coles · Machine Learning</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚔️ Match Predictor (Versus)",
        "🏆 Tabla & Proyección Monte Carlo",
        "🧠 Importancia de Variables",
        "📈 Validación en Vivo"
    ])

    # -------------------------------------------------------------
    # TAB 1: MATCH PREDICTOR
    # -------------------------------------------------------------
    with tab1:
        partidos_df = M["df_partidos"]
        equipos_activos = sorted(list(set(partidos_df[partidos_df.temporada == partidos_df.temporada.max()]["local"])))
        if not equipos_activos:
            equipos_activos = sorted(list(tracker.elos.keys()))

        fix_path = mo.DATA / "fixture.csv"
        fixture_disp = pd.DataFrame()
        if fix_path.exists():
            fix_raw = pd.read_csv(fix_path)
            temp_max = partidos_df["temporada"].max()
            if "temporada" in fix_raw.columns:
                fixture_disp = fix_raw[fix_raw.temporada == temp_max].copy()
            else:
                fixture_disp = fix_raw.copy()

        def_local_idx = 0
        def_visita_idx = 1 if len(equipos_activos) > 1 else 0

        # Cargar partido rápido desde Fixture
        if not fixture_disp.empty:
            st.markdown("#### ⚡ Próximos Partidos Oficiales de Danish Superliga")
            opciones_fixture = ["-- Seleccionar del calendario oficial --"] + [
                f"{r.local} vs {r.visita} ({pd.to_datetime(r.fecha).strftime('%d/%m %H:%M') if pd.notna(r.fecha) else 'Fecha TBD'})"
                for _, r in fixture_disp.head(20).iterrows()
            ]
            partido_elegido = st.selectbox("Cargar cotejo programado:", opciones_fixture, index=0)
            if partido_elegido != "-- Seleccionar del calendario oficial --":
                l_nom = partido_elegido.split(" vs ")[0]
                v_nom = partido_elegido.split(" vs ")[1].split(" (")[0]
                if l_nom in equipos_activos: def_local_idx = equipos_activos.index(l_nom)
                if v_nom in equipos_activos: def_visita_idx = equipos_activos.index(v_nom)

        st.markdown('<div class="sec-title">Configuración del Encuentro</div>', unsafe_allow_html=True)
        col_loc, col_mid, col_vis = st.columns([1.2, 0.4, 1.2])

        with col_loc:
            st.markdown('<div class="card-title">🏠 Club Local</div>', unsafe_allow_html=True)
            local = st.selectbox("Selecciona Local:", equipos_activos, index=def_local_idx, format_func=lambda t: fmt_opcion(equipos_data, t), key="sb_loc")
            escudo_l = logo_html(equipos_data, local, size=75)
            if escudo_l: st.markdown(f"<div style='text-align:center;margin-top:5px;'>{escudo_l}</div>", unsafe_allow_html=True)
            elo_l = tracker.elos[local]
            st.metric("ELO Rating", f"{elo_l:.0f} pts")

        with col_mid:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

        with col_vis:
            st.markdown('<div class="card-title">✈️ Club Visitante</div>', unsafe_allow_html=True)
            visita = st.selectbox("Selecciona Visitante:", equipos_activos, index=def_visita_idx, format_func=lambda t: fmt_opcion(equipos_data, t), key="sb_vis")
            escudo_v = logo_html(equipos_data, visita, size=75)
            if escudo_v: st.markdown(f"<div style='text-align:center;margin-top:5px;'>{escudo_v}</div>", unsafe_allow_html=True)
            elo_v = tracker.elos[visita]
            st.metric("ELO Rating", f"{elo_v:.0f} pts")

        if local == visita:
            st.warning("⚠️ Debes seleccionar dos clubes diferentes para evaluar el cotejo.")
        else:
            p, la, lb = mo.predecir_match(M, local, visita, modelo=modelo_tipo)

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

            st.markdown("#### ⚽ Goles Esperados (xG Bivariado Poisson)")
            cx1, cx2 = st.columns(2)
            cx1.metric(f"xG {local}", f"{la:.2f}")
            cx2.metric(f"xG {visita}", f"{lb:.2f}")

            # Cuotas implícitas y mercados
            st.markdown("#### 💰 Mercados de Apuestas Probabilísticas")
            cm1, cm2, cm3, cm4 = st.columns(4)
            p_over25 = 1.0 - sum(mo.matriz_marcador_exacto(la, lb)[i, j] for i in range(7) for j in range(7) if i + j <= 2)
            p_btts = sum(mo.matriz_marcador_exacto(la, lb)[i, j] for i in range(1, 7) for j in range(1, 7))

            cm1.metric("Cuota 1X2 (Local)", f"{1/p[0]:.2f}" if p[0] > 0.01 else ">100")
            cm2.metric("Cuota 1X2 (Empate)", f"{1/p[1]:.2f}" if p[1] > 0.01 else ">100")
            cm3.metric("Más de 2.5 Goles", f"{p_over25:.1%}")
            cm4.metric("Ambos Anotan (BTTS)", f"{p_btts:.1%}")

            # Matriz de calor marcador exacto
            with st.expander("🎲 Matriz de Marcador Exacto (Dixon-Coles)"):
                mat = mo.matriz_marcador_exacto(la, lb, max_goles=5)
                df_mat = pd.DataFrame(mat * 100, index=[f"{local} {i}" for i in range(mat.shape[0])], columns=[f"{visita} {j}" for j in range(mat.shape[1])])
                st.dataframe(df_mat.style.format("{:.1f}%").background_gradient(cmap="Reds"), width='stretch')

            # Variables clave del duelo
            with st.expander("🔍 Métricas y Variables del Enfrentamiento"):
                feats = tracker.get_features_for_match(local, visita, 2026)
                col_f1, col_f2, col_f3 = st.columns(3)
                col_f1.metric("Distancia de Viaje", f"{feats['distance_km']:.0f} km")
                col_f2.metric("Diferencia de Altitud", f"{feats['altitude_diff']:+.0f} m")
                col_f3.metric("Diferencia de Valor de Plantilla", f"{np.exp(feats['squad_value_diff']):.2f}x")

                col_f4, col_f5, col_f6 = st.columns(3)
                col_f4.metric("Diferencial de Forma (5 PJ)", f"{feats['form_diff']:+.2f}")
                col_f5.metric("Diferencial Pi-Rating", f"{feats['pi_diff']:+.2f}")
                col_f6.metric("Historial H2H (Goles Netos)", f"{feats['h2h_diff']:+.2f}")

    # -------------------------------------------------------------
    # TAB 2: TABLA & MONTE CARLO
    # -------------------------------------------------------------
    with tab2:
        st.markdown('<div class="sec-title">🏆 Tabla Real y Proyección Monte Carlo (50.000 Sims)</div>', unsafe_allow_html=True)
        tab_actual = mo.obtener_tabla_actual(M)

        col_t1, col_t2 = st.columns([1, 1.2])
        with col_t1:
            st.markdown("#### Posiciones Actuales (Temporada en Curso)")
            st.dataframe(tab_actual[["equipo", "pj", "puntos", "dg", "gf"]], hide_index=False, width='stretch')

        with col_t2:
            st.markdown("#### Proyección de Final de Temporada (Monte Carlo)")
            df_mc = simular_liga(M, "sim_key_den", modelo_tipo)
            df_mc_disp = df_mc[["equipo", "Puntos esperados", "P_campeon", "P_champions", "P_copas", "P_descenso"]].copy()

            st.dataframe(
                df_mc_disp.style.format({
                    "Puntos esperados": "{:.1f}",
                    "P_campeon": "{:.1%}",
                    "P_champions": "{:.1%}",
                    "P_copas": "{:.1%}",
                    "P_descenso": "{:.1%}"
                }).background_gradient(subset=["P_campeon"], cmap="YlOrRd")
                  .background_gradient(subset=["P_descenso"], cmap="Blues"),
                hide_index=True, width='stretch'
            )

        st.info("ℹ️ **Reglas de Danish Superliga**: Los 6 primeros disputan la Ronda de Campeonato por el título y cupos UEFA (1-2 a Champions League, 3 a Conference League, 4 disputa playoff europeo). Los 6 últimos disputan la Ronda de Relegación, donde los puestos 11º y 12º descienden directo a la 1. Division.")

    # -------------------------------------------------------------
    # TAB 3: IMPORTANCIA DE VARIABLES
    # -------------------------------------------------------------
    with tab3:
        st.markdown('<div class="sec-title">🧠 Arquitectura Analítica e Importancia de Variables</div>', unsafe_allow_html=True)

        met = M.get("metricas", {})
        if met:
            st.markdown("#### Rendimiento Out-of-Sample (Test $\ge$ 2025)")
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
            ax.barh(df_imp["Variable"][::-1], df_imp["Importancia"][::-1], color="#C60C30")
            ax.set_title("Top 15 Variables Predictivas Más Influyentes (Random Forest)")
            ax.set_xlabel("Importancia Relativa")
            plt.tight_layout()
            st.pyplot(fig)

    # -------------------------------------------------------------
    # TAB 4: VALIDACIÓN EN VIVO
    # -------------------------------------------------------------
    with tab4:
        st.markdown('<div class="sec-title">📈 Backtesting y Validación en Vivo</div>', unsafe_allow_html=True)
        st.caption("Evaluación de calibración y acierto match por match en los cotejos disputados en la temporada actual.")

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
                st.line_chart(df_evol.set_index("partidos")[["accuracy"]])

            with st.expander("📋 Ver detalle de predicciones partido a partido"):
                df_val_show = df_val[["fecha", "local", "visita", "goles_local", "goles_visita", "Prob_Local", "Prob_Empate", "Prob_Visita", "Prediccion"]].copy()
                map_res = {0: "Local", 1: "Empate", 2: "Visita"}
                df_val_show["Resultado Real"] = df_val_show.apply(lambda r: "Local" if r["goles_local"] > r["goles_visita"] else ("Empate" if r["goles_local"] == r["goles_visita"] else "Visita"), axis=1)
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
