"""
Aplicación Streamlit para la simulación y predicciones de LaLiga (España).
Visualizaciones premium de Versus, Mercados, Tabla de Posiciones y Proyecciones de Monte Carlo.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Insertar el directorio esp en el path para asegurar la importación del motor
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
    background:linear-gradient(135deg,#d90429,#2b2d42); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#d90429;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#d90429; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Emojis e identidades visuales de los clubes españoles
TEAM_DETAILS = {
    "Real Madrid": {"flag": "👑", "color": "#00529f"},
    "Barcelona": {"flag": "🔵🔴", "color": "#a50044"},
    "Atlético Madrid": {"flag": "🔴⚪", "color": "#cb3524"},
    "Real Sociedad": {"flag": "⚪🔵", "color": "#0000ff"},
    "Athletic Club": {"flag": "🦁", "color": "#ee1c25"},
    "Real Betis": {"flag": "🟢⚪", "color": "#009639"},
    "Villarreal": {"flag": "🟡", "color": "#ffec00"},
    "Girona": {"flag": "🔴🟡", "color": "#d91c1c"},
    "Sevilla": {"flag": "⚪🔴", "color": "#c41c1c"},
    "Valencia": {"flag": "🦇", "color": "#ff7c00"},
    "Celta Vigo": {"flag": "🩵", "color": "#008ac9"},
    "Osasuna": {"flag": "🐮", "color": "#e30613"},
    "Getafe": {"flag": "🔵", "color": "#0000ee"},
    "Mallorca": {"flag": "👹", "color": "#e20613"},
    "Las Palmas": {"flag": "🌴", "color": "#ffe800"},
    "Rayo Vallecano": {"flag": "⚡", "color": "#ff0000"},
    "Alavés": {"flag": "🦊", "color": "#0055a5"},
    "Leganés": {"flag": "🥒", "color": "#005da4"},
    "Valladolid": {"flag": "🟣⚪", "color": "#6c207e"},
    "Espanyol": {"flag": "🦜", "color": "#008ac9"},
    "Granada": {"flag": "🦁", "color": "#c8102e"},
    "Cádiz": {"flag": "🟡🔵", "color": "#ffcc00"},
    "Almería": {"flag": "🔴⚪", "color": "#e03a3e"},
    "Elche": {"flag": "🌴", "color": "#007d4c"},
    "Levante": {"flag": "🐸", "color": "#004d9c"},
    "Eibar": {"flag": "🔵🔴", "color": "#a50044"},
    "Huesca": {"flag": "🔵🔴", "color": "#a50044"}
}


def get_label(team):
    info = TEAM_DETAILS.get(team, {"flag": "⚽"})
    flag = info.get("flag_emoji", info.get("flag", "⚽"))
    return f"{flag} {team}"


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Corriendo simulaciones de Monte Carlo (3.000 iteraciones)...")
def simular_liga(_M, key, modelo_tipo):
    return mo.simular_campeonato(_M, n_sims=3000, modelo_tipo=modelo_tipo)


def run_app():
    st.sidebar.markdown("### 🛠️ Controles del Modelo")
    
    # Selector de modelo activo
    modelo_sel = st.sidebar.selectbox(
        "🤖 Modelo Predictivo:",
        ["Random Forest (Recomendado)", "Regresión Logística (LASSO L1)"],
        index=0
    )
    modelo_tipo = "rf" if "Random Forest" in modelo_sel else "lasso"
    
    # Botón para actualizar resultados en vivo
    if st.sidebar.button("🔄 Actualizar ESPN y Re-entrenar", type="primary"):
        with st.spinner("Descargando últimos resultados de ESPN..."):
            rec.recolectar()
            rec_box.recolectar()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    M = get_motor()
    
    # Encabezado
    st.markdown('<div class="main-title">🇪🇸 Portal de Predicción LaLiga</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Modelo de Regularización LASSO (L1) + Simulación de Campeón, Europa y Descenso</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚽ Predicción Versus", "📊 Tabla y Proyecciones", "🔬 Importancia de Variables", "🎯 Validación vs Realidad"])
    
    # ============================================================================
    # TAB 1: Predicción Versus
    # ============================================================================
    with tab1:
        st.markdown('<div class="sec-title">Analizador de Enfrentamientos</div>', unsafe_allow_html=True)
        
        # Filtrar opciones basadas en los equipos que existen en partidos.csv de 2026 o en el dataset
        opciones = sorted(list(M["df_features"]["local"].unique()))
        
        c1, cvs, c2 = st.columns([5, 1, 5])
        with c1:
            a = st.selectbox("Equipo Local", opciones, index=opciones.index("Real Madrid") if "Real Madrid" in opciones else 0, key="sel_a")
        with cvs:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        with c2:
            b = st.selectbox("Equipo Visitante", opciones, index=opciones.index("Barcelona") if "Barcelona" in opciones else 0, key="sel_b")
            
        if a == b:
            st.error("Selecciona dos equipos distintos.")
        else:
            # Calcular predicción 1X2 y marcadores
            p = mo.predecir_match(M, a, b, modelo_tipo=modelo_tipo)
            mix = mo.grilla_goles(M, a, b, modelo_tipo=modelo_tipo)
            
            tracker = M["tracker"]
            elo_diff = tracker.elos[a] - tracker.elos[b]
            la = np.exp(M["g_const"] + M["g_d"] * elo_diff)
            lb = np.exp(M["g_const"] - M["g_d"] * elo_diff)
            
            la_lbl = get_label(a)
            lb_lbl = get_label(b)
            
            # Alertas de Heurísticas de Alta Efectividad
            if p[0] > 0.55 and (la - lb) > 1.0:
                st.success(f"🔥 **ALERTA DE ALTA CONFIANZA (Acierto 81.4% Histórico):** Consenso perfecto entre Machine Learning (>55%) y modelo Poisson (>1.0 goles dif) a favor de victoria de **{a}**.")
            elif p[0] > 0.60:
                st.info(f"💪 **FAVORITO CLARO (Acierto 73% Histórico):** El modelo de Machine Learning asigna más del 60% de probabilidad de victoria a **{a}**.")
            elif p[2] > 0.50:
                st.success(f"⚠️ **VISITA FUERTE (Acierto 68% Histórico):** Probabilidad >50% para el equipo visitante (**{b}**). Este tipo de predicciones en España son muy rentables.")
            
            # Mostrar Probabilidades
            col_probs, col_stats = st.columns(2)
            
            with col_probs:
                st.markdown(f'<div class="card-title">Probabilidades de Victoria</div>', unsafe_allow_html=True)
                
                # Barras de progreso de probabilidad
                for label, prob, col in [(f"Victoria {a}", p[0], "#d90429"), ("Empate", p[1], "#64748b"), (f"Victoria {b}", p[2], "#2b2d42")]:
                    st.markdown(f"**{label}: {prob:.1%}** (Cuota Justa: `{mo.cuota(prob):.2f}`)")
                    st.progress(float(prob))
                    
                st.markdown("---")
                p_avanza_a = p[0] + p[1] * 0.5  # aproximación rápida
                st.caption(f"Expectativa de clasificación en eliminación directa neutral: **{a} {p_avanza_a:.1%}** / {b} {1-p_avanza_a:.1%}")
                
            with col_stats:
                st.markdown(f'<div class="card-title">Goles Esperados y Marcadores</div>', unsafe_allow_html=True)
                st.markdown(f"📈 **Goles esperados (Poisson):**")
                st.markdown(f"*   {la_lbl}: `{la:.2f}` goles")
                st.markdown(f"*   {lb_lbl}: `{lb:.2f}` goles")
                
                st.markdown("🎯 **Marcadores más probables:**")
                mk = mo.mercados(mix)
                for g1, g2, pr in mk["_top_marcadores"][:4]:
                    st.markdown(f"*   `{g1} - {g2}`: **{pr:.1%}** (Cuota: `{mo.cuota(pr):.1f}`)")
                    
            # Mercados de apuestas
            st.markdown('<div class="sec-title">Mercados de Goles y Apuestas Especiales</div>', unsafe_allow_html=True)
            filas = []
            for ln in (1.5, 2.5, 3.5):
                for lado in ("Over", "Under"):
                    pr = mk[f"{lado} {ln}"]
                    filas.append({"Mercado": f"{lado} {ln} goles", "Prob.": f"{pr:.1%}", "Cuota justa": f"{mo.cuota(pr):.2f}"})
            for et, key in (("Ambos marcan: Sí", "Ambos marcan (BTTS sí)"), ("Ambos marcan: No", "BTTS no")):
                filas.append({"Mercado": et, "Prob.": f"{mk[key]:.1%}", "Cuota justa": f"{mo.cuota(mix.ravel()[0]):.2f}" if "no" in key else f"{mo.cuota(mk[key]):.2f}"})
                
            # Doble Oportunidad
            p_1x = p[0] + p[1]
            p_x2 = p[2] + p[1]
            p_12 = p[0] + p[2]
            filas.append({"Mercado": "Doble Oportunidad: Local o Empate (1X)", "Prob.": f"{p_1x:.1%}", "Cuota justa": f"{mo.cuota(p_1x):.2f}"})
            filas.append({"Mercado": "Doble Oportunidad: Visita o Empate (X2)", "Prob.": f"{p_x2:.1%}", "Cuota justa": f"{mo.cuota(p_x2):.2f}"})
            filas.append({"Mercado": "Doble Oportunidad: Local o Visita (12)", "Prob.": f"{p_12:.1%}", "Cuota justa": f"{mo.cuota(p_12):.2f}"})
            
            # Sin Empate (DNB)
            denom = p[0] + p[2]
            p_dnb1 = p[0] / denom if denom > 0 else 0.5
            p_dnb2 = p[2] / denom if denom > 0 else 0.5
            filas.append({"Mercado": f"Sin Empate: {a} (DNB 1)", "Prob.": f"{p_dnb1:.1%}", "Cuota justa": f"{mo.cuota(p_dnb1):.2f}"})
            filas.append({"Mercado": f"Sin Empate: {b} (DNB 2)", "Prob.": f"{p_dnb2:.1%}", "Cuota justa": f"{mo.cuota(p_dnb2):.2f}"})
                
            mc1, mc2 = st.columns(2)
            mc1.dataframe(pd.DataFrame(filas[:7]), hide_index=True, width='stretch')
            mc2.dataframe(pd.DataFrame(filas[7:]), hide_index=True, width='stretch')
            
            # Gráfica de la matriz Dixon-Coles
            st.markdown('<div class="sec-title">Matriz de Goles Exactos (Dixon-Coles)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            m6 = mix[:6, :6]
            im = ax.imshow(m6, cmap="Reds")
            ax.set_xticks(range(6)); ax.set_xticklabels(range(6))
            ax.set_yticks(range(6)); ax.set_yticklabels(range(6))
            ax.set_xlabel(f"Goles de {b}"); ax.set_ylabel(f"Goles de {a}")
            fig.colorbar(im, ax=ax, label="Probabilidad")
            
            imax, jmax = np.unravel_index(m6.argmax(), m6.shape)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{m6[i,j]:.1%}", ha="center", va="center", color="white" if m6[i,j] > m6.max()*0.6 else "black", fontsize=8)
            st.pyplot(fig)
            
    # ============================================================================
    # TAB 2: Tabla y Proyecciones de Monte Carlo
    # ============================================================================
    with tab2:
        st.markdown('<div class="sec-title">Tabla General y Proyecciones Monte Carlo</div>', unsafe_allow_html=True)
        
        col_act, col_proj = st.columns(2)
        
        # 1. Tabla Actual
        df_actual = mo.obtener_tabla_actual(M)
        df_actual_vis = df_actual.copy()
        df_actual_vis["Equipo"] = df_actual_vis["equipo"].apply(get_label)
        df_actual_vis = df_actual_vis[["Equipo", "pj", "puntos", "dif_goles", "goles_favor"]]
        df_actual_vis = df_actual_vis.rename(columns={"pj": "PJ", "puntos": "PTS", "dif_goles": "DG", "goles_favor": "GF"})
        
        with col_act:
            st.markdown('<div class="card-title">Tabla de Posiciones Actual (Real)</div>', unsafe_allow_html=True)
            st.dataframe(df_actual_vis, hide_index=True, width='stretch', height=500)
            
        # 2. Proyecciones
        partidos_rec = pd.read_csv(mo.DATA / "partidos.csv")
        last_key = f"{len(partidos_rec)}-{partidos_rec.fecha.max()}"
        df_proy = simular_liga(M, last_key, modelo_tipo)
        
        df_proy_visual = df_proy.copy()
        df_proy_visual["Equipo"] = df_proy_visual["equipo"].apply(get_label)
        df_proy_visual = df_proy_visual[["Equipo", "P_campeon", "P_copas", "P_descenso"]]
        df_proy_visual = df_proy_visual.rename(columns={
            "P_campeon": "🏆 P(Campeón)",
            "P_copas": "🇪🇺 P(Copas)",
            "P_descenso": "🔻 P(Descenso)"
        })
        
        with col_proj:
            st.markdown('<div class="card-title">Proyecciones de Fin de Temporada (Monte Carlo)</div>', unsafe_allow_html=True)
            st.dataframe(
                df_proy_visual.style.format({
                    "🏆 P(Campeón)": "{:.1%}",
                    "🇪🇺 P(Copas)": "{:.1%}",
                    "🔻 P(Descenso)": "{:.1%}"
                }).background_gradient(subset=["🏆 P(Campeón)"], cmap="Reds")
                  .background_gradient(subset=["🔻 P(Descenso)"], cmap="OrRd"),
                hide_index=True,
                width='stretch',
                height=500
            )
            
    # ============================================================================
    # TAB 3: Importancia de Variables
    # ============================================================================
    with tab3:
        if modelo_tipo == "lasso":
            st.markdown('<div class="sec-title">Explicabilidad del Modelo LASSO (L1)</div>', unsafe_allow_html=True)
            st.markdown("La regularización **LASSO (L1)** penaliza los coeficientes de las variables redundantes o no informativas hasta reducirlas exactamente a cero, dejando solo los predictores de mayor peso out-of-sample.")
            
            # Mostrar la lista de variables seleccionadas
            importancia = []
            pipe = M["pipe_lasso"]
            lr = pipe.named_steps["lr"]
            coefs = lr.coef_
            avg_coef = np.mean(np.abs(coefs), axis=0)
            
            for feat, val in zip(M["cols"], avg_coef):
                if val > 1e-4:
                    importancia.append({"Variable": feat, "Peso Absoluto Promedio": round(val, 4)})
            col_val_name = "Peso Absoluto Promedio"
            title_graph = "Top 15 Características Predictoras (LASSO L1)"
        else:
            st.markdown('<div class="sec-title">Importancia de Características: Random Forest</div>', unsafe_allow_html=True)
            st.markdown("La importancia de características en **Random Forest** se calcula a partir de la reducción promedio de la impureza de Gini que aporta cada variable al realizar las divisiones tácticas en el ensamble de árboles.")
            
            importancia = []
            pipe = M["pipe_rf"]
            rf = pipe.named_steps["rf"]
            importances = rf.feature_importances_
            
            for feat, val in zip(M["cols"], importances):
                importancia.append({"Variable": feat, "Importancia (Gini)": round(val, 4)})
            col_val_name = "Importancia (Gini)"
            title_graph = "Top 15 Características Predictoras (Random Forest Gini)"
            
        df_imp = pd.DataFrame(importancia).sort_values(by=col_val_name, ascending=False).reset_index(drop=True)
        
        col_t, col_g = st.columns([5, 7])
        with col_t:
            st.dataframe(df_imp, hide_index=True, width='stretch')
            
        with col_g:
            fig, ax = plt.subplots(figsize=(6, 5))
            top_n = df_imp.head(15)
            ax.barh(top_n["Variable"][::-1], top_n[col_val_name][::-1], color="#d90429")
            ax.set_title(title_graph)
            st.pyplot(fig)
            
    # ============================================================================
    # TAB 4: Validación vs Realidad
    # ============================================================================
    with tab4:
        st.markdown('<div class="sec-title">El Modelo contra la Realidad (Out-of-sample)</div>', unsafe_allow_html=True)
        st.markdown("Comparación de la predicción **pre-partido** del modelo contra el **resultado real** para los partidos ya jugados.")
        
        # Seleccion de temporada
        temporadas_disponibles = sorted(M["df_features"]["temporada"].unique(), reverse=True)
        temporada_sel = st.selectbox("Selecciona la temporada a validar:", temporadas_disponibles, index=0)
        
        df_val, met, evol = mo.validacion_en_vivo(M, temporada_val=temporada_sel, modelo_tipo=modelo_tipo)
        
        if df_val is None or len(df_val) == 0:
            st.info("Aún no hay partidos finalizados en la temporada para validar.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Partidos Jugados", met["n"])
            m2.metric("Acierto (1X2)", f"{met['acierto']:.1%}")
            m3.metric("Log-loss modelo", f"{met['logloss']:.3f}", 
                      f"{met['logloss'] - met['logloss_base']:+.3f} vs baseline", delta_color="inverse")
            m4.metric("Log-loss baseline", f"{met['logloss_base']:.3f}")
            
            if met["logloss"] < met["logloss_base"]:
                st.success(f"El modelo va **por encima** del baseline en {met['n']} partidos reales de esta temporada. 👍")
            else:
                st.warning(f"⚠️ El modelo va por debajo del baseline, pero puede ser por la baja cantidad de partidos.")
                
            # Tabla detallada
            st.markdown("##### Historial de Predicciones")
            df_show = df_val[["fecha", "local", "visita", "goles_local", "goles_visita", "resultado", "Prediccion", "Prob_Local", "Prob_Empate", "Prob_Visita"]].copy()
            df_show["Acierto"] = (df_show["resultado"] == df_show["Prediccion"]).replace({True: "✅", False: "❌"})
            df_show["Prob_Local"] = df_show["Prob_Local"].apply(lambda x: f"{x:.1%}")
            df_show["Prob_Empate"] = df_show["Prob_Empate"].apply(lambda x: f"{x:.1%}")
            df_show["Prob_Visita"] = df_show["Prob_Visita"].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(df_show, hide_index=True, width='stretch')
            
            c_plot, c_table = st.columns([6, 4])
            with c_plot:
                if met["n"] >= 3:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(evol["partido"], evol["logloss_acum"], "o-", color="#d90429", label="Modelo (acumulado)")
                    ax.axhline(met["logloss_base"], color="#2b2d42", ls="--", lw=1.5, label="Baseline")
                    ax.set_xlabel("Partidos jugados (cronológico)")
                    ax.set_ylabel("Log-loss acumulado")
                    ax.set_title("Evolución del Log-loss en la Temporada")
                    ax.legend()
                    st.pyplot(fig)
                    
            with c_table:
                st.markdown("##### % Acierto por Equipo")
                # Calcular acierto donde el equipo participó
                team_stats = []
                equipos = set(df_val["local"]).union(set(df_val["visita"]))
                for eq in equipos:
                    df_eq = df_val[(df_val["local"] == eq) | (df_val["visita"] == eq)]
                    if len(df_eq) > 0:
                        aciertos = (df_eq["resultado"] == df_eq["Prediccion"]).sum()
                        team_stats.append({
                            "Equipo": eq,
                            "Partidos": len(df_eq),
                            "Aciertos": aciertos,
                            "% Acierto": aciertos / len(df_eq)
                        })
                if team_stats:
                    df_teams = pd.DataFrame(team_stats).sort_values("% Acierto", ascending=False)
                    st.dataframe(
                        df_teams.style.format({"% Acierto": "{:.1%}"}).background_gradient(subset=["% Acierto"], cmap="OrRd"),
                        hide_index=True, width='stretch'
                    )
