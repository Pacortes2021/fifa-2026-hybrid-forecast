"""
Aplicación Streamlit para la simulación y predicciones de la Primera División de Chile.
Visualizaciones premium de Versus, Mercados, Tabla de Posiciones y Proyecciones de Copas y Descenso.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Insertar el directorio chile en el path para asegurar la importación del motor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor as mo
import recolectar as rec
import recolectar_boxscore as rec_box

st.set_page_config(
    page_title="Simulador Liga Chilena 2026",
    page_icon="🇨🇱",
    layout="wide"
)

# Estilizado CSS Premium
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main-title { text-align:center; font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg,#0039a6,#d52b1e); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#0039a6;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#0039a6; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Emojis e identidades visuales de los clubes chilenos
TEAM_DETAILS = {
    "Colo Colo": {"flag": "⚪⚫", "color": "#000000"},
    "Universidad Católica": {"flag": "🔵⚪", "color": "#0039a6"},
    "Universidad de Chile": {"flag": "🔵", "color": "#00205b"},
    "Unión La Calera": {"flag": "🔴", "color": "#dd0000"},
    "Unión Española": {"flag": "🔴🟡", "color": "#ffd700"},
    "Everton CD": {"flag": "🟡🔵", "color": "#002d62"},
    "Audax Italiano": {"flag": "🟢", "color": "#008a4a"},
    "Palestino": {"flag": "🇵🇸", "color": "#007a33"},
    "O'Higgins": {"flag": "🔵", "color": "#0099ff"},
    "Huachipato": {"flag": "🔵⚫", "color": "#002f6c"},
    "Cobresal": {"flag": "🟠", "color": "#ff8c00"},
    "Ñublense": {"flag": "🔴", "color": "#d30a14"},
    "Coquimbo Unido": {"flag": "🟡⚫", "color": "#ffcc00"},
    "Deportes Concepcion": {"flag": "🟣", "color": "#6f2da8"},
    "Deportes Limache": {"flag": "🍅", "color": "#ff6347"},
    "Universidad de Concepción": {"flag": "🟡🔵", "color": "#ffd700"},
    "La Serena": {"flag": "🔴", "color": "#dd0000"},
    "Antofagasta": {"flag": "🔵", "color": "#0056a3"},
    "Curicó Unido": {"flag": "🔴", "color": "#dd0000"},
    "Melipilla": {"flag": "⚪", "color": "#000000"},
    "Santiago Wanderers": {"flag": "🟢", "color": "#008a4a"},
    "Copiapó": {"flag": "🟢", "color": "#007a33"},
    "Magallanes": {"flag": "🔵⚪", "color": "#0099ff"},
    "Unión Wanderers": {"flag": "🔴", "color": "#dd0000"},
    "Deportes Iquique": {"flag": "🔵", "color": "#00bfff"},
    "Cobreloa": {"flag": "🟠", "color": "#ff8c00"},
    "Ceará": {"flag": "⚫", "color": "#000000"}
}


def get_label(team):
    info = TEAM_DETAILS.get(team, {"flag": "⚽"})
    return f"{info['flag']} {team}"


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Corriendo simulaciones de Monte Carlo (4.000 iteraciones)...")
def simular_campeonato(_M, key):
    return mo.simular_campeonato(_M, n_sims=4000)


# Sidebar de controles
st.sidebar.markdown("### 🛠️ Controles del Modelo")

if st.sidebar.button("🔄 Actualizar ESPN y Re-entrenar", key="refresh_chile", type="primary"):
    with st.spinner("Descargando últimos resultados de la Liga Chilena..."):
        rec.recolectar()
        rec_box.recolectar()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

M = get_motor()

# Encabezado
st.markdown('<div class="main-title">🇨🇱 Portal de Predicción Liga Chilena</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Modelo de Regularización LASSO (L1) con SAGA + Simulación de Campeonato Completo</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚽ Predicción Versus", "📊 Tabla y Proyecciones", "🔬 Importancia de Variables", "🎯 Validación vs Realidad"])

# ============================================================================
# TAB 1: Predicción Versus
# ============================================================================
with tab1:
    st.markdown('<div class="sec-title">Analizador de Enfrentamientos</div>', unsafe_allow_html=True)
    
    opciones = sorted(list(TEAM_DETAILS.keys()))
    partidos_rec = M["partidos"]
    p_actuales = partidos_rec[partidos_rec.temporada == 2026]
    fix_rec = pd.read_csv(mo.DATA / "fixture.csv") if (mo.DATA / "fixture.csv").exists() else pd.DataFrame()
    equipos_activos = sorted(list(set(p_actuales["local"]).union(set(p_actuales["visita"])).union(set(fix_rec["local"] if not fix_rec.empty else []))))
    if not equipos_activos:
        equipos_activos = opciones
        
    c1, cvs, c2 = st.columns([5, 1, 5])
    with c1:
        a = st.selectbox("Equipo Local", equipos_activos, index=equipos_activos.index("Colo Colo") if "Colo Colo" in equipos_activos else 0, key="sel_a")
    with cvs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with c2:
        b = st.selectbox("Equipo Visitante", equipos_activos, index=equipos_activos.index("Universidad de Chile") if "Universidad de Chile" in equipos_activos else 0, key="sel_b")
        
    if a == b:
        st.error("Selecciona dos equipos distintos.")
    else:
        # Calcular grilla de goles
        mix, p, (la, lb) = mo.grilla_goles(M, a, b)
        
        la_lbl = get_label(a)
        lb_lbl = get_label(b)
        
        # Alertas de Heurísticas de Alta Efectividad
        if p[0] > 0.55 and (la - lb) > 1.0:
            st.success(f"🔥 **ALERTA DE ALTA CONFIANZA (Acierto 82.6% Histórico):** Consenso perfecto entre Machine Learning (>55%) y modelo Poisson (>1.0 goles dif) a favor de victoria de **{a}**.")
        elif p[0] > 0.60:
            st.info(f"💪 **FAVORITO CLARO (Acierto 71% Histórico):** El modelo de Machine Learning asigna más del 60% de probabilidad de victoria a **{a}**.")
        elif p[2] > 0.50:
            st.success(f"⚠️ **VISITA FUERTE (Acierto 67% Histórico):** Probabilidad >50% para el equipo visitante (**{b}**). Es un evento raro y suele ser muy rentable.")

        # Mostrar Probabilidades
        col_probs, col_stats = st.columns(2)
        
        with col_probs:
            st.markdown(f'<div class="card-title">Probabilidades de Victoria</div>', unsafe_allow_html=True)
            for label, prob in [(f"Victoria {a}", p[0]), ("Empate", p[1]), (f"Victoria {b}", p[2])]:
                st.markdown(f"**{label}: {prob:.1%}** (Cuota Justa: `{mo.cuota(prob):.2f}`)")
                st.progress(float(prob))
                
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
        st.markdown('<div class="sec-title">Mercados de Goles y Hándicaps</div>', unsafe_allow_html=True)
        filas = []
        for ln in (1.5, 2.5, 3.5):
            for lado in ("Over", "Under"):
                pr = mk[f"{lado} {ln}"]
                filas.append({"Mercado": f"{lado} {ln} goles", "Prob.": f"{pr:.1%}", "Cuota justa": f"{mo.cuota(pr):.2f}"})
        for et, key in (("Ambos marcan: Sí", "Ambos marcan (BTTS sí)"), ("Ambos marcan: No", "BTTS no")):
            filas.append({"Mercado": et, "Prob.": f"{mk[key]:.1%}", "Cuota justa": f"{mo.cuota(mix.ravel()[0]):.2f}" if "no" in key else f"{mo.cuota(mk[key]):.2f}"})
            
        mc1, mc2 = st.columns(2)
        mc1.dataframe(pd.DataFrame(filas[:4]), hide_index=True, width='stretch')
        mc2.dataframe(pd.DataFrame(filas[4:]), hide_index=True, width='stretch')
        
        # Gráfica de la matriz Dixon-Coles
        st.markdown('<div class="sec-title">Matriz de Goles Exactos (Dixon-Coles)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        m6 = mix[:6, :6]
        im = ax.imshow(m6, cmap="YlGn")
        ax.set_xticks(range(6)); ax.set_xticklabels(range(6))
        ax.set_yticks(range(6)); ax.set_yticklabels(range(6))
        ax.set_xlabel(f"Goles de {b}"); ax.set_ylabel(f"Goles de {a}")
        fig.colorbar(im, ax=ax, label="Probabilidad")
        
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
    df_actual_vis["Equipo"] = df_actual_vis["Selección"].apply(get_label)
    df_actual_vis = df_actual_vis[["Equipo", "PTS", "DG", "PG", "PE", "PP", "GF"]]
    
    with col_act:
        st.markdown('<div class="card-title">Tabla de Posiciones Actual (Real)</div>', unsafe_allow_html=True)
        st.dataframe(df_actual_vis, hide_index=True, width='stretch', height=500)
        
    # 2. Proyecciones
    last_key = f"{len(partidos_rec)}-{partidos_rec.fecha.max()}"
    df_proy = simular_campeonato(M, last_key)
    
    df_proy_visual = df_proy.copy()
    df_proy_visual["Equipo"] = df_proy_visual["Selección"].apply(get_label)
    df_proy_visual = df_proy_visual[[
        "Equipo", "Puntos esperados", "P_campeon", "P_libertadores_directo", 
        "P_libertadores_total", "P_sudamericana", "P_descenso"
    ]]
    df_proy_visual = df_proy_visual.rename(columns={
        "Puntos esperados": "Pts Esperados",
        "P_campeon": "🏆 Campeón",
        "P_libertadores_directo": "Libertadores (G.G.)",
        "P_libertadores_total": "Libertadores (Total)",
        "P_sudamericana": "Sudamericana",
        "P_descenso": "Descenso ⬇️"
    })
    
    with col_proj:
        st.markdown('<div class="card-title">Proyecciones de Fin de Temporada (Monte Carlo)</div>', unsafe_allow_html=True)
        st.dataframe(
            df_proy_visual.style.format({
                "Pts Esperados": "{:.1f}",
                "🏆 Campeón": "{:.1%}",
                "Libertadores (G.G.)": "{:.1%}",
                "Libertadores (Total)": "{:.1%}",
                "Sudamericana": "{:.1%}",
                "Descenso ⬇️": "{:.1%}"
            }).background_gradient(subset=["🏆 Campeón"], cmap="YlGn")
              .background_gradient(subset=["Descenso ⬇️"], cmap="OrRd"),
            hide_index=True,
            width='stretch',
            height=500
        )

# ============================================================================
# TAB 3: Importancia de Variables
# ============================================================================
with tab3:
    st.markdown('<div class="sec-title">Explicabilidad del Modelo LASSO (L1 - SAGA)</div>', unsafe_allow_html=True)
    st.markdown("La regularización **LASSO** penaliza coeficientes nulos, identificando las características con mayor peso predictivo para el fútbol chileno.")
    
    importancia = []
    pipe = M["pipe"]
    lr = pipe.named_steps["lr"]
    coefs = lr.coef_
    avg_coef = np.mean(np.abs(coefs), axis=0)
    
    for feat, val in zip(M["features"], avg_coef):
        if val > 1e-4:
            importancia.append({"Variable": feat, "Peso Absoluto Promedio": round(val, 4)})
            
    df_imp = pd.DataFrame(importancia).sort_values(by="Peso Absoluto Promedio", ascending=False).reset_index(drop=True)
    
    col_t, col_g = st.columns([5, 7])
    with col_t:
        st.dataframe(df_imp, hide_index=True, width='stretch')
        
    with col_g:
        fig, ax = plt.subplots(figsize=(6, 5))
        top_n = df_imp.head(15)
        ax.barh(top_n["Variable"][::-1], top_n["Peso Absoluto Promedio"][::-1], color="#0039a6")
        ax.set_title("Top 15 Características Predictoras (Chile)")
        st.pyplot(fig)

# ============================================================================
# TAB 4: Validación vs Realidad
# ============================================================================
with tab4:
    st.markdown('<div class="sec-title">El Modelo contra la Realidad (Out-of-sample)</div>', unsafe_allow_html=True)
    st.markdown("Comparación de la predicción del modelo contra el resultado real para los partidos de la liga chilena.")
    
    temporadas_disponibles = sorted(M["df_dataset"]["temporada"].unique(), reverse=True)
    temporada_sel = st.selectbox("Selecciona la temporada a validar:", temporadas_disponibles, index=0)
    
    df_val, met, evol = mo.validacion_en_vivo(M, temporada_val=temporada_sel)
    
    if df_val is None or len(df_val) == 0:
        st.info("Aún no hay partidos finalizados en la temporada seleccionada para validar.")
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
            st.warning(f"⚠️ El modelo va por debajo del baseline.")
            
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
                ax.plot(evol["partido"], evol["logloss_acum"], "o-", color="#0039a6", label="Modelo")
                ax.axhline(met["logloss_base"], color="#dc2626", ls="--", label="Baseline")
                ax.set_xlabel("Partidos jugados (cronológico)")
                ax.set_ylabel("Log-loss acumulado")
                ax.legend()
                st.pyplot(fig)
                
        with c_table:
            st.markdown("##### % Acierto por Equipo")
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
                    df_teams.style.format({"% Acierto": "{:.1%}"}).background_gradient(subset=["% Acierto"], cmap="YlGn"),
                    hide_index=True, width='stretch'
                )
