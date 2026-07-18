"""
Aplicación Streamlit para la simulación y predicciones de la Liga MX (México).
Visualizaciones premium de Versus, Mercados, Tabla de Posiciones y Simulación de Liguilla.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Insertar el directorio mex en el path para asegurar la importación del motor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor as mo
import recolectar as rec
import recolectar_boxscore as rec_box

st.set_page_config(
    page_title="Simulador Liga MX 2026",
    page_icon="🏆",
    layout="wide"
)

# Estilizado CSS Premium
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main-title { text-align:center; font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg,#005b41,#008170); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#005b41;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#005b41; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Emojis e identidades visuales de los clubes mexicanos
TEAM_DETAILS = {
    "América": {"flag": "🦅", "color": "#f8e000"},
    "Atlas": {"flag": "🦊", "color": "#d62728"},
    "Atlético de San Luis": {"flag": "🔴", "color": "#e02424"},
    "Cruz Azul": {"flag": "🚂", "color": "#0b3d91"},
    "FC Juarez": {"flag": "🐎", "color": "#3f6212"},
    "Guadalajara": {"flag": "🐐", "color": "#b91c1c"},
    "León": {"flag": "🦁", "color": "#15803d"},
    "Mazatlán FC": {"flag": "⚓", "color": "#6b21a8"},
    "Monterrey": {"flag": "🤠", "color": "#1e3a8a"},
    "Necaxa": {"flag": "⚡", "color": "#e11d48"},
    "Pachuca": {"flag": "🐹", "color": "#0284c7"},
    "Puebla": {"flag": "🎽", "color": "#06b6d4"},
    "Pumas UNAM": {"flag": "🐾", "color": "#ca8a04"},
    "Querétaro": {"flag": " Rooster", "flag_emoji": "🐓", "color": "#2563eb"},
    "Santos": {"flag": "⚔️", "color": "#16a34a"},
    "Tigres UANL": {"flag": "🐯", "color": "#eab308"},
    "Tijuana": {"flag": "🐕", "color": "#b91c1c"},
    "Toluca": {"flag": "😈", "color": "#dc2626"},
    "Atlante": {"flag": "🐎", "color": "#0d9488"}
}


def get_label(team):
    info = TEAM_DETAILS.get(team, {"flag": "⚽"})
    flag = info.get("flag_emoji", info.get("flag", "⚽"))
    return f"{flag} {team}"


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Corriendo simulaciones de Monte Carlo (4.000 iteraciones)...")
def simular_liga(_M, key):
    return mo.monte_carlo(_M, n_sims=4000)


# Sidebar de controles
st.sidebar.markdown("### 🛠️ Controles del Modelo")

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
st.markdown('<div class="main-title">🏆 Portal de Predicción Liga MX</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Modelo de Regularización LASSO (L1) + Simulación de Liguilla y Play-In</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["⚽ Predicción Versus", "📊 Tabla y Proyecciones", "🔬 Importancia de Variables"])

# ============================================================================
# TAB 1: Predicción Versus
# ============================================================================
with tab1:
    st.markdown('<div class="sec-title">Analizador de Enfrentamientos</div>', unsafe_allow_html=True)
    
    opciones = sorted(list(TEAM_DETAILS.keys()))
    
    c1, cvs, c2 = st.columns([5, 1, 5])
    with c1:
        a = st.selectbox("Equipo Local", opciones, index=opciones.index("América"), key="sel_a")
    with cvs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with c2:
        b = st.selectbox("Equipo Visitante", opciones, index=opciones.index("Guadalajara"), key="sel_b")
        
    if a == b:
        st.error("Selecciona dos equipos distintos.")
    else:
        # Calcular grilla de goles
        mix, p, (la, lb) = mo.grilla_goles(M, a, b)
        
        la_lbl = get_label(a)
        lb_lbl = get_label(b)
        
        # Mostrar Probabilidades
        col_probs, col_stats = st.columns(2)
        
        with col_probs:
            st.markdown(f'<div class="card-title">Probabilidades de Victoria</div>', unsafe_allow_html=True)
            
            # Barras de progreso de probabilidad
            for label, prob, col in [(f"Victoria {a}", p[0], "#005b41"), ("Empate", p[1], "#64748b"), (f"Victoria {b}", p[2], "#008170")]:
                st.markdown(f"**{label}: {prob:.1%}** (Cuota Justa: `{mo.cuota(prob):.2f}`)")
                st.progress(float(prob))
                
            st.markdown("---")
            # Avanza en caso de liguilla (Ida y vuelta empatada - avanza mejor posicionado / penales)
            p_avanza_a = p[0] + p[1] * 0.5  # aproximación rápida
            st.caption(f"Expectativa de clasificación en duelo de eliminación directa: **{a} {p_avanza_a:.1%}** / {b} {1-p_avanza_a:.1%}")
            
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
        
        imax, jmax = np.unravel_index(m6.argmax(), m6.shape)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{m6[i,j]:.1%}", ha="center", va="center", color="white" if m6[i,j] > m6.max()*0.6 else "black", fontsize=8)
        st.pyplot(fig)

# ============================================================================
# TAB 2: Tabla y Proyecciones de Monte Carlo
# ============================================================================
with tab2:
    st.markdown('<div class="sec-title">Proyecciones de Fin de Temporada</div>', unsafe_allow_html=True)
    st.markdown("Se simulan **4.000 torneos completos** a partir de la tabla actual, incorporando la liguilla y el Play-In para obtener el porcentaje de campeonar.")
    
    # Obtener última fecha/evento para cache key
    partidos_rec = M["partidos"]
    last_key = f"{len(partidos_rec)}-{partidos_rec.fecha.max()}"
    
    df_proy = simular_liga(M, last_key)
    
    # Emojis en la visualización de tabla
    df_proy_visual = df_proy.copy()
    df_proy_visual["Equipo"] = df_proy_visual["Selección"].apply(get_label)
    df_proy_visual = df_proy_visual.drop(columns=["Selección"])
    
    # Reordenar columnas para una visualización premium
    df_proy_visual = df_proy_visual[["Equipo", "Puntos esperados", "P_directo_QF", "P_Liguilla_total", "P_campeon"]]
    df_proy_visual = df_proy_visual.rename(columns={
        "P_directo_QF": "P(Top 6 Directo)",
        "P_Liguilla_total": "P(Clasificar Liguilla)",
        "P_campeon": "🏆 P(Campeón)"
    })
    
    st.dataframe(
        df_proy_visual.style.format({
            "P(Top 6 Directo)": "{:.1%}",
            "P(Clasificar Liguilla)": "{:.1%}",
            "🏆 P(Campeón)": "{:.1%}"
        }).background_gradient(subset=["🏆 P(Campeón)"], cmap="YlGn"),
        hide_index=True,
        width='stretch'
    )

# ============================================================================
# TAB 3: Importancia de Variables
# ============================================================================
with tab3:
    st.markdown('<div class="sec-title">Explicabilidad del Modelo LASSO (L1)</div>', unsafe_allow_html=True)
    st.markdown("La regularización **LASSO (L1)** penaliza los coeficientes de las variables redundantes o no informativas hasta reducirlas exactamente a cero, dejando solo los predictores de mayor peso out-of-sample.")
    
    # Mostrar la lista de variables seleccionadas
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
        ax.barh(top_n["Variable"][::-1], top_n["Peso Absoluto Promedio"][::-1], color="#008170")
        ax.set_xlabel("Importancia Relativa (L1 Coef)")
        ax.set_title("Top 15 Características Predictoras")
        st.pyplot(fig)
