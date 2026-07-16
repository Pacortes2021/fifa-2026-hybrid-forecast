"""
🇨🇱 Predictor — Primera División de Chile 2026
App Streamlit: tabla y proyección del campeonato (¿quién sale campeón? ¿quién desciende?),
predictor de partidos con mercados, y validación del modelo. Datos: API pública de ESPN.

Correr local:  streamlit run chile/app_chile.py
Actualizar datos:  python3 chile/recolectar.py  (y commitear los CSV)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import motor as mo

st.set_page_config(page_title="🇨🇱 Predictor Liga Chilena", page_icon="🇨🇱", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { text-align:center; font-size:2.4rem; font-weight:700;
    background:linear-gradient(135deg,#d52b1e,#0039a6); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.2rem; }
.main-subtitle { text-align:center; font-size:1.05rem; color:#64748b; margin-bottom:1.5rem; }
.sec-title { font-size:1.4rem; font-weight:700; color:#0039a6; margin:0.5rem 0 0.5rem 0; }
.vs-text { text-align:center; font-size:2rem; font-weight:800; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { box-shadow:0 4px 6px -1px rgb(0 0 0/0.08); border-radius:12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_motor():
    return mo.cargar(en_vivo=True)


@st.cache_data(show_spinner="Simulando 10.000 campeonatos…")
def get_sim():
    return mo.simular_campeonato(get_motor(), 10000)


@st.cache_data(show_spinner="Analizando variables…")
def get_analisis():
    return mo.analisis_variables(get_motor())


# Botón de refresco en la barra lateral
if st.sidebar.button("🔄 Actualizar Resultados (ESPN)", key="refresh_chile"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

M = get_motor()
EQUIPOS = M["equipos_2026"]

st.markdown('<div class="main-title">🇨🇱 Predictor — Primera División de Chile 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Elo + modelo de resultado + simulación de Monte Carlo · '
            'datos en vivo desde ESPN</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏆 Campeonato", "⚽ Predecir partido", "🎯 El modelo"])

# ============================================================================
# TAB 1 · Campeonato (tabla actual + proyección)
# ============================================================================
with tab1:
    tabla = mo.tabla_actual(M)
    sim = get_sim()
    jugados = int(tabla.PJ.max()); total = jugados + len(M["fixture"]) // 8
    st.markdown(f'<div class="sec-title">Proyección del título — quedan {len(M["fixture"])} partidos por jugar</div>',
                unsafe_allow_html=True)
    proy = sim.merge(tabla[["Equipo", "Pos", "PJ"]], on="Equipo").sort_values("P_campeon", ascending=False)

    col1, col2 = st.columns([1.05, 1])
    with col1:
        st.markdown("**Tabla actual** (🟢 zona de copas · 🔴 zona de descenso)")
        def pinta(row):
            if row["Pos"] <= mo.CUPOS_COPA: return ["background-color:#e7f6ec"] * len(row)
            if row["Pos"] > len(tabla) - mo.DESCIENDEN: return ["background-color:#fde7e7"] * len(row)
            return [""] * len(row)
        st.dataframe(tabla[["Pos", "Equipo", "PJ", "G", "E", "P", "GF", "GC", "DG", "Pts"]].style.apply(pinta, axis=1),
                     hide_index=True, height=600, width='stretch')
    with col2:
        st.markdown("**Probabilidades (10.000 simulaciones)**")
        vis = proy.copy()
        vis["🏆 campeón"] = vis.P_campeon.map(lambda x: f"{x:.1%}")
        vis["copas"] = vis.P_copa.map(lambda x: f"{x:.0%}")
        vis["descenso"] = vis.P_descenso.map(lambda x: f"{x:.0%}")
        vis["pts proy."] = vis.pts_proy.round(0).astype(int)
        st.dataframe(vis[["Equipo", "🏆 campeón", "copas", "descenso", "pts proy."]],
                     hide_index=True, height=600, width='stretch')

    st.markdown('<div class="sec-title">Favoritos al título</div>', unsafe_allow_html=True)
    top = proy[proy.P_campeon > 0.001].head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.5 * len(top))))
    ax.barh(top.Equipo, top.P_campeon * 100, color="#d52b1e")
    for y, v in enumerate(top.P_campeon * 100):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center")
    ax.set_xlabel("Probabilidad de salir campeón (%)"); ax.margins(x=0.12)
    plt.tight_layout(); st.pyplot(fig)

# ============================================================================
# TAB 2 · Predecir partido
# ============================================================================
with tab2:
    c1, cvs, c2 = st.columns([5, 1, 5])
    with c1:
        local = st.selectbox("Local", EQUIPOS, index=EQUIPOS.index("Colo Colo") if "Colo Colo" in EQUIPOS else 0)
    with cvs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with c2:
        idx_v = EQUIPOS.index("Universidad de Chile") if "Universidad de Chile" in EQUIPOS else 1
        visita = st.selectbox("Visita", EQUIPOS, index=idx_v)

    if local == visita:
        st.error("Elige dos equipos distintos.")
    else:
        mix, p, (la, lb) = mo.grilla(M, local, visita)
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Gana {local}", f"{p[0]:.0%}", f"cuota {1/max(p[0],1e-6):.2f}", delta_color="off")
        k2.metric("Empate", f"{p[1]:.0%}", f"cuota {1/max(p[1],1e-6):.2f}", delta_color="off")
        k3.metric(f"Gana {visita}", f"{p[2]:.0%}", f"cuota {1/max(p[2],1e-6):.2f}", delta_color="off")
        st.caption(f"Goles esperados: {local} {la:.2f} — {lb:.2f} {visita}  ·  incluye ventaja de localía")

        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.markdown("**Mercados de goles**")
            n = mix.shape[0]; gi, gj = np.indices((n, n)); tot = gi + gj
            filas = []
            for ln in (1.5, 2.5, 3.5):
                po = float(mix[tot > ln].sum())
                filas.append({"Mercado": f"Over {ln}", "Prob.": f"{po:.0%}", "Cuota": f"{1/max(po,1e-6):.2f}"})
            btts = float(mix[(gi >= 1) & (gj >= 1)].sum())
            filas.append({"Mercado": "Ambos marcan", "Prob.": f"{btts:.0%}", "Cuota": f"{1/max(btts,1e-6):.2f}"})
            st.dataframe(pd.DataFrame(filas), hide_index=True, width='stretch')
            flat = mix.ravel(); top = flat.argsort()[::-1][:4]
            st.markdown("**Marcadores más probables:** " + " · ".join(
                f"`{i//n}-{i%n} ({flat[i]:.0%})`" for i in top))
        with cc2:
            m6 = np.zeros((6, 6)); m6[:5, :5] = mix[:5, :5]
            m6[5, :5] = mix[5:, :5].sum(0); m6[:5, 5] = mix[:5, 5:].sum(1); m6[5, 5] = mix[5:, 5:].sum()
            fig, ax = plt.subplots(figsize=(5, 4.2)); ax.imshow(m6, cmap="Reds")
            et = [str(i) for i in range(5)] + ["5+"]
            ax.set_xticks(range(6)); ax.set_xticklabels(et); ax.set_yticks(range(6)); ax.set_yticklabels(et)
            ax.set_xlabel(f"Goles {visita}"); ax.set_ylabel(f"Goles {local}")
            im, jm = np.unravel_index(m6.argmax(), m6.shape)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{m6[i,j]:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if m6[i, j] > m6.max() * 0.6 else "black",
                            fontweight="bold" if (i, j) == (im, jm) else "normal")
            ax.add_patch(plt.Rectangle((jm-.5, im-.5), 1, 1, fill=False, edgecolor="#0039a6", lw=2))
            st.pyplot(fig)

        # Panel de estadísticas esperadas del partido (box score de ESPN)
        se = mo.stats_esperadas(M, local, visita)
        if se:
            st.markdown('<div class="sec-title">Estadísticas esperadas del partido</div>', unsafe_allow_html=True)
            st.caption("Un modelo por estadística (Poisson/lineal con dominio + forma reciente del box score "
                       "de ESPN). Incluye **tarjetas**, que en clubes sí tenemos. Tómalo como tendencia.")
            from scipy.stats import poisson as _pois
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                tab = pd.DataFrame({
                    "Estadística": ["⚽ Goles esperados (xG)", "⛳ Córners", "🎯 Tiros al arco", "🟨 Faltas", "🟨 T. amarillas", "📊 Posesión %"],
                    local: [f"{se['xg'][0]:.2f}", f"{se['wonCorners'][0]:.1f}", f"{se['shotsOnTarget'][0]:.1f}",
                            f"{se['foulsCommitted'][0]:.1f}", f"{se['yellowCards'][0]:.1f}", f"{se['possessionPct'][0]:.0f}%"],
                    visita: [f"{se['xg'][1]:.2f}", f"{se['wonCorners'][1]:.1f}", f"{se['shotsOnTarget'][1]:.1f}",
                             f"{se['foulsCommitted'][1]:.1f}", f"{se['yellowCards'][1]:.1f}", f"{se['possessionPct'][1]:.0f}%"]})
                st.dataframe(tab, hide_index=True, width='stretch')
            with sc2:
                filas = []
                for nm, key, lineas in [("Córners", "wonCorners", [8.5, 9.5, 10.5]),
                                        ("T. amarillas", "yellowCards", [3.5, 4.5, 5.5])]:
                    tot = sum(se[key]);
                    for ln in lineas:
                        po = 1 - _pois.cdf(int(ln), tot)
                        filas.append({"Mercado": f"{nm} Over {ln}", "Prob.": f"{po:.0%}", "Cuota": f"{1/max(po,1e-6):.2f}"})
                st.dataframe(pd.DataFrame(filas), hide_index=True, width='stretch')
                st.caption(f"Totales esperados — córners {sum(se['wonCorners']):.1f} · amarillas {sum(se['yellowCards']):.1f}")

# ============================================================================
# TAB 3 · El modelo
# ============================================================================
with tab3:
    st.markdown('<div class="sec-title">¿Cómo funciona y qué tan confiable es?</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Datos:** 1.433 partidos de la Primera División (2021–2026) traídos de la API de ESPN.
    - **Elo:** rating cronológico con ventaja de localía (+55) y multiplicador de goleada.
    - **Modelo:** logística multinomial (gana local / empate / gana visita) con `elo_diff` (con localía),
      forma reciente (puntos/partido últimos 5) y head-to-head.
    - **Simulación:** parte de la tabla real y juega el fixture restante 10.000 veces.
    """)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import log_loss, accuracy_score
    part, F = M["part"], M["FEATS"]
    tr, te = part[part.fecha < "2025-07-01"], part[part.fecha >= "2025-07-01"]
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(tr[F], tr.resultado)
    P = pipe.predict_proba(te[F])
    base = np.tile(np.bincount(tr.resultado) / len(tr), (len(te), 1))
    m1, m2, m3 = st.columns(3)
    m1.metric("Acierto en test", f"{accuracy_score(te.resultado, P.argmax(1)):.0%}", f"{len(te)} partidos")
    m2.metric("Log-loss modelo", f"{log_loss(te.resultado, P, labels=[0,1,2]):.3f}", "menor = mejor", delta_color="off")
    m3.metric("Log-loss baseline", f"{log_loss(te.resultado, base, labels=[0,1,2]):.3f}", "frecuencias", delta_color="off")
    st.caption(f"Validación temporal (entrena hasta jun-2025, evalúa jul-2025 en adelante). El fútbol de clubes "
               f"es más impredecible que las selecciones (el local gana solo el {(part.resultado==2).mean():.0%}), "
               "pero el modelo le gana al baseline.")

    st.markdown('<div class="sec-title">Matriz de confusión (test temporal)</div>', unsafe_allow_html=True)
    from sklearn.metrics import confusion_matrix
    cmcol1, cmcol2 = st.columns([1, 1])
    with cmcol1:
        et = ["Victoria local", "Empate", "Gana visita"]
        cm = confusion_matrix(te.resultado, P.argmax(1), labels=[2, 1, 0])
        fig, ax = plt.subplots(figsize=(5.2, 4.4)); ax.imshow(cm, cmap="Reds")
        ax.set_xticks(range(3)); ax.set_xticklabels(et, rotation=15, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(et, fontsize=8)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                        fontweight="bold" if i == j else "normal")
        plt.tight_layout(); st.pyplot(fig)
    with cmcol2:
        n_emp = int((P.argmax(1) == 1).sum())
        st.markdown(f"""
        **El modelo casi nunca predice empate por argmax** ({n_emp} de {len(te)} partidos).
        No es un error: el empate es la clase *del medio*, así que rara vez es la **más probable** —
        siempre hay una de las otras dos un poco por encima.

        Pero el modelo **sí estima bien la probabilidad** de empate (está calibrada), que es lo que
        usan las simulaciones. Por eso un modelo de fútbol **no se evalúa con accuracy** sino con
        **log-loss / RPS**, y se usan las probabilidades, no el resultado más probable.

        El modelo acierta sobre todo las **victorias locales** (la localía es la señal más fuerte).
        """)

    st.markdown('<div class="sec-title">Selección de variables (con rigor, no a ojo)</div>', unsafe_allow_html=True)
    st.caption("Mismo protocolo que el modelo del Mundial: candidatas point-in-time → VIF (multicolinealidad) "
               "→ selección forward con CV temporal → comparación de modelos en el hold-out.")
    df_cand, sel, df_mod = get_analisis()
    ca1, ca2 = st.columns(2)
    with ca1:
        st.markdown("**Variables candidatas**")
        st.dataframe(df_cand, hide_index=True, width='stretch')
        st.caption(f"El forward elige: **{', '.join(sel)}**. El Elo concentra casi toda la señal; la forma y "
                   "el head-to-head aportan dentro del ruido (no superan el umbral) — igual que en el Mundial.")
    with ca2:
        st.markdown("**Comparación de modelos (hold-out temporal)**")
        st.dataframe(df_mod, hide_index=True, width='stretch')
        st.caption("Las logísticas lineales ganan; Random Forest y Gradient Boosting quedan **peores** — con "
                   "poca señal, la flexibilidad extra solo memoriza ruido. Por eso el modelo es una logística.")

    st.markdown('<div class="sec-title">Ranking Elo actual</div>', unsafe_allow_html=True)
    elo_df = pd.DataFrame([{"Equipo": t, "Elo": round(e)} for t, e in
                          sorted(M["elo"].items(), key=lambda x: -x[1]) if t in EQUIPOS])
    st.dataframe(elo_df, hide_index=True, width='stretch', height=400)
