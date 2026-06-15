"""
🧪 LABORATORIO — Simulador Mundial 2026 (versión avanzada, provisoria)

App alternativa que NO reemplaza a app.py (el deploy en producción). Añade cuatro frentes
sobre el mismo modelo base/híbrido con ponderación K-factor:

  1. Partido + mercados de apuestas (Over/Under, BTTS, marcador exacto, hándicap).
  2. Torneo en vivo: cargar resultados reales, actualizar Elo/forma y re-simular lo que falta.
  3. Validación: calibración en el hold-out 2025-26 y backtesting económico (value betting).
  4. Robustez: intervalos de confianza del Monte Carlo y análisis de sensibilidad.

Correr local:  streamlit run lab/app_lab.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import motor as mo
import espn_live

st.set_page_config(page_title="🧪 Lab Mundial 2026", page_icon="🧪", layout="wide")

# Mapeo español -> inglés con bandera (coherente con app.py)
COUNTRIES_ES = {
    'Alemania': ('Germany', '🇩🇪'), 'Argelia': ('Algeria', '🇩🇿'), 'Argentina': ('Argentina', '🇦🇷'),
    'Australia': ('Australia', '🇦🇺'), 'Austria': ('Austria', '🇦🇹'), 'Bélgica': ('Belgium', '🇧🇪'),
    'Bosnia y Herzegovina': ('Bosnia and Herzegovina', '🇧🇦'), 'Brasil': ('Brazil', '🇧🇷'),
    'Cabo Verde': ('Cape Verde', '🇨🇻'), 'Canadá': ('Canada', '🇨🇦'), 'Catar': ('Qatar', '🇶🇦'),
    'Colombia': ('Colombia', '🇨🇴'), 'Corea del Sur': ('South Korea', '🇰🇷'), 'Costa de Marfil': ('Ivory Coast', '🇨🇮'),
    'Croacia': ('Croatia', '🇭🇷'), 'Curazao': ('Curaçao', '🇨🇼'), 'Ecuador': ('Ecuador', '🇪🇨'),
    'Egipto': ('Egypt', '🇪🇬'), 'Escocia': ('Scotland', '🏴'), 'España': ('Spain', '🇪🇸'),
    'Estados Unidos': ('United States', '🇺🇸'), 'Francia': ('France', '🇫🇷'), 'Ghana': ('Ghana', '🇬🇭'),
    'Haití': ('Haiti', '🇭🇹'), 'Inglaterra': ('England', '🏴'), 'Irak': ('Iraq', '🇮🇶'),
    'Irán': ('Iran', '🇮🇷'), 'Japón': ('Japan', '🇯🇵'), 'Jordania': ('Jordan', '🇯🇴'),
    'Marruecos': ('Morocco', '🇲🇦'), 'México': ('Mexico', '🇲🇽'), 'Noruega': ('Norway', '🇳🇴'),
    'Nueva Zelanda': ('New Zealand', '🇳🇿'), 'Países Bajos': ('Netherlands', '🇳🇱'), 'Panamá': ('Panama', '🇵🇦'),
    'Paraguay': ('Paraguay', '🇵🇾'), 'Portugal': ('Portugal', '🇵🇹'), 'República Checa': ('Czech Republic', '🇨🇿'),
    'Rep. Democrática del Congo': ('DR Congo', '🇨🇩'), 'Senegal': ('Senegal', '🇸🇳'), 'Sudáfrica': ('South Africa', '🇿🇦'),
    'Suecia': ('Sweden', '🇸🇪'), 'Suiza': ('Switzerland', '🇨🇭'), 'Túnez': ('Tunisia', '🇹🇳'),
    'Turquía': ('Turkey', '🇹🇷'), 'Uruguay': ('Uruguay', '🇺🇾'), 'Uzbekistán': ('Uzbekistan', '🇺🇿'),
    'Arabia Saudita': ('Saudi Arabia', '🇸🇦'),
}
EN2ES = {en: (es, fl) for es, (en, fl) in COUNTRIES_ES.items()}
OPC = sorted(f"{fl} {es}" for es, (en, fl) in COUNTRIES_ES.items())


def es2en(label):
    es = label.split(" ", 1)[1]
    return COUNTRIES_ES[es][0]


def etiqueta(en):
    es, fl = EN2ES.get(en, (en, ""))
    return f"{fl} {es}"


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Simulando 8.000 mundiales…")
def mc_base(modelo):
    return mo.monte_carlo(get_motor(), n_sims=8000, modelo=modelo)


@st.cache_data(ttl=900, show_spinner=False)
def cargar_espn():
    """Resultados reales del Mundial desde ESPN (cacheados 15 min). Vacío si la API falla."""
    try:
        return espn_live.traer_resultados(), None
    except Exception as e:  # red caída, formato cambiado, etc. — la app sigue funcionando
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=120, show_spinner=False)
def cargar_envivo():
    try:
        return espn_live.partidos_en_vivo()
    except Exception:
        return pd.DataFrame()


M = get_motor()
ESPN_DF, ESPN_ERR = cargar_espn()

st.title("🧪 Laboratorio — Mundial 2026")
st.caption("Versión avanzada y **provisoria**. No reemplaza a la app principal; explora cuatro extensiones "
           "del modelo: mercados de apuestas, torneo en vivo, validación económica y robustez.")

modelo = st.sidebar.radio("Modelo", ["base", "hyb"],
                          format_func=lambda x: "Base (Elo+H2H+valor)" if x == "base" else "Híbrido (+ forma)")
st.sidebar.markdown("---")
st.sidebar.info("Todos los modelos usan **ponderación K-factor** (los partidos en serio pesan más).")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚽ Partido + Mercados", "🔴 Torneo en vivo", "🎯 Validación", "📊 Robustez"])

# ============================================================================
# FRENTE 1 · Partido + mercados
# ============================================================================
with tab1:
    st.subheader("Predicción de un partido y mercados de apuestas")
    c1, cvs, c2 = st.columns([5, 1, 5])
    with c1:
        a = es2en(st.selectbox("Equipo 1", OPC, index=OPC.index("🇪🇸 España"), key="m_a"))
    with cvs:
        st.markdown("<h2 style='text-align:center;margin-top:1.5rem'>VS</h2>", unsafe_allow_html=True)
    with c2:
        b = es2en(st.selectbox("Equipo 2", OPC, index=OPC.index("🇦🇷 Argentina"), key="m_b"))
    cancha = st.radio("Cancha", ["Automática (anfitrión de local)", "Neutral",
                                 "Local equipo 1", "Local equipo 2"], horizontal=True)
    cmode = {"Automática (anfitrión de local)": "auto", "Neutral": "neutral",
             "Local equipo 1": "1", "Local equipo 2": "2"}[cancha]

    na, nb = EN2ES.get(a, (a,))[0], EN2ES.get(b, (b,))[0]
    if a == b:
        st.error("Elige dos selecciones distintas.")
    else:
        # Últimos partidos de cada selección (historial + Mundial en curso desde ESPN)
        lc1, lc2 = st.columns(2)
        for col, eq, nm in ((lc1, a, na), (lc2, b, nb)):
            with col:
                up = mo.ultimos_partidos(M, eq, n=6, extra=ESPN_DF)
                with st.expander(f"📋 Últimos 6 de {nm}", expanded=True):
                    if len(up):
                        st.dataframe(up[["res", "loc", "rival", "marcador", "fecha"]],
                                     hide_index=True, width='stretch')
                    else:
                        st.caption("Sin partidos en el historial.")

        mix, p, (la, lb) = mo.grilla(M, a, b, cmode, modelo)
        mk = mo.mercados(mix)
        st.markdown("#### Resultado (1X2) — probabilidad y cuota justa")
        st.caption("La **cuota justa** es 1 ÷ probabilidad. Si la casa paga MÁS que esta cuota, hay *value* "
                   "(infravalorado); si paga menos, está sobrevalorado.")
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Gana {na}", f"{p[0]:.1%}", f"cuota {mo.cuota(p[0]):.2f}", delta_color="off")
        k2.metric("Empate", f"{p[1]:.1%}", f"cuota {mo.cuota(p[1]):.2f}", delta_color="off")
        k3.metric(f"Gana {nb}", f"{p[2]:.1%}", f"cuota {mo.cuota(p[2]):.2f}", delta_color="off")
        st.caption(f"Goles esperados (Poisson): {la:.2f} — {lb:.2f}")

        st.markdown("#### Mercados — probabilidad y cuota justa")
        filas = []
        for ln in (1.5, 2.5, 3.5):
            for lado in ("Over", "Under"):
                pr = mk[f"{lado} {ln}"]
                filas.append({"Mercado": f"{lado} {ln} goles", "Prob.": f"{pr:.1%}", "Cuota justa": f"{mo.cuota(pr):.2f}"})
        for lado, key in (("Ambos marcan (sí)", "Ambos marcan (BTTS sí)"), ("Ambos marcan (no)", "BTTS no")):
            filas.append({"Mercado": lado, "Prob.": f"{mk[key]:.1%}", "Cuota justa": f"{mo.cuota(mk[key]):.2f}"})
        for ln in (-2, -1, 1):
            hc = mo.handicap_asiatico(mix, ln)
            sg = f"+{ln}" if ln > 0 else str(ln)
            filas.append({"Mercado": f"Hándicap {na} {sg}", "Prob.": f"{hc['A cubre']:.1%}",
                          "Cuota justa": f"{mo.cuota(hc['A cubre']):.2f}"})
        mcols = st.columns([2, 2])
        with mcols[0]:
            st.dataframe(pd.DataFrame(filas[:8]), hide_index=True, width='stretch')
        with mcols[1]:
            st.dataframe(pd.DataFrame(filas[8:]), hide_index=True, width='stretch')
            st.markdown("**Marcadores más probables**")
            st.write(" · ".join(f"`{i}-{j} ({pr:.0%}, cuota {mo.cuota(pr):.1f})`"
                                for i, j, pr in mk["_top_marcadores"][:3]))

        st.markdown("#### Matriz de marcadores")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        m6 = np.zeros((7, 7)); m6[:6, :6] = mix[:6, :6]
        m6[6, :6] = mix[6:, :6].sum(0); m6[:6, 6] = mix[:6, 6:].sum(1); m6[6, 6] = mix[6:, 6:].sum()
        im = ax.imshow(m6, cmap="Blues")
        et = [str(i) for i in range(6)] + ["6+"]
        ax.set_xticks(range(7)); ax.set_xticklabels(et); ax.set_yticks(range(7)); ax.set_yticklabels(et)
        ax.set_xlabel(f"Goles {EN2ES.get(b,(b,))[0]}"); ax.set_ylabel(f"Goles {EN2ES.get(a,(a,))[0]}")
        imax, jmax = np.unravel_index(m6.argmax(), m6.shape)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{m6[i,j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if m6[i, j] > m6.max() * 0.6 else "black",
                        fontweight="bold" if (i, j) == (imax, jmax) else "normal")
        ax.add_patch(plt.Rectangle((jmax - .5, imax - .5), 1, 1, fill=False, edgecolor="#d62728", lw=2))
        st.pyplot(fig)

# ============================================================================
# FRENTE 2 · Torneo en vivo
# ============================================================================
with tab2:
    st.subheader("Modelo vivo: resultados reales del Mundial → re-simulación")
    st.markdown(
        "El estado de los equipos está congelado al **corte de junio 2026**. Esta pestaña trae los "
        "resultados **reales** del Mundial desde la **API de ESPN**, **actualiza el Elo y la forma** de "
        "cada selección, **fija** los partidos de grupo ya jugados y **re-simula el torneo restante**.")

    fuente = st.radio("Fuente de resultados", ["🛰️ Automática (ESPN, en vivo)", "✍️ Manual"], horizontal=True)

    if fuente.startswith("🛰️"):
        if ESPN_ERR:
            st.error(f"No se pudo contactar a ESPN ({ESPN_ERR}). Usa el modo manual.")
            res = pd.DataFrame()
        elif len(ESPN_DF) == 0:
            st.info("ESPN aún no reporta partidos finalizados del Mundial. Vuelve cuando empiece.")
            res = pd.DataFrame()
        else:
            res = ESPN_DF.copy()
            st.success(f"{len(res)} partidos finalizados traídos de ESPN (cache 15 min).")
            envivo = cargar_envivo()
            if len(envivo):
                st.caption("🔴 En juego ahora: " + " · ".join(
                    f"{r.local} {r.marcador} {r.visita} ({r.minuto})" for r in envivo.itertuples(index=False)))
            st.dataframe(res[["fecha", "local", "goles_local", "goles_visita", "visita"]],
                         hide_index=True, width='stretch', height=240)
    else:
        plantilla = pd.DataFrame({"local": pd.Series(dtype="str"), "visita": pd.Series(dtype="str"),
                                  "goles_local": pd.Series(dtype="int"), "goles_visita": pd.Series(dtype="int")})
        edit = st.data_editor(
            plantilla, num_rows="dynamic", width='stretch', key="vivo",
            column_config={
                "local": st.column_config.SelectboxColumn("Local", options=mo.MUNDIALISTAS, required=True),
                "visita": st.column_config.SelectboxColumn("Visita", options=mo.MUNDIALISTAS, required=True),
                "goles_local": st.column_config.NumberColumn("Goles local", min_value=0, max_value=15, step=1),
                "goles_visita": st.column_config.NumberColumn("Goles visita", min_value=0, max_value=15, step=1)})
        res = edit.dropna(subset=["local", "visita", "goles_local", "goles_visita"])
        res = res[res.local != res.visita]
        if len(res):
            st.success(f"{len(res)} resultado(s) cargado(s).")

    if st.button("🔄 Actualizar y re-simular (4.000 mundiales)", type="primary"):
        if len(res) == 0:
            st.warning("No hay resultados para incorporar.")
        else:
            st2 = mo.actualizar_estados(M, res)
            fijos = {(r.local, r.visita): (int(r.goles_local), int(r.goles_visita))
                     for r in res.itertuples(index=False)
                     if mo.GRUPO_DE.get(r.local) == mo.GRUPO_DE.get(r.visita)}
            with st.spinner("Re-simulando con los estados actualizados…"):
                r_pre = mo.monte_carlo(M, 4000, modelo)
                r_post = mo.monte_carlo(M, 4000, modelo, states=st2, fijos=fijos)
            comp = r_pre[["Selección", "P_campeon"]].rename(columns={"P_campeon": "pre"}).merge(
                r_post[["Selección", "P_campeon"]].rename(columns={"P_campeon": "post"}), on="Selección")
            comp["Δ"] = comp.post - comp.pre
            comp = comp.sort_values("post", ascending=False).head(12)
            comp["Selección"] = comp["Selección"].map(etiqueta)
            st.markdown("##### Probabilidad de campeón: antes vs. después de los resultados")
            st.dataframe(comp.style.format({"pre": "{:.1%}", "post": "{:.1%}", "Δ": "{:+.1%}"})
                         .background_gradient(subset=["Δ"], cmap="RdYlGn"), width='stretch')
            subio = comp.loc[comp["Δ"].idxmax()]
            st.caption(f"Mayor salto: {subio['Selección']} ({subio['Δ']:+.1%}).")

# ============================================================================
# FRENTE 3 · Validación
# ============================================================================
with tab3:
    st.subheader("¿Es confiable el modelo? Calibración y prueba económica")
    P, y, te = mo.backtest_test(M, modelo)
    st.markdown("##### 1. Calibración en el hold-out temporal (2025–26, nunca visto)")
    st.caption("Si el modelo dice «60%», ¿ocurre el 60% de las veces? La curva debe pegarse a la diagonal.")
    cols = st.columns(3)
    for k, clase in mo.ETIQUETAS.items():
        xs, ys, ns, ece = mo.curva_calibracion(P, y, k)
        with cols[k]:
            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            ax.plot([0, 1], [0, 1], "k--", lw=.8)
            ax.plot(xs, ys, "o-", color="#2a7ae2")
            ax.set_title(f"{clase}  (ECE={ece:.3f})", fontsize=10)
            ax.set_xlabel("predicho"); ax.set_ylabel("observado")
            lim = max(xs.max(), ys.max()) * 1.1
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            st.pyplot(fig)
    st.info("**ECE** (Expected Calibration Error) bajo = probabilidades confiables. Es lo que hace que el "
            "Monte Carlo tenga sentido.")

    st.markdown("##### 2. Backtesting económico — *value betting*")
    margen = st.slider("Margen del bookmaker (overround)", 0.02, 0.10, 0.05, 0.01)
    resu, curva = mo.value_betting(M, modelo, margen=margen)
    if resu["n_apuestas"]:
        m1, m2, m3 = st.columns(3)
        m1.metric("Apuestas de valor", resu["n_apuestas"])
        m2.metric("ROI", f"{resu['roi']:+.1%}")
        m3.metric("% acierto", f"{resu['acierto']:.1%}")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(range(len(curva)), curva["acumulado"], color="#2a9d5c")
        ax.axhline(0, color="grey", lw=.8, ls="--")
        ax.set_xlabel("apuestas (orden cronológico)"); ax.set_ylabel("ganancia acumulada (u.)")
        ax.set_title("Curva de ganancia simulada")
        st.pyplot(fig)
    st.warning("⚠️ **Honestidad:** no hay cuotas reales de casas de apuestas en el proyecto. El «mercado» es "
               "**sintético** — las probabilidades de un modelo simple (solo Elo) más un margen. Que el modelo "
               "completo le saque ROI positivo prueba que la información extra (H2H, valor de plantilla) **aporta "
               "valor sobre el Elo solo**, no que le ganarías a Bet365. Con cuotas reales el número cambiaría.")

# ============================================================================
# FRENTE 4 · Robustez
# ============================================================================
with tab4:
    st.subheader("¿Cuánta confianza darle a las probabilidades?")
    res = mc_base(modelo)
    st.markdown("##### 1. Intervalo de confianza del Monte Carlo (8.000 simulaciones)")
    st.caption("Cada probabilidad es una estimación con error muestral. Barras = IC 95% (error binomial).")
    top = res.head(10).copy()
    los, his = zip(*[mo.ic_montecarlo(p, 8000) for p in top.P_campeon])
    fig, ax = plt.subplots(figsize=(9, 5))
    yp = np.arange(len(top))[::-1]
    ax.barh(yp, top.P_campeon * 100, color="#0b3d91",
            xerr=[(top.P_campeon - np.array(los)) * 100, (np.array(his) - top.P_campeon) * 100],
            error_kw=dict(ecolor="#d62728", capsize=4))
    ax.set_yticks(yp); ax.set_yticklabels([etiqueta(t) for t in top["Selección"]])
    ax.set_xlabel("P(campeón) %")
    st.pyplot(fig)
    st.caption("Con 8.000 simulaciones el error es de ±0.4–0.8 pts para los favoritos: suficiente para "
               "distinguir a España de Argentina, pero no para separar al 10° del 12°.")

    st.markdown("##### 2. Análisis de sensibilidad — ¿qué pasa si un equipo cambia de nivel?")
    st.caption("Mueve el Elo de una selección (p. ej. una baja importante = −Elo, o un buen momento = +Elo) "
               "y mira cómo cambian sus probabilidades en un cruce concreto. Instantáneo (nivel partido).")
    sc1, sc2 = st.columns(2)
    with sc1:
        eq = es2en(st.selectbox("Selección a ajustar", OPC, index=OPC.index("🇪🇸 España"), key="s_eq"))
        delta = st.slider("Ajuste de Elo", -200, 200, 0, 10)
    with sc2:
        rival = es2en(st.selectbox("Rival (cancha neutral)", OPC, index=OPC.index("🇫🇷 Francia"), key="s_riv"))
    if eq != rival:
        st_mod = M["states"].copy()
        st_mod.loc[eq, "elo"] += delta
        p0 = mo.prob_partido(M, eq, rival, "neutral", modelo)
        p1 = mo.prob_partido(M, eq, rival, "neutral", modelo, states=st_mod)
        d1, d2, d3 = st.columns(3)
        d1.metric(f"Gana {EN2ES.get(eq,(eq,))[0]}", f"{p1[0]:.1%}", f"{(p1[0]-p0[0])*100:+.1f} pts")
        d2.metric("Empate", f"{p1[1]:.1%}", f"{(p1[1]-p0[1])*100:+.1f} pts")
        d3.metric(f"Gana {EN2ES.get(rival,(rival,))[0]}", f"{p1[2]:.1%}", f"{(p1[2]-p0[2])*100:+.1f} pts")
        st.caption(f"Elo {EN2ES.get(eq,(eq,))[0]}: {M['states'].loc[eq,'elo']:.0f} → "
                   f"{M['states'].loc[eq,'elo']+delta:.0f}  (las flechas comparan contra el Elo original)")
