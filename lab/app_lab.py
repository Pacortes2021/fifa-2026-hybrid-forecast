"""
🧪 LABORATORIO — Simulador Mundial 2026 (versión avanzada, provisoria)

App alternativa que NO reemplaza a app.py (el deploy en producción). Reusa el estilo visual
"premium" de la app principal y añade cinco frentes sobre el modelo base/híbrido (K-factor):

  ⚽ Partido + mercados + cuotas justas (Base vs Híbrido lado a lado)
  📊 Fase de grupos (tabla de posiciones real, en vivo desde ESPN)
  🔴 Torneo en vivo (resultados reales de ESPN → re-simulación)
  🎯 Validación (calibración + value betting)
  📈 Robustez (intervalos de confianza + sensibilidad)

Correr local:  streamlit run lab/app_lab.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import motor as mo
import espn_live

st.set_page_config(page_title="🧪 Lab Mundial 2026", page_icon="🏆", layout="wide",
                   initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- #
#  Estilo premium (heredado de la app principal)
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { text-align:center; font-size:2.5rem; font-weight:700;
    background:linear-gradient(135deg,#0b3d91,#7a3b91); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.2rem; }
.main-subtitle { text-align:center; font-size:1.05rem; color:#64748b; margin-bottom:1.5rem; }
.card-title-base { font-size:1.25rem; font-weight:600; color:#0b3d91;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.card-title-hybrid { font-size:1.25rem; font-weight:600; color:#7a3b91;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.5rem; font-weight:700; color:#0b3d91; margin:0.4rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2rem; font-weight:800; color:#cbd5e1; margin-top:1.6rem; }
/* sombra premium a los contenedores con borde */
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 4px 6px -1px rgb(0 0 0/0.08),0 2px 4px -2px rgb(0 0 0/0.08);
    border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# Mapeo español -> inglés con bandera
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
    return COUNTRIES_ES[label.split(" ", 1)[1]][0]


def nombre(en):
    return EN2ES.get(en, (en,))[0]


def etiqueta(en):
    es, fl = EN2ES.get(en, (en, ""))
    return f"{fl} {es}"


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Simulando 8.000 mundiales…")
def mc_base(modelo):
    return mo.monte_carlo(get_motor(), n_sims=8000, modelo=modelo)


@st.cache_data(show_spinner="Re-simulando con resultados reales…")
def mc_vivo(modelo, key):
    """Monte Carlo incorporando los resultados reales de ESPN (estados + grupos fijados)."""
    M = get_motor()
    if len(ESPN_DF) == 0:
        return mc_base(modelo)
    st2 = mo.actualizar_estados(M, ESPN_DF)
    fijos = {(r.local, r.visita): (int(r.goles_local), int(r.goles_visita))
             for r in ESPN_DF.itertuples(index=False)
             if mo.GRUPO_DE.get(r.local) == mo.GRUPO_DE.get(r.visita)}
    return mo.monte_carlo(M, 6000, modelo, states=st2, fijos=fijos)


@st.cache_data(ttl=900, show_spinner=False)
def cargar_espn():
    try:
        return espn_live.traer_resultados(), None
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=120, show_spinner=False)
def cargar_envivo():
    try:
        return espn_live.partidos_en_vivo()
    except Exception:
        return pd.DataFrame()


M = get_motor()
ESPN_DF, ESPN_ERR = cargar_espn()
ESPN_KEY = "" if len(ESPN_DF) == 0 else f"{len(ESPN_DF)}-{ESPN_DF.fecha.max()}"

st.markdown('<div class="main-title">🏆 Laboratorio — Copa Mundial 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Mercados y cuotas · Fase de grupos en vivo · Re-simulación con '
            'resultados reales · Validación · Robustez</div>', unsafe_allow_html=True)

modelo = st.sidebar.radio("Modelo (para Grupos / En vivo / Validación / Robustez)", ["base", "hyb"],
                          format_func=lambda x: "Base (Elo+H2H+valor)" if x == "base" else "Híbrido (+ forma)")
st.sidebar.info("Todos los modelos usan **ponderación K-factor**: los partidos en serio (Mundial, "
                "clasificatorias) pesan más que los amistosos.")
if len(ESPN_DF):
    st.sidebar.success(f"🛰️ ESPN: {len(ESPN_DF)} partidos reales cargados.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["⚽ Partido + Mercados", "📊 Fase de grupos", "🔴 Torneo en vivo", "🎯 Validación", "📈 Robustez"])

# ============================================================================
# TAB 1 · Partido + mercados (comparación Base vs Híbrido, estilo app principal)
# ============================================================================
with tab1:
    c1, cvs, c2 = st.columns([5, 1, 5])
    with c1:
        a = es2en(st.selectbox("Selección 1", OPC, index=OPC.index("🇪🇸 España"), key="m_a"))
    with cvs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with c2:
        b = es2en(st.selectbox("Selección 2", OPC, index=OPC.index("🇦🇷 Argentina"), key="m_b"))
    cancha = st.radio("Cancha", ["Automática (anfitrión de local)", "Neutral",
                                 f"Local {nombre(a)}", f"Local {nombre(b)}"], horizontal=True)
    cmode = {"Automática (anfitrión de local)": "auto", "Neutral": "neutral",
             f"Local {nombre(a)}": "1", f"Local {nombre(b)}": "2"}.get(cancha, "auto")

    if a == b:
        st.error("Elige dos selecciones distintas.")
    else:
        na, nb = nombre(a), nombre(b)
        # Últimos partidos
        lc1, lc2 = st.columns(2)
        for col, eq, nm in ((lc1, a, na), (lc2, b, nb)):
            with col:
                up = mo.ultimos_partidos(M, eq, n=6, extra=ESPN_DF)
                with st.expander(f"📋 Forma reciente — {nm}", expanded=False):
                    st.dataframe(up[["res", "loc", "rival", "marcador", "fecha"]] if len(up) else up,
                                 hide_index=True, width='stretch')

        def tarjeta(modelo_id, titulo, css):
            with st.container(border=True):
                st.markdown(f'<div class="{css}">{titulo}</div>', unsafe_allow_html=True)
                mix, p, (la, lb) = mo.grilla(M, a, b, cmode, modelo_id)
                for nom_lado, prob in ((f"Victoria {na}", p[0]), ("Empate", p[1]), (f"Victoria {nb}", p[2])):
                    st.markdown(f"**{nom_lado}: {prob:.1%}**  ·  cuota justa `{mo.cuota(prob):.2f}`")
                    st.progress(float(prob))
                pav = p[0] + p[1] * p[0] / (p[0] + p[2])
                st.caption(f"Si fuese eliminatoria, avanza **{na} {pav:.0%}** / {nb} {1-pav:.0%}")
                st.caption(f"Goles esperados (Poisson): {na} {la:.2f} — {lb:.2f} {nb}")
                return mix, p

        col_b, col_h = st.columns(2)
        with col_b:
            mix_b, _ = tarjeta("base", "🔵 Modelo Base (Elo + H2H + valor)", "card-title-base")
        with col_h:
            mix_h, _ = tarjeta("hyb", "🟣 Modelo Híbrido (Base + forma reciente)", "card-title-hybrid")

        # Mercados (del modelo del sidebar) + cuotas
        st.markdown(f'<div class="sec-title">Mercados — modelo {"Base" if modelo=="base" else "Híbrido"} '
                    f'(probabilidad y cuota justa)</div>', unsafe_allow_html=True)
        st.caption("Compara la **cuota justa** (1 ÷ probabilidad) con la de tu casa de apuestas: si la casa "
                   "paga más, el resultado está *infravalorado* (hay value); si paga menos, sobrevalorado.")
        mix = mix_b if modelo == "base" else mix_h
        mk = mo.mercados(mix)
        filas = []
        for ln in (1.5, 2.5, 3.5):
            for lado in ("Over", "Under"):
                pr = mk[f"{lado} {ln}"]
                filas.append({"Mercado": f"{lado} {ln} goles", "Prob.": f"{pr:.1%}", "Cuota justa": f"{mo.cuota(pr):.2f}"})
        for et, key in (("Ambos marcan: Sí", "Ambos marcan (BTTS sí)"), ("Ambos marcan: No", "BTTS no")):
            filas.append({"Mercado": et, "Prob.": f"{mk[key]:.1%}", "Cuota justa": f"{mo.cuota(mk[key]):.2f}"})
        for ln in (-2, -1, 1):
            hc = mo.handicap_asiatico(mix, ln)
            sg = f"+{ln}" if ln > 0 else str(ln)
            filas.append({"Mercado": f"Hándicap {na} {sg}", "Prob.": f"{hc['A cubre']:.1%}",
                          "Cuota justa": f"{mo.cuota(hc['A cubre']):.2f}"})
        mc1, mc2 = st.columns(2)
        mc1.dataframe(pd.DataFrame(filas[:8]), hide_index=True, width='stretch')
        mc2.dataframe(pd.DataFrame(filas[8:]), hide_index=True, width='stretch')
        st.markdown("**Marcadores más probables:** " + " · ".join(
            f"`{i}-{j} ({pr:.0%}, cuota {mo.cuota(pr):.1f})`" for i, j, pr in mk["_top_marcadores"][:4]))

        # Estadísticas esperadas del partido (córners, tiros al arco, faltas, posesión)
        st.markdown('<div class="sec-title">Estadísticas esperadas del partido</div>', unsafe_allow_html=True)
        st.caption("Modelo *ataque × defensa*: lo que cada equipo genera × lo que el rival concede, "
                   "normalizado. Basado en el box score de ESPN (cobertura ≈48%), así que tómalo como "
                   "estimación de tendencia, no exacta.")
        se = mo.stats_esperadas(M, a, b)
        se1, se2 = st.columns([1, 1])
        with se1:
            df_se = pd.DataFrame({
                "Estadística": ["⛳ Córners", "🎯 Tiros al arco", "🟨 Faltas", "📊 Posesión %"],
                na: [f"{se['corners'][0]:.1f}", f"{se['tiros_arco'][0]:.1f}",
                     f"{se['faltas'][0]:.1f}", f"{se['posesion'][0]:.0f}%"],
                nb: [f"{se['corners'][1]:.1f}", f"{se['tiros_arco'][1]:.1f}",
                     f"{se['faltas'][1]:.1f}", f"{se['posesion'][1]:.0f}%"]})
            st.dataframe(df_se, hide_index=True, width='stretch')
        with se2:
            filas_ou = []
            for nm_st, lineas in (("Córners", [8.5, 9.5, 10.5]), ("Tiros al arco", [6.5, 7.5, 8.5])):
                key = "corners" if nm_st == "Córners" else "tiros_arco"
                ou = mo.over_under_total(*se[key], lineas)
                tot = sum(se[key])
                for ln, (po, pu) in ou.items():
                    filas_ou.append({"Mercado": f"{nm_st} Over {ln}", "Prob.": f"{po:.0%}",
                                     "Cuota justa": f"{mo.cuota(po):.2f}"})
            st.dataframe(pd.DataFrame(filas_ou), hide_index=True, width='stretch')
            st.caption(f"Totales esperados — córners: **{sum(se['corners']):.1f}**  ·  "
                       f"tiros al arco: **{sum(se['tiros_arco']):.1f}**  ·  faltas: **{sum(se['faltas']):.1f}**")

        # Matrices de marcadores lado a lado
        st.markdown('<div class="sec-title">Matrices de marcadores</div>', unsafe_allow_html=True)
        gx1, gx2 = st.columns(2)
        for col, mix_x, tit, cmap in ((gx1, mix_b, "Base", "Blues"), (gx2, mix_h, "Híbrido", "Purples")):
            with col:
                fig, ax = plt.subplots(figsize=(5.4, 4.6))
                m6 = np.zeros((7, 7)); m6[:6, :6] = mix_x[:6, :6]
                m6[6, :6] = mix_x[6:, :6].sum(0); m6[:6, 6] = mix_x[:6, 6:].sum(1); m6[6, 6] = mix_x[6:, 6:].sum()
                ax.imshow(m6, cmap=cmap)
                etq = [str(i) for i in range(6)] + ["6+"]
                ax.set_xticks(range(7)); ax.set_xticklabels(etq); ax.set_yticks(range(7)); ax.set_yticklabels(etq)
                ax.set_xlabel(f"Goles {nb}"); ax.set_ylabel(f"Goles {na}"); ax.set_title(tit, fontsize=11)
                imax, jmax = np.unravel_index(m6.argmax(), m6.shape)
                for i in range(7):
                    for j in range(7):
                        ax.text(j, i, f"{m6[i,j]:.0%}", ha="center", va="center", fontsize=7.5,
                                color="white" if m6[i, j] > m6.max() * 0.6 else "black",
                                fontweight="bold" if (i, j) == (imax, jmax) else "normal")
                ax.add_patch(plt.Rectangle((jmax-.5, imax-.5), 1, 1, fill=False, edgecolor="#d62728", lw=2))
                st.pyplot(fig)

# ============================================================================
# TAB 2 · Fase de grupos (tabla de posiciones real + prob. de clasificar)
# ============================================================================
with tab2:
    st.markdown('<div class="sec-title">Fase de grupos — tabla de posiciones</div>', unsafe_allow_html=True)
    if len(ESPN_DF):
        st.caption(f"Posiciones reales según los **{len(ESPN_DF)} partidos ya jugados** (ESPN). La columna "
                   "**P(clasif.)** es la probabilidad de avanzar a octavos según el modelo, ya actualizado "
                   "con estos resultados. Verde = puestos de clasificación directa (1° y 2°).")
    else:
        st.caption("El Mundial aún no reporta partidos. Las tablas arrancan en cero; cuando se jueguen, se "
                   "llenan solas desde ESPN.")
    tablas = mo.tablas_grupos(ESPN_DF)
    res_v = mc_vivo(modelo, ESPN_KEY)
    pclasif = res_v.set_index("Selección")["P_octavos"].to_dict()

    def pinta(row):
        if row["Pos"] <= 2:
            return ["background-color:#e7f6ec"] * len(row)
        if row["Pos"] == 3:
            return ["background-color:#fdf6e3"] * len(row)
        return [""] * len(row)

    letras = list(tablas.keys())
    for fila in range(0, 12, 3):
        cols = st.columns(3)
        for k, g in enumerate(letras[fila:fila + 3]):
            with cols[k]:
                with st.container(border=True):
                    st.markdown(f'<div class="card-title-base">Grupo {g}</div>', unsafe_allow_html=True)
                    d = tablas[g].copy()
                    d["Equipo"] = d["Equipo"].map(etiqueta)
                    d["P(clasif.)"] = [f"{pclasif.get(t, 0):.0%}" for t in tablas[g]["Equipo"]]
                    d = d[["Pos", "Equipo", "PJ", "G", "E", "P", "GF", "GC", "DG", "Pts", "P(clasif.)"]]
                    st.dataframe(d.style.apply(pinta, axis=1), hide_index=True, width='stretch')

# ============================================================================
# TAB 3 · Torneo en vivo
# ============================================================================
with tab3:
    st.markdown('<div class="sec-title">Modelo vivo: resultados reales → re-simulación</div>', unsafe_allow_html=True)
    st.markdown("Trae los resultados **reales** del Mundial desde la **API de ESPN**, actualiza el Elo y la "
                "forma de cada selección, **fija** los partidos de grupo jugados y **re-simula el resto**.")

    fuente = st.radio("Fuente de resultados", ["🛰️ Automática (ESPN)", "✍️ Manual"], horizontal=True)
    if fuente.startswith("🛰️"):
        if ESPN_ERR:
            st.error(f"No se pudo contactar a ESPN ({ESPN_ERR}). Usa el modo manual.")
            res = pd.DataFrame()
        elif len(ESPN_DF) == 0:
            st.info("ESPN aún no reporta partidos finalizados. Vuelve cuando empiece el Mundial.")
            res = pd.DataFrame()
        else:
            res = ESPN_DF.copy()
            st.success(f"{len(res)} partidos finalizados traídos de ESPN (cache 15 min).")
            ev = cargar_envivo()
            if len(ev):
                st.caption("🔴 En juego: " + " · ".join(
                    f"{r.local} {r.marcador} {r.visita} ({r.minuto})" for r in ev.itertuples(index=False)))
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
            with st.spinner("Re-simulando…"):
                r_pre = mo.monte_carlo(M, 4000, modelo)
                r_post = mo.monte_carlo(M, 4000, modelo, states=st2, fijos=fijos)
            comp = r_pre[["Selección", "P_campeon"]].rename(columns={"P_campeon": "pre"}).merge(
                r_post[["Selección", "P_campeon"]].rename(columns={"P_campeon": "post"}), on="Selección")
            comp["Δ"] = comp.post - comp.pre
            comp = comp.sort_values("post", ascending=False).head(12)
            comp["Selección"] = comp["Selección"].map(etiqueta)
            st.markdown("##### Probabilidad de campeón: antes vs. después")
            st.dataframe(comp.style.format({"pre": "{:.1%}", "post": "{:.1%}", "Δ": "{:+.1%}"})
                         .background_gradient(subset=["Δ"], cmap="RdYlGn"), width='stretch', hide_index=True)
            subio = comp.loc[comp["Δ"].idxmax()]
            st.caption(f"Mayor salto: {subio['Selección']} ({subio['Δ']:+.1%}).")

# ============================================================================
# TAB 4 · Validación
# ============================================================================
with tab4:
    st.markdown('<div class="sec-title">¿Es confiable el modelo?</div>', unsafe_allow_html=True)
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
            lim = max(xs.max(), ys.max()) * 1.1; ax.set_xlim(0, lim); ax.set_ylim(0, lim)
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
        st.pyplot(fig)
    st.warning("⚠️ **Honestidad:** no hay cuotas reales de casas de apuestas. El «mercado» es **sintético** "
               "(un modelo simple de solo Elo + margen). Un ROI positivo prueba que la información extra del "
               "modelo completo (H2H, valor de plantilla) **aporta valor sobre el Elo solo**, no que le "
               "ganarías a Bet365. Con cuotas reales el número cambiaría.")

# ============================================================================
# TAB 5 · Robustez
# ============================================================================
with tab5:
    st.markdown('<div class="sec-title">¿Cuánta confianza darle a las probabilidades?</div>', unsafe_allow_html=True)
    res = mc_base(modelo)
    st.markdown("##### 1. Intervalo de confianza del Monte Carlo (8.000 simulaciones)")
    st.caption("Cada probabilidad es una estimación con error muestral. Barras rojas = IC 95% (binomial).")
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

    st.markdown("##### 2. Análisis de sensibilidad — ¿y si un equipo cambia de nivel?")
    st.caption("Mueve el Elo de una selección (una baja importante = −Elo; un buen momento = +Elo) y mira el "
               "efecto inmediato en un cruce concreto.")
    sc1, sc2 = st.columns(2)
    with sc1:
        eq = es2en(st.selectbox("Selección a ajustar", OPC, index=OPC.index("🇪🇸 España"), key="s_eq"))
        delta = st.slider("Ajuste de Elo", -200, 200, 0, 10)
    with sc2:
        rival = es2en(st.selectbox("Rival (cancha neutral)", OPC, index=OPC.index("🇫🇷 Francia"), key="s_riv"))
    if eq != rival:
        st_mod = M["states"].copy(); st_mod.loc[eq, "elo"] += delta
        p0 = mo.prob_partido(M, eq, rival, "neutral", modelo)
        p1 = mo.prob_partido(M, eq, rival, "neutral", modelo, states=st_mod)
        d1, d2, d3 = st.columns(3)
        d1.metric(f"Gana {nombre(eq)}", f"{p1[0]:.1%}", f"{(p1[0]-p0[0])*100:+.1f} pts")
        d2.metric("Empate", f"{p1[1]:.1%}", f"{(p1[1]-p0[1])*100:+.1f} pts")
        d3.metric(f"Gana {nombre(rival)}", f"{p1[2]:.1%}", f"{(p1[2]-p0[2])*100:+.1f} pts")
        st.caption(f"Elo {nombre(eq)}: {M['states'].loc[eq,'elo']:.0f} → {M['states'].loc[eq,'elo']+delta:.0f}")
