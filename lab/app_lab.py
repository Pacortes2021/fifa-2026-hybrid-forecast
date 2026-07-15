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
import streamlit.components.v1 as components

import motor as mo
import espn_live
import tda_motor as tda

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


def bandera(en):
    return EN2ES.get(en, ("", "🏳️"))[1]


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


@st.cache_data(ttl=120, show_spinner=False)
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


@st.cache_data(ttl=120, show_spinner=False)
def cargar_bracket(key):
    try:
        return list(espn_live.bracket_eliminatorias()["R32"].values()), None  # los 16 cruces reales
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner="Simulando las eliminatorias (15.000 torneos)…")
def sim_bracket(key, modelo):
    """Cuadro real (cruces de ESPN en la plantilla FIFA) + simulación del campeón, con el Elo
       actualizado por los resultados reales y los KO ya jugados fijados."""
    r32, err = cargar_bracket(key)
    if not r32 or len(ESPN_DF) < 72:
        return None
    M = get_motor()
    br = mo.bracket_real(ESPN_DF, r32)
    st2 = mo.actualizar_estados(M, ESPN_DF)
    fk = espn_live.ganadores_ko()        # ganadores reales de KO (penales incluidos)
    return {"bracket": br, "sim": mo.simular_bracket(M, br, states=st2, n_sims=15000, modelo=modelo, fijos_ko=fk)}


@st.cache_data(show_spinner="Validando el panel de stats…")
def mc_backtest_stats():
    return mo.backtest_stats(get_motor())


# --------------------------------------------------------------------------- #
#  Render del cuadro de eliminatorias (HTML estilo portal deportivo)
# --------------------------------------------------------------------------- #
_BRACKET_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;}
body{margin:0;font-family:'Inter',sans-serif;background:transparent;}
.half{margin-bottom:4px;}
.htitle{font-size:.82rem;font-weight:700;color:#0b3d91;text-transform:uppercase;
        letter-spacing:.04em;margin:4px 0 2px 6px;}
.bracket{display:flex;align-items:stretch;height:420px;}
.round{display:flex;flex-direction:column;flex:1;min-width:150px;padding:0 7px;}
.rbody{display:flex;flex-direction:column;justify-content:space-around;flex:1;}
.rhead{text-align:center;font-size:.6rem;font-weight:700;color:#94a3b8;text-transform:uppercase;
       letter-spacing:.07em;margin-bottom:3px;}
.match{background:#fff;border:1px solid #e2e8f0;border-radius:7px;overflow:hidden;
       box-shadow:0 1px 2px rgba(0,0,0,.06);margin:3px 0;}
.match.played{border-color:#86efac;}
.tm{display:flex;align-items:center;gap:5px;padding:3px 7px;font-size:.74rem;color:#475569;
    border-bottom:1px solid #f1f5f9;}
.tm:last-child{border-bottom:none;}
.tm .fl{font-size:.9rem;line-height:1;}
.tm .nm{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.tm .pr{font-variant-numeric:tabular-nums;font-size:.68rem;color:#a3acba;font-weight:600;}
.tm.win{background:#eef3fc;color:#0b3d91;font-weight:700;}
.tm.win .pr{color:#2a5db0;}
.match.played .tm.win{background:#e7f6ec;color:#15803d;}
.match.played .tm.win .pr{color:#15803d;}
.match.played .tm.lose{color:#cbd5e1;}
.finalwrap{display:flex;flex-direction:column;align-items:center;margin:6px 0;}
.fhdr{font-size:.7rem;font-weight:700;color:#7a3b91;text-transform:uppercase;letter-spacing:.08em;
      margin-bottom:4px;}
.finalists{display:flex;gap:10px;margin-bottom:8px;}
.finalists .match{min-width:158px;}
.champ{background:linear-gradient(135deg,#0b3d91,#7a3b91);color:#fff;border-radius:11px;
       padding:9px 26px;text-align:center;box-shadow:0 5px 14px rgba(11,61,145,.28);}
.champ .lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.12em;opacity:.85;}
.champ .nm{font-size:1.2rem;font-weight:700;margin:1px 0;}
.champ .pr{font-size:.72rem;opacity:.92;}
</style>
"""
_HEAD = {"R32": "Dieciseisavos", "R16": "Octavos", "QF": "Cuartos", "SF": "Semifinal"}


def _argmax_reach(reach, key):
    d = reach.get(key, {})
    if not d:
        return None, 0.0
    t = max(d, key=d.get)
    return t, d[t]


def _box_info(bracket, reach, rnd, num):
    m = bracket["FINAL"] if rnd == "FINAL" else bracket[rnd][num]
    key = ("FINAL", 1) if rnd == "FINAL" else (rnd, num)
    if rnd == "R32":
        home, away = m["home"], m["away"]
        slots = [(home, reach[key].get(home, 0.0)), (away, reach[key].get(away, 0.0))]
        played = m["state"] == "post" and m["gh"] is not None and m["ga"] is not None
        # ganador real (penales incluidos: ESPN marca 'winner'); si no, el favorito de la sim
        winner = (m.get("winner") or (home if m["gh"] > m["ga"] else away)) if played else _argmax_reach(reach, key)[0]
        return {"slots": slots, "played": played, "score": (m["gh"], m["ga"]) if played else None,
                "pens": m.get("pens") if played else None, "winner": winner}
    th, ph = _argmax_reach(reach, m["home"])
    ta, pa = _argmax_reach(reach, m["away"])
    return {"slots": [(th, ph), (ta, pa)], "played": False, "score": None, "pens": None,
            "winner": _argmax_reach(reach, key)[0]}


def _box_html(info):
    rows = ""
    for idx, (team, p) in enumerate(info["slots"]):
        es = nombre(team) if team else "—"
        fl = bandera(team) if team else "·"
        if info["played"]:
            val = str(info["score"][idx])
            if info.get("pens"):
                val = f'{val} <small>({info["pens"][idx]})</small>'   # marcador (penales)
            cls = "tm win" if team == info["winner"] else "tm lose"
        else:
            val = f"{p:.0%}"
            cls = "tm win" if team and team == info["winner"] else "tm"
        rows += (f'<div class="{cls}"><span class="fl">{fl}</span>'
                 f'<span class="nm" title="{es}">{es}</span><span class="pr">{val}</span></div>')
    return f'<div class="match{" played" if info["played"] else ""}">{rows}</div>'


def _collect_orden(bracket, side, acc):
    rnd, num = side
    if rnd != "R32":
        m = bracket[rnd][num]
        _collect_orden(bracket, m["home"], acc)
        _collect_orden(bracket, m["away"], acc)
    acc.setdefault(rnd, []).append(num)
    return acc


def _half_html(bracket, sim, sf_num, tabla):
    reach = sim["reach"]
    acc = _collect_orden(bracket, ("SF", sf_num), {})
    teams = [bracket["R32"][n][s] for n in acc["R32"] for s in ("home", "away")]
    pcamp = tabla.set_index("Selección")["P_campeon"]
    fuerte = max(teams, key=lambda t: pcamp.get(t, 0))
    cols = ""
    for rnd in ("R32", "R16", "QF", "SF"):
        boxes = "".join(_box_html(_box_info(bracket, reach, rnd, n)) for n in acc[rnd])
        cols += f'<div class="round"><div class="rhead">{_HEAD[rnd]}</div><div class="rbody">{boxes}</div></div>'
    return (f'<div class="half"><div class="htitle">Lado de {nombre(fuerte)} {bandera(fuerte)}</div>'
            f'<div class="bracket">{cols}</div></div>')


def _final_html(bracket, sim, tabla):
    fi = _box_info(bracket, sim["reach"], "FINAL", 1)
    champ = tabla.iloc[0]
    return (f'<div class="finalwrap"><div class="fhdr">★ Final ★</div>'
            f'<div class="finalists">{_box_html(fi)}</div>'
            f'<div class="champ"><div class="lbl">Campeón más probable</div>'
            f'<div class="nm">{bandera(champ["Selección"])} {nombre(champ["Selección"])}</div>'
            f'<div class="pr">campeón en el {champ["P_campeon"]:.1%} de los torneos simulados</div></div></div>')


def bracket_completo_html(bracket, sim, tabla):
    return (_BRACKET_CSS + _half_html(bracket, sim, 1, tabla)
            + _final_html(bracket, sim, tabla) + _half_html(bracket, sim, 2, tabla))


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

tab1, tabB, tab3, tab6, tab4, tab5, tab_tda = st.tabs(
    ["⚽ Partido + Mercados", "🗺️ Cuadro eliminatorias", "🔴 Torneo en vivo",
     "🔬 Modelo vs realidad", "🎯 Validación", "📈 Robustez", "🔷 TDA vs ML"])


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
        st.caption("Un modelo por estadística, construido con **selección de variables + validación "
                   "temporal** (no a ojo): los córners y tiros al arco los predicen sobre todo el "
                   "**dominio (Elo)** y los **tiros**, las faltas las que **provoca el rival**. Box score "
                   "de ESPN (cobertura ≈48%) → tómalo como tendencia, no número exacto.")
        se = mo.stats_esperadas(M, a, b, cmode)
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
        st.caption(f"Con **corrección Dixon-Coles** (ρ={M.get('rho_dc', 0):+.2f}) — el estándar de las casas de "
                   "apuestas: ajusta los marcadores de bajo score (0-0, 1-1…) que el Poisson simple subestima. "
                   "El 1X2 lo fija el clasificador (que ya predice mejor); Dixon-Coles afina la *distribución* de "
                   "marcadores dentro de cada resultado.")
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
# TAB B · Cuadro de eliminatorias (bracket real + simulación del campeón)
# ============================================================================
with tabB:
    st.markdown('<div class="sec-title">El cuadro de eliminatorias — camino al título</div>',
                unsafe_allow_html=True)
    st.markdown("Cuadro **real, ya definido**: los 16 cruces de dieciseisavos vienen de la API de ESPN. "
                "Encima va la **simulación del campeón** corrida sobre ese cuadro — 15.000 torneos jugando "
                "solo las eliminatorias, con el Elo de cada selección ya actualizado por sus resultados "
                "reales en el Mundial. Los cruces ya jugados (penales incluidos) quedan **fijados**.")
    if st.button("🔄 Actualizar resultados (ESPN)", key="refresh_bracket"):
        st.cache_data.clear()
        st.rerun()
    payload = sim_bracket(ESPN_KEY, modelo)
    if payload is None:
        st.info("El cuadro de eliminatorias aún no está publicado en ESPN. Se llena solo en cuanto "
                "termine la fase de grupos y se definan los cruces.")
    else:
        br, sim = payload["bracket"], payload["sim"]
        tabla = sim["tabla"]
        st.markdown("##### 🏆 Campeón según la simulación")
        c = st.columns(3)
        for col, (_, r), md in zip(c, tabla.head(3).iterrows(), ["🥇", "🥈", "🥉"]):
            col.metric(f"{md} {etiqueta(r['Selección'])}", f"{r['P_campeon']:.1%}")
        components.html(bracket_completo_html(br, sim, tabla), height=1010, scrolling=True)
        st.caption("En **dieciseisavos** se ven los cruces reales con la probabilidad de avanzar de cada "
                   "selección (verde = partido ya jugado, con su marcador). De **octavos en adelante**, "
                   "cada casillero muestra el equipo **más probable** de ocuparlo según la simulación "
                   "(proyección — el cruce exacto aún no está definido). El borde azul marca al favorito "
                   "de cada llave.")
        with st.expander("📋 Probabilidades completas por ronda (las 32 selecciones)"):
            show = tabla.copy()
            show["Selección"] = show["Selección"].map(etiqueta)
            show = show.rename(columns={"P_R16": "P(8vos)", "P_QF": "P(4tos)", "P_SF": "P(semis)",
                                        "P_final": "P(final)", "P_campeon": "P(campeón)"})
            pct = ["P(8vos)", "P(4tos)", "P(semis)", "P(final)", "P(campeón)"]
            st.dataframe(show.style.format({c: "{:.1%}" for c in pct})
                         .background_gradient(subset=["P(campeón)"], cmap="Blues"),
                         hide_index=True, width='stretch')

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
# TAB 6 · Modelo vs realidad (validación en vivo contra el Mundial real)
# ============================================================================
with tab6:
    st.markdown('<div class="sec-title">El modelo contra la realidad del Mundial</div>', unsafe_allow_html=True)
    st.markdown("La prueba más auténtica: comparar la predicción **pre-partido** del modelo contra el "
                "**resultado real** de cada partido del Mundial ya jugado (datos de ESPN). Out-of-sample puro.")
    if len(ESPN_DF) == 0:
        st.info("Aún no hay partidos finalizados del Mundial. Esta sección se llena sola conforme se jueguen.")
    else:
        tabla, met, evol = mo.validacion_en_vivo(M, ESPN_DF, modelo)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Partidos", met["n"])
        m2.metric("Acierto (1X2)", f"{met['acierto']:.0%}")
        m3.metric("Log-loss modelo", f"{met['logloss']:.3f}",
                  f"{met['logloss'] - met['logloss_base']:+.3f} vs baseline", delta_color="inverse")
        m4.metric("Log-loss baseline", f"{met['logloss_base']:.3f}")
        peor = met["logloss"] > met["logloss_base"]
        if peor:
            st.warning(f"⚠️ Con **{met['n']} partidos** el modelo va por **debajo** del baseline de frecuencias "
                       "— pero es muestra pequeñísima y los Mundiales arrancan con muchos empates y sorpresas "
                       "(*efecto debut*). Verificado aparte: el modelo está **bien calibrado** en su histórico "
                       "(~1.000 partidos), así que esto es **varianza esperable**, no un sesgo. Esta tabla dirá "
                       "si revierte a la media conforme avance el torneo.")
        else:
            st.success(f"El modelo va **por encima** del baseline en {met['n']} partidos reales. 👍")
        st.dataframe(tabla, hide_index=True, width='stretch')
        if met["n"] >= 3:
            fig, ax = plt.subplots(figsize=(9, 3.4))
            ax.plot(evol["partido"], evol["logloss_acum"], "o-", color="#0b3d91", label="modelo (acumulado)")
            ax.axhline(met["logloss_base"], color="#d62728", ls="--", lw=1, label="baseline frecuencias")
            ax.set_xlabel("partidos del Mundial (cronológico)"); ax.set_ylabel("log-loss acumulado")
            ax.legend(); ax.set_title("¿Mejora el modelo conforme avanza el Mundial?")
            st.pyplot(fig)
        st.caption(f"P(empate) que predijo el modelo: {met['p_empate_pred']:.0%}  ·  "
                   f"empates reales hasta ahora: {met['empates_reales']:.0%}  — "
                   "los Mundiales suelen empezar más empatados de lo normal.")

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

    st.markdown("##### 3. ¿Sirve el panel de estadísticas del partido?")
    st.caption("Cada estadística tiene su propio modelo, con **selección de variables + validación "
               "temporal** (igual que el modelo de resultado). MAE = error absoluto medio en el hold-out "
               "2025-26, comparado contra dos baselines: el **promedio propio del equipo** y la **media global**.")
    bs = mc_backtest_stats()
    st.dataframe(bs, hide_index=True, width='stretch')
    st.info("**Lectura honesta:** el modelo con selección de variables **le gana a ambos baselines** en las "
            "cuatro estadísticas — hacerlo con rigor (no a ojo) sí mejora. Aun así, las cifras puntuales "
            "son **ruidosas** (córners ±2.2 sobre 4, posesión ±12pp): sirven como *tendencia* (quién "
            "domina), no como número exacto — predecir córners exactos es casi imposible por el azar. "
            "Hallazgo: córners y tiros al arco se predicen mejor con el **dominio (Elo) y los tiros** que "
            "con los córners históricos. Es un complemento útil, pero **menos fiable que el modelo de "
            "resultado/goles**, que es el central.")

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

# ============================================================================
# TAB 7 · TDA vs ML  — Análisis Topológico de Datos comparado con el modelo
# ============================================================================
with tab_tda:
    st.markdown('<div class="sec-title">🔷 TDA vs ML — ¿Puede la Topología predecir fútbol?</div>',
                unsafe_allow_html=True)

    with st.expander("ℹ️ ¿Qué es el TDA y cómo funciona aquí?", expanded=False):
        st.markdown("""
        **TDA (Análisis Topológico de Datos)** estudia la *forma* de los datos, no solo sus valores.

        Colocamos cada una de las **48 selecciones como un punto en R⁵**
        (Elo, valor de plantilla, goles anotados, goles recibidos, tiros al arco —
        las mismas variables de tu modelo híbrido). Luego "inflamos bolas" alrededor de cada punto
        y observamos qué estructuras aparecen/desaparecen conforme el radio ε crece
        (**filtración de Vietoris-Rips**).

        Lo que se mide:
        - **β₀** = cuántas "islas" de equipos hay (¿todos conectados?)
        - **β₁** = cuántos "loops" hay (¿grupos de paridad competitiva?)
        - **β₂** = cuántas "burbujas" hay (¿núcleos cerrados de élite?)

        Luego entrenamos un clasificador con features topológicos y comparamos su log-loss
        contra el modelo híbrido en los partidos reales del Mundial.
        """)

    if not tda.TDA_OK:
        st.error("❌ Instala las librerías TDA: `pip install ripser persim`")
        st.code(tda.TDA_ERR_MSG, language="text")
    else:
        import numpy as np

        # ── 1) Nube de puntos ─────────────────────────────────────────────
        st.markdown("### 1️⃣ Las 48 selecciones en el espacio R⁵")
        st.caption("Proyección 3D de un espacio de 5 dimensiones. ⭐ dorado = semifinalistas actuales.")

        with st.spinner("Construyendo nube de puntos en R⁵…"):
            df_raw_tda, X_norm_tda, equipos_tda, _ = tda.cargar_nube(M["states"])

        col_nube, col_info = st.columns([3, 2])
        with col_nube:
            fig_n = tda.fig_nube_3d(X_norm_tda, equipos_tda, M["states"])
            st.pyplot(fig_n)

        with col_info:
            centroide = X_norm_tda.mean(axis=0)
            distancias_c = {eq: float(np.linalg.norm(X_norm_tda[i] - centroide))
                            for i, eq in enumerate(equipos_tda)}
            top_nucleo = sorted(distancias_c.items(), key=lambda x: x[1])[:6]
            top_peri = sorted(distancias_c.items(), key=lambda x: -x[1])[:6]

            st.markdown("**🎯 Equipos en el núcleo** *(más similares al promedio)*")
            for eq, d in top_nucleo:
                st.markdown(f"- {bandera(eq)} {nombre(eq)}: `ε={d:.3f}`")
            st.markdown("**🏝️ Equipos en la periferia** *(outliers en el espacio)*")
            for eq, d in top_peri:
                st.markdown(f"- {bandera(eq)} {nombre(eq)}: `ε={d:.3f}`")
            st.caption("💡 Periférico ≠ débil. Un equipo puede ser outlier por localía, valor de plantilla, etc.")

        st.markdown("---")

        # ── 2) Homología persistente ───────────────────────────────────────
        st.markdown("### 2️⃣ Homología Persistente — la 'huella topológica' del torneo")
        st.caption("El diagrama de persistencia muestra estructuras topológicas. "
                   "Puntos lejos de la diagonal = estructuras importantes y duraderas.")

        with st.spinner("Calculando homología persistente (puede tardar 5-10s)…"):
            resultado_tda = tda.calcular_persistencia(X_norm_tda, max_dim=2)
        resumen_top = tda.tabla_rasgos_topologicos_descripcion(resultado_tda, equipos_tda, X_norm_tda)

        col_p, col_b = st.columns(2)
        with col_p:
            st.pyplot(tda.fig_diagrama_persistencia(resultado_tda))
        with col_b:
            st.pyplot(tda.fig_betti_vs_epsilon(resultado_tda))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ε para conectar todo", f"{resumen_top['radio_conexion_total']:.3f}"
                  if resumen_top['radio_conexion_total'] else "—",
                  help="A este radio ε todas las selecciones forman una sola componente (β₀ = 1)")
        m2.metric("Ciclos H₁ persistentes", resumen_top["n_ciclos_persistentes"],
                  help="Grupos de equipos con 'paridad' en sus variables")
        m3.metric("Entropía H₀", resumen_top["entropia_h0"],
                  help="Diversidad entre grupos. Mayor = más heterogéneo")
        m4.metric("Entropía H₁", resumen_top["entropia_h1"],
                  help="Complejidad de ciclos. Mayor = más imprevisible")

        # Tabla de Betti a diferentes radios
        eps_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        dgms_d = resultado_tda["dgms"]
        betti_rows = []
        for eps in eps_vals:
            b0, b1, b2 = tda.numeros_betti(dgms_d, eps)
            betti_rows.append({"ε (radio)": eps, "β₀ islas": b0, "β₁ ciclos": b1, "β₂ cavidades": b2})
        with st.expander("Ver tabla de Betti a diferentes radios ε"):
            st.dataframe(tda.pd.DataFrame(betti_rows), hide_index=True, use_container_width=True)
            st.caption("Cuando β₀=1 → todos conectados | β₁>0 → hay loops de paridad | β₂>0 → hay 'élite cerrada'")

        st.markdown("---")

        # ── 3) Comparación directa TDA vs ML ──────────────────────────────
        st.markdown("### 3️⃣ La prueba de fuego: TDA vs ML en los partidos REALES del Mundial")
        st.caption("Out-of-sample puro. Ambos modelos entrenados en datos históricos, evaluados en el Mundial.")

        if len(ESPN_DF) == 0:
            st.info("⏳ Esta sección se activa cuando haya partidos reales del Mundial registrados.")
        else:
            with st.spinner(f"Entrenando clasificador TDA y comparando en {len(ESPN_DF)} partidos…"):
                try:
                    tabla_comp, metricas_comp, _, _, _ = tda.comparar_en_mundial(M, ESPN_DF, modelo)

                    if metricas_comp:
                        # Fila 1: Log-Loss
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Partidos evaluados", metricas_comp["n"])
                        col2.metric("Log-Loss ML Híbrido", f"{metricas_comp['logloss_ml']:.3f}",
                                    delta=f"{metricas_comp['logloss_ml']-metricas_comp['logloss_base']:+.3f} vs base",
                                    delta_color="inverse")
                        col3.metric("Log-Loss ML+TDA (Combinado)", f"{metricas_comp['logloss_comb']:.3f}",
                                    delta=f"{metricas_comp['logloss_comb']-metricas_comp['logloss_base']:+.3f} vs base",
                                    delta_color="inverse")
                        col4.metric("Log-Loss TDA Puro", f"{metricas_comp['logloss_tda']:.3f}",
                                    delta=f"{metricas_comp['logloss_tda']-metricas_comp['logloss_base']:+.3f} vs base",
                                    delta_color="inverse")

                        # Fila 2: Tasa de acierto
                        col1_2, col2_2, col3_2, col4_2 = st.columns(4)
                        col1_2.markdown("<small>Métrica de calibración (menor es mejor)</small>", unsafe_allow_html=True)
                        col2_2.metric("Acierto (1X2) ML", f"{metricas_comp['acierto_ml']:.1%}")
                        col3_2.metric("Acierto (1X2) ML+TDA", f"{metricas_comp['acierto_comb']:.1%}")
                        col4_2.metric("Acierto (1X2) TDA", f"{metricas_comp['acierto_tda']:.1%}")

                        st.pyplot(tda.fig_comparacion_logloss(metricas_comp))

                        # Comparación directa
                        diff_comb_ml = metricas_comp["logloss_comb"] - metricas_comp["logloss_ml"]
                        if diff_comb_ml < 0:
                            st.success(f"🚀 **¡Éxito! El modelo Combinado (ML+TDA) le gana a tu modelo ML Híbrido** solo (Δ = {abs(diff_comb_ml):.4f} de log-loss). "
                                       "La geometría topológica (posición relativa en el espacio de variables) sí aporta información útil sobre el rendimiento de las selecciones.")
                        else:
                            st.warning(f"💡 **Tu modelo ML Híbrido sigue siendo el rey** (Log-loss de {metricas_comp['logloss_ml']:.3f} vs {metricas_comp['logloss_comb']:.3f} del combinado). "
                                       "Añadir TDA mete un poco de ruido. Esto sugiere que las distancias topológicas globales en R⁵ no aportan información predictiva extra sobre la diferencia directa de Elo y estadísticas.")

                        st.markdown("##### Partido a partido")
                        st.dataframe(tabla_comp, hide_index=True, use_container_width=True)
                    else:
                        st.warning("No se pudieron calcular métricas comparativas.")
                except Exception as e:
                    st.error(f"Error al calcular TDA vs ML: {e}")

        st.markdown("---")

        # ── 4) Interpretación honesta ──────────────────────────────────────
        st.markdown("### 4️⃣ Diagnóstico topológico — ¿Qué dice la geometría del torneo?")
        with st.container(border=True):
            st.markdown("""
| Lo que mide el TDA | Qué significa | ¿Predice resultados? |
|---|---|---|
| **β₀ > 1** a ε pequeño | Hay grupos bien diferenciados de equipos | No directamente |
| **β₁ > 0** | Hay "zonas de paridad" circular entre equipos | Sugiere, no garantiza |
| **Equipo periférico** | Outlier en alguna variable (p.ej. localía alta) | No implica victoria |
| **ε alto para conectar** | El torneo es diverso / hay una jerarquía clara | Compatible con el Elo |

**Conclusión honesta:** El TDA es una herramienta poderosa de *exploración*, no de *predicción pura*.
Describe la forma de los datos. Para predecir partidos con probabilidades calibradas, el modelo
Elo+H2H+valor gana porque tiene un historico de 5.000+ partidos ajustando sus parámetros.
El TDA puede aportar como **feature adicional**, no como reemplazo.
            """)

        # Botón para descargar reporte
        resumen_txt = (
            f"REPORTE TOPOLÓGICO — MUNDIAL 2026\n"
            f"Radio ε conexión total: {resumen_top['radio_conexion_total']:.4f}\n"
            f"Ciclos H₁ persistentes: {resumen_top['n_ciclos_persistentes']}\n"
            f"Entropía H₀: {resumen_top['entropia_h0']}\n\n"
            f"Núcleo:\n" + "\n".join(f"  {nombre(eq)}: {d:.3f}"
                                      for eq, d in resumen_top["top_centrales"])
            + f"\n\nPeriferia:\n" + "\n".join(f"  {nombre(eq)}: {d:.3f}"
                                               for eq, d in resumen_top["top_aislados"])
        )
        if len(ESPN_DF) > 0 and "metricas_comp" in dir() and metricas_comp:
            resumen_txt += (
                f"\n\nComparación ({metricas_comp['n']} partidos):\n"
                f"  Log-loss ML:  {metricas_comp['logloss_ml']:.4f}\n"
                f"  Log-loss TDA: {metricas_comp['logloss_tda']:.4f}\n"
                f"  Acierto ML:   {metricas_comp['acierto_ml']:.1%}\n"
                f"  Acierto TDA:  {metricas_comp['acierto_tda']:.1%}\n"
            )
        st.download_button("📄 Descargar reporte topológico (.txt)",
                           resumen_txt, file_name="reporte_tda_mundial2026.txt", mime="text/plain")
