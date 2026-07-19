"""
Interfaz de visualización para la Copa Mundial de la FIFA 2026.
Se integra como un módulo dentro del Portal Maestro.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# Asegurar importación de motores locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor as mo
import espn_live

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
def mc_vivo(modelo, key, ESPN_DF):
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
        return list(espn_live.bracket_eliminatorias()["R32"].values()), None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner="Simulando las eliminatorias (15.000 torneos)…")
def sim_bracket(key, modelo, ESPN_DF):
    r32, err = cargar_bracket(key)
    if not r32 or len(ESPN_DF) < 72:
        return None
    M = get_motor()
    br = mo.bracket_real(ESPN_DF, r32)
    st2 = mo.actualizar_estados(M, ESPN_DF)
    fk = espn_live.ganadores_ko()
    return {"bracket": br, "sim": mo.simular_bracket(M, br, states=st2, n_sims=15000, modelo=modelo, fijos_ko=fk)}


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
                val = f'{val} <small>({info["pens"][idx]})</small>'
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


def run_app():
    M = get_motor()
    ESPN_DF, ESPN_ERR = cargar_espn()
    ESPN_KEY = "" if len(ESPN_DF) == 0 else f"{len(ESPN_DF)}-{ESPN_DF.fecha.max()}"
    
    st.markdown('<div class="main-title">🏆 Simulador de la Copa Mundial 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Predicción Versus · Cruces de eliminatorias reales · Simulación de campeonato en vivo</div>', unsafe_allow_html=True)
    
    modelo = st.sidebar.radio(
        "Modelo de predicción", 
        ["base", "hyb", "two_stage"],
        format_func=lambda x: "Base (Elo + H2H + valor)" if x == "base" else ("Híbrido (+ forma reciente)" if x == "hyb" else "Híbrido 2 Etapas (Táctico)"),
        key="wm_model"
    )
    st.sidebar.info("Todos los modelos usan **ponderación K-factor**: los partidos oficiales pesan más que los amistosos.")
    if len(ESPN_DF):
        st.sidebar.success(f"🛰️ ESPN: {len(ESPN_DF)} partidos reales cargados.")
        
    tab1, tabB, tab3 = st.tabs([
        "⚽ Partido + Mercados", 
        "🗺️ Cuadro de eliminatorias", 
        "🔴 Torneo en vivo"
    ])
    
    with tab1:
        c1, cvs, c2 = st.columns([5, 1, 5])
        with c1:
            a = es2en(st.selectbox("Selección 1", OPC, index=OPC.index("🇪🇸 España"), key="m_a"))
        with cvs:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        with c2:
            b = es2en(st.selectbox("Selección 2", OPC, index=OPC.index("🇦🇷 Argentina"), key="m_b"))
        
        cancha = st.radio("Cancha", ["Automática (anfitrión de local)", "Neutral",
                                     f"Local {nombre(a)}", f"Local {nombre(b)}"], horizontal=True, key="wm_cancha")
        cmode = {"Automática (anfitrión de local)": "auto", "Neutral": "neutral",
                 f"Local {nombre(a)}": "1", f"Local {nombre(b)}": "2"}.get(cancha, "auto")

        if a == b:
            st.error("Elige dos selecciones distintas.")
        else:
            na, nb = nombre(a), nombre(b)
            
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

            col_b, col_h, col_ts = st.columns(3)
            with col_b:
                mix_b, p_b = tarjeta("base", "🔵 Modelo Base (Elo + H2H + valor)", "card-title-base")
            with col_h:
                mix_h, p_h = tarjeta("hyb", "🟣 Modelo Híbrido (Base + forma)", "card-title-hybrid")
            with col_ts:
                mix_ts, p_ts = tarjeta("two_stage", "🟢 Híbrido 2 Etapas (Táctico)", "card-title-two-stage")

            lbl_mod = "Base" if modelo == "base" else ("Híbrido" if modelo == "hyb" else "Híbrido 2 Etapas")
            st.markdown(f'<div class="sec-title">Mercados — modelo {lbl_mod} (probabilidad y cuota justa)</div>', unsafe_allow_html=True)
            
            mix = mix_b if modelo == "base" else (mix_h if modelo == "hyb" else mix_ts)
            p = p_b if modelo == "base" else (p_h if modelo == "hyb" else p_ts)
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
            filas.append({"Mercado": f"Sin Empate: {na} (DNB 1)", "Prob.": f"{p_dnb1:.1%}", "Cuota justa": f"{mo.cuota(p_dnb1):.2f}"})
            filas.append({"Mercado": f"Sin Empate: {nb} (DNB 2)", "Prob.": f"{p_dnb2:.1%}", "Cuota justa": f"{mo.cuota(p_dnb2):.2f}"})
                
            mc1, mc2 = st.columns(2)
            mc1.dataframe(pd.DataFrame(filas[:8]), hide_index=True, width='stretch')
            mc2.dataframe(pd.DataFrame(filas[8:]), hide_index=True, width='stretch')
            st.markdown("**Marcadores más probables:** " + " · ".join(
                f"`{i}-{j} ({pr:.0%}, cuota {mo.cuota(pr):.1f})`" for i, j, pr in mk["_top_marcadores"][:4]))

            st.markdown('<div class="sec-title">Estadísticas esperadas del partido</div>', unsafe_allow_html=True)
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
                    for ln, (po, pu) in ou.items():
                        filas_ou.append({"Mercado": f"{nm_st} Over {ln}", "Prob.": f"{po:.0%}",
                                         "Cuota justa": f"{mo.cuota(po):.2f}"})
                st.dataframe(pd.DataFrame(filas_ou), hide_index=True, width='stretch')
                st.caption(f"Totales esperados — córners: **{sum(se['corners']):.1f}**  ·  "
                           f"tiros al arco: **{sum(se['tiros_arco']):.1f}**  ·  faltas: **{sum(se['faltas']):.1f}**")

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

    with tabB:
        st.markdown('<div class="sec-title">El cuadro de eliminatorias — camino al título</div>',
                    unsafe_allow_html=True)
        st.markdown("Cuadro **real, ya definido**: los 16 cruces de dieciseisavos vienen de la API de ESPN. "
                    "Encima va la **simulación del campeón** corrida sobre ese cuadro — 15.000 torneos jugando "
                    "solo las eliminatorias.")
        payload = sim_bracket(ESPN_KEY, modelo, ESPN_DF)
        if payload is None:
            st.info("El cuadro de eliminatorias aún no está publicado en ESPN.")
        else:
            br, sim = payload["bracket"], payload["sim"]
            tabla = sim["tabla"]
            st.markdown("##### 🏆 Campeón según la simulación")
            c = st.columns(3)
            for col, (_, r), md in zip(c, tabla.head(3).iterrows(), ["🥇", "🥈", "🥉"]):
                col.metric(f"{md} {etiqueta(r['Selección'])}", f"{r['P_campeon']:.1%}")
            components.html(bracket_completo_html(br, sim, tabla), height=1010, scrolling=True)
            with st.expander("📋 Probabilidades completas por ronda (las 32 selecciones)"):
                show = tabla.copy()
                show["Selección"] = show["Selección"].map(etiqueta)
                show = show.rename(columns={"P_R16": "P(8vos)", "P_QF": "P(4tos)", "P_SF": "P(semis)",
                                             "P_final": "P(final)", "P_campeon": "P(campeón)"})
                pct = ["P(8vos)", "P(4tos)", "P(semis)", "P(final)", "P(campeón)"]
                st.dataframe(show.style.format({c: "{:.1%}" for c in pct})
                             .background_gradient(subset=["P(campeón)"], cmap="Blues"),
                             hide_index=True, width='stretch')

    with tab3:
        st.markdown('<div class="sec-title">Modelo vivo: resultados reales → re-simulación</div>', unsafe_allow_html=True)
        
        fuente = st.radio("Fuente de resultados", ["🛰️ Automática (ESPN)", "✍️ Manual"], horizontal=True, key="wm_fuente")
        if fuente.startswith("🛰️"):
            if ESPN_ERR:
                st.error(f"No se pudo contactar a ESPN ({ESPN_ERR}). Usa el modo manual.")
                res = pd.DataFrame()
            elif len(ESPN_DF) == 0:
                st.info("ESPN aún no reporta partidos finalizados.")
                res = pd.DataFrame()
            else:
                res = ESPN_DF.copy()
                st.success(f"{len(res)} partidos finalizados traídos de ESPN.")
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
                plantilla, num_rows="dynamic", width='stretch', key="wm_vivo",
                column_config={
                    "local": st.column_config.SelectboxColumn("Local", options=mo.MUNDIALISTAS, required=True),
                    "visita": st.column_config.SelectboxColumn("Visita", options=mo.MUNDIALISTAS, required=True),
                    "goles_local": st.column_config.NumberColumn("Goles local", min_value=0, max_value=15, step=1),
                    "goles_visita": st.column_config.NumberColumn("Goles visita", min_value=0, max_value=15, step=1)})
            res = edit.dropna(subset=["local", "visita", "goles_local", "goles_visita"])
            res = res[res.local != res.visita]
            if len(res):
                st.success(f"{len(res)} resultado(s) cargado(s).")
                
        if st.button("🔄 Actualizar y re-simular (4.000 mundiales)", type="primary", key="wm_resim"):
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
