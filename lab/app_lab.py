"""
Interfaz de visualización para la Copa Mundial de la FIFA 2026.
Se integra como un módulo dentro del Portal Maestro.
"""
import os
import sys
from pathlib import Path
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


@st.cache_data(show_spinner=False)
def cargar_proyecciones_torneo(modelo):
    """Carga las probabilidades del torneo simulado por Monte Carlo precalculado."""
    OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
    if modelo == "hyb" and (OUTPUTS / "probabilidades_torneo_hibrido.csv").exists():
        return pd.read_csv(OUTPUTS / "probabilidades_torneo_hibrido.csv")
    elif (OUTPUTS / "probabilidades_torneo.csv").exists():
        return pd.read_csv(OUTPUTS / "probabilidades_torneo.csv")
    return mc_base(modelo)


@st.cache_data(show_spinner=False)
def cargar_predicciones_grupos(modelo):
    """Carga las predicciones de los 72 partidos de fase de grupos."""
    OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
    if modelo == "hyb" and (OUTPUTS / "predicciones_fase_grupos_hibrido.csv").exists():
        return pd.read_csv(OUTPUTS / "predicciones_fase_grupos_hibrido.csv")
    elif (OUTPUTS / "predicciones_fase_grupos.csv").exists():
        return pd.read_csv(OUTPUTS / "predicciones_fase_grupos.csv")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def cargar_proyecciones_bracket_real():
    """Carga la simulación del bracket de eliminatorias oficial."""
    OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
    if (OUTPUTS / "probabilidades_campeon_bracket_real.csv").exists():
        return pd.read_csv(OUTPUTS / "probabilidades_campeon_bracket_real.csv")
    return pd.DataFrame()


@st.cache_data(show_spinner="Evaluando modelo en hold-out temporal...")
def get_backtest_holdout(modelo):
    M = get_motor()
    mod_id = "base" if modelo == "base" else "hyb"
    return mo.backtest_test(M, mod_id)


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
        
    tab1, tab2, tabB, tab3, tab4 = st.tabs([
        "⚽ Partido + Mercados", 
        "📊 Proyecciones y Grupos",
        "🗺️ Cuadro de eliminatorias", 
        "🔴 Torneo en vivo",
        "🎯 Validación vs Realidad"
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

    with tab2:
        st.markdown('<div class="sec-title">Proyecciones del Torneo y Fase de Grupos</div>', unsafe_allow_html=True)
        st.markdown(
            "Registro cuantitativo de **10.000 simulaciones Monte Carlo** del torneo completo, "
            "probabilidades de título y rondas eliminatorias para las 48 selecciones, junto a las predicciones de los 72 partidos de fase de grupos."
        )

        df_proy_raw = cargar_proyecciones_torneo(modelo)
        df_proy = df_proy_raw.copy()
        if "elo" not in df_proy.columns and "states" in M:
            df_proy["elo"] = df_proy["Selección"].map(M["states"]["elo"]).fillna(1500).astype(int)

        top_teams = df_proy.sort_values("P_campeon", ascending=False).reset_index(drop=True)

        # Tarjetas de resumen métrico
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        if len(top_teams) >= 4:
            m_col1.metric("🥇 Máximo Favorito", f"{etiqueta(top_teams.loc[0, 'Selección'])}", f"{top_teams.loc[0, 'P_campeon']:.1%}")
            m_col2.metric("🥈 Segundo Favorito", f"{etiqueta(top_teams.loc[1, 'Selección'])}", f"{top_teams.loc[1, 'P_campeon']:.1%}")
            m_col3.metric("🥉 Tercer Favorito", f"{etiqueta(top_teams.loc[2, 'Selección'])}", f"{top_teams.loc[2, 'P_campeon']:.1%}")
            m_col4.metric("⭐ Cuarto Candidato", f"{etiqueta(top_teams.loc[3, 'Selección'])}", f"{top_teams.loc[3, 'P_campeon']:.1%}")

        st.markdown("---")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "🏆 Probabilidades del Torneo (48 Selecciones)",
            "🌐 Fase de Grupos (Grupos A a L)",
            "📅 Pronóstico de Partidos (72 Encuentros)"
        ])

        with sub_tab1:
            col_chart, col_tbl = st.columns([5, 6])
            with col_chart:
                st.markdown("##### 📊 Top 12 Candidatos al Título (IC 95%)")
                top12 = top_teams.head(12).copy()
                los, his = zip(*[mo.ic_montecarlo(p, 10000) for p in top12["P_campeon"]])
                fig_bar, ax_bar = plt.subplots(figsize=(6, 6))
                yp = np.arange(len(top12))[::-1]
                err_low = (top12["P_campeon"] - np.array(los)) * 100
                err_high = (np.array(his) - top12["P_campeon"]) * 100
                ax_bar.barh(yp, top12["P_campeon"] * 100, color="#0b3d91", alpha=0.85,
                            xerr=[err_low, err_high], capsize=3, ecolor="#d62728")
                ax_bar.set_yticks(yp)
                ax_bar.set_yticklabels([etiqueta(t) for t in top12["Selección"]], fontsize=9)
                ax_bar.set_xlabel("P(Campeón) %", fontsize=9)
                ax_bar.set_title("Probabilidad de Campeón ± Intervalo 95%", fontsize=10, fontweight="bold")
                ax_bar.grid(axis="x", ls=":", alpha=0.6)
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with col_tbl:
                st.markdown("##### 📋 Registro Completo de Probabilidades")
                df_disp = top_teams.copy()
                df_disp["Equipo"] = df_disp["Selección"].map(etiqueta)
                df_disp["ELO"] = df_disp["elo"]
                cols_show = ["Equipo", "grupo", "ELO", "P_campeon", "P_final", "P_semi", "P_octavos"]
                df_disp = df_disp[[c for c in cols_show if c in df_disp.columns]]
                df_disp = df_disp.rename(columns={
                    "grupo": "Grupo",
                    "P_campeon": "🏆 P(Campeón)",
                    "P_final": "🥈 P(Final)",
                    "P_semi": "🥉 P(Semis)",
                    "P_octavos": "⚔️ P(Avanzar)"
                })
                pct_cols = [c for c in ["🏆 P(Campeón)", "🥈 P(Final)", "🥉 P(Semis)", "⚔️ P(Avanzar)"] if c in df_disp.columns]
                st.dataframe(
                    df_disp.style.format({c: "{:.1%}" for c in pct_cols})
                           .background_gradient(subset=["🏆 P(Campeón)"], cmap="YlOrRd"),
                    hide_index=True, width='stretch', height=480
                )

        with sub_tab2:
            st.markdown("##### 🌐 Estructura de Grupos y Chances de Clasificación")
            st.caption("P(Avanzar) refleja la probabilidad de clasificar a Dieciseisavos (1º, 2º o los 8 mejores terceros).")
            dict_oct = top_teams.set_index("Selección")["P_octavos"].to_dict()
            letras = list(mo.GRUPOS.keys())
            for fila in range(0, 12, 3):
                cols_g = st.columns(3)
                for k, g in enumerate(letras[fila:fila + 3]):
                    with cols_g[k]:
                        with st.container(border=True):
                            st.markdown(f'<div class="card-title-base">Grupo {g}</div>', unsafe_allow_html=True)
                            filas_g = []
                            for eq in mo.GRUPOS[g]:
                                elo_eq = int(M["states"].loc[eq, "elo"]) if eq in M["states"].index else 1500
                                p_adv = dict_oct.get(eq, 0.0)
                                filas_g.append({
                                    "Selección": etiqueta(eq),
                                    "ELO": elo_eq,
                                    "P(Avanzar)": p_adv
                                })
                            df_g = pd.DataFrame(filas_g).sort_values("P(Avanzar)", ascending=False).reset_index(drop=True)
                            df_g.insert(0, "Pos Proy", df_g.index + 1)
                            st.dataframe(
                                df_g.style.format({"P(Avanzar)": "{:.1%}"})
                                          .background_gradient(subset=["P(Avanzar)"], cmap="Greens"),
                                hide_index=True, width='stretch'
                            )

        with sub_tab3:
            st.markdown("##### 📅 Pronóstico de los 72 Partidos de Fase de Grupos")
            st.caption("Probabilidades pre-partido calculadas con el motor Dixon-Coles y features avanzadas.")
            df_partidos_fg = cargar_predicciones_grupos(modelo)
            if not df_partidos_fg.empty:
                col_filtro1, col_filtro2 = st.columns([3, 4])
                with col_filtro1:
                    grupos_unicos = ["Todos"] + sorted(list(df_partidos_fg["grupo"].unique()))
                    sel_grp = st.selectbox("Filtrar por Grupo", grupos_unicos, key="fg_grp_sel")
                with col_filtro2:
                    todas_sel = ["Todas"] + OPC
                    sel_team_f = st.selectbox("Filtrar por Selección", todas_sel, key="fg_team_sel")

                df_fg_show = df_partidos_fg.copy()
                if sel_grp != "Todos":
                    df_fg_show = df_fg_show[df_fg_show["grupo"] == sel_grp]
                if sel_team_f != "Todas":
                    en_sel = es2en(sel_team_f)
                    df_fg_show = df_fg_show[(df_fg_show["equipo_1"] == en_sel) | (df_fg_show["equipo_2"] == en_sel)]

                df_fg_show["Local"] = df_fg_show["equipo_1"].map(etiqueta)
                df_fg_show["Visita"] = df_fg_show["equipo_2"].map(etiqueta)
                df_fg_show["xG Local"] = df_fg_show["goles_esp_1"].map(lambda x: f"{x:.2f}")
                df_fg_show["xG Visita"] = df_fg_show["goles_esp_2"].map(lambda x: f"{x:.2f}")
                df_fg_show = df_fg_show.rename(columns={
                    "grupo": "Grupo",
                    "cancha": "Cancha",
                    "P(gana 1)": "P(Local)",
                    "P(empate)": "P(Empate)",
                    "P(gana 2)": "P(Visita)"
                })
                cols_fg = ["Grupo", "Local", "xG Local", "P(Local)", "P(Empate)", "P(Visita)", "xG Visita", "Visita", "Cancha"]
                st.dataframe(
                    df_fg_show[[c for c in cols_fg if c in df_fg_show.columns]]
                        .style.format({"P(Local)": "{:.1%}", "P(Empate)": "{:.1%}", "P(Visita)": "{:.1%}"})
                        .background_gradient(subset=["P(Local)"], cmap="Blues")
                        .background_gradient(subset=["P(Visita)"], cmap="Purples"),
                    hide_index=True, width='stretch', height=450
                )

    with tabB:
        st.markdown('<div class="sec-title">El cuadro de eliminatorias — camino al título</div>',
                    unsafe_allow_html=True)
        st.markdown("Cuadro **real, ya definido**: los 16 cruces de dieciseisavos vienen de la API de ESPN. "
                    "Encima va la **simulación del campeón** corrida sobre ese cuadro — 15.000 torneos jugando "
                    "solo las eliminatorias.")
        payload = sim_bracket(ESPN_KEY, modelo, ESPN_DF)
        if payload is None:
            st.info("ℹ️ El cuadro oficial de eliminatorias de ESPN se activará automáticamente una vez concluyan los 72 partidos de la Fase de Grupos.")
            df_br = cargar_proyecciones_bracket_real()
            if not df_br.empty:
                st.markdown("##### 🏆 Proyección Pre-Torneo del Cuadro de Eliminatorias")
                st.caption("Simulación de 20.000 torneos jugando las eliminatorias con la estructura oficial del cuadro (FIFA 2026):")
                df_br_show = df_br.copy()
                df_br_show["Selección"] = df_br_show["Selección"].map(etiqueta)
                df_br_show = df_br_show.rename(columns={"P_campeon": "🏆 P(Campeón en Cuadro)"})
                st.dataframe(
                    df_br_show.head(16).style.format({"🏆 P(Campeón en Cuadro)": "{:.1%}"})
                                             .background_gradient(subset=["🏆 P(Campeón en Cuadro)"], cmap="Blues"),
                    hide_index=True, width='stretch'
                )
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

    with tab4:
        st.markdown('<div class="sec-title">El Modelo contra la Realidad</div>', unsafe_allow_html=True)
        
        if len(ESPN_DF) > 0:
            st.markdown("Compara las predicciones pre-partido del modelo con los resultados reales del torneo recopilados de ESPN.")
            df_val, met, evol = mo.validacion_en_vivo(M, ESPN_DF, modelo)
            
            if len(df_val) == 0:
                st.info("No hay partidos jugados por selecciones mundialistas válidas aún.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Partidos Jugados", met["n"])
                m2.metric("Acierto (1X2)", f"{met['acierto']:.1%}")
                m3.metric("Log-loss modelo", f"{met['logloss']:.3f}", 
                          f"{met['logloss'] - met['logloss_base']:+.3f} vs baseline", delta_color="inverse")
                m4.metric("Log-loss baseline", f"{met['logloss_base']:.3f}")
                
                if met["logloss"] < met["logloss_base"]:
                    st.success(f"El modelo va **por encima** del baseline en {met['n']} partidos del Mundial. 👍")
                else:
                    st.warning(f"⚠️ El modelo va por debajo del baseline.")
                    
                st.markdown("##### Historial de Predicciones del Mundial")
                st.dataframe(df_val, hide_index=True, width='stretch')
                
                c_plot, c_table = st.columns([6, 4])
                with c_plot:
                    if met["n"] >= 3:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(evol["partido"], evol["logloss_acum"], "o-", color="#0b3d91", label="Modelo (acumulado)")
                        ax.axhline(evol["baseline"].iloc[0], color="#dc2626", ls="--", label="Baseline")
                        ax.set_xlabel("Partidos jugados (cronológico)")
                        ax.set_ylabel("Log-loss acumulado")
                        ax.legend()
                        st.pyplot(fig)
                        
                with c_table:
                    st.markdown("##### % Acierto por Selección")
                    team_stats = []
                    equipos = set(df_val["Local"]).union(set(df_val["Visita"]))
                    for eq in equipos:
                        df_eq = df_val[(df_val["Local"] == eq) | (df_val["Visita"] == eq)]
                        if len(df_eq) > 0:
                            aciertos = (df_eq["Acierto"] == "✅").sum()
                            team_stats.append({
                                "Selección": eq,
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
        else:
            st.info("🛰️ El Mundial 2026 aún no comienza en ESPN. A continuación se presenta el **registro empírico de validación out-of-sample** obtenido en el conjunto de prueba temporal (partidos internacionales 2025–2026, nunca vistos por el entrenamiento del modelo).")
            
            P_val, y_val, te_val = get_backtest_holdout(modelo)
            n_val = len(y_val)
            pred_val = P_val.argmax(axis=1)
            aciertos_val = int((pred_val == y_val).sum())
            acc_val = aciertos_val / n_val
            from sklearn.metrics import log_loss
            ll_val = float(log_loss(y_val, P_val, labels=[0, 1, 2]))
            base_val = np.tile([0.279, 0.275, 0.446], (n_val, 1))
            ll_base_val = float(log_loss(y_val, base_val, labels=[0, 1, 2]))
            
            vm1, vm2, vm3, vm4 = st.columns(4)
            vm1.metric("Partidos Evaluados (OOS)", f"{n_val:,}")
            vm2.metric("Tasa de Acierto (1X2)", f"{acc_val:.1%}")
            vm3.metric("Log-Loss Modelo", f"{ll_val:.3f}", f"{ll_val - ll_base_val:+.3f} vs baseline", delta_color="inverse")
            vm4.metric("Log-Loss Baseline", f"{ll_base_val:.3f}")
            
            if ll_val < ll_base_val:
                st.success(f"🏆 El modelo supera con creces al baseline ingenuo en {n_val:,} partidos de selecciones (ganancia de {ll_base_val - ll_val:.3f} pts de log-loss y {acc_val:.1%} de acierto).")
            
            st.markdown("##### 📋 Historial de Validación Out-of-Sample (Hold-out 2025–2026)")
            te_show = pd.DataFrame({
                "Fecha": pd.to_datetime(te_val["fecha"]).dt.strftime("%Y-%m-%d"),
                "Competición": te_val["competicion"],
                "Local": te_val["local"].map(etiqueta),
                "Visita": te_val["visita"].map(etiqueta),
                "Resultado Real": [mo.ETIQUETAS.get(y, "—") for y in y_val],
                "Predicción": [mo.ETIQUETAS.get(p, "—") for p in pred_val],
                "P(Local)": [f"{P_val[i, 2]:.0%}" for i in range(n_val)],
                "P(Empate)": [f"{P_val[i, 1]:.0%}" for i in range(n_val)],
                "P(Visita)": [f"{P_val[i, 0]:.0%}" for i in range(n_val)],
                "Acierto": ["✅" if pred_val[i] == y_val[i] else "❌" for i in range(n_val)]
            })
            st.dataframe(te_show, hide_index=True, width='stretch', height=350)
            
            st.markdown("##### 📈 Curvas de Calibración de Probabilidades")
            st.caption("Si el modelo predice 60% de probabilidad, el evento debe ocurrir el 60% de las veces. La curva debe alinearse con la diagonal ideal.")
            c_cal1, c_cal2, c_cal3 = st.columns(3)
            cols_cal = [c_cal1, c_cal2, c_cal3]
            for k, (clase_idx, clase_nom) in enumerate([(2, "Victoria Local"), (1, "Empate"), (0, "Victoria Visita")]):
                xs, ys, ns, ece = mo.curva_calibracion(P_val, y_val, clase_idx)
                with cols_cal[k]:
                    fig_c, ax_c = plt.subplots(figsize=(4, 3.5))
                    ax_c.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Ideal")
                    ax_c.plot(xs, ys, "o-", color="#0b3d91" if clase_idx==2 else ("#7a3b91" if clase_idx==1 else "#2a9d5c"), lw=2, label="Modelo")
                    ax_c.set_title(f"{clase_nom} (ECE={ece:.3f})", fontsize=10, fontweight="bold")
                    ax_c.set_xlabel("Probabilidad Predicha", fontsize=8)
                    ax_c.set_ylabel("Frecuencia Observada", fontsize=8)
                    ax_c.set_xlim(0, 1)
                    ax_c.set_ylim(0, 1)
                    ax_c.grid(True, ls=":", alpha=0.5)
                    ax_c.legend(fontsize=8)
                    st.pyplot(fig_c)
                    plt.close(fig_c)
