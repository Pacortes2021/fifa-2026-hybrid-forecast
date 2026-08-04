"""
Aplicación Streamlit para la simulación y predicciones de la Bundesliga (Alemania).
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

# Emojis e identidades visuales de los clubes ingleses
TEAM_DETAILS = {
    "Bayern Munich": {"flag": "🔴", "color": "#dc052d"},
    "RB Leipzig": {"flag": "🔴⚪", "color": "#dd0741"},
    "Bayer Leverkusen": {"flag": "⚫🔴", "color": "#e32221"},
    "Borussia Dortmund": {"flag": "🟡⚫", "color": "#fde100"},
    "VfB Stuttgart": {"flag": "🔴⚪", "color": "#e32219"},
    "Eintracht Frankfurt": {"flag": "⚫🔴", "color": "#e1000f"},
    "TSG Hoffenheim": {"flag": "🔵", "color": "#1c63b7"},
    "SC Freiburg": {"flag": "⚫🔴", "color": "#a6122d"},
    "Mainz": {"flag": "🔴⚪", "color": "#c3141f"},
    "FC Augsburg": {"flag": "🟢⚪", "color": "#ba3733"},
    "Werder Bremen": {"flag": "🟢⚪", "color": "#1d9053"},
    "Borussia Mönchengladbach": {"flag": "⚫🟢", "color": "#000000"},
    "FC Cologne": {"flag": "🔴⚪", "color": "#ed1c24"},
    "1. FC Union Berlin": {"flag": "🔴", "color": "#e30613"},
    "Hamburg SV": {"flag": "🔵⚪", "color": "#00549a"},
    "Schalke 04": {"flag": "🔵⚪", "color": "#004d9d"},
    "SV Elversberg": {"flag": "⚫", "color": "#231f20"},
    "SC Paderborn 07": {"flag": "⚫🔵", "color": "#005ca9"},
    "1. FC Heidenheim 1846": {"flag": "🔴🔵", "color": "#e30613"},
    "Holstein Kiel": {"flag": "🔵⚪", "color": "#00377e"},
    "VfL Bochum": {"flag": "🔵", "color": "#005ca9"},
    "SV Darmstadt 98": {"flag": "🔵", "color": "#00519e"},
    "SpVgg Greuther Fürth": {"flag": "🟢⚪", "color": "#00843d"},
    "Arminia Bielefeld": {"flag": "🔵⚫", "color": "#21468b"},
    "Hertha Berlin": {"flag": "🔵⚪", "color": "#1351a3"},
    "St. Pauli": {"flag": "🟤", "color": "#5b3416"},
    "VfL Wolfsburg": {"flag": "🟢⚪", "color": "#65b32e"}
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
    """Devuelve el <img> del escudo del club desde data/equipos.csv (ID estable de ESPN)."""
    url = logo_url(equipos, team)
    if not url:
        return ""
    return f'<img src="{url}" width="{size}" style="border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.15);">'


def fmt_opcion(equipos, team):
    """Etiqueta de opcion del selectbox: abreviatura del club + nombre.
    Streamlit renderiza las opciones como texto plano (no admite HTML/emojis)."""
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
    flag = info.get("flag_emoji", info.get("flag", "⚽"))
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
        ["🌲 Random Forest (Recomendado)", "📐 LASSO L1 (Regresión)", "🔀 Stacking (Ensemble óptimo)"],
        index=0
    )
    if "LASSO" in modelo_sel:
        modelo_tipo = "lasso"
    elif "Stacking" in modelo_sel:
        modelo_tipo = "stacking"
    elif "XGB" in modelo_sel:
        modelo_tipo = "xgb"
    else:
        modelo_tipo = "rf"
    
    # Botón para actualizar resultados en vivo
    if st.sidebar.button("🔄 Actualizar ESPN y Re-entrenar", type="primary"):
        with st.spinner("Descargando últimos resultados de ESPN..."):
            rec.recolectar()
            rec_box.recolectar()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    M = get_motor()
    equipos = M.get("equipos", {})
    try:
        _partidos_df = pd.read_csv(mo.DATA / "partidos.csv", parse_dates=["fecha"])
        _ult_fecha = pd.to_datetime(_partidos_df["fecha"].max()).date().strftime("%d/%m/%Y")
        st.sidebar.caption(f"🗓️ Datos actualizados: {_ult_fecha} · {len(_partidos_df)} partidos")
    except Exception:
        pass

    # ── Métricas por modelo (sidebar)
    met_all = M.get("metricas", {})
    if met_all:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📊 Métricas Out-of-Sample (2025+)")
        mejor_ll = min(met_all[k]["logloss"] for k in met_all)
        for nombre, clave in [("LASSO", "lasso"), ("RF", "rf"), ("XGB", "xgb"), ("Stacking", "stacking")]:
            if clave not in met_all: continue
            m = met_all[clave]
            star = " ⭐" if m["logloss"] == mejor_ll else ""
            w_str = f" (w={m['w']})" if clave == "stacking" and "w" in m else ""
            st.sidebar.caption(f"**{nombre}{w_str}{star}** — LL: `{m['logloss']:.4f}` | Acc: `{m['accuracy']:.1f}%`")

    # Encabezado
    nombre_modelo = "🌲 Random Forest" if modelo_tipo == "rf" else ("🔀 Stacking" if modelo_tipo == "stacking" else ("🚀 XGBoost" if modelo_tipo == "xgb" else "📐 LASSO L1"))
    st.markdown('<div class="main-title">🇩🇪 Portal de Predicción Bundesliga</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Modelo activo: <b>{nombre_modelo}</b> — LASSO + RF + Simulación de Campeón, Europa y Descenso</div>', unsafe_allow_html=True)
    
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
            a = st.selectbox("Equipo Local", opciones, index=opciones.index("Bayern Munich") if "Bayern Munich" in opciones else 0, key="sel_a", format_func=lambda t: fmt_opcion(equipos, t))
            st.markdown(f'<div style=\"text-align:center;margin-top:0.2rem;\">{logo_html(equipos, a, 64)}</div>', unsafe_allow_html=True)
        with cvs:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        with c2:
            b = st.selectbox("Equipo Visitante", opciones, index=opciones.index("Borussia Dortmund") if "Borussia Dortmund" in opciones else 0, key="sel_b", format_func=lambda t: fmt_opcion(equipos, t))
            st.markdown(f'<div style=\"text-align:center;margin-top:0.2rem;\">{logo_html(equipos, b, 64)}</div>', unsafe_allow_html=True)
            
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
                st.success(f"⚠️ **VISITA FUERTE (Acierto 68% Histórico):** Probabilidad >50% para el equipo visitante (**{b}**). Este tipo de predicciones en Inglaterra son muy rentables.")
            
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

            # ── Comparativa Directa de los 3 Modelos ────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="sec-title">🤖 Comparativa Directa entre Modelos para este Partido</div>', unsafe_allow_html=True)
            p_lasso = mo.predecir_match(M, a, b, modelo_tipo="lasso")
            p_rf    = mo.predecir_match(M, a, b, modelo_tipo="rf")
            p_stk   = mo.predecir_match(M, a, b, modelo_tipo="stacking")
            p_xgb   = mo.predecir_match(M, a, b, modelo_tipo="xgb")
            
            df_comp_mod = pd.DataFrame([
                {
                    "Modelo Predictivo": "📐 LASSO L1 (Regresión)",
                    f"Victoria {a}": f"{p_lasso[0]:.1%}",
                    "Empate": f"{p_lasso[1]:.1%}",
                    f"Victoria {b}": f"{p_lasso[2]:.1%}",
                    "Log-Loss Out-of-Sample (2025+)": f"{met_all.get('lasso',{}).get('logloss','-'):.4f}" if 'lasso' in met_all else "-",
                    "Accuracy Out-of-Sample": f"{met_all.get('lasso',{}).get('accuracy','-'):.1f}%" if 'lasso' in met_all else "-"
                },
                {
                    "Modelo Predictivo": "🌲 Random Forest",
                    f"Victoria {a}": f"{p_rf[0]:.1%}",
                    "Empate": f"{p_rf[1]:.1%}",
                    f"Victoria {b}": f"{p_rf[2]:.1%}",
                    "Log-Loss Out-of-Sample (2025+)": f"{met_all.get('rf',{}).get('logloss','-'):.4f}" if 'rf' in met_all else "-",
                    "Accuracy Out-of-Sample": f"{met_all.get('rf',{}).get('accuracy','-'):.1f}%" if 'rf' in met_all else "-"
                },
                {"Modelo Predictivo": "🚀 XGBoost",
                    f"Victoria {a}": f"{p_xgb[0]:.1%}",
                    "Empate": f"{p_xgb[1]:.1%}",
                    f"Victoria {b}": f"{p_xgb[2]:.1%}",
                    "Log-Loss Out-of-Sample (2025+)": f"{met_all.get('xgb',{}).get('logloss','-'):.4f}" if 'xgb' in met_all else "-",
                    "Accuracy Out-of-Sample": f"{met_all.get('xgb',{}).get('accuracy','-'):.1f}%" if 'xgb' in met_all else "-"
                },
                {
                    "Modelo Predictivo": f"🔀 Stacking (w={met_all.get('stacking',{}).get('w',[0.5,0.5,0.0])})" if 'w' in met_all.get('stacking',{}) else "🔀 Stacking",
                    f"Victoria {a}": f"{p_stk[0]:.1%}",
                    "Empate": f"{p_stk[1]:.1%}",
                    f"Victoria {b}": f"{p_stk[2]:.1%}",
                    "Log-Loss Out-of-Sample (2025+)": f"{met_all.get('stacking',{}).get('logloss','-'):.4f}" if 'stacking' in met_all else "-",
                    "Accuracy Out-of-Sample": f"{met_all.get('stacking',{}).get('accuracy','-'):.1f}%" if 'stacking' in met_all else "-"
                }
            ])
            st.dataframe(df_comp_mod, use_container_width=True, hide_index=True)


            # ── DETALLES DEL PARTIDO ──────────────────────────────────────────
            with st.expander("🔍 Detalles del Partido — Variables del Modelo", expanded=False):
                tracker = M["tracker"]
                N = 5

                elo_a  = tracker.elos.get(a, 1500.0)
                elo_b  = tracker.elos.get(b, 1500.0)
                vl_a   = mo.get_squad_value(a, mo._temporada_actual())
                vl_b   = mo.get_squad_value(b, mo._temporada_actual())
                fa_d   = mo.get_advanced_features(a, mo._temporada_actual())
                fb_d   = mo.get_advanced_features(b, mo._temporada_actual())
                form_a = float(np.mean(list(tracker.recent_results[a])[-N:])) if tracker.recent_results[a] else 0.333
                form_b = float(np.mean(list(tracker.recent_results[b])[-N:])) if tracker.recent_results[b] else 0.333
                gfa    = float(np.mean(list(tracker.recent_gf[a])[-N:])) if tracker.recent_gf[a] else 1.0
                gfb    = float(np.mean(list(tracker.recent_gf[b])[-N:])) if tracker.recent_gf[b] else 1.0
                gaa    = float(np.mean(list(tracker.recent_ga[a])[-N:])) if tracker.recent_ga[a] else 1.0
                gab    = float(np.mean(list(tracker.recent_ga[b])[-N:])) if tracker.recent_ga[b] else 1.0
                ppg_a  = (tracker.season_pts[a] / tracker.season_matches[a]) if tracker.season_matches[a] > 0 else 1.33
                ppg_b  = (tracker.season_pts[b] / tracker.season_matches[b]) if tracker.season_matches[b] > 0 else 1.33
                pi_a   = tracker.pi_tracker.r_home.get(a, 0.0)
                pi_b   = tracker.pi_tracker.r_home.get(b, 0.0)
                d_km   = mo.get_distance_km(a, b)

                def _barra(va, vb, fmt=".0f", invert=False):
                    total = va + vb if (va + vb) > 0 else 1
                    pa = va / total; pb = vb / total
                    if invert: pa, pb = pb, pa
                    ca = "#10b981" if pa >= pb else "#9ca3af"
                    cb = "#10b981" if pb > pa  else "#9ca3af"
                    return (f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
                            f"<span style='min-width:75px;text-align:right;font-weight:700;color:{ca};font-size:.9rem;'>{va:{fmt}}</span>"
                            f"<div style='flex:1;background:#e5e7eb;border-radius:8px;height:10px;overflow:hidden;'>"
                            f"<div style='width:{pa*100:.1f}%;background:{ca};height:100%;border-radius:8px 0 0 8px;float:left;'></div>"
                            f"<div style='width:{pb*100:.1f}%;background:{cb};height:100%;border-radius:0 8px 8px 0;float:right;'></div>"
                            f"</div>"
                            f"<span style='min-width:75px;font-weight:700;color:{cb};font-size:.9rem;'>{vb:{fmt}}</span>"
                            f"</div>")

                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;font-weight:800;font-size:1rem;"
                    f"padding:6px 0 8px 0;border-bottom:2px solid #e5e7eb;margin-bottom:8px;'>"
                    f"<span style='color:#1d4ed8;'>🏠 {a}</span>"
                    f"<span style='color:#6b7280;font-size:.8rem;'>Variable</span>"
                    f"<span style='color:#dc2626;'>✈️ {b}</span></div>", unsafe_allow_html=True)

                filas_d = [
                    ("⚡ ELO Rating",              elo_a,  elo_b,  ".0f",  False),
                    ("💰 Valor Plantilla (M€)",     vl_a,   vl_b,   ".1f",  False),
                    ("🏆 Pi-Rating (local)",        pi_a,   pi_b,   ".3f",  False),
                    ("📈 Puntos por Partido",       ppg_a,  ppg_b,  ".2f",  False),
                    ("🔥 Forma últimos 5 (0–1)",    form_a, form_b, ".3f",  False),
                    ("⚽ GF últimos 5",             gfa,    gfb,    ".2f",  False),
                    ("🛡️ GA últimos 5",             gaa,    gab,    ".2f",  True),
                    ("🎂 Edad promedio",             fa_d.get("avg_age", 0),        fb_d.get("avg_age", 0),        ".1f", True),
                    ("🌍 % Extranjeros",            fa_d.get("pct_foreigners",0)*100, fb_d.get("pct_foreigners",0)*100, ".1f", False),
                    ("🏟️ Capacidad estadio",        fa_d.get("stadium_capacity", 0), fb_d.get("stadium_capacity", 0), ".0f", False),
                    ("👥 Asistencia promedio",      fa_d.get("avg_attendance", 0),   fb_d.get("avg_attendance", 0),   ".0f", False),
                ]
                for lbl, va, vb, fmt, inv in filas_d:
                    c_lbl, c_bar = st.columns([1.6, 3.4])
                    with c_lbl:
                        st.markdown(f"<span style='font-size:.83rem;color:#374151;'>{lbl}</span>", unsafe_allow_html=True)
                    with c_bar:
                        st.markdown(_barra(va, vb, fmt=fmt, invert=inv), unsafe_allow_html=True)

                st.markdown(
                    f"<div style='margin-top:10px;padding:8px 14px;background:#f0f9ff;"
                    f"border-radius:8px;font-size:.82rem;color:#0369a1;'>"
                    f"📍 Distancia de viaje del visitante: <b>{d_km:.0f} km</b></div>",
                    unsafe_allow_html=True)
            # ─────────────────────────────────────────────────────────────────

            # Mercados de apuestas
            st.markdown('<div class="sec-title">Mercados de Goles y Apuestas Especiales</div>', unsafe_allow_html=True)

            filas = []
            for ln in (1.5, 2.5, 3.5):
                for lado in ("Over", "Under"):
                    pr = mk[f"{lado} {ln}"]
                    filas.append({"Mercado": f"{lado} {ln} goles", "Prob.": f"{pr:.1%}", "Cuota justa": f"{mo.cuota(pr):.2f}"})
            for et, key in (("Ambos marcan: Sí", "Ambos marcan (BTTS sí)"), ("Ambos marcan: No", "BTTS no")):
                filas.append({"Mercado": et, "Prob.": f"{mk[key]:.1%}", "Cuota justa": f"{mo.cuota(mk[key]):.2f}"})
                
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
        df_actual_vis["Escudo"] = df_actual_vis["equipo"].apply(lambda t: logo_url(equipos, t))
        df_actual_vis["Equipo"] = df_actual_vis["equipo"].apply(lambda t: label_tabla(equipos, t))
        df_actual_vis = df_actual_vis[["Escudo", "Equipo", "pj", "puntos", "dif_goles", "goles_favor"]]
        df_actual_vis = df_actual_vis.rename(columns={"pj": "PJ", "puntos": "PTS", "dif_goles": "DG", "goles_favor": "GF"})
        
        with col_act:
            st.markdown('<div class="card-title">Tabla de Posiciones Actual (Real)</div>', unsafe_allow_html=True)
            st.dataframe(df_actual_vis, hide_index=True, width='stretch', height=500,
                         column_config={"Escudo": st.column_config.ImageColumn("", width="small")})
            
        # 2. Proyecciones
        partidos_rec = pd.read_csv(mo.DATA / "partidos.csv")
        last_key = f"{len(partidos_rec)}-{partidos_rec.fecha.max()}"
        df_proy = simular_liga(M, last_key, modelo_tipo)
        
        df_proy_visual = df_proy.copy()
        df_proy_visual["Escudo"] = df_proy_visual["equipo"].apply(lambda t: logo_url(equipos, t))
        df_proy_visual["Equipo"] = df_proy_visual["equipo"].apply(lambda t: label_tabla(equipos, t))
        df_proy_visual = df_proy_visual[["Escudo", "Equipo", "P_campeon", "P_copas", "P_descenso"]]
        df_proy_visual = df_proy_visual.rename(columns={
            "P_campeon": "🏆 P(Campeón)",
            "P_copas": "🇪🇺 P(Copas)",
            "P_descenso": "🔻 P(Descenso)"
        })
        
        with col_proj:
            st.markdown('<div class="card-title">Proyección de la Temporada en Curso (Monte Carlo)</div>', unsafe_allow_html=True)
            st.caption("Simula el resto de la temporada 2026/27 a partir del fixture publicado hasta la fecha.")
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
        # La validación es honesta solo sobre temporadas out-of-sample (test >= 2025).
        temporadas_disponibles = sorted(M["df_features"]["temporada"].unique(), reverse=True)
        temporadas_disponibles = [t for t in temporadas_disponibles if t >= 2025]
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
                eqs_val = set(df_val["local"]).union(set(df_val["visita"]))
                for eq in eqs_val:
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





if __name__ == "__main__":
    run_app()
