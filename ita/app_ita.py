"""
Aplicación Streamlit para la simulación y predicciones de la Serie A (Italia).
Visualizaciones premium de Versus, Mercados, Tabla de Posiciones y Proyecciones de Monte Carlo.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Insertar el directorio ita en el path para asegurar la importación del motor
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
    background:linear-gradient(135deg,#008C45,#003399); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#003399;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.6rem; font-weight:800; color:#003399; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Emojis e identidades visuales de los clubes italianos
TEAM_DETAILS = {
    "AC Milan": {"flag": "🔴⚫", "color": "#e4002b"},
    "AS Roma": {"flag": "🟡🔴", "color": "#990a2c"},
    "Atalanta": {"flag": "🔵⚫", "color": "#1157bf"},
    "Bologna": {"flag": "🔴🔵", "color": "#04043d"},
    "Cagliari": {"flag": "🔴🔵", "color": "#282846"},
    "Como": {"flag": "🔵⚪", "color": "#4169e1"},
    "Fiorentina": {"flag": "🟣", "color": "#4c1d84"},
    "Frosinone": {"flag": "🟡🔵", "color": "#facf08"},
    "Genoa": {"flag": "🔴🔵", "color": "#08305d"},
    "Inter": {"flag": "🔵⚫", "color": "#00239c"},
    "Juventus": {"flag": "⚪⚫", "color": "#000000"},
    "Lazio": {"flag": "🦅", "color": "#74bde7"},
    "Lecce": {"flag": "🟡🔴", "color": "#fced0b"},
    "Monza": {"flag": "🔴⚪", "color": "#c8142f"},
    "Napoli": {"flag": "🔵", "color": "#0677d2"},
    "Parma": {"flag": "🟡🔵", "color": "#19161d"},
    "Sassuolo": {"flag": "🟢⚫", "color": "#0fa653"},
    "Torino": {"flag": "🐂", "color": "#9f0000"},
    "Udinese": {"flag": "⚪⚫", "color": "#19161d"},
    "Venezia": {"flag": "🟠🟢⚫", "color": "#000000"},
    "Verona": {"flag": "🟡🔵", "color": "#00239c"},
    "Empoli": {"flag": "🔵", "color": "#005bdd"},
    "Cremonese": {"flag": "🔴⚪", "color": "#ff0000"},
    "Pisa": {"flag": "🔵⚫", "color": "#1a1a1a"},
    "Sampdoria": {"flag": "🔵⚪🔴⚫", "color": "#2234a4"},
    "Benevento": {"flag": "🟡🔴", "color": "#c60000"},
    "Salernitana": {"flag": "🟤", "color": "#990000"},
    "Spezia": {"flag": "⚪⚫", "color": "#ffffff"},
    "Crotone": {"flag": "🔴🔵", "color": "#275ba2"}
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
    """Etiqueta de opción del selectbox: abreviatura del club + nombre."""
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
        [
            "🌲 Random Forest (Recomendado)",
            "📐 LASSO L1 (Regresión)",
            "🔀 Stacking (Ensemble óptimo)",
            "🚀 XGBoost (Gradient Boosting)"
        ],
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
        mejor_ll = min(met_all[k]["logloss"] for k in met_all if "logloss" in met_all[k])
        for nombre, clave in [("LASSO", "lasso"), ("RF", "rf"), ("XGB", "xgb"), ("Stacking", "stacking")]:
            if clave not in met_all: continue
            m = met_all[clave]
            star = " ⭐" if m["logloss"] == mejor_ll else ""
            w_str = f" (w={m['w']})" if clave == "stacking" and "w" in m else ""
            st.sidebar.caption(f"**{nombre}{w_str}{star}** — LL: `{m['logloss']:.4f}` | Acc: `{m['accuracy']:.1f}%`")

    # Encabezado
    nombre_modelo = "🌲 Random Forest" if modelo_tipo == "rf" else ("🔀 Stacking" if modelo_tipo == "stacking" else ("🚀 XGBoost" if modelo_tipo == "xgb" else "📐 LASSO L1"))
    st.markdown('<div class="main-title">🇮🇹 Portal de Predicción Serie A</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Modelo activo: <b>{nombre_modelo}</b> — LASSO + RF + XGBoost + Simulación de Scudetto, Champions y Descenso</div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚽ Predicción Versus", "📊 Tabla y Proyecciones", "🔬 Importancia de Variables", "🎯 Validación vs Realidad"])

    # ============================================================================
    # TAB 1: Predicción Versus
    # ============================================================================
    with tab1:
        st.markdown('<div class="sec-title">Analizador de Enfrentamientos</div>', unsafe_allow_html=True)

        opciones = sorted(list(M["df_features"]["local"].unique()))

        fix_rec = pd.read_csv(mo.DATA / "fixture.csv") if (mo.DATA / "fixture.csv").exists() else pd.DataFrame()
        if not fix_rec.empty:
            df_fix_sorted = fix_rec.copy()
            df_fix_sorted["fecha_dt"] = pd.to_datetime(df_fix_sorted["fecha"])
            proximos = df_fix_sorted.sort_values("fecha_dt").head(25)

            fix_map = {}
            opciones_fixture = ["— Seleccionar partido programado —"]
            for _, r in proximos.iterrows():
                f_str = r["fecha_dt"].strftime("%d/%m %H:%M")
                lbl = f"📅 {f_str} | {r['local']} vs {r['visita']}"
                opciones_fixture.append(lbl)
                fix_map[lbl] = (r["local"], r["visita"])

            def _on_fixture_change_ita():
                sel = st.session_state.get("sel_fix_ita")
                if sel and sel in fix_map:
                    l_sel, v_sel = fix_map[sel]
                    if l_sel in opciones and v_sel in opciones:
                        st.session_state["sel_a_ita"] = l_sel
                        st.session_state["sel_b_ita"] = v_sel

            st.selectbox(
                "⚡ Cargar Partido de la Próxima Fecha:",
                opciones_fixture,
                key="sel_fix_ita",
                on_change=_on_fixture_change_ita,
                help="Elige un partido oficial programado para cargar ambos equipos automáticamente."
            )

        c1, cvs, c2 = st.columns([5, 1, 5])
        with c1:
            def_a = "Inter" if "Inter" in opciones else opciones[0]
            a = st.selectbox("Equipo Local", opciones, index=opciones.index(def_a), key="sel_a_ita", format_func=lambda t: fmt_opcion(equipos, t))
            st.markdown(f'<div style="text-align:center;margin-top:0.2rem;">{logo_html(equipos, a, 64)}</div>', unsafe_allow_html=True)
        with cvs:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        with c2:
            def_b = "Juventus" if "Juventus" in opciones else ("AC Milan" if "AC Milan" in opciones else opciones[1])
            b = st.selectbox("Equipo Visitante", opciones, index=opciones.index(def_b), key="sel_b_ita", format_func=lambda t: fmt_opcion(equipos, t))
            st.markdown(f'<div style="text-align:center;margin-top:0.2rem;">{logo_html(equipos, b, 64)}</div>', unsafe_allow_html=True)

        if a == b:
            st.error("Selecciona dos equipos distintos.")
        else:
            # Calcular predicción 1X2 y marcadores
            p = mo.predecir_match(M, a, b, modelo_tipo=modelo_tipo)
            mix = mo.grilla_goles(M, a, b, modelo_tipo=modelo_tipo)

            tracker = M["tracker"]
            elo_diff = tracker.elos[a] - tracker.elos[b]
            la = np.exp(M["g_const"] + M["g_d"] * elo_diff + M.get("g_home", 0.0) * 1)
            lb = np.exp(M["g_const"] - M["g_d"] * elo_diff + M.get("g_home", 0.0) * 0)

            la_lbl = get_label(a)
            lb_lbl = get_label(b)

            # Alertas de Heurísticas de Alta Efectividad
            if p[0] > 0.55 and (la - lb) > 1.0:
                st.success(f"🔥 **ALERTA DE ALTA CONFIANZA (Acierto >80% Histórico):** Consenso perfecto entre Machine Learning (>55%) y modelo Poisson (>1.0 goles dif) a favor de victoria de **{a}**.")
            elif p[0] > 0.60:
                st.info(f"💪 **FAVORITO CLARO (Acierto >72% Histórico):** El modelo de Machine Learning asigna más del 60% de probabilidad de victoria a **{a}**.")
            elif p[2] > 0.50:
                st.success(f"⚠️ **VISITA FUERTE (Acierto >65% Histórico):** Probabilidad >50% para el equipo visitante (**{b}**). Este tipo de predicciones en Serie A tienen alto valor.")

            # Mostrar Probabilidades
            col_probs, col_stats = st.columns(2)

            with col_probs:
                st.markdown('<div class="card-title">Probabilidades de Victoria</div>', unsafe_allow_html=True)
                for label, prob, col in [(f"Victoria {a}", p[0], "#008C45"), ("Empate", p[1], "#64748b"), (f"Victoria {b}", p[2], "#003399")]:
                    st.markdown(f"**{label}: {prob:.1%}** (Cuota Justa: `{mo.cuota(prob):.2f}`)")
                    st.progress(float(prob))

                st.markdown("---")
                p_avanza_a = p[0] + p[1] * 0.5
                st.caption(f"Expectativa de clasificación en eliminación directa neutral: **{a} {p_avanza_a:.1%}** / {b} {1-p_avanza_a:.1%}")

            with col_stats:
                st.markdown('<div class="card-title">Goles Esperados y Marcadores</div>', unsafe_allow_html=True)
                st.markdown(f"📈 **Goles esperados (Poisson):**")
                st.markdown(f"*   {la_lbl}: `{la:.2f}` goles")
                st.markdown(f"*   {lb_lbl}: `{lb:.2f}` goles")

                st.markdown("🎯 **Marcadores más probables:**")
                mk_temp = mo.mercados(mix)
                # Extraer marcadores top de la grilla
                marcadores = []
                for i in range(min(6, mix.shape[0])):
                    for j in range(min(6, mix.shape[1])):
                        marcadores.append((i, j, mix[i, j]))
                marcadores.sort(key=lambda x: x[2], reverse=True)
                for g1, g2, pr in marcadores[:4]:
                    st.markdown(f"*   `{g1} - {g2}`: **{pr:.1%}** (Cuota: `{mo.cuota(pr):.1f}`)")

            # ── Comparativa Directa de los Modelos ────────────────────────────────
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
                {
                    "Modelo Predictivo": "🚀 XGBoost",
                    f"Victoria {a}": f"{p_xgb[0]:.1%}",
                    "Empate": f"{p_xgb[1]:.1%}",
                    f"Victoria {b}": f"{p_xgb[2]:.1%}",
                    "Log-Loss Out-of-Sample (2025+)": f"{met_all.get('xgb',{}).get('logloss','-'):.4f}" if 'xgb' in met_all else "-",
                    "Accuracy Out-of-Sample": f"{met_all.get('xgb',{}).get('accuracy','-'):.1f}%" if 'xgb' in met_all else "-"
                },
                {
                    "Modelo Predictivo": f"🔀 Stacking (w={met_all.get('stacking',{}).get('w',[0.258, 0.385, 0.356])})" if 'w' in met_all.get('stacking',{}) else "🔀 Stacking",
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
                    f"<span style='color:#008C45;'>🏠 {a}</span>"
                    f"<span style='color:#6b7280;font-size:.8rem;'>Variable</span>"
                    f"<span style='color:#003399;'>✈️ {b}</span></div>", unsafe_allow_html=True)

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

            # Mercados de apuestas
            st.markdown('<div class="sec-title">Mercados de Goles y Apuestas Especiales</div>', unsafe_allow_html=True)
            mk = mo.mercados(mix)

            filas = []
            for ln in ("1.5", "2.5", "3.5"):
                pr_over = mk["OverUnder"][f"O{ln}"]
                pr_under = mk["OverUnder"][f"U{ln}"]
                filas.append({"Mercado": f"Over {ln} goles", "Prob.": f"{pr_over:.1%}", "Cuota justa": f"{mo.cuota(pr_over):.2f}"})
                filas.append({"Mercado": f"Under {ln} goles", "Prob.": f"{pr_under:.1%}", "Cuota justa": f"{mo.cuota(pr_under):.2f}"})

            filas.append({"Mercado": "Ambos marcan: Sí", "Prob.": f"{mk['BTTS']['Si']:.1%}", "Cuota justa": f"{mo.cuota(mk['BTTS']['Si']):.2f}"})
            filas.append({"Mercado": "Ambos marcan: No", "Prob.": f"{mk['BTTS']['No']:.1%}", "Cuota justa": f"{mo.cuota(mk['BTTS']['No']):.2f}"})

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
            im = ax.imshow(m6, cmap="Blues")
            ax.set_xticks(range(6)); ax.set_xticklabels(range(6))
            ax.set_yticks(range(6)); ax.set_yticklabels(range(6))
            ax.set_xlabel(f"Goles de {b}"); ax.set_ylabel(f"Goles de {a}")
            fig.colorbar(im, ax=ax, label="Probabilidad")

            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{m6[i,j]:.1%}", ha="center", va="center", color="white" if m6[i,j] > m6.max()*0.6 else "black", fontsize=8)
            st.pyplot(fig)

            # ── HISTORIAL HEAD-TO-HEAD (ENFRENTAMIENTOS DIRECTOS) ─────────
            st.markdown("---")
            st.markdown('<div class="sec-title">⚔️ Historial Head-to-Head (Enfrentamientos Directos)</div>', unsafe_allow_html=True)

            partidos_df_ita = pd.read_csv(mo.DATA / "partidos.csv")
            mask_h2h = ((partidos_df_ita["local"] == a) & (partidos_df_ita["visita"] == b)) | \
                       ((partidos_df_ita["local"] == b) & (partidos_df_ita["visita"] == a))
            df_h2h = partidos_df_ita[mask_h2h].copy()

            if df_h2h.empty:
                st.info(f"ℹ️ No se registran enfrentamientos directos oficiales entre **{a}** y **{b}** en el dataset reciente (2021-2026).")
            else:
                df_h2h["fecha_dt"] = pd.to_datetime(df_h2h["fecha"])
                df_h2h = df_h2h.sort_values("fecha_dt", ascending=False)

                vic_a = sum(1 for _, r in df_h2h.iterrows() if (r["local"] == a and r["goles_local"] > r["goles_visita"]) or (r["visita"] == a and r["goles_visita"] > r["goles_local"]))
                vic_b = sum(1 for _, r in df_h2h.iterrows() if (r["local"] == b and r["goles_local"] > r["goles_visita"]) or (r["visita"] == b and r["goles_visita"] > r["goles_local"]))
                emp = len(df_h2h) - vic_a - vic_b

                gol_a = sum(int(r["goles_local"] if r["local"] == a else r["goles_visita"]) for _, r in df_h2h.iterrows())
                gol_b = sum(int(r["goles_local"] if r["local"] == b else r["goles_visita"]) for _, r in df_h2h.iterrows())

                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                with col_h1:
                    st.metric("Total Duelos", len(df_h2h))
                with col_h2:
                    pct_a = (vic_a / len(df_h2h)) * 100
                    st.metric(f"Victorias {a}", f"{vic_a} ({pct_a:.0f}%)", f"{gol_a} goles")
                with col_h3:
                    pct_e = (emp / len(df_h2h)) * 100
                    st.metric("Empates", f"{emp} ({pct_e:.0f}%)")
                with col_h4:
                    pct_b = (vic_b / len(df_h2h)) * 100
                    st.metric(f"Victorias {b}", f"{vic_b} ({pct_b:.0f}%)", f"{gol_b} goles")

                filas_hist = []
                for _, r in df_h2h.head(8).iterrows():
                    gl, gv = int(r["goles_local"]), int(r["goles_visita"])
                    if gl > gv:
                        ganador = f"🟢 Gana {r['local']}"
                    elif gv > gl:
                        ganador = f"🟢 Gana {r['visita']}"
                    else:
                        ganador = "⚪ Empate"
                    filas_hist.append({
                        "Fecha": r["fecha_dt"].strftime("%d/%m/%Y"),
                        "Temporada": r["temporada"],
                        "Local": r["local"],
                        "Marcador": f"{gl} - {gv}",
                        "Visitante": r["visita"],
                        "Resultado": ganador
                    })
                st.dataframe(pd.DataFrame(filas_hist), hide_index=True, use_container_width=True)

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
        df_proy_visual = df_proy_visual[["Escudo", "Equipo", "P_campeon", "P_champions", "P_copas", "P_descenso"]]
        df_proy_visual = df_proy_visual.rename(columns={
            "P_campeon": "🏆 P(Scudetto)",
            "P_champions": "🌟 P(Champions)",
            "P_copas": "🇪🇺 P(Copas)",
            "P_descenso": "🔻 P(Descenso)"
        })

        with col_proj:
            st.markdown('<div class="card-title">Proyección de la Temporada en Curso (Monte Carlo)</div>', unsafe_allow_html=True)
            st.caption("Simula el resto de la temporada 2026/27 a partir del fixture oficial con 50.000 iteraciones.")
            st.dataframe(
                df_proy_visual.style.format({
                    "🏆 P(Scudetto)": "{:.1%}",
                    "🌟 P(Champions)": "{:.1%}",
                    "🇪🇺 P(Copas)": "{:.1%}",
                    "🔻 P(Descenso)": "{:.1%}"
                }).background_gradient(subset=["🏆 P(Scudetto)"], cmap="Greens")
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
        elif modelo_tipo == "xgb":
            st.markdown('<div class="sec-title">Importancia de Características: XGBoost</div>', unsafe_allow_html=True)
            st.markdown("La importancia de características en **XGBoost** mide la ganancia promedio de información que aporta cada variable en los árboles de decisión graduados.")

            importancia = []
            pipe = M["pipe_xgb"]
            xgb_m = pipe.named_steps["xgb"]
            importances = xgb_m.feature_importances_

            for feat, val in zip(M["cols"], importances):
                importancia.append({"Variable": feat, "Importancia (Ganancia)": round(val, 4)})
            col_val_name = "Importancia (Ganancia)"
            title_graph = "Top 15 Características Predictoras (XGBoost)"
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
            ax.barh(top_n["Variable"][::-1], top_n[col_val_name][::-1], color="#008C45")
            ax.set_title(title_graph)
            st.pyplot(fig)

    # ============================================================================
    # TAB 4: Validación vs Realidad
    # ============================================================================
    with tab4:
        st.markdown('<div class="sec-title">El Modelo contra la Realidad (Out-of-sample)</div>', unsafe_allow_html=True)
        st.markdown("Comparación de la predicción **pre-partido** del modelo contra el **resultado real** para los partidos ya jugados.")

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
                    ax.plot(evol["partidos"], evol["logloss"], "o-", color="#008C45", label="Modelo (acumulado)")
                    ax.axhline(met["logloss_base"], color="#003399", ls="--", lw=1.5, label="Baseline")
                    ax.set_xlabel("Partidos jugados (cronológico)")
                    ax.set_ylabel("Log-loss acumulado")
                    ax.set_title("Evolución del Log-loss en la Temporada")
                    ax.legend()
                    st.pyplot(fig)

            with c_table:
                st.markdown("##### % Acierto por Equipo")
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
