"""
Interfaz de usuario en Streamlit para el Portal de Predicción de la Liga Profesional de Fútbol de Argentina.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import arg.motor as mo
import arg.recolectar as rec
import arg.recolectar_boxscore as rec_box

# Configuración de página
st.set_page_config(
    page_title="Predicción Liga Profesional Argentina 🇦🇷",
    page_icon="🇦🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

TEAM_DETAILS = {
    "Boca Juniors": {"flag": "💙💛💙", "desc": "La Bombonera — Buenos Aires", "color": "#0033a0"},
    "River Plate": {"flag": "⚪🔴⚪", "desc": "MÁS Monumental — Buenos Aires", "color": "#dc2626"},
    "Racing Club": {"flag": "🩵⚪🩵", "desc": "El Cilindro de Avellaneda", "color": "#00a3e0"},
    "Independiente": {"flag": "🔴⚪🔴", "desc": "Estadio Libertadores de América", "color": "#b91c1c"},
    "San Lorenzo": {"flag": "🔵🔴🔵", "desc": "Pedro Bidegain (Nuevo Gasómetro)", "color": "#1d4ed8"},
    "Vélez Sarsfield": {"flag": "⚪🔵⚪", "desc": "José Amalfitani — Liniers", "color": "#2563eb"},
    "Estudiantes de La Plata": {"flag": "🔴⚪🔴", "desc": "Estadio UNO Jorge Luis Hirschi", "color": "#991b1b"},
    "Gimnasia La Plata": {"flag": "🔵⚪🔵", "desc": "Juan Carmelo Zerillo (El Bosque)", "color": "#1e3a8a"},
    "Rosario Central": {"flag": "🟡🔵🟡", "desc": "Gigante de Arroyito — Rosario", "color": "#eab308"},
    "Newell's Old Boys": {"flag": "🔴⚫🔴", "desc": "Coloso Marcelo Bielsa — Rosario", "color": "#7f1d1d"},
    "Talleres de Córdoba": {"flag": "🔵⚪🔵", "desc": "Mario Alberto Kempes — Córdoba", "color": "#1e40af"},
    "Belgrano": {"flag": "🩵⚪🩵", "desc": "Julio César Villagra (El Gigante de Alberdi)", "color": "#38bdf8"},
    "Instituto": {"flag": "🔴⚪🔴", "desc": "Juan Domingo Perón — Córdoba", "color": "#ef4444"},
    "Godoy Cruz": {"flag": "🔵⚪🔵", "desc": "Malvinas Argentinas — Mendoza", "color": "#3b82f6"},
    "Huracán": {"flag": "⚪🔴⚪", "desc": "Tomás Adolfo Ducó (El Palacio)", "color": "#dc2626"},
    "Lanús": {"flag": "🟣⚪🟣", "desc": "Néstor Díaz Pérez (La Fortaleza)", "color": "#701a75"},
    "Banfield": {"flag": "🟢⚪🟢", "desc": "Florencio Sola (El Taladro)", "color": "#15803d"},
    "Argentinos Juniors": {"flag": "🔴⚪🔴", "desc": "Diego Armando Maradona — La Paternal", "color": "#b91c1c"},
    "Defensa y Justicia": {"flag": "🟢🟡🟢", "desc": "Norberto Tomaghello — Florencio Varela", "color": "#16a34a"},
    "Platense": {"flag": "🤎⚪🤎", "desc": "Ciudad de Vicente López", "color": "#78350f"},
    "Tigre": {"flag": "🔵🔴🔵", "desc": "José Dellagiovanna — Victoria", "color": "#1d4ed8"},
    "Central Córdoba": {"flag": "⚫⚪⚫", "desc": "Madre de Ciudades — Santiago del Estero", "color": "#18181b"},
    "Atlético Tucumán": {"flag": "🩵⚪🩵", "desc": "Monumental José Fierro — Tucumán", "color": "#0ea5e9"},
    "Unión de Santa Fe": {"flag": "🔴⚪🔴", "desc": "15 de Abril — Santa Fe", "color": "#ef4444"},
    "Sarmiento": {"flag": "🟢⚪🟢", "desc": "Eva Perón — Junín", "color": "#15803d"},
    "Barracas Central": {"flag": "🔴⚪🔴", "desc": "Claudio Chiqui Tapia", "color": "#b91c1c"},
    "Deportivo Riestra": {"flag": "⚫⚪⚫", "desc": "Guillermo Laza — Flores", "color": "#27272a"},
    "Independiente Rivadavia": {"flag": "🔵⚪🔵", "desc": "Bautista Gargantini — Mendoza", "color": "#1e3a8a"}
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
    info = TEAM_DETAILS.get(team, {"flag": "⚽", "desc": team})
    return f"{info['flag']} {team}"


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Corriendo simulaciones de Monte Carlo (50.000 iteraciones)...")
def simular_campeonato(_M, key, modelo):
    return mo.simular_campeonato(_M, n_sims=50000, modelo=modelo)


def run_app():
    st.sidebar.markdown("### 🛠️ Controles del Modelo (Argentina)")

    OPCIONES_MOD = ["🌲 Random Forest (Recomendado)", "📐 LASSO L1 (Regresión)", "🔀 Stacking (Ensemble óptimo)"]
    modelo_sel = st.sidebar.selectbox("🤖 Modelo Predictivo:", OPCIONES_MOD, index=0)
    modelo = "lasso" if "LASSO" in modelo_sel else ("stacking" if "Stacking" in modelo_sel else "rf")

    if st.sidebar.button("🔄 Actualizar ESPN y Re-entrenar", key="refresh_arg", type="primary"):
        with st.spinner("Descargando últimos resultados de la Liga Profesional de Argentina..."):
            rec.recolectar()
            rec_box.recolectar()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    M = get_motor()
    equipos = M.get("equipos", {})

    if "metricas" in M:
        met_all = M["metricas"]
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📊 Métricas Out-of-Sample (2025+)")
        mejor_ll = min(met_all[k]["logloss"] for k in met_all)
        for nombre, clave in [("LASSO", "lasso"), ("RF", "rf"), ("XGB", "xgb"), ("Stacking", "stacking")]:
            if clave not in met_all: continue
            m = met_all[clave]
            star = " ⭐" if m["logloss"] == mejor_ll else ""
            w_str = f" (w={m['w']})" if clave == "stacking" and "w" in m else ""
            st.sidebar.caption(f"**{nombre}{w_str}{star}** — LL: `{m['logloss']:.4f}` | Acc: `{m['accuracy']:.1f}%`")

    nombre_modelo = "🌲 Random Forest" if modelo == "rf" else ("🔀 Stacking" if modelo == "stacking" else ("🚀 XGBoost" if modelo == "xgb" else "📐 LASSO L1"))
    st.markdown('<div class="main-title">🇦🇷 Portal de Predicción Liga Profesional Argentina</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Modelo activo: <b>{nombre_modelo}</b> — LASSO + RF + Simulación de Campeonato Completo</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["⚽ Predicción Versus", "📊 Tabla y Proyecciones", "🔬 Importancia de Variables", "🎯 Validación vs Realidad"])

    # TAB 1: Predicción Versus
    with tab1:
        st.markdown('<div class="sec-title">Analizador de Enfrentamientos Directos</div>', unsafe_allow_html=True)

        opciones = sorted(list(TEAM_DETAILS.keys()))
        partidos_rec = M["partidos"]
        p_actuales = partidos_rec[partidos_rec.temporada == mo._temporada_actual()]
        fix_rec = pd.read_csv(mo.DATA / "fixture.csv") if (mo.DATA / "fixture.csv").exists() else pd.DataFrame()
        equipos_activos = sorted(list(set(p_actuales["local"]).union(set(p_actuales["visita"])).union(set(fix_rec["local"] if not fix_rec.empty else []))))
        if not equipos_activos:
            equipos_activos = opciones

        c1, cvs, c2 = st.columns([5, 1, 5])
        with c1:
            a = st.selectbox("Equipo Local", equipos_activos, index=equipos_activos.index("Boca Juniors") if "Boca Juniors" in equipos_activos else 0, key="sel_a_arg", format_func=lambda t: fmt_opcion(equipos, t))
            st.markdown(f'<div style=\"text-align:center;margin-top:0.2rem;\">{logo_html(equipos, a, 64)}</div>', unsafe_allow_html=True)
        with cvs:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        with c2:
            b = st.selectbox("Equipo Visitante", equipos_activos, index=equipos_activos.index("River Plate") if "River Plate" in equipos_activos else 1, key="sel_b_arg", format_func=lambda t: fmt_opcion(equipos, t))
            st.markdown(f'<div style=\"text-align:center;margin-top:0.2rem;\">{logo_html(equipos, b, 64)}</div>', unsafe_allow_html=True)

        if a == b:
            st.error("Selecciona dos equipos distintos.")
        else:
            mix, p, (la, lb) = mo.grilla_goles(M, a, b, modelo=modelo)
            la_lbl = get_label(a); lb_lbl = get_label(b)

            if p[0] > 0.55 and (la - lb) > 1.0:
                st.success(f"🔥 **ALERTA DE ALTA CONFIANZA:** Consenso perfecto entre Machine Learning (>55%) y Poisson (>1.0 goles dif) a favor de **{a}**.")
            elif p[0] > 0.60:
                st.info(f"💪 **FAVORITO CLARO:** El modelo asigna más del 60% de probabilidad de victoria a **{a}**.")
            elif p[2] > 0.50:
                st.success(f"⚠️ **VISITA FUERTE:** Probabilidad >50% para el equipo visitante (**{b}**).")

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

            st.markdown("---")
            st.markdown('<div class="sec-title">🤖 Comparativa Directa entre Modelos para este Partido</div>', unsafe_allow_html=True)
            p_lasso, _, _ = mo.predecir_match(M, a, b, modelo="lasso")
            p_rf, _, _    = mo.predecir_match(M, a, b, modelo="rf")
            p_stk, _, _   = mo.predecir_match(M, a, b, modelo="stacking")
            p_xgb, _, _   = mo.predecir_match(M, a, b, modelo="xgb")

            df_comp_mod = pd.DataFrame([
                {
                    "Modelo Predictivo": "📐 LASSO L1 (Regresión)",
                    f"Victoria {a}": f"{p_lasso[0]:.1%}", "Empate": f"{p_lasso[1]:.1%}", f"Victoria {b}": f"{p_lasso[2]:.1%}",
                    "Log-Loss Out-of-Sample": f"{met_all.get('lasso',{}).get('logloss','-'):.4f}" if 'lasso' in met_all else "-",
                    "Accuracy Out-of-Sample": f"{met_all.get('lasso',{}).get('accuracy','-'):.1f}%" if 'lasso' in met_all else "-"
                },
                {
                    "Modelo Predictivo": "🌲 Random Forest",
                    f"Victoria {a}": f"{p_rf[0]:.1%}", "Empate": f"{p_rf[1]:.1%}", f"Victoria {b}": f"{p_rf[2]:.1%}",
                    "Log-Loss Out-of-Sample": f"{met_all.get('rf',{}).get('logloss','-'):.4f}" if 'rf' in met_all else "-",
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
                    f"Victoria {a}": f"{p_stk[0]:.1%}", "Empate": f"{p_stk[1]:.1%}", f"Victoria {b}": f"{p_stk[2]:.1%}",
                    "Log-Loss Out-of-Sample": f"{met_all.get('stacking',{}).get('logloss','-'):.4f}" if 'stacking' in met_all else "-",
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

    # TAB 2: Tabla y Proyecciones
    with tab2:
        st.markdown('<div class="sec-title">Proyección de Tabla Liga Profesional Argentina 2026</div>', unsafe_allow_html=True)
        st.markdown("Simulación estocástica de Monte Carlo (4.000 iteraciones del torneo completo).")

        partidos_rec = M["partidos"]
        last_key = f"{len(partidos_rec)}-{partidos_rec.fecha.max()}"
        df_proy = simular_campeonato(M, last_key, modelo)

        def style_prob(val):
            if isinstance(val, (int, float)):
                if val > 0.4: return "background-color: #d1fae5; color: #065f46; font-weight: bold;"
                elif val > 0.15: return "background-color: #ecfdf5; color: #047857;"
            return ""

        st.dataframe(
            df_proy.style.format({
                "Puntos esperados": "{:.1f}",
                "P_campeon": "{:.1%}",
                "P_copas": "{:.1%}",
                "P_descenso": "{:.1%}"
            }).map(style_prob, subset=["P_campeon", "P_copas"]),
            use_container_width=True,
            hide_index=True
        )

    # TAB 3: Importancia de Variables
    with tab3:
        st.markdown('<div class="sec-title">¿Qué Variables Pesan Más en el Fútbol Argentino?</div>', unsafe_allow_html=True)

        col_t, col_g = st.columns([5, 5])
        pipe_rf = M["pipe_rf"]
        if hasattr(pipe_rf, "estimator"):
            base_pipe = pipe_rf.estimator
            if hasattr(base_pipe, "estimator"): # Handle FrozenEstimator
                base_pipe = base_pipe.estimator
        elif hasattr(pipe_rf, "base_estimator"):
            base_pipe = pipe_rf.base_estimator
        else:
            base_pipe = pipe_rf
        rf_model = base_pipe.named_steps["rf"]
        cols = M["features"]
        importances = rf_model.feature_importances_

        df_imp = pd.DataFrame({"Variable": cols, "Peso Absoluto Promedio": importances}).sort_values(by="Peso Absoluto Promedio", ascending=False).reset_index(drop=True)

        with col_t:
            st.markdown("##### 🏆 Ranking de Importancia (Random Forest)")
            st.dataframe(df_imp.head(15).style.format({"Peso Absoluto Promedio": "{:.4f}"}), use_container_width=True, hide_index=True)

        with col_g:
            fig, ax = plt.subplots(figsize=(6, 5))
            top_n = df_imp.head(15)
            ax.barh(top_n["Variable"][::-1], top_n["Peso Absoluto Promedio"][::-1], color="#74acdf")
            ax.set_title("Top 15 Características Predictoras (Argentina)")
            st.pyplot(fig)

    # TAB 4: Validación vs Realidad
    with tab4:
        st.markdown('<div class="sec-title">El Modelo contra la Realidad (Out-of-sample)</div>', unsafe_allow_html=True)
        st.markdown("Comparación de la predicción del modelo contra el resultado real para los partidos de la liga argentina.")

        # La validación es honesta solo sobre temporadas out-of-sample (test >= 2025).
        temporadas_disponibles = sorted(M["df_dataset"]["temporada"].unique(), reverse=True)
        temporadas_disponibles = [t for t in temporadas_disponibles if t >= 2025]
        temporada_sel = st.selectbox("Selecciona la temporada a validar:", temporadas_disponibles, index=0)

        df_val, met, evol = mo.validacion_en_vivo(M, temporada_val=temporada_sel)

        if df_val is None or len(df_val) == 0:
            st.info("Aún no hay partidos finalizados en la temporada seleccionada para validar.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Partidos Jugados", met["n"])
            m2.metric("Acierto (1X2)", f"{met['acierto']:.1%}")
            m3.metric("Log-loss modelo", f"{met['logloss']:.3f}", f"{met['logloss'] - met['logloss_base']:+.3f} vs baseline", delta_color="inverse")
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
            st.dataframe(df_show, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    run_app()
