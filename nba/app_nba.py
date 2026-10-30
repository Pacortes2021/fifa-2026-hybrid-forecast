"""
Aplicación Streamlit para la simulación, proyecciones y predicciones de la NBA.
Visualizaciones analíticas de Enfrentamientos Versus, Hándicaps (Spread), Totales (Over/Under),
Dean Oliver Four Factors, Clasificación Este/Oeste y Simulación Monte Carlo de Playoffs y Finales NBA.
"""
from datetime import datetime
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Insertar el directorio nba en el path para asegurar la importación del motor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motor as mo
import recolectar as rec
import recolectar_boxscore as rec_box
from motor import cuota, cuota_americana

# Estilizado CSS Premium para NBA
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main-title { text-align:center; font-size:2.8rem; font-weight:900;
    background:linear-gradient(135deg,#1d428a,#c8102e); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
.main-subtitle { text-align:center; font-size:1.1rem; color:#64748b; margin-bottom:1.8rem; }
.card-title { font-size:1.25rem; font-weight:700; color:#1d428a;
    border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:0.8rem; }
.sec-title { font-size:1.5rem; font-weight:800; color:#1d428a; margin:0.8rem 0 0.6rem 0; }
.vs-text { text-align:center; font-size:2.2rem; font-weight:900; color:#cbd5e1; margin-top:1.6rem; }
.metric-box { background:#f8fafc; border-radius:12px; padding:12px; border:1px solid #e2e8f0; text-align:center; margin-bottom:10px; }
.metric-val { font-size:1.8rem; font-weight:800; color:#0f172a; }
.metric-lbl { font-size:0.85rem; color:#64748b; text-transform:uppercase; font-weight:600; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    border-radius:16px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


def logo_url(equipos, team):
    """URL del logo oficial de la franquicia desde data/equipos.csv."""
    if not equipos or team not in equipos:
        return None
    return equipos[team].get("logo")


def logo_html(equipos, team, size=64):
    """Etiqueta <img> con el logo en alta resolución."""
    url = logo_url(equipos, team)
    if not url:
        return "🏀"
    return f'<img src="{url}" width="{size}" style="border-radius:10px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.15));">'


def fmt_opcion(equipos, team):
    """Formato de selección: Abreviatura · Nombre de franquicia."""
    if equipos and team in equipos:
        abbr = equipos[team].get("abbreviation")
        if abbr:
            return f"{abbr} · {team}"
    return team


@st.cache_resource
def get_motor():
    return mo.cargar()


@st.cache_data(show_spinner="Corriendo simulaciones de Monte Carlo para la NBA (2.000 iteraciones)...")
def simular_temporada_cached(_M, key):
    return mo.simular_temporada(_M, n_sims=2000)


def run_app():
    st.sidebar.markdown("### 🏀 Controles del Modelo NBA")

    # Selector de modelo predictivo
    OPCIONES_MOD = [
        "🔀 Stacking (Ensemble Óptimo) [Recomendado]",
        "🚀 XGBoost (Gradient Boosting)",
        "🌲 Random Forest (Ensemble no lineal)",
        "📐 LASSO L1 (Regresión Logística Esparsa)"
    ]
    modelo_sel = st.sidebar.selectbox("🤖 Modelo Predictivo:", OPCIONES_MOD, index=0)
    modelo = "stacking" if "Stacking" in modelo_sel else ("xgb" if "XGB" in modelo_sel else ("rf" if "Random Forest" in modelo_sel else "lasso"))

    if st.sidebar.button("🔄 Actualizar ESPN y Re-entrenar", key="refresh_nba", type="primary"):
        with st.spinner("Descargando últimos resultados y box scores de la NBA..."):
            rec.recolectar()
            rec_box.recolectar(limite=200)
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

    # Métricas del modelo en el sidebar
    met_all = M.get("metricas", {})
    if met_all:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📊 Rendimiento Out-of-Sample (Test 2025-26+)")
        mejor_ll = min(met_all[k]["logloss"] for k in ("lasso", "rf", "xgb", "stacking") if k in met_all)
        for nombre, clave in [("Stacking", "stacking"), ("XGBoost", "xgb"), ("LASSO L1", "lasso"), ("Random Forest", "rf")]:
            if clave not in met_all:
                continue
            m = met_all[clave]
            star = " ⭐" if m["logloss"] == mejor_ll else ""
            st.sidebar.caption(f"**{nombre}{star}** — Log-Loss: `{m['logloss']:.4f}` | Acc: `{m['accuracy']:.1f}%`")
        if "stacking" in met_all:
            st.sidebar.caption(f"🎯 Error Spread: `±{met_all['stacking']['mae_spread']:.1f} pts` | Totales: `±{met_all['stacking']['mae_totals']:.1f} pts`")

    # Encabezado principal
    nombre_modelo = "🔀 Stacking Ensemble" if modelo == "stacking" else ("🚀 XGBoost" if modelo == "xgb" else ("🌲 Random Forest" if modelo == "rf" else "📐 LASSO L1"))
    st.markdown('<div class="main-title">🏀 Portal de Simulación y Predicción NBA</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Modelo activo: <b>{nombre_modelo}</b> — Four Factors de Dean Oliver + Ratings de Posesión + Monte Carlo Playoff</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏀 Predicción Versus", "📊 Clasificación y Playoffs", "🔬 Factores e Importancia", "🎯 Validación vs Realidad"
    ])

    # ============================================================================
    # TAB 1: Predicción Versus
    # ============================================================================
    with tab1:
        st.markdown('<div class="sec-title">Analizador de Enfrentamientos NBA</div>', unsafe_allow_html=True)
        
        lista_equipos = sorted(list(equipos.keys())) if equipos else sorted([
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
            "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
            "Houston Rockets", "Indiana Pacers", "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
            "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
            "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
        ])

        # Cargar partidos del fixture oficial
        fix_path = mo.DATA / "fixture.csv"
        df_fix = pd.read_csv(fix_path) if fix_path.exists() else pd.DataFrame()

        if not df_fix.empty:
            df_fix_sorted = df_fix.copy()
            df_fix_sorted["fecha_dt"] = pd.to_datetime(df_fix_sorted["fecha"])
            proximos = df_fix_sorted[df_fix_sorted["fecha_dt"] >= pd.Timestamp.now() - pd.Timedelta(days=1)].sort_values("fecha_dt").head(30)
            
            if not proximos.empty:
                fix_map = {}
                opciones_fix = ["— Seleccionar partido del calendario oficial —"]
                for _, r in proximos.iterrows():
                    f_str = r["fecha_dt"].strftime("%d/%m %H:%M")
                    lbl = f"📅 {f_str} ET | {r['visita']} en {r['local']}"
                    opciones_fix.append(lbl)
                    fix_map[lbl] = (r["local"], r["visita"])

                def _on_fix_change():
                    sel = st.session_state.get("sel_fix_nba")
                    if sel and sel in fix_map:
                        l_sel, v_sel = fix_map[sel]
                        st.session_state["sel_home_nba"] = l_sel
                        st.session_state["sel_away_nba"] = v_sel

                st.selectbox(
                    "⚡ Cargar Partido Programado de la Temporada 2026-27:",
                    opciones_fix,
                    key="sel_fix_nba",
                    on_change=_on_fix_change,
                    help="Carga automáticamente los equipos local y visitante de la cartelera oficial."
                )

        c1, cvs, c2 = st.columns([5, 1, 5])
        with c1:
            idx_h = lista_equipos.index("Boston Celtics") if "Boston Celtics" in lista_equipos else 0
            home_team = st.selectbox(
                "Equipo Local (Home)", lista_equipos, index=idx_h, key="sel_home_nba",
                format_func=lambda t: fmt_opcion(equipos, t)
            )
            st.markdown(f'<div style="text-align:center;margin-top:0.3rem;">{logo_html(equipos, home_team, 75)}</div>', unsafe_allow_html=True)
        with cvs:
            st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        with c2:
            idx_a = lista_equipos.index("Los Angeles Lakers") if "Los Angeles Lakers" in lista_equipos else 1
            away_team = st.selectbox(
                "Equipo Visitante (Away)", lista_equipos, index=idx_a, key="sel_away_nba",
                format_func=lambda t: fmt_opcion(equipos, t)
            )
            st.markdown(f'<div style="text-align:center;margin-top:0.3rem;">{logo_html(equipos, away_team, 75)}</div>', unsafe_allow_html=True)

        if home_team == away_team:
            st.warning("Selecciona dos franquicias distintas para analizar el enfrentamiento.")
        else:
            # Simulación interactiva de Bajas de Jugadores Estrella
            stars_h = mo.NBA_FRANCHISE_STARS.get(home_team, ["Jugador Franquicia"])
            stars_a = mo.NBA_FRANCHISE_STARS.get(away_team, ["Jugador Franquicia"])

            with st.expander("🚑 Simulación de Bajas de Jugadores Estrella (Injury Impact)", expanded=False):
                st.caption("Selecciona si una de las franquicias no contará con su estrella principal para calibrar el impacto en la probabilidad y hándicap:")
                col_inj_h, col_inj_a = st.columns(2)
                with col_inj_h:
                    baja_h = st.checkbox(
                        f"Baja de figura en {home_team} ({', '.join(stars_h)})",
                        help=f"Simula la ausencia de figuras clave de {home_team}",
                        key="inj_h_cb"
                    )
                with col_inj_a:
                    baja_a = st.checkbox(
                        f"Baja de figura en {away_team} ({', '.join(stars_a)})",
                        help=f"Simula la ausencia de figuras clave de {away_team}",
                        key="inj_a_cb"
                    )

            # Predicción con el motor NBA (incluyendo ajuste dinámico por bajas)
            pred_mc = mo.simular_partido_mc(
                M, home_team, away_team, n_sims=4000, modelo=modelo,
                baja_estrella_local=baja_h, baja_estrella_visita=baja_a
            )
            pred = pred_mc["pred"]
            p_h = pred["p_local"]
            p_a = pred["p_visita"]
            feats = pred["feats"]

            # Alertas heurísticas de bajas
            if baja_h:
                st.error(f"🚑 **BAJA CLAVE ({home_team}):** Simulando ausencia de figura estelar ({', '.join(stars_h)}). Se aplica penalización empírica de ~5.5 pts en Net Rating y -75 pts Elo.")
            if baja_a:
                st.error(f"🚑 **BAJA CLAVE ({away_team}):** Simulando ausencia de figura estelar ({', '.join(stars_a)}). Se aplica penalización empírica de ~5.5 pts en Net Rating y -75 pts Elo.")

            # Alertas heurísticas de baloncesto (fatiga y favoritos)
            if feats["b2b_local"] and not feats["b2b_visita"]:
                st.warning(f"⚠️ **ALERTA DE FATIGA (Back-to-Back):** {home_team} juega sin descanso (0 días), lo que históricamente reduce el ritmo y la efectividad en tiros de campo.")
            elif feats["b2b_visita"] and not feats["b2b_local"]:
                st.info(f"✈️ **DESVENTAJA VISITANTE (Back-to-Back + Viaje):** {away_team} llega en segunda noche consecutiva, otorgando ventaja física a {home_team}.")

            if p_h >= 0.70:
                st.success(f"🔥 **FAVORITO SÓLIDO:** El modelo asigna **{p_h:.1%}** de victoria a **{home_team}** con una proyección de margen de `{pred['proj_spread']:+.1f}` puntos.")
            elif p_a >= 0.58:
                st.info(f"💎 **OPORTUNIDAD DE VALOR VISITANTE:** **{away_team}** registra **{p_a:.1%}** de probabilidad de triunfo fuera de casa (Cuota justa: `{pred['cuota_visita']}`).")

            # Fila de métricas clave del partido
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Victoria {home_team}</div>
                    <div class="metric-val">{p_h:.1%}</div>
                    <div style="color:#64748b;font-size:0.85rem;">Cuota: <b>{pred['cuota_local']}</b> ({pred['cuota_am_local']})</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Victoria {away_team}</div>
                    <div class="metric-val">{p_a:.1%}</div>
                    <div style="color:#64748b;font-size:0.85rem;">Cuota: <b>{pred['cuota_visita']}</b> ({pred['cuota_am_visita']})</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Marcador Proyectado</div>
                    <div class="metric-val">{pred['proj_pts_local']:.0f} - {pred['proj_pts_visita']:.0f}</div>
                    <div style="color:#64748b;font-size:0.85rem;">Hándicap (Spread): <b>{pred['proj_spread']:+.1f}</b></div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Total Proyectado</div>
                    <div class="metric-val">{pred['proj_total']:.1f}</div>
                    <div style="color:#64748b;font-size:0.85rem;">Pace estimado: <b>{feats['exp_game_pace']:.1f} pos</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Puntuación proyectada por cuartos
            st.markdown('<div class="card-title">🏀 Proyección de Anotación por Cuartos</div>', unsafe_allow_html=True)
            df_cuartos = pd.DataFrame({
                "Cuarto": ["Q1", "Q2", "Q3", "Q4", "Total Proyectado"],
                f"🏠 {home_team}": pred_mc["cuartos_local"] + [pred["proj_pts_local"]],
                f"✈️ {away_team}": pred_mc["cuartos_visita"] + [pred["proj_pts_visita"]]
            })
            st.dataframe(df_cuartos.set_index("Cuarto"), use_container_width=True)

            # Distribuciones de Monte Carlo: Spreads y Totales
            col_spread, col_ou = st.columns(2)
            with col_spread:
                st.markdown(f'<div class="card-title">📊 Probabilidades de Hándicap (Spread {home_team})</div>', unsafe_allow_html=True)
                filas_sp = []
                for line, p_cov in pred_mc["probs_spread"].items():
                    sign = "+" if line > 0 else ""
                    filas_sp.append({
                        "Línea de Puntos": f"{home_team} {sign}{line}",
                        "Probabilidad de Cubrir": f"{p_cov:.1%}",
                        "Cuota Justa": mo.cuota(p_cov)
                    })
                st.dataframe(pd.DataFrame(filas_sp), use_container_width=True, hide_index=True)

            with col_ou:
                st.markdown('<div class="card-title">📈 Probabilidades Over/Under (Totales)</div>', unsafe_allow_html=True)
                filas_ou = []
                for ou_line, p_over in pred_mc["probs_ou"].items():
                    p_under = 1.0 - p_over
                    filas_ou.append({
                        "Línea Total": f"{ou_line} Pts",
                        "Prob. Over": f"{p_over:.1%}",
                        "Cuota Over": mo.cuota(p_over),
                        "Prob. Under": f"{p_under:.1%}",
                        "Cuota Under": mo.cuota(p_under)
                    })
                st.dataframe(pd.DataFrame(filas_ou), use_container_width=True, hide_index=True)

            # Four Factors de Dean Oliver y Tendencias (Comparativa Analítica)
            st.markdown('<div class="card-title">🔬 Four Factors, Tendencias y Métricas de Localía</div>', unsafe_allow_html=True)
            df_factors = pd.DataFrame({
                "Métrica / Factor": [
                    "Effective FG% (eFG% Ofensivo)",
                    "Opponent eFG% (eFG% Defensivo)",
                    "Offensive Rating (ORTG / 100 pos)",
                    "Defensive Rating (DRTG / 100 pos)",
                    "Net Rating Global (NetRTG)",
                    "Ritmo de Posesiones (Pace)",
                    "Free Throw Rate (FTA / FGA)",
                    "Tendencia Reciente (Momentum L5 vs L20)",
                    "Récord Reciente Casa / Ruta",
                    "Net Rating Reciente Casa / Ruta",
                    "Rating Elo Puro (Sin Localía)",
                    "Rating Elo con Localía"
                ],
                f"🏠 {home_team}": [
                    f"{feats['efg_local']:.1%}",
                    f"{feats['opp_efg_local']:.1%}",
                    f"{feats['ortg_local']:.1f}",
                    f"{feats['drtg_local']:.1f}",
                    f"{feats['netrtg_local']:+.1f}",
                    f"{feats['pace_local']:.1f}",
                    f"{feats['ft_rate_local']:.3f}",
                    f"{feats.get('trend_local', 0.0):+.1f} pts",
                    f"{feats.get('l5_home_win_pct', 0.55):.0%} (en casa)",
                    f"{feats.get('l5_home_netrtg', 2.5):+.1f} pts",
                    f"{feats['elo_local'] - mo.ELO_HOME_ADV:.0f}",
                    f"{feats['elo_local']:.0f}"
                ],
                f"✈️ {away_team}": [
                    f"{feats['efg_visita']:.1%}",
                    f"{feats['opp_efg_visita']:.1%}",
                    f"{feats['ortg_visita']:.1f}",
                    f"{feats['drtg_visita']:.1f}",
                    f"{feats['netrtg_visita']:+.1f}",
                    f"{feats['pace_visita']:.1f}",
                    f"{feats['ft_rate_visita']:.3f}",
                    f"{feats.get('trend_visita', 0.0):+.1f} pts",
                    f"{feats.get('l5_away_win_pct', 0.45):.0%} (de visita)",
                    f"{feats.get('l5_away_netrtg', -2.5):+.1f} pts",
                    f"{feats['elo_visita']:.0f}",
                    f"{feats['elo_visita']:.0f}"
                ]
            })
            st.dataframe(df_factors, use_container_width=True, hide_index=True)

    # ============================================================================
    # TAB 2: Clasificación y Simulación Playoff (Monte Carlo)
    # ============================================================================
    with tab2:
        st.markdown('<div class="sec-title">Proyecciones de Temporada Regular, Play-In y Playoffs NBA</div>', unsafe_allow_html=True)
        st.markdown("Simulación estocástica de Monte Carlo (2.000 iteraciones) de los partidos restantes del calendario 2026-27, torneo Play-In y series al mejor de 7 partidos.")

        df_sim = simular_temporada_cached(M, f"{len(df_fix)}_{modelo}")

        tab_east, tab_west, tab_champ = st.tabs(["🏛️ Conferencia Este", "🤠 Conferencia Oeste", "🏆 Probabilidades de Campeón"])
        
        with tab_east:
            st.markdown("#### 🏛️ Eastern Conference (Proyección Final)")
            df_east = df_sim[df_sim["Conferencia"] == "East"].drop(columns=["Conferencia"]).reset_index(drop=True)
            df_east["Posición Proyectada"] = range(1, len(df_east) + 1)
            # Formatear porcentajes
            for c in ["P_Playoff_Directo", "P_PlayIn", "P_Finalista_Conf", "P_Campeon"]:
                df_east[c] = df_east[c].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_east[[
                "Posición Proyectada", "Equipo", "Victorias Esperadas", "Derrotas Esperadas",
                "P_Playoff_Directo", "P_PlayIn", "P_Finalista_Conf", "P_Campeon"
            ]], use_container_width=True, hide_index=True)

        with tab_west:
            st.markdown("#### 🤠 Western Conference (Proyección Final)")
            df_west = df_sim[df_sim["Conferencia"] == "West"].drop(columns=["Conferencia"]).reset_index(drop=True)
            df_west["Posición Proyectada"] = range(1, len(df_west) + 1)
            for c in ["P_Playoff_Directo", "P_PlayIn", "P_Finalista_Conf", "P_Campeon"]:
                df_west[c] = df_west[c].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_west[[
                "Posición Proyectada", "Equipo", "Victorias Esperadas", "Derrotas Esperadas",
                "P_Playoff_Directo", "P_PlayIn", "P_Finalista_Conf", "P_Campeon"
            ]], use_container_width=True, hide_index=True)

        with tab_champ:
            st.markdown("#### 🏆 Candidatos al Anillo de Campeón NBA 2026-27")
            top_champs = df_sim.sort_values(by="P_Campeon", ascending=False).head(12)
            
            fig, ax = plt.subplots(figsize=(10, 4.8))
            y_pos = np.arange(len(top_champs))
            ax.barh(y_pos, top_champs["P_Campeon"] * 100.0, color="#1d428a", edgecolor="#c8102e", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_champs["Equipo"], fontsize=10, fontweight="bold")
            ax.invert_yaxis()
            ax.set_xlabel("Probabilidad de Título NBA (%)", fontsize=11, fontweight="bold")
            for i, v in enumerate(top_champs["P_Campeon"]):
                ax.text(v * 100.0 + 0.3, i + 0.15, f"{v:.1%}", color="#0f172a", fontweight="bold", fontsize=9.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)

    # ============================================================================
    # TAB 3: Factores e Importancia
    # ============================================================================
    with tab3:
        st.markdown('<div class="sec-title">Importancia de Variables y Rendimiento de Modelos</div>', unsafe_allow_html=True)
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Log-Loss Ensamble</div>
                <div class="metric-val">{met_all.get('stacking', {}).get('logloss', 0.60):.4f}</div>
                <div style="color:#10b981;font-size:0.85rem;">vs Baseline: {met_all.get('stacking', {}).get('logloss_base', 0.69):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Acierto Ganador (Accuracy)</div>
                <div class="metric-val">{met_all.get('stacking', {}).get('accuracy', 67.5):.1f}%</div>
                <div style="color:#64748b;font-size:0.85rem;">Test Out-of-Sample (1.394 partidos)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Error Absoluto Medio (MAE)</div>
                <div class="metric-val">±{met_all.get('stacking', {}).get('mae_spread', 11.5):.1f} pts</div>
                <div style="color:#64748b;font-size:0.85rem;">Desviación Hándicap: {met_all.get('stacking', {}).get('sigma_spread', 14.7):.1f} pts</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="card-title">Ranking de Variables Más Influyentes (LASSO y Random Forest)</div>', unsafe_allow_html=True)
        feats_rank = met_all.get("features_rank", [])
        if feats_rank:
            df_rf = pd.DataFrame(feats_rank).head(15)
            df_rf.columns = ["Variable", "Coeficiente LASSO L1", "Importancia Random Forest"]
            st.dataframe(df_rf, use_container_width=True, hide_index=True)

    # ============================================================================
    # TAB 4: Validación vs Realidad y Acierto por Franquicia
    # ============================================================================
    with tab4:
        st.markdown('<div class="sec-title">Validación Empírica y Rendimiento por Franquicia</div>', unsafe_allow_html=True)

        col_periodo, _ = st.columns([2, 1])
        with col_periodo:
            periodo = st.radio(
                "Período a Evaluar:",
                ["🏀 Temporada 2026-27 (En Vivo / Out-of-Sample)", "📚 Histórico Completo (2022-2027)"],
                horizontal=True,
                key="nba_val_periodo"
            )

        if "2026-27" in periodo:
            df_val, met_val, evol = mo.validacion_en_vivo(M, temporada=2027)
        else:
            features = M["features"]
            pipe = M["pipe_lasso"]
            df_val = M["df_dataset"].copy()
            X_all = df_val[features].fillna(0.0)
            y_all = df_val["target_win_local"]
            probs_all = pipe.predict_proba(X_all)
            preds_all = pipe.predict(X_all)
            df_val["Prob_Local"] = probs_all[:, 1]
            df_val["Prob_Visita"] = probs_all[:, 0]
            df_val["Prediccion"] = preds_all

            from sklearn.metrics import log_loss, accuracy_score
            ll = log_loss(y_all, probs_all)
            acc = accuracy_score(y_all, preds_all)
            freq = y_all.mean()
            base_p = np.full_like(probs_all, 0.5)
            base_p[:, 1] = freq
            base_p[:, 0] = 1.0 - freq
            ll_base = log_loss(y_all, base_p)
            met_val = {
                "n": len(df_val),
                "acierto": round(float(acc * 100.0), 2),
                "logloss": round(float(ll), 4),
                "logloss_base": round(float(ll_base), 4)
            }
            evol = pd.DataFrame()

        # Métricas superiores
        v1, v2, v3 = st.columns(3)
        with v1:
            st.metric("Partidos Evaluados", met_val["n"])
        with v2:
            st.metric("% Acierto General", f"{met_val['acierto']:.1f}%")
        with v3:
            st.metric("Log-Loss", f"{met_val['logloss']:.4f}", delta=f"{(met_val['logloss'] - met_val['logloss_base']):.4f} vs base", delta_color="inverse")

        # Columnas: Curva de Error y Tabla de Acierto por Equipo
        c_graf, c_table = st.columns([1, 1])

        with c_graf:
            st.markdown('<div class="card-title">📈 Curva de Log-Loss Acumulado</div>', unsafe_allow_html=True)
            if not evol.empty and "logloss_acum" in evol.columns:
                fig, ax = plt.subplots(figsize=(6, 4.4))
                ax.plot(evol["partido"], evol["logloss_acum"], "o-", color="#1d428a", label="Modelo (acumulado)", markersize=4)
                ax.axhline(met_val["logloss_base"], color="#c8102e", ls="--", lw=1.6, label=f"Baseline ({met_val['logloss_base']:.3f})")
                ax.set_xlabel("Partidos Jugados (Cronológico)", fontsize=9)
                ax.set_ylabel("Log-Loss Acumulado", fontsize=9)
                ax.set_title("Evolución del Error Probabilístico", fontsize=10, fontweight="bold")
                ax.legend(fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig)
            else:
                st.info("La curva acumulada cronológica se grafica en la evaluación de la temporada 2026-27 en vivo.")

        with c_table:
            st.markdown('<div class="card-title">🎯 Porcentaje de Acierto por Franquicia</div>', unsafe_allow_html=True)
            eqs_official = sorted(list(equipos.keys())) if equipos else sorted(list(set(df_val["local"]).union(set(df_val["visita"]))))
            team_stats = []
            for eq in eqs_official:
                df_eq = df_val[(df_val["local"] == eq) | (df_val["visita"] == eq)]
                pj = len(df_eq)
                if pj > 0:
                    aciertos = int((df_eq["Prediccion"] == df_eq["target_win_local"]).sum())
                    fallos = pj - aciertos
                    acc = aciertos / pj

                    df_h = df_val[df_val["local"] == eq]
                    acc_h = (df_h["Prediccion"] == df_h["target_win_local"]).sum() / len(df_h) if len(df_h) > 0 else None

                    df_a = df_val[df_val["visita"] == eq]
                    acc_a = (df_a["Prediccion"] == df_a["target_win_local"]).sum() / len(df_a) if len(df_a) > 0 else None

                    team_stats.append({
                        "Equipo": eq,
                        "PJ": pj,
                        "Aciertos": aciertos,
                        "Fallos": fallos,
                        "% Acierto": acc,
                        "% Casa": f"{acc_h:.0%}" if acc_h is not None else "—",
                        "% Ruta": f"{acc_a:.0%}" if acc_a is not None else "—"
                    })

            if team_stats:
                df_teams = pd.DataFrame(team_stats).sort_values(by=["% Acierto", "PJ"], ascending=[False, False])
                st.dataframe(
                    df_teams.style.format({"% Acierto": "{:.1%}"}).background_gradient(subset=["% Acierto"], cmap="YlGn"),
                    use_container_width=True, hide_index=True
                )

        # Historial de partidos con filtro por franquicia
        st.markdown('<div class="card-title">📜 Historial de Partidos y Pronósticos Realizados</div>', unsafe_allow_html=True)

        filtro_eq = st.selectbox(
            "🔍 Filtrar partidos disputados por franquicia:",
            ["— Todas las franquicias —"] + sorted(list(equipos.keys())),
            key="filtro_hist_nba"
        )

        df_disp = df_val.copy()
        if filtro_eq != "— Todas las franquicias —":
            df_disp = df_disp[(df_disp["local"] == filtro_eq) | (df_disp["visita"] == filtro_eq)]

        if not df_disp.empty:
            df_val_disp = df_disp[[
                "fecha", "local", "visita", "puntos_local", "puntos_visita",
                "Prob_Local", "Prob_Visita", "Prediccion", "target_win_local"
            ]].copy().sort_values("fecha", ascending=False)

            df_val_disp["Resultado"] = df_val_disp.apply(lambda r: f"{int(r['puntos_local'])} - {int(r['puntos_visita'])}", axis=1)
            df_val_disp["Pronóstico"] = df_val_disp["Prediccion"].apply(lambda p: "🏠 Local" if p == 1 else "✈️ Visita")
            df_val_disp["Acierto"] = df_val_disp["Prediccion"] == df_val_disp["target_win_local"]
            df_val_disp["Acierto Icono"] = df_val_disp["Acierto"].apply(lambda ok: "✅ Acierto" if ok else "❌ Fallo")
            df_val_disp["Prob. Local"] = df_val_disp["Prob_Local"].apply(lambda p: f"{p:.1%}")
            df_val_disp["Prob. Visita"] = df_val_disp["Prob_Visita"].apply(lambda p: f"{p:.1%}")

            st.dataframe(df_val_disp[[
                "fecha", "local", "visita", "Resultado", "Pronóstico", "Prob. Local", "Prob. Visita", "Acierto Icono"
            ]], use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron partidos para la selección.")


if __name__ == "__main__":
    run_app()
