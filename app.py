import os
import sys
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from collections import Counter
from scipy.stats import poisson
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# La API de ESPN (cuadro real) y el armado del bracket (plantilla FIFA) viven en lab/ — sin entrenar modelo.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab'))
import espn_live
import motor as _motor  # solo se usan bracket_real / ko_fijos / tablas_grupos (nada que entrene)

# Configuración de página con título y layout
st.set_page_config(
    page_title="Simulador Mundial 2026 — Base vs Híbrido",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilizado CSS Premium personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0b3d91, #7a3b91);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .main-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .card-title-base {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0b3d91;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .card-title-hybrid {
        font-size: 1.3rem;
        font-weight: 600;
        color: #7a3b91;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .vs-text {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #cbd5e1;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 48 Países participantes con traducción y bandera emoji
COUNTRIES_ES = {
    'Alemania': {'en': 'Germany', 'flag': '🇩🇪'},
    'Argelia': {'en': 'Algeria', 'flag': '🇩🇿'},
    'Argentina': {'en': 'Argentina', 'flag': '🇦🇷'},
    'Australia': {'en': 'Australia', 'flag': '🇦🇺'},
    'Austria': {'en': 'Austria', 'flag': '🇦🇹'},
    'Bélgica': {'en': 'Belgium', 'flag': '🇧🇪'},
    'Bosnia y Herzegovina': {'en': 'Bosnia and Herzegovina', 'flag': '🇧🇦'},
    'Brasil': {'en': 'Brazil', 'flag': '🇧🇷'},
    'Cabo Verde': {'en': 'Cape Verde', 'flag': '🇨🇻'},
    'Canadá': {'en': 'Canada', 'flag': '🇨🇦'},
    'Catar': {'en': 'Qatar', 'flag': '🇶🇦'},
    'Colombia': {'en': 'Colombia', 'flag': '🇨🇴'},
    'Corea del Sur': {'en': 'South Korea', 'flag': '🇰🇷'},
    'Costa de Marfil': {'en': 'Ivory Coast', 'flag': '🇨🇮'},
    'Croacia': {'en': 'Croatia', 'flag': '🇭🇷'},
    'Curazao': {'en': 'Curaçao', 'flag': '🇨🇼'},
    'Ecuador': {'en': 'Ecuador', 'flag': '🇪🇨'},
    'Egipto': {'en': 'Egypt', 'flag': '🇪🇬'},
    'Escocia': {'en': 'Scotland', 'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿'},
    'España': {'en': 'Spain', 'flag': '🇪🇸'},
    'Estados Unidos': {'en': 'United States', 'flag': '🇺🇸'},
    'Francia': {'en': 'France', 'flag': '🇫🇷'},
    'Ghana': {'en': 'Ghana', 'flag': '🇬🇭'},
    'Haití': {'en': 'Haiti', 'flag': '🇭🇹'},
    'Inglaterra': {'en': 'England', 'flag': '🏴\U000e0067\U000e0062\U000e006e\U000e0067\U000e007f'},
    'Irak': {'en': 'Iraq', 'flag': '🇮🇶'},
    'Irán': {'en': 'Iran', 'flag': '🇮🇷'},
    'Japón': {'en': 'Japan', 'flag': '🇯🇵'},
    'Jordania': {'en': 'Jordan', 'flag': '🇯🇴'},
    'Marruecos': {'en': 'Morocco', 'flag': '🇲🇦'},
    'México': {'en': 'Mexico', 'flag': '🇲🇽'},
    'Noruega': {'en': 'Norway', 'flag': '🇳🇴'},
    'Nueva Zelanda': {'en': 'New Zealand', 'flag': '🇳🇿'},
    'Países Bajos': {'en': 'Netherlands', 'flag': '🇳🇱'},
    'Panamá': {'en': 'Panama', 'flag': '🇵🇦'},
    'Paraguay': {'en': 'Paraguay', 'flag': '🇵🇾'},
    'Portugal': {'en': 'Portugal', 'flag': '🇵🇹'},
    'República Checa': {'en': 'Czech Republic', 'flag': '🇨🇿'},
    'Rep. Democrática del Congo': {'en': 'DR Congo', 'flag': '🇨🇩'},
    'Senegal': {'en': 'Senegal', 'flag': '🇸🇳'},
    'Sudáfrica': {'en': 'South Africa', 'flag': '🇿🇦'},
    'Suecia': {'en': 'Sweden', 'flag': '🇸🇪'},
    'Suiza': {'en': 'Switzerland', 'flag': '🇨🇭'},
    'Túnez': {'en': 'Tunisia', 'flag': '🇹🇳'},
    'Turquía': {'en': 'Turkey', 'flag': '🇹🇷'},
    'Uruguay': {'en': 'Uruguay', 'flag': '🇺🇾'},
    'Uzbekistán': {'en': 'Uzbekistan', 'flag': '🇺🇿'},
    'Arabia Saudita': {'en': 'Saudi Arabia', 'flag': '🇸🇦'},
}

# Inverso del mapa para decodificar selección
MUNDIALISTAS = [v['en'] for v in COUNTRIES_ES.values()]
ANFITRIONES = {'United States', 'Canada', 'Mexico'}
BASE_VARS_G = ['elo_diff', 'h2h_diff', 'squad_value_diff']

# Inverso inglés -> (español, bandera) para el cuadro de eliminatorias
EN2ES = {v['en']: (k, v['flag']) for k, v in COUNTRIES_ES.items()}


def _nombre(en):
    return EN2ES.get(en, (en, ''))[0]


def _bandera(en):
    return EN2ES.get(en, ('', '🏳️'))[1]

@st.cache_resource
def load_data_and_train():
    # 1. Carga de datos base
    df = pd.read_csv('data/modelado_espn.csv', parse_dates=['fecha']).sort_values('fecha').reset_index(drop=True)
    states = pd.read_csv('data/team_states.csv').set_index('team')
    hist = pd.read_csv('data/results.csv', parse_dates=['date']).dropna(subset=['home_score', 'away_score'])

    # Calcular H2H
    m48 = set(MUNDIALISTAS)
    duelos = hist[hist.home_team.isin(m48) & hist.away_team.isin(m48)]
    H2H = {}
    for r in duelos.itertuples(index=False):
        d = r.home_score - r.away_score
        for a, b, s in ((r.home_team, r.away_team, d), (r.away_team, r.home_team, -d)):
            H2H.setdefault((a, b), []).append(s)
    H2H = {k: float(np.mean(v)) for k, v in H2H.items()}

    # Definir variables de cada modelo
    BASE_VARS = ['elo_diff', 'h2h_diff', 'squad_value_diff']
    ALL_VARS = BASE_VARS + ['goles_anotados_diff', 'goles_recibidos_diff', 'tiros_diff', 'tiros_arco_diff', 'corners_diff', 'posesion_diff', 'faltas_diff']
    HYBRID_VARS = ['elo_diff', 'h2h_diff', 'squad_value_diff', 'goles_anotados_diff', 'goles_recibidos_diff', 'tiros_arco_diff']

    # Filtrar datos de entrenamiento correspondientes a cada notebook
    data_base = df.dropna(subset=BASE_VARS + ['ea_overall_diff']).reset_index(drop=True)
    data_hybrid = df.dropna(subset=ALL_VARS).reset_index(drop=True)

    # Ponderación por K-factor: los partidos en serio pesan más que los amistosos (Mundial 3x un amistoso)
    def tipo_competicion(c):
        c = str(c).lower()
        if 'amistoso' in c: return 'amistoso'
        if 'clasif' in c: return 'clasificatoria'
        if 'nations league' in c: return 'nations_league'
        if 'mundial' in c: return 'mundial'
        return 'continental'
    K_FACTOR = {'amistoso': 1.0, 'clasificatoria': 2.0, 'nations_league': 2.0, 'continental': 2.5, 'mundial': 3.0}
    w_base = data_base.competicion.map(tipo_competicion).map(K_FACTOR).values
    w_hybrid = data_hybrid.competicion.map(tipo_competicion).map(K_FACTOR).values

    # Entrenar Clasificador A (Base)
    pipe_base = Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=2000))])
    pipe_base.fit(data_base[BASE_VARS], data_base['resultado'], m__sample_weight=w_base)

    # Entrenar Clasificador B (Híbrido)
    pipe_hybrid = Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=2000))])
    pipe_hybrid.fit(data_hybrid[HYBRID_VARS], data_hybrid['resultado'], m__sample_weight=w_hybrid)

    # Entrenar Poisson A (Base - basado solo en Elo)
    espn = pd.read_csv('data/espn_stats.csv', parse_dates=['fecha'])
    espn = espn[(espn.fecha >= '2019-01-01') & espn.goles_local.notna()]
    largo_base = pd.concat([
        pd.DataFrame({'g': espn.goles_local.values, 'd': (espn.elo_local - espn.elo_visita).values}),
        pd.DataFrame({'g': espn.goles_visita.values, 'd': (espn.elo_visita - espn.elo_local).values})
    ])
    gp = sm.GLM(largo_base['g'], sm.add_constant(largo_base[['d']]), family=sm.families.Poisson()).fit()
    gb0, gb1 = gp.params['const'], gp.params['d']

    # Entrenar Poisson B (Híbrido - basado en las 6 variables híbridas)
    goles = pd.read_csv('data/espn_stats.csv', parse_dates=['fecha'])[['fecha', 'local', 'visita', 'goles_local', 'goles_visita']]
    datag = data_hybrid.merge(goles, on=['fecha', 'local', 'visita'], how='left')
    largo_hybrid = pd.concat([
        datag[HYBRID_VARS].assign(g=datag.goles_local.values),
        (-datag[HYBRID_VARS]).assign(g=datag.goles_visita.values)
    ], ignore_index=True)
    pois_engine = sm.GLM(largo_hybrid['g'], sm.add_constant(largo_hybrid[HYBRID_VARS]), family=sm.families.Poisson()).fit()
    b0s = pois_engine.params['const']; bets = pois_engine.params[HYBRID_VARS].values

    return pipe_base, pipe_hybrid, gb0, gb1, b0s, bets, H2H, states

# Cargar recursos una sola vez
pipe_base, pipe_hybrid, gb0, gb1, b0s, bets, H2H, states = load_data_and_train()

# Render de Títulos e Introducción
st.markdown('<div class="main-title">🏆 Simulador de Partidos — Copa Mundial 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Comparación de Predicciones: Modelo Completo Base vs. Modelo Híbrido (Stats + Prior)</div>', unsafe_allow_html=True)

# Controles de Entrada del Versus
dropdown_options = sorted([f"{v['flag']} {k}" for k, v in COUNTRIES_ES.items()])

col_in1, col_vs, col_in2 = st.columns([4, 1, 4])

with col_in1:
    team_a_sel = st.selectbox("Selección 1 (Local administrativo)", dropdown_options, index=dropdown_options.index("🇪🇸 España"))
    es_name_a = team_a_sel.split(" ", 1)[1]
    en_name_a = COUNTRIES_ES[es_name_a]['en']
    flag_a = COUNTRIES_ES[es_name_a]['flag']

with col_vs:
    st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

with col_in2:
    team_b_sel = st.selectbox("Selección 2 (Visita administrativa)", dropdown_options, index=dropdown_options.index("🇦🇷 Argentina"))
    es_name_b = team_b_sel.split(" ", 1)[1]
    en_name_b = COUNTRIES_ES[es_name_b]['en']
    flag_b = COUNTRIES_ES[es_name_b]['flag']

cancha_sel = st.radio(
    "Tipo de Cancha / Localía",
    ["Automática (regla del Mundial: anfitriones de local)", "Cancha Neutral",
     f"Localía para {flag_a} {es_name_a}", f"Localía para {flag_b} {es_name_b}"],
    horizontal=True
)

if en_name_a == en_name_b:
    st.error("Por favor, selecciona dos países distintos para realizar el versus.")
    st.stop()

# Reconstruir Cancha
if cancha_sel.startswith("Automática"):
    cancha_mode = 'auto'
elif cancha_sel == "Cancha Neutral":
    cancha_mode = 'neutral'
elif cancha_sel.startswith("Localía para " + flag_a):
    cancha_mode = '1'
else:
    cancha_mode = '2'

# Función de cálculo de versus para ambos modelos
def predecir_duelo(a, b, cancha):
    sa, sb = states.loc[a], states.loc[b]
    
    # 1. Variables
    feats_full = pd.DataFrame([{
        'elo_diff': sa.elo - sb.elo,
        'squad_value_diff': np.log(sa.squad_value) - np.log(sb.squad_value),
        'h2h_diff': H2H.get((a, b), 0.0),
        'goles_anotados_diff': sa.goles_anotados_avg - sb.goles_anotados_avg,
        'goles_recibidos_diff': sa.goles_recibidos_avg - sb.goles_recibidos_avg,
        'tiros_arco_diff': sa.tiros_arco_avg - sb.tiros_arco_avg,
    }])

    BASE_VARS = ['elo_diff', 'h2h_diff', 'squad_value_diff']
    HYBRID_VARS = ['elo_diff', 'h2h_diff', 'squad_value_diff', 'goles_anotados_diff', 'goles_recibidos_diff', 'tiros_arco_diff']

    # --- MODELO A (BASE) ---
    pa_base = pipe_base.predict_proba(feats_full[BASE_VARS])[0]
    pb_base = pipe_base.predict_proba(pd.DataFrame([{
        'elo_diff': sb.elo - sa.elo,
        'h2h_diff': H2H.get((b, a), 0.0),
        'squad_value_diff': np.log(sb.squad_value) - np.log(sa.squad_value)
    }])[BASE_VARS])[0]
    
    va_base = np.array([pa_base[2], pa_base[1], pa_base[0]])
    vb_base = np.array([pb_base[0], pb_base[1], pb_base[2]])
    
    if cancha == '1' or (cancha == 'auto' and a in ANFITRIONES and b not in ANFITRIONES):
        p_base = va_base
    elif cancha == '2' or (cancha == 'auto' and b in ANFITRIONES and a not in ANFITRIONES):
        p_base = vb_base
    else:
        p_base = (va_base + vb_base) / 2

    # --- MODELO B (HÍBRIDO) ---
    pa_hybrid = pipe_hybrid.predict_proba(feats_full[HYBRID_VARS])[0]
    pb_hybrid = pipe_hybrid.predict_proba(pd.DataFrame([{
        'elo_diff': sb.elo - sa.elo,
        'h2h_diff': H2H.get((b, a), 0.0),
        'squad_value_diff': np.log(sb.squad_value) - np.log(sa.squad_value),
        'goles_anotados_diff': sb.goles_anotados_avg - sa.goles_anotados_avg,
        'goles_recibidos_diff': sb.goles_recibidos_avg - sa.goles_recibidos_avg,
        'tiros_arco_diff': sb.tiros_arco_avg - sa.tiros_arco_avg
    }])[HYBRID_VARS])[0]

    va_hybrid = np.array([pa_hybrid[2], pa_hybrid[1], pa_hybrid[0]])
    vb_hybrid = np.array([pb_hybrid[0], pb_hybrid[1], pb_hybrid[2]])

    if cancha == '1' or (cancha == 'auto' and a in ANFITRIONES and b not in ANFITRIONES):
        p_hybrid = va_hybrid
    elif cancha == '2' or (cancha == 'auto' and b in ANFITRIONES and a not in ANFITRIONES):
        p_hybrid = vb_hybrid
    else:
        p_hybrid = (va_hybrid + vb_hybrid) / 2

    # 2. Goles esperados (Poisson)
    # Poisson A (Base - Elo)
    d_elo = sa.elo - sb.elo
    la_base = float(np.exp(gb0 + gb1 * d_elo))
    lb_base = float(np.exp(gb0 - gb1 * d_elo))
    
    # Poisson B (Híbrido - Híbridas)
    fr = feats_full[['elo_diff', 'h2h_diff', 'squad_value_diff', 'goles_anotados_diff', 'goles_recibidos_diff', 'tiros_arco_diff']].values[0].astype(float)
    s = float(np.dot(bets, fr))
    la_hybrid = float(np.exp(b0s + s))
    lb_hybrid = float(np.exp(b0s - s))

    return p_base, p_hybrid, la_base, lb_base, la_hybrid, lb_hybrid

# Ejecutar Predicción
p_base, p_hybrid, la_base, lb_base, la_hybrid, lb_hybrid = predecir_duelo(en_name_a, en_name_b, cancha_mode)

# Render de Resultados Comparativos en Columnas
col1, col2 = st.columns(2)

def render_prob_bars(probs, t1, t2):
    st.markdown(f"**Victoria {t1}:** {probs[0]:.1%}")
    st.progress(float(probs[0]))
    st.markdown(f"**Empate:** {probs[1]:.1%}")
    st.progress(float(probs[1]))
    st.markdown(f"**Victoria {t2}:** {probs[2]:.1%}")
    st.progress(float(probs[2]))

def get_top_scores(la, lb, probs):
    GRID_MAX = 10
    gidx = np.arange(GRID_MAX + 1)
    grid = np.outer(poisson.pmf(gidx, la), poisson.pmf(gidx, lb))
    grid /= grid.sum()
    gi, gj = np.indices(grid.shape)
    masks = (gi > gj, gi == gj, gi < gj)
    mix = sum(probs[k] * (grid * mk) / (grid * mk).sum() for k, mk in enumerate(masks))
    flat = mix.ravel()
    top_indices = flat.argsort()[::-1][:5]
    top_scores = []
    for ix in top_indices:
        g1, g2 = divmod(ix, GRID_MAX + 1)
        top_scores.append((f"{g1} - {g2}", flat[ix]))
    return top_scores, mix

with col1:
    st.markdown(f'<div class="card"><div class="card-title-base">🔵 Modelo Completo Base (Original)</div>', unsafe_allow_html=True)
    render_prob_bars(p_base, es_name_a, es_name_b)
    
    # Avanza
    p_adv_a_base = p_base[0] + p_base[1] * p_base[0] / (p_base[0] + p_base[2])
    st.markdown(f"⚽ **Si es eliminatoria avanza:** {es_name_a} **{p_adv_a_base:.1%}** / {es_name_b} **{1-p_adv_a_base:.1%}**")
    st.markdown(f"📊 **Goles esperados (Poisson):** {es_name_a} `{la_base:.2f}` — `{lb_base:.2f}` {es_name_b}")
    
    # Marcadores más probables
    scores_base, mix_base = get_top_scores(la_base, lb_base, p_base)
    st.markdown("**Marcadores más probables:**")
    st.markdown(" / ".join([f"`{sc} ({pr:.1%})`" for sc, pr in scores_base]))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="card"><div class="card-title-hybrid">🟣 Modelo Híbrido (Stats + Prior)</div>', unsafe_allow_html=True)
    render_prob_bars(p_hybrid, es_name_a, es_name_b)
    
    # Avanza
    p_adv_a_hybrid = p_hybrid[0] + p_hybrid[1] * p_hybrid[0] / (p_hybrid[0] + p_hybrid[2])
    st.markdown(f"⚽ **Si es eliminatoria avanza:** {es_name_a} **{p_adv_a_hybrid:.1%}** / {es_name_b} **{1-p_adv_a_hybrid:.1%}**")
    st.markdown(f"📊 **Goles esperados (Poisson):** {es_name_a} `{la_hybrid:.2f}` — `{lb_hybrid:.2f}` {es_name_b}")
    
    # Marcadores más probables
    scores_hybrid, mix_hybrid = get_top_scores(la_hybrid, lb_hybrid, p_hybrid)
    st.markdown("**Marcadores más probables:**")
    st.markdown(" / ".join([f"`{sc} ({pr:.1%})`" for sc, pr in scores_hybrid]))
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
#  CUADRO DE ELIMINATORIAS (en vivo desde ESPN) + simulación del campeón
#  Usa el MODELO BASE de esta misma app, con el Elo actualizado por los
#  resultados reales del Mundial. El cuadro real se lee de la API de ESPN.
# ===========================================================================
def _prob_base_st(a, b, cancha, st_df):
    """[P(gana a), empate, P(gana b)] del modelo Base con states arbitrarios (Elo ya actualizado)."""
    sa, sb = st_df.loc[a], st_df.loc[b]
    fa = pd.DataFrame([{'elo_diff': sa.elo - sb.elo, 'h2h_diff': H2H.get((a, b), 0.0),
                        'squad_value_diff': np.log(sa.squad_value) - np.log(sb.squad_value)}])
    fb = pd.DataFrame([{'elo_diff': sb.elo - sa.elo, 'h2h_diff': H2H.get((b, a), 0.0),
                        'squad_value_diff': np.log(sb.squad_value) - np.log(sa.squad_value)}])
    pa = pipe_base.predict_proba(fa[BASE_VARS_G])[0]
    pb = pipe_base.predict_proba(fb[BASE_VARS_G])[0]
    va = np.array([pa[2], pa[1], pa[0]]); vb = np.array([pb[0], pb[1], pb[2]])
    if cancha == '1' or (cancha == 'auto' and a in ANFITRIONES and b not in ANFITRIONES): return va
    if cancha == '2' or (cancha == 'auto' and b in ANFITRIONES and a not in ANFITRIONES): return vb
    return (va + vb) / 2


def _elo_update(ea, eb, ga, gb, k=60.0):
    we = 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))
    w = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
    gd = abs(ga - gb)
    mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8.0))
    delta = k * mult * (w - we)
    return ea + delta, eb - delta


def _actualizar_estados(res):
    """Copia de states con el Elo actualizado por los resultados reales (el Base solo usa Elo+h2h+valor,
       y h2h/valor no cambian → basta con mover el Elo)."""
    st_df = states.copy()
    for r in res.itertuples(index=False):
        a, b = r.local, r.visita
        if a not in st_df.index or b not in st_df.index:
            continue
        ga, gb = int(r.goles_local), int(r.goles_visita)
        ea, eb = _elo_update(st_df.loc[a, 'elo'], st_df.loc[b, 'elo'], ga, gb)
        st_df.loc[a, 'elo'], st_df.loc[b, 'elo'] = ea, eb
    return st_df


def _p_adv(a, b, st_df):
    p = _prob_base_st(a, b, 'auto', st_df)
    s = p[0] + p[2]
    return p[0] + p[1] * p[0] / s if s > 0 else 0.5


def _simular_bracket(bracket, st_df, n_sims=8000, seed=42, fijos_ko=None):
    rng = np.random.default_rng(seed)
    R32, R16, QF, SF, FINAL = bracket['R32'], bracket['R16'], bracket['QF'], bracket['SF'], bracket['FINAL']
    fijos_ko = fijos_ko or {}
    cache = {}

    def padv(a, b):
        if (a, b) not in cache:
            v = _p_adv(a, b, st_df); cache[(a, b)] = v; cache[(b, a)] = 1.0 - v
        return cache[(a, b)]

    def jugar(a, b):
        if a is None or b is None:
            return a if b is None else b
        fij = fijos_ko.get(frozenset({a, b}))
        if fij is not None:
            return fij
        return a if rng.random() < padv(a, b) else b

    win = {('R32', n): Counter() for n in R32}
    for rk, rd in (('R16', R16), ('QF', QF), ('SF', SF)):
        win.update({(rk, n): Counter() for n in rd})
    win[('FINAL', 1)] = Counter()
    for _ in range(n_sims):
        W = {'R32': {}, 'R16': {}, 'QF': {}, 'SF': {}}
        for n, m in R32.items():
            w = jugar(m['home'], m['away'])
            W['R32'][n] = w; win[('R32', n)][w] += 1
        for rk, rd in (('R16', R16), ('QF', QF), ('SF', SF)):
            for n, m in rd.items():
                a = W[m['home'][0]][m['home'][1]]; b = W[m['away'][0]][m['away'][1]]
                w = jugar(a, b); W[rk][n] = w; win[(rk, n)][w] += 1
        a = W[FINAL['home'][0]][FINAL['home'][1]]; b = W[FINAL['away'][0]][FINAL['away'][1]]
        win[('FINAL', 1)][jugar(a, b)] += 1
    reach = {k: {t: c / n_sims for t, c in cnt.items()} for k, cnt in win.items()}
    r32_de = {}
    for n, m in R32.items():
        r32_de[m['home']] = n; r32_de[m['away']] = n

    def suma(rk, rd, t):
        return sum(reach[(rk, n)].get(t, 0.0) for n in rd)

    teams = sorted(r32_de)
    filas = [{'Selección': t, 'P_R16': reach[('R32', r32_de[t])].get(t, 0.0),
              'P_QF': suma('R16', R16, t), 'P_SF': suma('QF', QF, t),
              'P_final': suma('SF', SF, t), 'P_campeon': reach[('FINAL', 1)].get(t, 0.0)} for t in teams]
    tabla = pd.DataFrame(filas).sort_values('P_campeon', ascending=False).reset_index(drop=True)
    return {'tabla': tabla, 'reach': reach}


# ---- Render del cuadro estilo portal deportivo ----
_BRACKET_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;}
body{margin:0;font-family:'Inter',sans-serif;background:transparent;}
.half{margin-bottom:4px;}
.htitle{font-size:.82rem;font-weight:700;color:#0b3d91;text-transform:uppercase;letter-spacing:.04em;margin:4px 0 2px 6px;}
.bracket{display:flex;align-items:stretch;height:420px;}
.round{display:flex;flex-direction:column;flex:1;min-width:150px;padding:0 7px;}
.rbody{display:flex;flex-direction:column;justify-content:space-around;flex:1;}
.rhead{text-align:center;font-size:.6rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px;}
.match{background:#fff;border:1px solid #e2e8f0;border-radius:7px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06);margin:3px 0;}
.match.played{border-color:#86efac;}
.tm{display:flex;align-items:center;gap:5px;padding:3px 7px;font-size:.74rem;color:#475569;border-bottom:1px solid #f1f5f9;}
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
.fhdr{font-size:.7rem;font-weight:700;color:#7a3b91;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}
.finalists{display:flex;gap:10px;margin-bottom:8px;}
.finalists .match{min-width:158px;}
.champ{background:linear-gradient(135deg,#0b3d91,#7a3b91);color:#fff;border-radius:11px;padding:9px 26px;text-align:center;box-shadow:0 5px 14px rgba(11,61,145,.28);}
.champ .lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.12em;opacity:.85;}
.champ .nm{font-size:1.2rem;font-weight:700;margin:1px 0;}
.champ .pr{font-size:.72rem;opacity:.92;}
</style>
"""
_HEAD = {'R32': 'Dieciseisavos', 'R16': 'Octavos', 'QF': 'Cuartos', 'SF': 'Semifinal'}


def _argmax_reach(reach, key):
    d = reach.get(key, {})
    if not d:
        return None, 0.0
    t = max(d, key=d.get)
    return t, d[t]


def _box_info(bracket, reach, rnd, num):
    m = bracket['FINAL'] if rnd == 'FINAL' else bracket[rnd][num]
    key = ('FINAL', 1) if rnd == 'FINAL' else (rnd, num)
    if rnd == 'R32':
        home, away = m['home'], m['away']
        slots = [(home, reach[key].get(home, 0.0)), (away, reach[key].get(away, 0.0))]
        played = m['state'] == 'post' and m['gh'] is not None and m['ga'] is not None
        winner = (home if m['gh'] > m['ga'] else away) if played else _argmax_reach(reach, key)[0]
        return {'slots': slots, 'played': played, 'score': (m['gh'], m['ga']) if played else None, 'winner': winner}
    th, ph = _argmax_reach(reach, m['home'])
    ta, pa = _argmax_reach(reach, m['away'])
    return {'slots': [(th, ph), (ta, pa)], 'played': False, 'score': None, 'winner': _argmax_reach(reach, key)[0]}


def _box_html(info):
    rows = ''
    for idx, (team, p) in enumerate(info['slots']):
        es = _nombre(team) if team else '—'
        fl = _bandera(team) if team else '·'
        val = str(info['score'][idx]) if info['played'] else f'{p:.0%}'
        if info['played']:
            cls = 'tm win' if team == info['winner'] else 'tm lose'
        else:
            cls = 'tm win' if team and team == info['winner'] else 'tm'
        rows += (f'<div class="{cls}"><span class="fl">{fl}</span>'
                 f'<span class="nm" title="{es}">{es}</span><span class="pr">{val}</span></div>')
    return f'<div class="match{" played" if info["played"] else ""}">{rows}</div>'


def _collect_orden(bracket, side, acc):
    rnd, num = side
    if rnd != 'R32':
        m = bracket[rnd][num]
        _collect_orden(bracket, m['home'], acc)
        _collect_orden(bracket, m['away'], acc)
    acc.setdefault(rnd, []).append(num)
    return acc


def _half_html(bracket, sim, sf_num):
    reach = sim['reach']
    acc = _collect_orden(bracket, ('SF', sf_num), {})
    teams = [bracket['R32'][n][s] for n in acc['R32'] for s in ('home', 'away')]
    pcamp = sim['tabla'].set_index('Selección')['P_campeon']
    fuerte = max(teams, key=lambda t: pcamp.get(t, 0))
    cols = ''
    for rnd in ('R32', 'R16', 'QF', 'SF'):
        boxes = ''.join(_box_html(_box_info(bracket, reach, rnd, n)) for n in acc[rnd])
        cols += f'<div class="round"><div class="rhead">{_HEAD[rnd]}</div><div class="rbody">{boxes}</div></div>'
    return (f'<div class="half"><div class="htitle">Lado de {_nombre(fuerte)} {_bandera(fuerte)}</div>'
            f'<div class="bracket">{cols}</div></div>')


def _final_html(bracket, sim):
    fi = _box_info(bracket, sim['reach'], 'FINAL', 1)
    champ = sim['tabla'].iloc[0]
    return (f'<div class="finalwrap"><div class="fhdr">★ Final ★</div>'
            f'<div class="finalists">{_box_html(fi)}</div>'
            f'<div class="champ"><div class="lbl">Campeón más probable</div>'
            f'<div class="nm">{_bandera(champ["Selección"])} {_nombre(champ["Selección"])}</div>'
            f'<div class="pr">campeón en el {champ["P_campeon"]:.1%} de los torneos simulados</div></div></div>')


def _bracket_completo_html(bracket, sim):
    return (_BRACKET_CSS + _half_html(bracket, sim, 1) + _final_html(bracket, sim)
            + _half_html(bracket, sim, 2))


@st.cache_data(ttl=900, show_spinner=False)
def cargar_espn_app():
    try:
        res = espn_live.traer_resultados()
        r32 = list(espn_live.bracket_eliminatorias()['R32'].values())  # los 16 cruces reales
        return res, r32, None
    except Exception as e:
        return pd.DataFrame(), None, str(e)


@st.cache_data(show_spinner='Simulando las eliminatorias (8.000 torneos)…')
def sim_campeon(key):
    res, r32, err = cargar_espn_app()
    if not r32 or len(res) < 72:
        return None
    # cuadro real = cruces de ESPN colocados en la plantilla FIFA (árbol oficial)
    br = _motor.bracket_real(res, r32)
    st_df = _actualizar_estados(res)
    fk = _motor.ko_fijos(res)            # fija los KO ya jugados (cualquier ronda)
    return {'bracket': br, 'sim': _simular_bracket(br, st_df, n_sims=8000, fijos_ko=fk)}


ESPN_RES, ESPN_BR, ESPN_ERR = cargar_espn_app()
ESPN_KEY = '' if len(ESPN_RES) == 0 else f'{len(ESPN_RES)}-{ESPN_RES.fecha.max()}'

# Pestañas de detalle
tab0, tab1, tab2, tab3 = st.tabs(["🗺️ Cuadro de eliminatorias", "📊 Comparativa de Atributos",
                                  "🔲 Matrices de Marcadores", "ℹ️ Metodología y Métricas"])

with tab0:
    st.markdown("### 🗺️ Cuadro de eliminatorias — camino al título")
    st.markdown("Cuadro **real, ya definido** (los 16 cruces de dieciseisavos vienen de la API de ESPN) "
                "y la **simulación del campeón** sobre él: 8.000 torneos jugando solo las eliminatorias, "
                "con el **modelo Base** y el Elo de cada selección ya actualizado por sus resultados "
                "reales en el Mundial.")
    if ESPN_ERR:
        st.warning(f"No se pudo contactar a ESPN ({ESPN_ERR}). El cuadro se mostrará cuando vuelva la conexión.")
    payload = sim_campeon(ESPN_KEY) if not ESPN_ERR else None
    if payload is None:
        st.info("El cuadro de eliminatorias aún no está publicado en ESPN. Se llena solo cuando termine "
                "la fase de grupos y se definan los cruces.")
    else:
        br_e, sim_e = payload['bracket'], payload['sim']
        tabla_e = sim_e['tabla']
        st.markdown("##### 🏆 Campeón según la simulación")
        cc = st.columns(3)
        for col, (_, r), md in zip(cc, tabla_e.head(3).iterrows(), ["🥇", "🥈", "🥉"]):
            col.metric(f"{md} {_bandera(r['Selección'])} {_nombre(r['Selección'])}", f"{r['P_campeon']:.1%}")
        components.html(_bracket_completo_html(br_e, sim_e), height=1010, scrolling=True)
        st.caption("En **dieciseisavos** se ven los cruces reales con la probabilidad de avanzar de cada "
                   "selección (verde = partido ya jugado, con su marcador). De **octavos en adelante**, "
                   "cada casillero muestra el equipo **más probable** de ocuparlo según la simulación "
                   "(proyección — el cruce exacto aún no está definido). El borde azul marca al favorito.")
        with st.expander("📋 Probabilidades completas por ronda (las 32 selecciones)"):
            show = tabla_e.copy()
            show['Selección'] = [f"{_bandera(t)} {_nombre(t)}" for t in show['Selección']]
            show = show.rename(columns={'P_R16': 'P(8vos)', 'P_QF': 'P(4tos)', 'P_SF': 'P(semis)',
                                        'P_final': 'P(final)', 'P_campeon': 'P(campeón)'})
            pct = ['P(8vos)', 'P(4tos)', 'P(semis)', 'P(final)', 'P(campeón)']
            st.dataframe(show.style.format({c: '{:.1%}' for c in pct})
                         .background_gradient(subset=['P(campeón)'], cmap='Blues'),
                         hide_index=True, width='stretch')

with tab1:
    sa, sb = states.loc[en_name_a], states.loc[en_name_b]
    
    comparativa_df = pd.DataFrame({
        'Atributo': [
            'Elo FIFA Rating',
            'Valor de Plantilla (Euros)',
            'H2H Histórico (Diferencia Goles)',
            'Goles Anotados Recientes (últimos 8)',
            'Goles Recibidos Recientes (últimos 8)',
            'Tiros al Arco Recientes (últimos 8)',
            'Corners Recientes (últimos 8)',
            'Posesión Reciente (últimos 8)'
        ],
        f'{flag_a} {es_name_a}': [
            f"{sa.elo:.1f}",
            f"€{sa.squad_value:,.0f}",
            "---",
            f"{sa.goles_anotados_avg:.2f}",
            f"{sa.goles_recibidos_avg:.2f}",
            f"{sa.tiros_arco_avg:.2f}",
            f"{sa.corners_avg:.2f}",
            f"{sa.posesion_avg:.1%}"
        ],
        f'{flag_b} {es_name_b}': [
            f"{sb.elo:.1f}",
            f"€{sb.squad_value:,.0f}",
            "---",
            f"{sb.goles_anotados_avg:.2f}",
            f"{sb.goles_recibidos_avg:.2f}",
            f"{sb.tiros_arco_avg:.2f}",
            f"{sb.corners_avg:.2f}",
            f"{sb.posesion_avg:.1%}"
        ],
        'Diferencia (Local − Visita)': [
            f"{sa.elo - sb.elo:+.1f}",
            f"€{sa.squad_value - sb.squad_value:,.0f}",
            f"{H2H.get((en_name_a, en_name_b), 0.0):+.2f}",
            f"{sa.goles_anotados_avg - sb.goles_anotados_avg:+.2f}",
            f"{sa.goles_recibidos_avg - sb.goles_recibidos_avg:+.2f}",
            f"{sa.tiros_arco_avg - sb.tiros_arco_avg:+.2f}",
            f"{sa.corners_avg - sb.corners_avg:+.2f}",
            f"{sa.posesion_avg - sb.posesion_avg:+.1%}"
        ]
    })
    
    st.table(comparativa_df)

with tab2:
    st.markdown("### Matriz de Marcadores del Partido (Corte en 5 Goles)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz Base
    sns.heatmap(mix_base[:6, :6], annot=True, fmt=".1%", cmap="Blues", ax=ax1, cbar=False)
    ax1.set_title(f"Modelo Completo Base — {es_name_a} vs {es_name_b}")
    ax1.set_xlabel(f"Goles de {es_name_b}")
    ax1.set_ylabel(f"Goles de {es_name_a}")
    
    # Matriz Híbrido
    sns.heatmap(mix_hybrid[:6, :6], annot=True, fmt=".1%", cmap="Purples", ax=ax2, cbar=False)
    ax2.set_title(f"Modelo Híbrido — {es_name_a} vs {es_name_b}")
    ax2.set_xlabel(f"Goles de {es_name_b}")
    ax2.set_ylabel(f"Goles de {es_name_a}")
    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.markdown("""
    ### Comparativa de Modelos Científicos

    Este panel interactivo permite enfrentar las predicciones de dos metodologías desarrolladas para el Mundial 2026:

    1. **Modelo Completo Base (Original)**:
       - **Variables**: `elo_diff`, `h2h_diff`, y `squad_value_diff` (elegidas por selección forward + VIF con validación temporal).
       - **Filosofía**: Se enfoca puramente en la jerarquía, poder financiero de la plantilla y el historial histórico de enfrentamientos de las selecciones a largo plazo.
       - **Rendimiento**: Log-Loss `0.8517` en test (2025–26) · `0.9064` en CV temporal walk-forward.

    2. **Modelo Híbrido (Stats + Prior)**:
       - **Variables**: Prior de calidad (`elo_diff`, `h2h_diff`, `squad_value_diff`) + Modificadores de forma reciente (`goles_anotados_diff`, `goles_recibidos_diff`, `tiros_arco_diff`).
       - **Filosofía**: Incorpora la inercia reciente del equipo (promedios móviles de los últimos 8 partidos) para ajustar la probabilidad, de modo que equipos en racha positiva o negativa sean penalizados/premiados en sus opciones del Mundial.
       - **Rendimiento**: Log-Loss `0.8507` en test (2025–26) · `0.9096` en CV temporal walk-forward.

    > **¿Cuál es mejor?** Estadísticamente están **empatados**: el híbrido gana por 0.001 en test pero
    > el base gana en validación cruzada — ambas diferencias quedan muy por debajo del ruido (±0.04
    > entre cortes de la CV). Por eso esta app los muestra **lado a lado** en vez de declarar un
    > ganador: cuando dos modelos honestos discrepan, la discrepancia es información.

    > **Nota sobre el Empate en Eliminatorias**: En caso de empate en tiempo regular, el porcentaje para clasificar (avanzar ronda) se calcula distribuyendo la probabilidad del empate de forma proporcional a las chances de victoria regular de cada equipo, simulando la prórroga y penaltis.
    """)
