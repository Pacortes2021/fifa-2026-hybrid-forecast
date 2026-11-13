"""
Portal Maestro de Predicciones y Proyecciones Deportivas (Machine Learning).
Enruta a los simuladores de la Copa Mundial 2026, Champions y Nations League, NBA y las ligas domésticas (ver TORNEOS).
"""
import hmac
import importlib
import os
import sys
import threading
import streamlit as st

# Configuración única de Streamlit (debe ser la primera llamada de st)
st.set_page_config(
    page_title="Portal de Predicciones Fútbol ML",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilizado CSS Premium Global para el Portal
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.sidebar .sidebar-content {
    background-color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)


BASE = os.path.dirname(os.path.abspath(__file__))

# Torneo -> (carpeta, módulo de la app). Cada carpeta trae módulos con nombres repetidos
# (motor.py, recolectar.py, ...), que Python guarda en sys.modules por nombre.
TORNEOS = {
    "🏆 Copa Mundial 2026": ("lab", "app_lab"),
    "⭐ UEFA Champions League": ("ucl", "app_ucl"),
    "🇪🇺 UEFA Nations League": ("unl", "app_unl"),
    "🏀 NBA (Estados Unidos)": ("nba", "app_nba"),
    "🇦🇷 Liga Profesional (Argentina)": ("arg", "app_arg"),
    "🇲🇽 Liga MX (México)": ("mex", "app_mex"),
    "🇧🇷 Brasileirão (Brasil)": ("bra", "app_bra"),
    "🇨🇱 Liga Chilena (Primera)": ("chile", "app_chile"),
    "🇪🇸 LaLiga (España)": ("esp", "app_esp"),
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Inglaterra)": ("eng", "app_eng"),
    "🇩🇪 Bundesliga (Alemania)": ("bund", "app_bund"),
    "🇮🇹 Serie A (Italia)": ("ita", "app_ita"),
    "🇫🇷 Ligue 1 (Francia)": ("fra", "app_fra"),
    "🇳🇱 Eredivisie (Países Bajos)": ("ned", "app_ned"),
    "🇵🇹 Primeira Liga (Portugal)": ("por", "app_por"),
    "🇧🇪 Jupiler Pro League (Bélgica)": ("bel", "app_bel"),
    "🇹🇷 Süper Lig (Turquía)": ("tur", "app_tur"),
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership (Escocia)": ("sco", "app_sco"),
    "🇦🇹 Austrian Bundesliga (Austria)": ("aut", "app_aut"),
    "🇩🇰 Danish Superliga (Dinamarca)": ("den", "app_den"),
    "🇬🇷 Super League (Grecia)": ("gre", "app_gre"),
    "🇳🇴 Eliteserien (Noruega)": ("nor", "app_nor"),
    "🇸🇪 Allsvenskan (Suecia)": ("swe", "app_swe"),
}
MODULOS_COMPARTIDOS = ("motor", "recolectar", "recolectar_boxscore", "recolectar_equipos",
                       "recolectar_squad_values", "espn_live", "tda_motor")
_DIRS_TORNEOS = {os.path.join(BASE, carpeta) for carpeta, _ in TORNEOS.values()}


def _estado_proceso():
    """Estado compartido por todas las sesiones del proceso.

    Streamlit atiende cada sesión en un hilo del mismo proceso, y sys.path / sys.modules son
    globales. Antes, dos usuarios abriendo ligas distintas a la vez podían importar el
    motor.py de la otra liga; además pickle resuelve las clases guardadas en caché por el
    nombre global "motor". El lock serializa el ruteo y la ejecución de cada página, y el
    registro guarda los módulos propios de cada torneo para reactivarlos en cada rerun.
    Vive en `sys` (no en st.cache_resource) para sobrevivir a st.cache_resource.clear();
    dict.setdefault es atómico, así que todos los hilos obtienen el mismo objeto.
    """
    return vars(sys).setdefault("_portal_predicciones_estado", {"lock": threading.RLock(), "modulos": {}})


def _activar_torneo(carpeta, modulo):
    """Deja sys.path y sys.modules como si solo existiera este torneo y devuelve su app.
    Debe llamarse con el lock tomado."""
    estado = _estado_proceso()
    sys.path[:] = [p for p in sys.path if p not in _DIRS_TORNEOS]
    if BASE not in sys.path:
        sys.path.append(BASE)  # para importar "<carpeta>.<app>" como paquete
    sys.path.insert(0, os.path.join(BASE, carpeta))
    for nombre in MODULOS_COMPARTIDOS:
        for k in [k for k in sys.modules if k == nombre or k.startswith(nombre + ".")]:
            del sys.modules[k]

    nombre_app = f"{carpeta}.{modulo}"
    if nombre_app not in estado["modulos"]:
        sys.modules.pop(nombre_app, None)  # importarla de nuevo para capturar sus módulos propios
        app_mod = importlib.import_module(nombre_app)
        propios = {n: sys.modules[n] for n in MODULOS_COMPARTIDOS if n in sys.modules}
        estado["modulos"][nombre_app] = (app_mod, propios)
    app_mod, propios = estado["modulos"][nombre_app]
    sys.modules.update(propios)
    return app_mod


# Barra lateral para navegación
logo_path = os.path.join(os.path.dirname(__file__), "assets", "sidebar_logo.jpg")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, caption="Predicciones ML", width="stretch")

st.sidebar.markdown("## 🎮 Navegación")

torneos_disponibles = list(TORNEOS) + ["📜 Archivo Prospectivo (Ciego)"]

# Detección inteligente por parámetro URL (ej: ?torneo=nba, ?torneo=mex, ?torneo=esp)
idx_def = 0
qp_torneo = str(st.query_params.get("torneo", "")).lower().strip()
if qp_torneo:
    if qp_torneo in ("archivo", "prospectivo", "log"):
        idx_def = len(torneos_disponibles) - 1
    else:
        _ALIAS_TORNEO = {
            "mundial": "lab", "fifa": "lab", "champions": "ucl", "nations": "unl",
            "basket": "nba", "chi": "chile",
        }
        qp_cod = _ALIAS_TORNEO.get(qp_torneo, qp_torneo)
        for i, (etq, (carpeta, _)) in enumerate(TORNEOS.items()):
            if qp_cod == carpeta or qp_torneo in etq.lower():
                idx_def = i
                break

torneo_seleccionado = st.sidebar.selectbox(
    "Selecciona el Torneo:",
    torneos_disponibles,
    index=idx_def
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre los Modelos")
st.sidebar.caption(
    "Todos los modelos de ligas domésticas utilizan regularización LASSO (L1) con optimizador SAGA, "
    "Random Forest y Stacking óptimo sobre variables de Pi-Ratings, Fatigue Factor, Distancia de Viajes (Haversine) "
    "y priors de ELO y Valor de Plantilla."
)

def _es_admin():
    """El botón de actualizar/re-entrenar descarga de ESPN, re-entrena y borra la caché de
    todas las ligas para todos los usuarios. En el portal público solo se muestra con
    ?admin=<clave> (clave en st.secrets["admin_key"]) o con PORTAL_ADMIN=1 al correr en local.
    Los datos se actualizan solos cada día con el workflow de GitHub Actions."""
    if os.environ.get("PORTAL_ADMIN") == "1":
        return True
    try:
        clave = str(st.secrets.get("admin_key", ""))
    except Exception:  # sin secrets.toml
        clave = ""
    dada = str(st.query_params.get("admin", ""))
    return bool(clave) and hmac.compare_digest(dada, clave)


st.session_state["portal_admin"] = _es_admin()

# Enrutador
if torneo_seleccionado == "📜 Archivo Prospectivo (Ciego)":
    st.markdown("## 📜 Auditoría Prospectiva Ciega (Predicciones Pre-Partido Inmutables)")
    from comun.archivo_ui import render_archivo_prospectivo
    render_archivo_prospectivo(competicion=None, key_prefix="portal_arch")
else:
    carpeta, modulo = TORNEOS[torneo_seleccionado]
    with _estado_proceso()["lock"]:
        _activar_torneo(carpeta, modulo).run_app()
