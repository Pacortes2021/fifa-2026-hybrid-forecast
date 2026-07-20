"""
Portal Maestro de Predicciones y Proyecciones Deportivas (Machine Learning).
Enruta a los simuladores de la Copa Mundial 2026, Liga MX, Brasileirão y Liga Chilena.
"""
import os
import sys
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


def limpiar_cache_importacion():
    """Limpia los módulos compartidos de sys.modules para evitar colisiones
    entre las distintas ligas que usan archivos con el mismo nombre (ej. motor.py)."""
    modulos_a_limpiar = [
        "motor", "recolectar", "recolectar_boxscore", 
        "espn_live", "app_lab", "app_mex", "app_bra", "app_chile", "app_esp"
    ]
    for mod in modulos_a_limpiar:
        if mod in sys.modules:
            del sys.modules[mod]
        # También limpiar submódulos si los hay
        keys_to_del = [k for k in sys.modules.keys() if k.startswith(mod + ".")]
        for k in keys_to_del:
            del sys.modules[k]


# Barra lateral para navegación
st.sidebar.image("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=200&auto=format&fit=crop", caption="Predicciones ML", use_container_width=True)
st.sidebar.markdown("## 🎮 Navegación")

torneo_seleccionado = st.sidebar.selectbox(
    "Selecciona el Torneo:",
    [
        "🏆 Copa Mundial 2026",
        "🇲🇽 Liga MX (México)",
        "🇧🇷 Brasileirão (Brasil)",
        "🇨🇱 Liga Chilena (Primera)",
        "🇪🇸 LaLiga (España)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre los Modelos")
st.sidebar.caption(
    "Todos los modelos de ligas domésticas utilizan regularización LASSO (L1) con optimizador SAGA "
    "sobre variables point-in-time de boxscore y priors de Elo y Valor de Plantilla. "
    "El Mundial utiliza un enfoque híbrido bayesiano basado en Elo."
)

# Enrutador
limpiar_cache_importacion()

if torneo_seleccionado == "🏆 Copa Mundial 2026":
    # Mover al directorio de lab
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))
    import lab.app_lab as lab_app
    lab_app.run_app()

elif torneo_seleccionado == "🇲🇽 Liga MX (México)":
    # Mover al directorio de mex
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mex"))
    import mex.app_mex as mex_app
    mex_app.run_app()

elif torneo_seleccionado == "🇧🇷 Brasileirão (Brasil)":
    # Mover al directorio de bra
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bra"))
    import bra.app_bra as bra_app
    bra_app.run_app()

elif torneo_seleccionado == "🇨🇱 Liga Chilena (Primera)":
    # Mover al directorio de chile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chile"))
    import chile.app_chile as chile_app
    chile_app.run_app()

elif torneo_seleccionado == "🇪🇸 LaLiga (España)":
    # Mover al directorio de esp
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "esp"))
    import esp.app_esp as esp_app
    esp_app.run_app()
