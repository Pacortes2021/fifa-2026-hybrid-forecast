# ⚽ FIFA 2026 & Domestic Leagues: Hybrid ML Forecasting Platform

Plataforma integral de analítica avanzada, modelado probabilístico y predicción de fútbol basada en **Machine Learning Híbrido**, **Matrices Bivariadas de Poisson / Dixon-Coles** y **Simulaciones Monte Carlo Vectorizadas**.

El sistema cuenta con un portal central en **Streamlit** que permite interactuar con **18 aplicaciones de predicción independientes**: la Copa Mundial de la FIFA 2026 y las **17 ligas domésticas más competitivas de Europa y América**.

---

## 🚀 Portal Web Interactivo en Vivo

👉 **[Acceder a la Plataforma Streamlit](https://pacortes2021-fifa-2026-hybrid-forecast-app-anrhbf.streamlit.app/)**

### Competiciones Disponibles
1. 🏆 **Copa Mundial de la FIFA 2026**: Bracket oficial de 48 selecciones, Elo histórico (1872–2026) y Homología Persistente (TDA experimental).
2. 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Premier League** (Inglaterra)
3. 🇪🇸 **LaLiga EA Sports** (España)
4. 🇩🇪 **Bundesliga** (Alemania)
5. 🇮🇹 **Serie A** (Italia)
6. 🇫🇷 **Ligue 1** (Francia)
7. 🇳🇱 **Eredivisie** (Países Bajos)
8. 🇵🇹 **Primeira Liga** (Portugal)
9. 🇧🇪 **Jupiler Pro League** (Bélgica)
10. 🇹🇷 **Trendyol Süper Lig** (Turquía)
11. 🏴󠁧󠁢󠁳󠁣󠁴󠁿 **Scottish Premiership** (Escocia)
12. 🇦🇹 **Austrian Bundesliga** (Austria)
13. 🇩🇰 **Danish Superliga** (Dinamarca)
14. 🇬🇷 **Super League** (Grecia)
15. 🇧🇷 **Brasileirão Série A** (Brasil)
16. 🇲🇽 **Liga MX** (México — Apertura / Clausura y Liguilla)
17. 🇨🇱 **Primera División de Chile**
18. 🇦🇷 **Liga Profesional de Fútbol Argentino** (Zonas, promedios y tabla anual)

---

## 🧠 Arquitectura del Sistema

El proyecto está diseñado bajo un patrón modular homogéneo donde cada liga cuenta con su propio ecosistema de datos, recolectores, motor analítico e interfaz de usuario, orquestados desde un enrutador central:

```
fifa-2026-hybrid-forecast/
├── app.py                     # Enrutador principal del portal Streamlit
├── requirements.txt           # Dependencias de producción
├── tests/
│   └── smoke.py              # Suite de smoke tests unificada (17 ligas)
├── .github/workflows/
│   └── refresh_data.yml      # Pipeline CI/CD: ingesta diaria automatizada (06:00 UTC)
│
├── [eng | esp | bund | ita | fra | ned | por | bel | tur | sco | aut | den | gre | bra | mex | chile | arg]/   # Módulos por liga
│   ├── motor.py               # StateTracker, Poisson Dixon-Coles, ML Stacking y Monte Carlo
│   ├── app_<liga>.py          # Interfaz Streamlit (Versus, H2H, Tabla MC, Mercados, Validación)
│   ├── recolectar.py          # Scraper/ingestor de partidos y fixture desde ESPN API
│   ├── recolectar_boxscore.py # Ingestor de estadísticas detalladas por cotejo
│   └── data/                  # Almacenamiento local versionado
│       ├── partidos.csv       # Historial de partidos disputados
│       ├── fixture.csv        # Calendario oficial y próximos encuentros
│       ├── equipos.csv        # Metadatos, colores y escudos
│       ├── box_score.csv      # Estadísticas avanzadas de partido (tiros, corners, pases)
│       └── advanced_features*.csv # Altitud, coordenadas geográficas y valor de plantilla
│
└── lab/                       # Laboratorio de I+D del Mundial 2026
    ├── motor.py               # Motor base vs híbrido para selecciones nacionales
    └── tda_motor.py           # Análisis Topológico de Datos (Vietoris-Rips / Ripser)
```

---

## 🔬 Metodología de Modelado (Por Liga)

Cada liga implementa un flujo riguroso para evitar fuga de información (*data leakage*) y garantizar calibración probabilística real:

### 1. Ingesta y `StateTracker` Cronológico
A medida que se recorre el calendario en orden temporal estricto:
- **Rating Elo Dinámico**: Adaptación continua con K-Factor ponderado y corrección por margen de victoria.
- **Forma Reciente**: Ventana móvil de los últimos 5 encuentros (puntos logrados, goles a favor y goles en contra).
- **Métricas de Rendimiento (Box Scores)**: Diferenciales de tiros al arco, bloqueos, centros efectivos, despejes e intercepciones.
- **Factores de Contexto**: Ventaja de localía empírica (`HOME_ADV`), distancias de viaje logarítmicas (cálculo geodésico Haversine) y diferenciales de altitud sobre el nivel del mar.
- **Jerarquía Financiera**: Logaritmo del valor de mercado de la plantilla extraído vía Transfermarkt.

### 2. Validación Temporal Honesta (Walk-Forward Split)
- **Train (Entrenamiento)**: Partidos de temporadas históricas (<= 2023).
- **Calibration (Calibración OOS)**: Partidos de temporada 2024.
- **Test (Prueba Ciega en Vivo)**: Partidos de temporadas >= 2025 y 2026.

### 3. Ensamble de Modelos (Stacking)
- **LASSO (Logistic Regression L1 via SAGA)**: Selección estricta de variables que anula coeficientes no informativos o redundantes.
- **Random Forest Classifier**: Captura de interacciones no lineales complejas entre variables de forma y contexto.
- **XGBoost**: Gradient Boosting calibrado.
- **Meta-Modelo Stacking**: Ponderación óptima calculada mediante optimización acotada (Nelder-Mead) sobre el conjunto de calibración fuera de muestra.

### 4. Modelo Bivariado de Goles (Dixon-Coles / Poisson)
- Modelado GLM de tasas de anotación esperadas (lambdas) condicionadas por el diferencial de Elo y la ventaja de localía.
- Generación de la **matriz bivariada de probabilidad 10x10** con ajuste de dependencia en marcadores bajos (0-0, 1-0, 0-1, 1-1).
- Re-calibración exacta de la matriz contra las probabilidades 1X2 del modelo Stacking para proyectar mercados derivados:
  - **1X2 Tradicional**
  - **Draw No Bet (DNB / Apuesta sin Empate)**
  - **Líneas Over / Under (1.5, 2.5, 3.5 goles)**
  - **Both Teams To Score (Ambos Anotan - Sí/No)**
  - **Cuotas Justas de Mercado** (1 / P).

### 5. Simulación Monte Carlo Vectorizada
- Simulación de **3,000 a 50,000 iteraciones de la temporada completa** utilizando vectorización pura en NumPy.
- Pre-cálculo de la matriz completa de emparejamientos y resolución simultánea de miles de calendarios en milisegundos (~1,200x más rápida que iteraciones basadas en bucles).
- Salida probabilística completa:
  - Probabilidad de coronarse **Campeón**.
  - Probabilidad de clasificación a **Copas Continentales** (Champions League, Europa League, Conference League, Copa Libertadores, Copa Sudamericana).
  - Probabilidad de **Descenso**.
  - Puntos esperados proyectados al final de la temporada.

---

## ⚡ Automatización e Infraestructura (CI/CD)

- **Actualización Diaria**: Un GitHub Action programado (`06:00 UTC`) ejecuta los scripts de recolección de todas las ligas, capturando marcadores recientes y actualizando los fixtures futuros.
- **Caché Criptográfico Inteligente**: Los motores calculan un hash SHA-256 (`_cache_key`) de los datasets y configuraciones. Si no existen datos nuevos, los modelos y simulaciones se cargan instantáneamente desde disco (`simulacion_mc.pkl`), evitando re-entrenamientos innecesarios.

---

## 💻 Instalación y Uso Local

### Prerrequisitos
Python 3.10 o superior instalado.

### 1. Clonar el repositorio
```bash
git clone https://github.com/Pacortes2021/fifa-2026-hybrid-forecast.git
cd fifa-2026-hybrid-forecast
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación
```bash
streamlit run app.py
```

### 4. Ejecutar la suite de pruebas (Smoke Tests)
Para validar que los motores de las 11 ligas cargan, predicen y simulan sin errores:
```bash
python3 tests/smoke.py
```
O probar ligas específicas:
```bash
python3 tests/smoke.py eng esp bund fra ned por
```

---

## 🏆 Respaldo Metodológico: Mundial FIFA 2026

Para la competición del Mundial de selecciones (`lab/`), el modelo procesó **49,373 partidos históricos internacionales desde 1872 hasta 2026**:
- Fusión jerárquica de ratings Elo con cobertura del 100% (cero nulos en 5,659 partidos recientes).
- Ponderación por K-factor de torneos oficiales FIFA frente a cotejos amistosos.
- Modelado del bracket de 48 participantes con simetría de localía para sedes neutrales en Estados Unidos, México y Canadá.
- Notebooks detallados y reproducibles en la carpeta [`notebooks/`](notebooks/).

---

## 📜 Licencia y Autoría

Desarrollado y mantenido por **Pablo Cortés** ([@Pacortes2021](https://github.com/Pacortes2021)).  
Diseñado con fines de investigación académica, ingeniería de machine learning y analítica deportiva avanzada.
