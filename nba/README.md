# 🏀 Predictor & Simulador NBA (National Basketball Association)

Módulo predictivo y simulador estocástico para la **NBA**, adaptado minuciosamente a la naturaleza y analítica avanzada del baloncesto profesional moderno.

Datos obtenidos desde la **API pública de ESPN** (`sports/basketball/nba`).

---

## 🚀 Cómo Ejecutar

```bash
# Lanzar la aplicación Streamlit de la NBA (desde la raíz del repositorio)
streamlit run nba/app_nba.py

# O desde el portal maestro con selector de ligas
streamlit run app.py

# Actualizar partidos y fixture oficial desde ESPN
python3 nba/recolectar.py

# Actualizar estadísticas detalladas de box score (robos, pérdidas, etc.)
python3 nba/recolectar_boxscore.py
```

---

## 🔬 Metodología y Trabajo Minucioso de Variables

El baloncesto difiere radicalmente del fútbol: no existen los empates, el ritmo de posesiones domina el marcador final y los equipos anotan entre 100 y 130 puntos por encuentro. El sistema modela esta dinámica a través de:

1. **Sistema Elo NBA (FiveThirtyEight methodology)**:
   - Ventaja de localía calibrada en **+100 puntos Elo** (~3.2 puntos de margen).
   - Multiplicador por margen de victoria (MOV):
     $$K = 20 \times \frac{(MOV + 3)^{0.8}}{7.5 + 0.006 \times |\text{Elo}_w - \text{Elo}_l|}$$
   - Penalización por fatiga y partidos en noches consecutivas (**Back-to-Back**: -40 puntos Elo si el equipo llega con 0 días de descanso).
   - Regresión inter-temporada al promedio histórico (75% Elo previo + 25% 1505).

2. **Dean Oliver Four Factors (Factores Clave de Eficiencia)**:
   - **Effective Field Goal % (eFG%)**: Pondera el valor añadido del tiro de 3 puntos $\frac{FGM + 0.5 \times 3PM}{FGA}$.
   - **Turnover Rate (TOV%)**: Tasa de pérdidas de balón por posesión.
   - **Offensive Rebound % (ORB%)**: Capacidad de generar segundas oportunidades.
   - **Free Throw Rate (FTRate)**: Ratio de tiros libres intentados frente a tiros de campo.

3. **Posesiones y Ritmo de Juego (Pace)**:
   - Estimación de posesiones rodantes por 48 minutos para proyectar el ritmo esperado del partido:
     $$\text{Pace}_{\text{exp}} = \text{Pace}_{\text{local}} + \text{Pace}_{\text{visita}} - \text{Pace}_{\text{liga}}$$

4. **Ratings Ofensivos y Defensivos (ORTG & DRTG)**:
   - Puntos anotados y recibidos normalizados por cada 100 posesiones de juego, eliminando el sesgo de equipos con ritmos rápidos o lentos.

5. **Modelos de Machine Learning**:
   - **Clasificación de Ganador (Moneyline)**: LASSO (L1 sparsi-fied), Random Forest, XGBoost y Stacking Ensemble óptimo.
   - **Regresión de Margen (Spread)** y **Totales (Over/Under)**: Modelos Ridge para estimar la diferencia de puntos y puntuación combinada.

6. **Simulador de Monte Carlo**:
   - **Por Partido**: 4.000 simulaciones estocásticas con distribución bivariada gaussiana para proyectar coberturas de spread (hándicap) y líneas Over/Under.
   - **Por Temporada**: Simulación del fixture completo, clasificación Este/Oeste, torneo Play-In (7 al 10) y series al mejor de 7 partidos de Playoffs hasta coronar al Campeón de la NBA.

---

## 📁 Estructura del Módulo

- [`recolectar_equipos.py`](file:///Users/pabloignaciocortesvielma/Documents/UdeC/2026-1/fifa-2026-hybrid-forecast/nba/recolectar_equipos.py): Extrae las 30 franquicias con sus IDs, logos HD, colores, conferencia y división.
- [`recolectar.py`](file:///Users/pabloignaciocortesvielma/Documents/UdeC/2026-1/fifa-2026-hybrid-forecast/nba/recolectar.py): Descarga el histórico de partidos (>5.300 juegos) y fixture oficial de la temporada.
- [`recolectar_boxscore.py`](file:///Users/pabloignaciocortesvielma/Documents/UdeC/2026-1/fifa-2026-hybrid-forecast/nba/recolectar_boxscore.py): Ingesta incremental concurrente de métricas detalladas de box score.
- [`motor.py`](file:///Users/pabloignaciocortesvielma/Documents/UdeC/2026-1/fifa-2026-hybrid-forecast/nba/motor.py): Núcleo matemático de features causales, entrenamiento ML, simulación Monte Carlo y validación out-of-sample.
- [`app_nba.py`](file:///Users/pabloignaciocortesvielma/Documents/UdeC/2026-1/fifa-2026-hybrid-forecast/nba/app_nba.py): Interfaz gráfica en Streamlit con diseño premium de analítica deportiva.
- `data/`: CSVs con franquicias, historial de partidos jugados, calendario por jugar y estadísticas de box score.
