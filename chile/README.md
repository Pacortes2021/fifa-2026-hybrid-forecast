# 🇨🇱 Predictor — Primera División de Chile 2026

Mismo enfoque que el predictor del Mundial, aplicado a la liga chilena de clubes (con localía real).
Datos de la **API pública de ESPN** (liga `chi.1`, sin API key).

## Correr

```bash
streamlit run chile/app_chile.py        # la app (desde la raíz del repo)
python3 chile/recolectar.py             # actualizar los datos (re-escribe data/*.csv)
```

## Qué hace

- **🏆 Campeonato:** tabla de posiciones real + proyección por Monte Carlo (10.000 simulaciones del
  fixture restante): probabilidad de salir campeón, clasificar a copas y descender, puntos y posición
  proyectados de cada equipo.
- **⚽ Predecir partido:** probabilidades 1X2 con localía, cuotas justas, goles esperados, mercados
  (Over/Under, ambos marcan), marcadores más probables y matriz.
- **🎯 El modelo:** validación temporal (acierto, log-loss vs baseline) y ranking Elo.

## Método

- **Elo** cronológico sobre 1.433 partidos (2021–2026) con ventaja de localía (+55) y multiplicador de goleada.
- **Modelo de resultado:** logística multinomial (gana local / empate / gana visita) con `elo_diff`
  (con localía), forma reciente (puntos/partido de los últimos 5) y head-to-head.
- **Poisson de goles** para los marcadores; **Monte Carlo** para el campeonato.

## Archivos

- `recolectar.py` — baja los datos de ESPN → `data/partidos.csv` (jugados) y `data/fixture.csv` (por jugar).
- `motor.py` — Elo, features, modelo, Poisson y simulador. Importable y testeable.
- `app_chile.py` — la interfaz Streamlit.

## Notas

- La app lee los CSV de `data/`. Para refrescar la tabla con resultados nuevos, corre `recolectar.py`
  y commitea los CSV (en local; en Streamlit Cloud se actualiza al hacer push).
- Validación temporal: ~53% de acierto y log-loss 0.996 (vs baseline 1.041). El fútbol de clubes es
  más impredecible que las selecciones — el local gana solo ~45% de los partidos.
