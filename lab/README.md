# 🧪 Laboratorio (provisorio) — Mundial 2026

App alternativa que **no reemplaza** a `app.py` (el deploy en producción). Vive aparte para
experimentar sin tocar lo que ya está online. Extiende el mismo modelo (base / híbrido, con
ponderación K-factor) con cuatro frentes nuevos.

## Correr localmente

```bash
pip install -r ../requirements.txt
streamlit run lab/app_lab.py        # desde la raíz del repo
```

## Los cuatro frentes

Hereda el **estilo visual premium** de la app principal (tipografía Inter, título con gradiente,
tarjetas, comparación Base vs Híbrido lado a lado).

| Pestaña | Qué hace |
|---|---|
| **⚽ Partido + Mercados** | Analizador de enfrentamiento directo entre 2 selecciones: probabilidades 1X2, cuota justa (1/prob), matriz bivariada Poisson Dixon-Coles (7x7) y estimación de estadísticas de partido (xG, tiros al arco, corners, faltas, posesión). |
| **📊 Proyecciones y Grupos** | Registro cuantitativo de **10.000 simulaciones Monte Carlo**: probabilidades de Campeón, Final, Semis y Clasificación para las 48 selecciones (con intervalos de confianza al 95%), tablas de los 12 grupos (A a L) y pronóstico de los 72 partidos de fase de grupos con xG. |
| **🗺️ Cuadro de eliminatorias** | Simulación del camino al título sobre los 16 cruces oficiales (15.000 simulaciones). Si el cuadro de ESPN aún no está publicado (pre-torneo), despliega la simulación precalculada de las eliminatorias oficiales FIFA (`probabilidades_campeon_bracket_real.csv`). |
| **🔴 Torneo en vivo** | Ingesta en vivo de resultados de ESPN o carga manual de resultados para re-simular el torneo fijando cotejos disputados y actualizando el Elo en caliente. |
| **🎯 Validación vs Realidad** | Evaluación empírica del modelo: durante el Mundial valida partido a partido contra ESPN; mientras no haya partidos jugados, despliega el **registro out-of-sample en el hold-out 2025–2026** (1.045 partidos, curvas de calibración y log-loss vs baseline). |

## 📊 Registro de Resultados Cuantitativos

### 1. Evaluación Out-of-Sample (Hold-out 2025–2026)
- **Partidos evaluados**: 1.045 cotejos internacionales reales
- **Tasa de acierto (1X2)**: **60.1%** (frente al 33.3% aleatorio)
- **Log-Loss del modelo**: **0.848** (frente al **1.099** del baseline uniforme histórico, ganancia de -0.251)
- **Calibración ECE**: Calibración monotónica alineada con la diagonal ideal en victorias locales, empates y visitas.

### 2. Proyecciones Monte Carlo del Mundial (10.000 Torneos)
| Selección | Grupo | ELO | P(Campeón) | P(Final) | P(Semis) | P(Avanzar) |
|---|---|---|---|---|---|---|
| 🇪🇸 España | H | 2223 | **30.1%** | 42.9% | 56.7% | 83.1% |
| 🇦🇷 Argentina | J | 2189 | **18.2%** | 32.3% | 47.5% | 73.4% |
| 🇫🇷 Francia | I | 2128 | **11.8%** | 21.6% | 39.5% | 78.3% |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra | L | 2090 | **10.0%** | 19.3% | 33.5% | 75.4% |
| 🇧🇷 Brasil | C | 2069 | **6.5%** | 13.4% | 26.9% | 69.7% |
| 🇵🇹 Portugal | K | 2056 | **4.4%** | 10.7% | 20.7% | 67.8% |
| 🇳🇱 Países Bajos | F | 2004 | **2.8%** | 7.0% | 16.9% | 52.7% |
| 🇩🇪 Alemania | E | 2000 | **2.8%** | 7.1% | 18.3% | 65.1% |
| 🇨🇴 Colombia | K | 2064 | **2.2%** | 5.9% | 13.4% | 56.2% |

## Archivos

- `motor.py` — toda la lógica (carga, modelos, Poisson, mercados, cuotas, modo vivo, simulación, validación). Importable y testeable.
- `espn_live.py` — conector con la API pública de ESPN (resultados reales del Mundial, sin API key).
- `app_lab.py` — interfaz Streamlit (5 pestañas canónicas).

## Notas de honestidad

- **Cuotas**: el proyecto **no tiene acceso a cuotas reales** (Bet365, etc.). Lo que mostramos es la **cuota justa del modelo** (1 ÷ probabilidad). Tú la comparas contra la cuota real de la casa: si la casa paga más, hay *value*.
- **Torneo en vivo**: trae resultados reales vía API de ESPN (`fifa.world`, sin key). La actualización de Elo usa K=60 (Mundial) y multiplicador de goleada; la forma se actualiza con media móvil de ventana ≈8. Si ESPN no responde, la pestaña cae a carga manual.
- **Backtesting económico**: como no hay cuotas reales, el "mercado" es **sintético** (un modelo simple de solo-Elo + margen). Un ROI positivo prueba que la información extra del modelo completo (H2H, valor de plantilla) **aporta valor sobre el Elo solo** — no que le ganarías a una casa real.

## Despliegue alternativo (opcional)

Para tener el lab online **sin tocar** el deploy principal, crea en Streamlit Cloud una segunda app
apuntando al mismo repo con **Main file path** = `lab/app_lab.py`. El `app.py` de producción queda intacto.
