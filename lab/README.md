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
| **⚽ Partido + Mercados** | Compara **Base vs Híbrido** lado a lado (estilo de la app principal): probabilidades V/E/D con barras + **cuota justa** (1/prob) de cada resultado y mercado, para detectar *value* contra una casa. Mercados: Over/Under (1.5/2.5/3.5), BTTS, hándicap, marcadores más probables y las dos matrices. Muestra los **últimos 6 partidos** de cada selección (historial + Mundial en curso). |
| **📊 Fase de grupos** | Las 12 **tablas de posiciones** (PJ, G/E/P, GF, GC, DG, Pts) construidas con los resultados reales de ESPN, con la **probabilidad de clasificar** de cada equipo según el modelo ya actualizado. Verde = puestos de clasificación directa. |
| **🔴 Torneo en vivo** | Trae los resultados **reales** del Mundial desde la **API de ESPN** (o carga manual): **actualiza el Elo y la forma**, **fija** los partidos de grupo jugados y **re-simula el resto**, mostrando cómo cambian las probabilidades de campeón. |
| **🎯 Validación** | (1) **Calibración** en el hold-out temporal 2025–26 (curvas + ECE). (2) **Backtesting económico** de *value betting* contra un mercado sintético. |
| **📈 Robustez** | (1) **Intervalos de confianza** del Monte Carlo (8.000 simulaciones). (2) **Análisis de sensibilidad**: mueve el Elo de una selección y observa el efecto inmediato. |

## Archivos

- `motor.py` — toda la lógica (carga, modelos, Poisson, mercados, cuotas, modo vivo, simulación, validación). Importable y testeable.
- `espn_live.py` — conector con la API pública de ESPN (resultados reales del Mundial, sin API key).
- `app_lab.py` — la interfaz Streamlit (4 pestañas).

## Notas de honestidad

- **Cuotas**: el proyecto **no tiene acceso a cuotas reales** (Bet365, etc.). Lo que mostramos es la **cuota justa del modelo** (1 ÷ probabilidad). Tú la comparas contra la cuota real de la casa: si la casa paga más, hay *value*.
- **Torneo en vivo**: trae resultados reales vía API de ESPN (`fifa.world`, sin key). La actualización de Elo usa K=60 (Mundial) y multiplicador de goleada; la forma se actualiza con media móvil de ventana ≈8. Si ESPN no responde, la pestaña cae a carga manual.
- **Backtesting económico**: como no hay cuotas reales, el "mercado" es **sintético** (un modelo simple de solo-Elo + margen). Un ROI positivo prueba que la información extra del modelo completo (H2H, valor de plantilla) **aporta valor sobre el Elo solo** — no que le ganarías a una casa real.

## Despliegue alternativo (opcional)

Para tener el lab online **sin tocar** el deploy principal, crea en Streamlit Cloud una segunda app
apuntando al mismo repo con **Main file path** = `lab/app_lab.py`. El `app.py` de producción queda intacto.
