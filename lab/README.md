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

| Pestaña | Qué hace |
|---|---|
| **⚽ Partido + Mercados** | Probabilidades V/E/D de cualquier cruce + mercados de apuestas derivados de la grilla Poisson: Over/Under (1.5/2.5/3.5), *ambos marcan* (BTTS), marcadores más probables, hándicap de líneas enteras, y la matriz de marcadores. |
| **🔴 Torneo en vivo** | Carga resultados reales del Mundial conforme se juegan: el modelo **actualiza el Elo y la forma** de cada selección, **fija** los partidos de grupo ya jugados y **re-simula el torneo restante**, mostrando cómo cambian las probabilidades de campeón. |
| **🎯 Validación** | (1) **Calibración** en el hold-out temporal 2025–26 (curvas de confiabilidad + ECE por clase). (2) **Backtesting económico** de *value betting* contra un mercado sintético, con curva de ganancia. |
| **📊 Robustez** | (1) **Intervalos de confianza** del Monte Carlo (error binomial sobre 8.000 simulaciones). (2) **Análisis de sensibilidad**: mueve el Elo de una selección y mira el efecto inmediato en un cruce. |

## Archivos

- `motor.py` — toda la lógica (carga, modelos, Poisson, mercados, modo vivo, simulación, validación). Importable y testeable.
- `app_lab.py` — la interfaz Streamlit (4 pestañas).

## Notas de honestidad

- **Torneo en vivo**: al corte de junio 2026 el Mundial aún no se jugaba (0 resultados reales en los datos), así que esta pestaña ofrece el **mecanismo** listo para usarse — ingresas los resultados a mano y re-simula. La actualización de Elo usa K=60 (Mundial) y multiplicador de goleada; la forma se actualiza con media móvil de ventana ≈8.
- **Backtesting económico**: el proyecto **no tiene cuotas reales** de casas de apuestas. El "mercado" es **sintético** (un modelo simple de solo-Elo + margen). Un ROI positivo prueba que la información extra del modelo completo (H2H, valor de plantilla) **aporta valor sobre el Elo solo** — no que le ganarías a una casa real. Con cuotas reales el número cambiaría.

## Despliegue alternativo (opcional)

Para tener el lab online **sin tocar** el deploy principal, crea en Streamlit Cloud una segunda app
apuntando al mismo repo con **Main file path** = `lab/app_lab.py`. El `app.py` de producción queda intacto.
