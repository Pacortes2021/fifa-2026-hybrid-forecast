# 🏆 Pipeline de Datos y Elo Histórico: Mundial 2026

Este directorio contiene los datasets finales preparados para el entrenamiento y simulación del modelo de predicción del Mundial de la FIFA 2026, junto con una explicación técnica detallada del procedimiento realizado.

---

## 🖥️ Aplicación Interactiva en Vivo (Streamlit)

Hemos desplegado una aplicación interactiva en la nube de Streamlit para que puedas simular cualquier partido del Mundial de forma interactiva y comparar los modelos en tiempo real:

👉 **[Simulador Web del Mundial 2026](https://pacortes2021-fifa-2026-hybrid-forecast-app-anrhbf.streamlit.app/)**

### ¿Qué puedes hacer en la aplicación?
* **Búsqueda Amigable en Español**: Selecciona las selecciones en tu idioma con banderas emoji (se mapean automáticamente a los nombres en inglés en el backend).
* **Comparación Directa de Modelos**: Compara el **Modelo Base** (jerarquía, finanzas e historial de largo plazo) frente al **Modelo Híbrido** (que incorpora la forma y volumen de juego recientes).
* **Pronósticos y Métricas**: Visualiza probabilidades de ganar/empatar/perder, el porcentaje para avanzar ronda si es eliminatoria, goles esperados (Poisson) y los 5 marcadores más probables.
* **Matrices de Goles**: Analiza la matriz cruzada de probabilidad de marcadores (del 0-0 al 5-5) mediante mapas de calor interactivos.

---

## 📂 Estructura del Repositorio

```
├── app.py                 # Aplicación Streamlit (Base vs Híbrido)
├── requirements.txt
├── data/                  # Datasets fuente
│   ├── espn_stats.csv         # 5,659 partidos 2018–2026 con Elo pre-partido (100% cobertura)
│   ├── modelado_espn.csv      # Dataset de entrenamiento: 5,023 partidos, 11 variables delta + target
│   ├── team_states.csv        # Estado actual de las 336 selecciones (corte jun-2026) — inicializa el simulador
│   └── results.csv            # Histórico 1872–2026 (martj42) — base del Elo y del head-to-head
├── notebooks/             # Análisis y modelado
│   ├── Mundial_2026_Metodologia.ipynb        # Entregable principal (v5): metodología completa
│   ├── Mundial_2026_Metodologia_Limpio.ipynb # Misma metodología con redacción condensada
│   ├── Mundial_2026_Hibrido.ipynb            # Modelo Híbrido: prior de calidad + forma reciente (6 vars)
│   └── mundial_solo_stats.ipynb              # Experimento: ¿cuánto predice SOLO el box score? (7 vars)
├── outputs/               # Salidas de los notebooks (CSV)
│   ├── predicciones_fase_grupos*.csv         # Los 72 partidos de grupos: P(V/E/D) + goles esperados
│   └── probabilidades_torneo*.csv            # P(octavos/semis/final/campeón) por modelo y motor
└── archivo/               # Notebooks de versiones anteriores (referencia histórica)
```

### Los modelos del proyecto

| Modelo | Variables | Log-Loss test 2025–26 | Log-Loss CV temporal | Notebook |
|---|---|---|---|---|
| **Base (principal)** | `elo_diff`, `h2h_diff`, `squad_value_diff` | 0.8517 | **0.9064** | `Mundial_2026_Metodologia.ipynb` |
| **Híbrido** | Base + `goles_anotados/recibidos_diff`, `tiros_arco_diff` | **0.8507** | 0.9096 | `Mundial_2026_Hibrido.ipynb` |
| Solo juego (experimento) | Las 7 variables de box score | 1.0041 | 1.0281 | `mundial_solo_stats.ipynb` |

> Base e Híbrido están **estadísticamente empatados** (el híbrido gana por 0.001 en test, el base gana
> en CV; ambas diferencias muy por debajo del ruido) — por eso la app los muestra lado a lado. El
> experimento "solo juego" cuantifica el valor de la fuerza acumulada: sin Elo/plantilla/H2H se pierde
> ~0.15 de log-loss y el pronóstico del torneo se distorsiona (las stats crudas no ajustan por la
> calidad del rival).

**Tipo de competición (§6c del notebook principal):** los amistosos son intrínsecamente menos
predecibles (log-loss ~0.92 vs ~0.78 en partidos competitivos). Corolario importante: **el pronóstico
del Mundial es mejor que la métrica titular** — el 0.85 global está arrastrado por amistosos; en
partidos en serio (lo que es el Mundial) el modelo rinde ~0.78. Por eso todos los modelos de simulación
se entrenan con **ponderación por K-factor** (Mundial 3× un amistoso), alineados con la filosofía del
Elo. El efecto en el pronóstico es marginal (España pasa de 28.4% a 29.2%) pero la metodología es más
coherente.

**El notebook principal** sigue el camino: EDA → selección de variables data-driven (VIF + forward + significancia, **solo con train y CV temporal**) → entrenamiento con split temporal (test = 2025–26) → matriz de confusión → tratamiento del empate (RPS) → ROC/calibración → comparación de 14 modelos con hiperparámetros por CV (lineales, Lasso/Ridge/Elastic Net, ordinal logit y probit, SVM, red neuronal, RF/GB/XGBoost) → simulación Monte Carlo (10,000 torneos, bracket oficial FIFA 48 con letras verificadas contra el calendario, simetrización de localía para cancha neutral) → **un Mundial de muestra por dentro** (tablas de grupo, terceros y bracket ronda a ronda con probabilidades) → **simulador manual `versus()`** (cualquier par de las 336 selecciones, con matriz de marcadores) → modelo Poisson de goles, ensamble y torneo re-simulado con motor Poisson como contraste.

---

## 🛠️ Procedimiento y Metodología Realizada

### 1. Cálculo de Elo Histórico
Para medir de forma precisa la fuerza de cada selección, implementamos un motor de Elo cronológico que procesó **49,373 partidos históricos** de fútbol internacional desde el primer partido en **1872** hasta **junio de 2026** (`results.csv`).
* **Parámetros**: Valor inicial de `1500.0` y ventaja de localía de `100.0` puntos (salvo en canchas neutrales).
* **Pesos K-Factor**: Amistosos (20.0), Clasificatorias (40.0), Copas Continentales (50.0) y Mundiales (60.0).
* **Multiplicador de Goles**: Amplifica el delta según la diferencia de goles para dar mayor peso a las goleadas.
* **Filtro crítico**: Se eliminaron los 72 registros futuros sin marcador del fixture del Mundial 2026 para evitar que propaguen valores nulos (`NaN`) en el cálculo final de los equipos participantes.

### 2. Normalización de Nombres de Selecciones
Para lograr un cruce del 100% de cobertura entre los partidos de ESPN y los históricos de `results.csv`, mapeamos y corregimos las discrepancias de nombres de 9 selecciones en la base de ESPN:
* `Chinese Taipei` ➡️ `Taiwan`
* `Brunei Darussalam` ➡️ `Brunei`
* `Kyrgyz Republic` ➡️ `Kyrgyzstan`
* `Sao Tome and Principe` ➡️ `São Tomé and Príncipe`
* `St. Kitts and Nevis` ➡️ `Saint Kitts and Nevis`
* `St. Lucia` ➡️ `Saint Lucia`
* `St. Martin` ➡️ `Saint Martin`
* `St. Vincent and the Grenadines` ➡️ `Saint Vincent and the Grenadines`
* `US Virgin Islands` ➡️ `United States Virgin Islands`

### 3. Fusión en Cascada de Elo
Asociamos a cada partido de ESPN su respectivo Elo pre-partido aplicando una búsqueda jerárquica:
1. **Match Exacto** en la misma fecha y equipos.
2. **Match Invertido** (para partidos en campo neutral donde ESPN y la base histórica difieren sobre quién es local y visitante).
3. **Margen de ±1 día** (para resolver desfases de zona horaria).
4. **Consulta Histórica Acumulada**: Si el partido no existe en el set histórico de resultados (ej. amistosos no oficiales), se busca el Elo del equipo justo después de su último partido registrado estrictamente antes de la fecha actual.

**Resultado:** Cobertura del **100% (0 nulos en 5,659 filas)**.

### 4. Pre-población de Historiales de Goles
Para evitar que los partidos de los primeros años (2018) comenzaran con un promedio de goles acumulado de `0.0` (debido a la falta de historial previo en ESPN), pre-poblamos las colas de partidos móviles de cada selección con los **goles anotados y recibidos de sus últimos 8 partidos en results.csv** antes del 7 de enero de 2018. Esto dotó al inicio de la base de datos de un contexto histórico real e inmediato.

### 5. Generación de Variables (Promedios Móviles y Deltas)
Recorrimos los partidos cronológicamente y mantuvimos una **ventana móvil Walk-Forward de los últimos 8 partidos** por selección para calcular:
* **Goles**: Promedio de anotados y recibidos (100% de cobertura).
* **Estadísticas Detalladas**: Promedio de tiros, tiros al arco, córners, posesión y faltas.
  * *Tratamiento de nulos*: ESPN no registra estadísticas de box score detallado en el 60.5% de su base de datos. Tratar los vacíos o valores de `0.0` como reales sesgaría los promedios (posesión del 0.0% no existe). El script los reemplaza por `NaN` y calcula un promedio móvil limpio (`np.nanmean`) sobre los partidos que sí tienen datos.
  * *Fallback Global*: Si un equipo no tiene ningún partido con estadísticas en su ventana de 8 partidos, se le asigna la media global (ej. 50% de posesión, 11.5 tiros, 12.9 faltas).
* **Cálculo de Diferencias (Deltas)**: Restamos el perfil de promedios del Local menos el Visitante. Esto genera variables simétricas directas del enfrentamiento (`elo_diff`, `tiros_diff`, etc.), que es lo que optimiza el entrenamiento de modelos supervisados.

### 6. Filtrado de Entrenamiento (2019-01-01)
Aunque usamos los datos de 2018 para calentar el historial y que los equipos acumularan estadísticas detalladas, **filtramos el dataset final de modelado a partir del 1 de enero de 2019**. Esto garantiza que las variables detalladas ya se encuentren estables, eliminando deltas en cero artificiales del periodo de inicio.

---

## 📈 Formato del Dataset de Entrenamiento (`modelado_espn.csv`)

El archivo cuenta con **5,023 filas** (partidos jugados entre el 02-01-2019 y el 09-06-2026) y las siguientes columnas:
* **`fecha`**: Fecha del encuentro.
* **`competicion`**: Nombre de la copa, torneo o amistoso.
* **`local` / `visita`**: Nombres de las selecciones.
* **`elo_diff`**: Diferencia de rating Elo pre-partido.
* **`squad_value_diff`**: Diferencia en el logaritmo del valor de la plantilla (Point-in-Time).
* **`ea_overall_diff`**: Diferencia de la media general en el videojuego FIFA/EA FC vigente.
* **`h2h_diff`**: Diferencia de goles promedio acumulada en enfrentamientos directos previos (desde 1872).
* **`goles_anotados_diff` / `goles_recibidos_diff`**: Diferencia de promedios de goles en los últimos 8 partidos.
* **`tiros_diff` / `tiros_arco_diff` / `corners_diff` / `posesion_diff` / `faltas_diff`**: Diferenciales de estadísticas detalladas promedio de la ventana móvil.
* **`resultado` (Target)**: Variable categórica a clasificar desde la perspectiva local:
  * `2` = Victoria Local
  * `1` = Empate
  * `0` = Victoria Visitante
