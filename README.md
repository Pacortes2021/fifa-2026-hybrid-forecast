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

## 📂 Contenido del Directorio

1. **`espn_stats.csv`**: Base de datos de partidos internacionales masculinos (2018–2026) recopilada de ESPN. Ha sido ordenada cronológicamente y enriquecida con las columnas `elo_local` y `elo_visita` calculadas pre-partido.
2. **`modelado_espn.csv`**: Dataset de entrenamiento final filtrado desde el **1 de enero de 2019**. Contiene las diferencias relativas (deltas) de las variables móviles y el target listo para alimentar el modelo de Machine Learning.
3. **`team_states.csv`**: El estado absoluto final de cada selección (al corte de junio de 2026). Contiene su Elo y el promedio móvil de goles, tiros, posesión, córners y faltas. **Este archivo es indispensable para inicializar el simulador de Monte Carlo del Mundial 2026**, ya que permite calcular las diferencias en tiempo real para cualquier partido nuevo (ej. Francia vs. Senegal).
4. **`results.csv`**: Histórico completo de partidos internacionales (1872–2026, fuente martj42/international_results). Base del cálculo de Elo y del head-to-head (`h2h_diff`) para cruces nuevos.
5. **`Mundial_2026_Metodologia.ipynb`**: El notebook metodológico completo (v5). Sigue el camino EDA → selección de variables data-driven (VIF + forward + significancia, **solo con train y CV temporal**) → entrenamiento con split temporal (test = 2025–26) → matriz de confusión → tratamiento del empate (RPS) → ROC/calibración → comparación de 14 modelos con hiperparámetros por CV (lineales, Lasso/Ridge/Elastic Net, ordinal logit y probit, SVM, red neuronal, RF/GB/XGBoost) → simulación Monte Carlo (10,000 torneos, bracket oficial FIFA 48, simetrización de localía para cancha neutral) → **un Mundial de muestra por dentro** (tablas de grupo con Pts/GF/GC/DG, terceros y bracket ronda a ronda) → **simulador manual `versus()`** (cualquier par de las 336 selecciones, con detalle del pronóstico) → modelo Poisson de goles y ensamble como contraste.
6. **`predicciones_fase_grupos.csv`** / **`probabilidades_torneo.csv`** / **`probabilidades_torneo_poisson.csv`**: Salidas del notebook — probabilidades V/E/D y goles esperados de cada partido de grupos, y probabilidades de cada selección de llegar a octavos/semis/final/título según los dos motores de simulación (clasificador logístico y Poisson de goles), comparados entre sí en §10.2.

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
