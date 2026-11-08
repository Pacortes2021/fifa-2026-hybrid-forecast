# Diseño: estándar de UI para las 23 páginas (portal de predicciones)

Rutas: `<liga>/app_<liga>.py` y `<liga>/motor.py`. Código analizado en el clon de solo lectura. Nota: la sonda con AppTest (`agentes/ui/probe.py`) re-entrenó y escribió cachés `.pkl` (ignorados por git) en lab/ucl/unl del clon; se detuvo. No se verificó por ejecución el resto de las ligas: todo lo demás es análisis estático. Existe ya `tests/ui_smoke.py`, que recorre todos los torneos con AppTest.

## 1. Matriz comparativa

### Familias de UI (el código es copia y pega dentro de cada familia)
| Familia | Páginas | Líneas | Pestañas (orden) |
|---|---|---|---|
| A "Eng-like" | eng, esp, bund, ita | ~690 | ⚽ Predicción Versus / 📊 Tabla y Proyecciones / 🔬 Importancia de Variables / 🎯 Validación vs Realidad (eng:188) |
| B "Radiografía" | fra, ned, por, bel | ~615 | igual que A |
| C "Sudamérica/Mex" | mex, bra, chile, arg | 500-650 | igual que A (mex:178, bra:180, chile:177, arg:147) |
| D "Lean inglés" | tur, sco, aut, den, gre, nor, swe | 365-385 | ⚔️ Match Predictor (Versus) / 🏆 Tabla & Proyección Monte Carlo / 🧠 Importancia de Variables / 📈 Validación en Vivo (tur:168) |
| E copas | ucl (862), unl (538) | | Versus / Tabla (UNL: Grupos) y Proyecciones / 📥 Exportar para IA (CSV) / Importancia / Validación (ucl:155, unl:137) |
| F | nba (641) | | 🏀 Predicción Versus / 📊 Clasificación y Playoffs (sub-tabs Este/Oeste/Campeón, nba:406) / 🔬 Factores e Importancia / 🎯 Validación (nba:136) |
| G | lab (777) | | ⚽ Partido + Mercados / 📊 Proyecciones y Grupos (3 sub-tabs, lab:471) / 🗺️ Cuadro de eliminatorias / 🔴 Torneo en vivo / 🎯 Validación (lab:308) |

### Sidebar y selector de modelo (default)
- A, B, C, arg: orden RF(Recomendado) / LASSO / Stacking / XGB, default **RF** (eng:132-141; mex:131; arg:111).
- D: orden Stacking / RF / LASSO / XGB, default **Stacking**, y la etiqueta RF dice "(Ensemble)" (tur:125-135).
- ucl: Stacking / RF / LASSO / XGB, default **Stacking**, otras etiquetas ("✨ Stacking Óptimo", "🎯 LASSO") (ucl:88-98).
- unl: RF / LASSO / XGB / SVM / Stacking, default RF (unl:63-70). Tiene SVM y métricas fallback hardcodeadas (unl:73-92).
- nba: stacking; lab: `st.sidebar.radio` base/hyb/two_stage (lab:300).
- Botón de actualizar ESPN:
  - A/B/C/arg: botón primario "🔄 Actualizar ESPN y Re-entrenar" visible si `portal_admin` (eng:152).
  - D/ucl: `st.sidebar.expander("🔄 Actualización de Datos ESPN")` (tur:138).
  - lab: sin botón.
- Caption "🗓️ Datos actualizados: fecha · N partidos": A/B/C/arg/unl/nba/ucl sí (eng:162-167); D no; lab no.
- Bloque "📊 Métricas Out-of-Sample" en sidebar: A/B/C/arg/ucl/unl/nba sí (eng:169-180); **D no** (muestra métricas dentro de la pestaña 3, tur:324-333). El rótulo cambia por página: "(2025+)", "(2026+)", "(Test 2026)", "(Test 2025-26+)".

### Componentes por página (✔ = presente)
| Componente | A (eng esp bund ita) | B (fra ned por bel) | C (mex bra chile) | arg | D (7 ligas) | ucl | unl | nba | lab |
|---|---|---|---|---|---|---|---|---|---|
| Selector "Cargar partido del fixture" | ✔ (con `on_change`) | ✔ | ✔ | ✔ | ✔ (sin `on_change`, parsea el texto, tur:199-210) | ✔ | ✔ (radio fixture/libre) | ✔ (solo futuros) | ✗ |
| Escudos (equipos.csv) en selector | ✔ 64px | ✔ | ✔ | ✔ | ✔ 75px | ✔ | ✔ + banderas | ✔ 75px | solo bandera emoji |
| Barras de probabilidad + cuota justa | ✔ `st.progress` (eng:269) | ✔ 3 `st.metric` con delta=cuota (fra:253) | ✔ | ✔ | `st.metric`+progress, cuota solo 1X2 local/empate (tur:244-262) | ✔ | `st.metric` + barra HTML 3 colores (unl:210-225) | tarjetas HTML `.metric-box` | 3 tarjetas, una por modelo (lab:357-361) |
| Alertas heurísticas | ✔ 3 (eng:255-260); ita con "Acierto >80% histórico" inventado (ita:255-259) | fra distinto (info/warning, fra:244-248) | ✔ | ✔ | ✗ | ✔ | ✗ | ✔ muchas, con % histórico (nba:239-285) | ✗ |
| Goles esperados y marcadores top | ✔ | `st.metric` xG (fra:262) | ✔ | ✔ | xG metric (tur:236-266) | ✔ | ✔ | proyección por cuartos | ✔ + córners/tiros/faltas/posesión (lab:404) |
| Comparativa directa de 4 modelos (tabla) | ✔ (eng:288-329) | ✗ | ✔ | ✔ | ✗ | ✔ | ✗ | ✗ | ✔ (3 tarjetas) |
| Detalle de variables (barras HTML `_barra`) | ✔ expander colapsado (eng:333) | ✔ "Radiografía Táctica" **expanded=True** (fra:268) | ✔ | ✔ | expander 6 `st.metric` vía `get_features_for_match` (tur:275) | ✔ similar a D (ucl:449) | `feats` en cabecera | tabla Four Factors | ✗ |
| Mercados O/U 1.5-3.5, BTTS | ✔ tabla 2 cols | ✔ (esquema nested) | ✔ | ✗ | solo O2.5 y BTTS como metric (tur:263-272) | ✔ | O/U 2.5 y BTTS | O/U de puntos | ✔ + hándicap asiático |
| Doble oportunidad + DNB | ✔ | ✔ | ✔ | ✗ | ✗ | ✔ | ✗ | ✗ | ✔ |
| Hándicap | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | spread por línea (nba:324) | ✔ asiático (lab:390) |
| Matriz de goles | imshow (eng:435) | ✔ | ✔ | ✗ | expander dataframe gradiente (tur:269) | ✔ | imshow | ✗ | doble imshow con "6+" (lab:427) |
| H2H | ✔ (eng:450-502) | ✔ | ✔ | ✔ | ✗ | ✔ | ✗ | ✗ | ✗ (forma reciente por selección, lab:339) |
| Tabla actual + Monte Carlo con escudos (`ImageColumn`) | ✔ | ✔ | ✔ (chile quita la col Escudo de la proyección) | proyección sola | ✗ sin escudos; `hide_index=False` | ✔ | por grupos | por conferencia | grupos |
| Importancia de variables | LASSO/RF; con stacking/xgb muestra RF en silencio (eng:559-587); ita/fra... sí XGB | ✔ LASSO/XGB/RF | solo LASSO (mex:545, bra:549) | solo RF | solo RF top 15 | ✔ | ✔ | tabla + heurísticas | ✗ |
| Validación: selector temporada | ✔ (≥2025) | ✔ | ✔ | ✔ | ✗ | ✔ | ✗ | radio período | ✗ |
| Curva log-loss acumulado + baseline | ✔ matplotlib (eng:650) | ✔ | ✔ | ✗ | `line_chart` de accuracy (tur:362) | ✔ | ✔ | ✔ | calibración |
| % acierto por equipo | ✔ (eng:660) | ✔ | ✔ | ✗ | ✗ | ✗ | ✗ | ✔ + % casa/ruta (nba:573-606) | ✗ |
| Filtro de partidos por equipo | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✔ (nba:608-625) | ✗ |
| Simulación de bajas | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✔ (nba:213) | ✗ |
| Exportar para IA + prompt maestro | ✗ | ✗ | ✗ | ✗ | ✗ | ✔ (ucl:521-740) | ✔ (unl:331) | ✗ | ✗ |

### Estilo, títulos y escudos
- Clases CSS comunes: `.main-title` (degradado), `.main-subtitle`, `.card-title`, `.sec-title`, `.vs-text`. Se inyectan por `_inyectar_estilos()` en cada `run_app` (eng:21). Excepción: **arg no define CSS** (arg:144 usa clases inexistentes), lab usa su propio CSS (`card-title-base`, etc.), nba añade `.metric-box`.
- Colores de acento hoy (se heredan a título/card/sec): eng/esp/bund `#d90429` (cuidado: esp y bund usan el rojo de Inglaterra); ita `#003399`; fra `#002395`; ned `#FF4F00`/`#21468B`; por `#006600`; bel `#C60000`; mex `#005b41`; bra `#00a859`; chile `#0039a6`; arg: ninguno (gráfico `#74acdf`); tur `#E30A17`; sco `#002B7F`; aut `#ED2939`; den `#C60C30`; gre `#0D5EAF`; nor `#BA0C2F`; swe `#005293`; ucl `#001438`; unl `#071530`; nba `#1d428a`. `.sec-title` en D es negro `#1a1a1a`; en A/C del color de acento.
- Idioma del título: "Portal de Predicción X" en A, B, C, arg y nba ("Portal de Simulación y Predicción NBA"). "X Predictor" en D (tur "Trendyol Süper Lig Predictor", sco, aut "Austrian Bundesliga Predictor", den, gre, nor, swe) y en ucl/unl. lab: "Simulador de la Copa Mundial 2026".
- Escudos: todas las páginas con equipos.csv usan `logo_url/logo_html/fmt_opcion/label_tabla/get_label` copiados en cada archivo (eng:76-114). D usa escudos solo en selector (no en tablas). unl y lab usan banderas emoji. `TEAM_DETAILS` (emoji+color) se mantiene como fallback en cada página.
- Cuota justa: A/B/C/arg/lab/ucl `mo.cuota`; D calcula `1/p` a mano (tur:268-269).

### Funciones únicas que vale la pena llevar a todas
1. Selector de fixture con `on_change` que fija `session_state` (eng:213-227) — reemplaza el parseo de texto de D.
2. Comparativa de modelos en un partido (eng:288-329; arg:231).
3. Detalle de variables con barras enfrentadas `_barra` (eng:355-401).
4. Mercados completos con DNB/doble oportunidad (eng:405-432, lab:388-420) y hándicap asiático (`handicap_asiatico` existe en eng/motor.py:~670 y lab).
5. Estadísticas esperadas córners/tiros/faltas/posesión (lab:404-424; requiere datos de boxscore, solo si el motor los expone).
6. Forma reciente (últimos 6 partidos) por equipo (lab:339).
7. Simulador de bajas (nba:213) — solo NBA, o como "bajas" genérico opcional.
8. % acierto por equipo con split casa/ruta (nba:573-606) y filtro de partidos por equipo (nba:608).
9. Curva log-loss con baseline (eng:650) y selector de temporada OOS (eng:609-613).
10. Exportar para IA + prompt maestro (ucl:521, unl:331), candidato a pestaña común opcional.
11. `plt.close(fig)` (unl), no usado en eng-like, que acumula figuras.

## 2. Inconsistencias y problemas de UX
1. **Cadena de selección de modelo distinta** (default RF vs Stacking; nombres de etiqueta distintos) — eng:132, tur:125, ucl:88, unl:63.
2. **arg no inyecta CSS** y llama a `st.set_page_config` otra vez (arg:14), pese a que `app.py:13` ya lo hace. Es el único módulo que lo repite. Además importa `arg.motor` (paquete) en vez de `motor` (arg:9), distinto de los demás; funciona por el enrutador pero rompe el patrón.
3. **arg:394** dice "4.000 iteraciones" y llama a `n_sims=50000` (arg:99); título hardcodea "2026" (arg:393); lab:450 dice "10.000 simulaciones" y corre 8.000 (lab:67); unl 5.000, ucl 10.000, nba 2.000, resto 50.000 (distinto por familia; el motor D tiene default 1000).
4. **arg carece de mercados, matriz de goles, doble oportunidad, DNB** (grep 0 resultados) y de curva log-loss/acierto por equipo (arg:454-491).
5. **Texto copiado de otra liga**:
   - eng:274-275, esp, bund, ita:270-272: "Expectativa de clasificación en eliminación directa neutral" (viene del Mundial, no aplica a liga).
   - esp/bund importan el rojo de Premier; chile y bra comentan "Copas" en docstring (chile:3, bra:3).
   - mex:243 "Es un evento poco frecuente en México" y ita:255-259 "(Acierto >80% / >72% / >65% Histórico)" son afirmaciones hardcodeadas sin respaldo en el código.
   - eng/esp/bund subtítulo "LASSO + RF" ignora XGB; ita/fra/ned/por/bel dicen "LASSO + RF + XGBoost"; D no tiene subtítulo del modelo activo.
   - fra:`#002395` etc. correcto, pero ned usa dos azules/naranjas distintos para card y sec.
6. **Importancia de variables engañosa**: en eng/esp/bund con stacking o XGB cae al `else` y muestra importancia de RF sin avisar (eng:559-587). mex/bra/chile solo LASSO (mex:545), arg solo RF (arg:425), D solo RF (tur:336).
7. **Validación ignora el modelo elegido** en C/arg: `mo.validacion_en_vivo(M, temporada_val=...)` no recibe `modelo` y usa `pipe_val`/`pipe` (mex:576-579 en UI, mex/motor.py:1298); el selector de la barra lateral no tiene efecto en la pestaña 4 aunque dice "modelo activo".
8. **Validación C/arg muestra enteros 0/1/2** sin leyenda en `resultado` y `Prediccion` (con 0 = visita, 2 = local; mex/motor.py:1325-1330). A/B muestran "Gana local/Empate/Gana visita"; D muestra "Local/Empate/Visita".
9. **Etiqueta/columnas de la validación**: acierto en NBA es porcentaje 0-100 (`f"{acierto:.1f}%"`, nba:~520) mientras en fútbol es 0-1 (`:.1%`). NBA con "Histórico Completo" evalúa sobre datos de entrenamiento (nba:~512-540): métrica in-sample presentada como validación.
10. **Cache key fija en simulaciones**: D usa `"sim_key_tur"` (tur:301) y ucl `"sim_key_ucl"`; no invalida al llegar resultados nuevos (A/B/C/arg usan `f"{len(partidos)}-{fecha.max()}"`, eng:527).
11. **Año hardcodeado**: `get_advanced_features(a, 2026)`/`get_squad_value(a, 2026)` en fra/ned/por/bel (fra:~271-275; 4 coincidencias por página) y `get_features_for_match(local, visita, 2026)` en D (tur:276); A usa `mo._temporada_actual()`. En ligas con temporada jul-jun esto es incorrecto en parte del año. D también pasa `temporada=2026` por defecto en `predecir_match` (tur/motor.py:547).
12. **Deprecación** `use_container_width` (se retira "después del 2025-12-31", hoy 2026-11-08 con Streamlit 1.50 aún emite aviso): nba 10, arg 5, mex/bra/chile/esp/eng/bund/ita 2 c/u (p.ej. eng:329, eng:502), ucl 1 (ucl:658). El resto ya usa `width='stretch'`. Reemplazar por `width="stretch"`.
13. **`mercados` con dos esquemas** (plano en A/C/arg/lab/ucl; anidado `OverUnder/BTTS/DNB` en ita/fra/ned/por/bel) y D no tiene `mercados` ni `grilla_goles` ni `cuota`.
14. **Defaults de equipos** hardcodeados por liga (eng:231 Arsenal/Liverpool; esp Real Madrid/Barcelona; ita Inter/Juventus; fra PSG/Marseille...). eng/esp/bund listan `df_features.local.unique()` (incluye descendidos); mex usa `TEAM_DETAILS.keys()` (lista estática, mex:~187); chile/bra/arg filtran temporada actual+fixture (chile:185-190); D filtra por `temporada.max()`. D no tiene valor por defecto de equipo conocido (índices 0 y 1 → equipos arbitrarios).
15. **Redundancias**: `st.progress` + `metric` duplican la probabilidad; en B el título "Victoria Local" como `card-title` encima de cada metric duplica el label. En ita:281 `mk_temp = mo.mercados(mix)` es una llamada muerta. `p_avanza_a` (arg/eng) y el expander "Radiografía" expandido por defecto empuja el contenido principal (mercados) fuera de pantalla.
16. **Datos dummy en UI**: unl:`st.sidebar.success("54 Selecciones · 670 Partidos · 148 en Fixture")` (unl:~124) y métricas fallback 0.7654/66.7 etc. (unl:73-92) son números fijos. nba usa fallbacks 0.60/0.69/67.5 y texto "1.394 partidos"/"5.359 partidos" fijo (nba:~455-492). NBA heurísticas con "Cumplimiento Histórico" fijos (nba:239-285, 487-495).
17. **"Puntos esperados" en ita/fra/ned/por/bel** es una aproximación `puntos + (38-pj)*1.35` hardcodeada (ita/motor.py:~758), no sale de la simulación, y el 38 no vale para ligas de 34 jornadas (fra/ned/por/bel/ita según equipos). Revisar antes de mostrarla en una UI común. En D proviene de la simulación.
18. **Rótulos confusos**: `P(Copas)` con 🇪🇺 genérico (eng:536) vs "Libertadores (G.G.)" (chile:514, "G.G." no se entiende) vs "P(Top 6)/P(Liguilla)" (mex) vs `P_champions` solo en ita+; "Modelo activo" en subtítulo difiere. "Equipo Local/Visitante" vs "Selecciona Local" vs "Club Local" (tur:215).
19. **Doble set de títulos de sección** con emojis solo en D/ucl/unl/nba; sin emojis en A/B/C. Mezcla ES/EN en D: "Match Predictor", "Tabla & Proyección Monte Carlo", "Validación en Vivo", "Predictor".
20. **Mutación de estado global**: `app.py` selecciona el torneo con el índice por parámetro URL solo para nba/chile/ucl/unl (`app.py:121-130`); las otras 19 ligas no se pueden abrir por URL.
21. `TEAM_DETAILS` + helpers de escudos duplicados 21 veces; `get_motor`, `simular_liga/simular_campeonato` también (nombres distintos: `simular_liga`, `simular_campeonato`, `simular_ucl`).

## 3. Propuesta de estándar

### 3.1 Módulos
- `comun/ui_liga.py`: `run_liga_app(cfg: LigaConfig, adapter: MotorAdapter)`; `comun/ui_componentes.py` (tarjetas, mercados, barras comparativas, tablas con escudo); `comun/estilos.py` (`inyectar_estilos(acento, acento2)`); `comun/escudos.py` (`logo_url/logo_html/fmt_opcion/label_tabla/get_label`).
- Cada `<liga>/app_<liga>.py` queda en ~40 líneas: `LigaConfig` + adapter + extensiones, y `run_app()` que llama a `run_liga_app`. `app.py` no cambia (sigue importando `<carpeta>.<app>.run_app`).
- `LigaConfig`: `codigo`, `nombre_corto`, `titulo` ("Portal de Predicción {nombre}"), `bandera`, `acento`, `acento2`, `n_sims=50000`, `n_jornadas`, `reglas_texto` (clasificación a copas/descenso), `columnas_proy` (lista de `(clave, etiqueta, cmap)`), `equipos_default=(local, visita)`, `formato_extra` (liguilla, zonas, playoffs, etc.).

### 3.2 Convenciones
- **Título**: `{bandera} Portal de Predicción {Nombre oficial}` siempre en español; subtítulo `Modelo activo: <b>{nombre}</b> — {descripción corta}` generado desde el modelo seleccionado (LASSO + RF + XGBoost + Stacking). Mismo estándar para nba ("Portal de Predicción NBA"); ucl/unl: "Portal de Predicción UEFA Champions League/Nations League".
- **Color de acento por liga** en `LigaConfig.acento`: gradiente del título (acento→acento2), `.card-title` y `.sec-title` acento; `st.progress` y barras matplotlib con el mismo acento. Paleta propuesta (principal/secundario): eng `#3d195b`/`#00ff85`; esp `#c60b1e`/`#ffc400`; bund `#d3010c`/`#1a1a1a`; ita `#008C45`/`#003399`; fra `#002395`/`#ED2939`; ned `#FF4F00`/`#21468B`; por `#006600`/`#FF0000`; bel `#C60000`/`#FFCD00`; tur/sco/aut/den/gre/nor/swe y mex/bra/chile/arg: los valores actuales; ucl `#001438`; unl `#071530`; nba `#1d428a`/`#c8102e`. (Mantener los actuales donde existan para minimizar cambios; esp/bund necesitan un acento propio.)
- **Selector de modelo**: orden **Stacking (Recomendado) / Random Forest / LASSO L1 / XGBoost** (+ SVM solo si `adapter.modelos_disponibles` lo incluye, como unl); default = el de mejor log-loss si hay `metricas`, si no Stacking. Etiquetas únicas en `comun`. `key=f"modelo_{cfg.codigo}"`.
- **Sidebar** (orden): 🛠️ Controles del Modelo → selector de modelo → 🔄 actualización (botón solo si `portal_admin`, dentro de expander con caption informativa si no) → caption de fecha/n partidos → 📊 Métricas Out-of-Sample por modelo (⭐ al mejor) → texto de nota del modelo. Sin números hardcodeados.
- **Pestañas** (fútbol de liga, 19 ligas): `⚽ Predicción Versus` / `📊 Tabla y Proyecciones` / `🔬 Importancia de Variables` / `🎯 Validación vs Realidad`. Sin cambiar el orden. Pestaña opcional `📥 Exportar para IA` solo cuando `adapter.soporta_export_ia` (ucl/unl; luego posible para ligas).
- **Layout Tab 1**: (1) sec-title "Analizador de Enfrentamientos"; (2) selector de fixture con `on_change`; (3) columnas [5,1,5] local/VS/visita con escudo 64px y ELO como `st.caption`; (4) alertas (reglas comunes parametrizadas por liga, sin porcentajes históricos hardcodeados) ; (5) fila de 3 tarjetas 1X2 con probabilidad y cuota justa (un solo componente; sin duplicar `progress`) + xG; (6) "Marcadores más probables" top 4; (7) expander colapsado "Comparativa de modelos para este partido"; (8) sec "Mercados": O/U 1.5/2.5/3.5, BTTS, doble oportunidad, DNB, hándicap asiático (-1, 0, +1) cuando el adapter provea la grilla; (9) matriz de goles 6x6 con acento del color; (10) expander colapsado "Detalle de variables" (barras `_barra`) y distancia de viaje; (11) H2H con 4 métricas + últimos 8; (12) opcional "Forma reciente" (últimos 5-6) si el adapter lo ofrece.
- **Layout Tab 2**: columnas 1:1 (o [1,1.2]), izquierda "Tabla de posiciones actual (real)" con `Escudo`/`Equipo`/`PJ`/`PTS`/`DG`/`GF` (+`PG/PE/PP` si existen); derecha "Proyección de la temporada (Monte Carlo)" con columnas definidas por `cfg.columnas_proy` y degradado en campeón/descenso; debajo `st.info(cfg.reglas_texto)`; caption con `n_sims` real (leído de `cfg.n_sims`, nunca hardcodeado en texto distinto).
- **Layout Tab 3**: selector implícito por modelo (LASSO coef abs. promedio; RF Gini; XGB ganancia; Stacking → mostrar los tres en sub-pestañas o usar el modelo base de mayor peso con aviso) con tabla [5,7] + barras top 15 en el acento; métricas OOS del modelo.
- **Layout Tab 4**: selector de temporada (≥ primer año OOS), 4 métricas (partidos, acierto, log-loss, baseline), success/warning, tabla de historial con etiquetas "Gana local/Empate/Gana visita", curva log-loss acumulado + baseline, `% Acierto por equipo` con casa/ruta, filtro por equipo del historial. **Siempre** respeta el modelo elegido (si el adapter no puede, mostrar `st.caption("Validación con el modelo X")`).
- **Escudos**: siempre `equipos.csv`; emoji solo de fallback (`label_tabla`). Banderas emoji solo en selecciones (lab, unl).
- **Convenciones de widgets**: `width="stretch"` en vez de `use_container_width`; `plt.close(fig)` tras `st.pyplot`; keys `f"{nombre}_{cfg.codigo}"`; textos de spinner/captions desde `cfg`.

### 3.3 Extensiones por liga (hooks `cfg.extensiones`)
- **mex**: liguilla + play-in (`P_directo_QF`, `P_Liguilla_total`, `simular_play_in_y_liguilla`, mex/motor.py:962); nota "Apertura 2026" en tabla.
- **arg**: zonas/promedios, copas por tabla anual; texto "Proyección ... 2026" desde cfg.
- **bra/chile**: columnas Libertadores (directo/total)/Sudamericana/Descenso.
- **den/gre/sco/aut**: split / playoffs de campeonato y descenso (texto en reglas; ideal columna extra `P(Ronda de campeonato)` si el motor la simula).
- **tur/D**: `P_champions`, `P_copas`, `P_descenso`.
- **ucl**: fase de liga suiza + top 8 / playoffs / eliminado, bracket, `is_knockout`/`is_neutral` en Versus, Exportar para IA.
- **unl**: grupos por división (Liga A-D), Final Four, ascensos/descensos, SVM, selección libre.
- **nba**: Versus propio (spread, cuartos, Four Factors, bajas, B2B), clasificación por conferencia + play-in, heurísticas (movidas a configuración con n y cumplimiento calculado en vez de texto fijo). NBA no usa 1X2: se renderiza con sus componentes propios y comparte con `comun` solo sidebar, estilos, selector de modelo, validación (con `% acierto por franquicia`).
- **lab**: 3 modelos base/hyb/two_stage, cancha auto/neutral/local, córners/tiros, cuadro de eliminatorias, torneo en vivo. Comparte solo `estilos`, componentes de mercados y matriz de goles.

## 4. Interfaz de adaptador del motor

### 4.1 Diferencias por familia (archivo:línea)
| Aspecto | **E1** eng/esp/bund (+ita/fra/ned/por/bel con variantes) | **E2** mex/bra/chile/arg | **E3** tur/sco/aut/den/gre/nor/swe | **E4** ucl, unl | nba, lab |
|---|---|---|---|---|---|
| kwarg del modelo | `modelo_tipo=` (o `modelo=`) | `modelo=` (o `modelo_tipo=`) en predecir_match/grilla (mex:729, 761); `monte_carlo(modelo=)` (mex:1225) | `modelo=` (tur:547); `simular_campeonato(modelo=, modelo_tipo=)` (tur:723) | ucl `modelo=`/`modelo_tipo` (ucl:878,1138); unl `predecir_match(modelo=)` pero `simular_campeonato(modelo_tipo=)` (unl:392,459) | nba `modelo=` (nba:659); lab `modelo=base/hyb/two_stage` |
| `predecir_match` retorno | **solo `p`** array [L,E,V] (eng:577-601; esp:552-576; ita:528-549; fra:522-543…) | `(p, la, lb)` (mex:758, arg:582) | `(p, la, lb)` (tur:578); den/gre/nor/swe `(np.array([p_h,p_d,p_a]), mu_l, mu_v)` (den:568) | `(p, la, lb)` (ucl:913; unl:426) | nba dict (nba:719); lab no existe (`grilla`) |
| Orden de clases del modelo | internamente `[V,E,L]` y reordena (`p_raw[2],p_raw[1],p_raw[0]`, eng:600) | igual (mex:753) | **sin reordenar**: usa `p_raw[0],[1],[2]` y etiqueta 0=local (tur:570; den:379 `resultado=0 si gana local`) | unl reordena (unl:414); ucl sin reorden | — |
| Etiqueta de `resultado` en dataset | 0=visita, 1=empate, 2=local (eng:395) | mismo (mex:1325) | 0=local, 1=empate, 2=visita (tur:385) | — | `target_win_local` (nba) |
| Grilla de goles | `grilla_goles(M,a,b,modelo_tipo=)` → **solo matriz** 10x10 (eng:604-646) | `grilla_goles(M,a,b,modelo=)` → **`(mix, p, (la, lb))`** (mex:794; arg:621) | **no existe**; `matriz_marcador_exacto(la,lb,max_goles)` (tur:580) devuelve `ndarray` 8x8 (den: 7x7) | ucl `matriz_marcador_exacto` (ucl:928); unl `calcular_mercados(la,lb)` → dict con `matrix` | lab `grilla(M,a,b,cancha,modelo)` → `(mix,p,(la,lb))` (lab:335) |
| `mercados` | dict **plano** `Over 1.5`, `Under 1.5`, `Ambos marcan (BTTS sí)`, `BTTS no`, `_top_marcadores` (eng:653; mex:801) | igual | **no existe** | ucl:946 plano; unl dict anidado propio | lab plano + `handicap_asiatico` |
| `mercados` en ita/fra/ned/por/bel | **anidado** `{'1X2','OverUnder':{'O1.5',..},'BTTS':{'Si','No'},'DNB':{'1','2'}}` sin `_top_marcadores` (ita:580-601) | | | | |
| `cuota` | `1/p` con 99 si 0 (eng:649); ita `1/max(p,1e-6)` (ita:577) | `cuota(p)` | **no existe** | ucl:942 / unl:749 (round 2) | nba:108 (margen) |
| xG `la, lb` en UI | recalculados en UI desde `M["g_const"], M["g_d"], M["g_home"]` y ELO (eng:246-249) | devueltos por motor | devueltos por motor | devueltos | — |
| Claves de M | `cols`, `df_features`, `pipe_lasso`, `pipe_rf`, (`pipe_xgb`), `stack_w` (+`alpha_stack`), `g_const/g_d/g_home`, `tracker`, `equipos`, `metricas` (eng:518-531); ita-bel añaden `partidos` | `features`, `pipe` (LASSO), `pipe_val` (mex, chile), `pipe_lasso/rf/xgb`, `stack_w`, `alpha_stack`, `poisson_params`, `partidos`, `df_dataset`, `tracker`, `equipos`, `metricas` (mex:664; arg:490) | `features`, `pipe_lasso/rf/xgb` (xgb **no** está en `metricas`, tur:499-505), `weights_opt` (tur/sco/aut) o `weights_stacking` (den/gre/nor/swe), `poiss_params` (tur) o `poisson_params` (den…), `df_partidos`, `df_dataset`, `tracker`, `equipos`, `metricas` (tur:529-535) | ucl `features`, `weights_stacking`, `poisson_params`, `df_dataset`, `df_partidos`; unl `cols_features`, `alpha_stack`, `equipos_info`, `pipe_svm` | nba `features`, `stack_w`, `df_dataset`, `pipe_spread/totals`; lab `df,states,hist,pipe_base,pipe_hyb` |
| Pesos stacking | `stack_w` ndarray [lasso, rf, xgb] | idem | `weights_opt` list o `weights_stacking` | `weights_stacking` / `alpha_stack` escalar | `stack_w` |
| `metricas` | `{lasso,rf,xgb,stacking}` → `{logloss, accuracy(0-100), w}`; sidebar eng no filtra no-dict (eng:174), ita/fra sí (ita diff) | igual | `{lasso, rf, stacking}` (sin xgb) | ucl/unl con claves extra (`best_c`, `w` raíz); UI filtra con `isinstance` | nba añade `mae_spread, mae_totals, features_rank` |
| `obtener_tabla_actual` columnas | `equipo, pj, puntos, dif_goles, goles_favor` (eng:768) | `Selección, PTS, PG, PE, PP, DG, GF` (**sin PJ**; mex:1366, bra:~982) | `equipo, pj, g?, e?, p?, puntos, gf, gc, dg` (tur:598-637: `dg`, `gf`) | ucl: `equipo, pj, puntos, dg, gf` | — |
| `simular_campeonato` (nombre y columnas) | `simular_campeonato(M,n_sims,fijos,modelo_tipo,modelo,seed)` → `equipo, P_campeon, P_copas, P_descenso` (eng:872-959) | mex `monte_carlo` → `Selección, Puntos esperados, P_directo_QF, P_Liguilla_total, P_campeon`; bra `monte_carlo` → `Selección, Puntos esperados, P_campeon, P_libertadores_directo/total, P_sudamericana, P_descenso`; chile `simular_campeonato` igual que bra; arg `monte_carlo` → `P_copas` (arg:819, 902) | `simular_campeonato(M,n_sims=1000)` → `equipo, Selección, Puntos esperados, P_campeon, P_champions, P_copas, P_descenso` (tur:723-804) | ucl `Selección, equipo, Puntos esperados, P_campeon, P_final, P_semi, P_cuartos, P_octavos, P_top8, P_playoffs, P_eliminado`; unl dict de DataFrames (`campeon`, `tablas_actuales`, `proyecciones_grupos`) | lab `monte_carlo` → `Selección, P_campeon, P_final...`; nba `simular_temporada` → `Equipo, Conferencia, Victorias Esperadas, P_*` |
| ita-fra-ned-por-bel simular | además `Selección`, `Puntos esperados` (aprox. fija 38 jornadas) y `P_champions` (ita:755-770) | | | | |
| `validacion_en_vivo` | `(M, temporada_val=None, modelo_tipo="rf")` → `(df_val, met, evol)`; vacío = `(DataFrame(), {}, DataFrame())`; `Prediccion/resultado` strings; evol cols `partido, logloss_acum, baseline` (eng:1000-1075) | `(M, temporada_val=2026)` (**sin modelo**); vacío = `(None,None,None)`; `resultado/Prediccion` enteros; evol `partido, logloss_acum` (mex:1298-1363; arg:907) | `(M, temporada_val=None, modelo="rf", modelo_tipo=None)`; `Prediccion` entero con 0=local; evol `partidos, logloss, accuracy` (tur:843-897) | ucl `(M, temporada_val, modelo, modelo_tipo)`; unl `(M, resultados=None, modelo)` → `evolucion` con `partido, logloss_acum` | nba `(M, temporada=2027)` (nba:1008); lab `(M, resultados, modelo)` |
| `met` | `n, acierto(0-1), logloss, logloss_base` | igual | igual | igual | nba: `acierto` en 0-100 |
| Features avanzadas | `get_advanced_features(team, season)`, `get_squad_value(team, season)`, `get_distance_km(a,b)`, `tracker.pi_tracker`, `tracker.recent_*`, `tracker.season_pts` (eng:337-353) | ídem | solo `tracker.get_features_for_match(local, visita, 2026)` → `distance_km, altitude_diff, squad_value_diff, form_diff, pi_diff, h2h_diff` (tur:276) | ucl igual a D con `is_knockout`/`is_neutral` | — |
| Temporada | `_temporada_actual()` en motor (eng:571) | idem (mex:724) | `_temporada_actual()` pero UI usa 2026 | ucl `_temporada_actual()` | — |
| Tablas dependientes de | `M["df_features"]["local"]` (equipos disponibles) | `M["partidos"]`, `TEAM_DETAILS` | `M["df_partidos"]` | — | — |

### 4.2 Adaptador normalizado
```python
@dataclass
class Pred1X2:
    p: np.ndarray      # float[3] en orden FIJO [Local, Empate, Visita]
    xg: tuple[float, float]  # (lambda_local, lambda_visita)
    grid: np.ndarray   # matriz goles (>= 7x7), filas=local, columnas=visita, suma 1

class MotorAdapter(Protocol):
    MODELOS: tuple[str, ...]              # subset de ("stacking","rf","lasso","xgb","svm")
    def cargar(self) -> dict: ...         # M original
    def equipos_disponibles(self, M) -> list[str]
    def equipos(self, M) -> dict          # norm_name -> {logo, abbreviation}
    def temporada_actual(self) -> int
    def predecir(self, M, local, visita, modelo) -> Pred1X2
    def mercados(self, grid, p, local, visita) -> dict   # contrato único (ver abajo)
    def metricas(self, M) -> dict[str, dict]   # {modelo: {logloss, accuracy(0-100), w?}}
    def pesos_stacking(self, M) -> list[float]|None
    def importancia(self, M, modelo) -> pd.DataFrame     # Variable, Valor, Tipo ("coef"/"gini"/"gain")
    def tabla_actual(self, M) -> pd.DataFrame  # equipo, pj, pg, pe, pp, gf, gc, dg, puntos (pg/pe/pp opcionales)
    def proyeccion(self, M, modelo, n_sims, cache_key) -> pd.DataFrame  # equipo + columnas P_* declaradas en cfg
    def validar(self, M, temporada, modelo) -> Validacion
    def detalle_variables(self, M, local, visita) -> list[tuple[str,float,float,str,bool]]  # (etiqueta, va, vb, fmt, invertir)
    def h2h(self, M, a, b) -> pd.DataFrame     # fecha, temporada, local, visita, goles_local, goles_visita
    def fixture(self, M, n=25) -> pd.DataFrame # fecha, local, visita
    def forma_reciente(self, M, equipo, n=5) -> pd.DataFrame | None  # opcional
    def distancia_km(self, a, b) -> float | None  # opcional

@dataclass
class Validacion:
    df: pd.DataFrame   # fecha, local, visita, goles_local, goles_visita, resultado, Prediccion (strings "Gana local"/"Empate"/"Gana visita"), Prob_Local, Prob_Empate, Prob_Visita
    met: dict          # n, acierto (0-1), logloss, logloss_base
    evol: pd.DataFrame # partido (1..n), logloss_acum, baseline
```
Contrato `mercados()` único (dict plano): `{"Over 1.5","Under 1.5",...,"BTTS sí","BTTS no","_top_marcadores":[(g1,g2,p)],"Hándicap -1" ...}` y la UI calcula 1X2, doble oportunidad y DNB desde `p`. Para ita/fra/ned/por/bel el adaptador traduce el esquema anidado; para D se implementa en `comun` desde la grilla (D no tiene `mercados`).

### 4.3 Adaptadores por familia (qué implementa cada uno)
- **A1 (eng, esp, bund)**: `predecir` = `predecir_match(M,a,b,modelo_tipo=m)` (retorna solo p ya en orden [L,E,V]) + `grilla_goles(M,a,b,modelo_tipo=m)` + `xg` recalculado con `M["g_const"], M["g_d"], M["g_home"]` y `tracker.elos` (copiar eng:246-249; ita lo escribe como `g_home*1`/`*0`, equivalente). `cols` → importancia. `tabla`: renombrar `dif_goles→dg`, `goles_favor→gf`. `proyeccion`: `equipo` + P_campeon/P_copas/P_descenso; `validar`: `validacion_en_vivo(M, temporada_val, modelo_tipo)` ya con strings; vacío = DataFrame vacío.
- **A2 (ita, fra, ned, por, bel)**: igual que A1, `mercados` nested→plano, `proyeccion` trae `Selección` y `Puntos esperados` aproximada (ocultar o recalcular; ver inconsistencia 17), `P_champions` extra, `metricas` filtrar claves sin `logloss`, `detalle_variables` con temporada actual (no 2026).
- **C (mex, bra, chile, arg)**: `predecir` = `grilla_goles(M,a,b,modelo=m)` → `(mix,p,(la,lb))` (ya incluye p); `temporada_actual` del motor; `equipos_disponibles` = `partidos` temporada actual + fixture (chile:185-190) en vez de `TEAM_DETAILS`; `proyeccion`: mex `monte_carlo`, bra `monte_carlo`, chile/arg `simular_campeonato`, todas con `modelo=`; renombrar `Selección→equipo`; `tabla`: `Selección→equipo`, `PTS→puntos`, calcular `pj=PG+PE+PP`, `DG→dg`, `GF→gf`; `importancia` con `pipe_lasso`/`pipe_rf`/`pipe_xgb` y `M["features"]` (mex:560 usa `M["pipe"]`); `validar`: envolver `validacion_en_vivo(M,temporada_val)` — **no depende de modelo** (documentar o ampliar motor); mapear `Prediccion` y `resultado` 0=visita,1=empate,2=local a strings y `Prob_*` ya vienen ordenados por columna; `(None,None,None)` → Validacion vacía.
- **D (tur, sco, aut, den, gre, nor, swe)**: `predecir` = `(p, la, lb) = predecir_match(M,a,b,temporada=_temporada_actual(),modelo=m)` (den/gre/nor/swe sin `fecha`); `grid = matriz_marcador_exacto(la,lb,max_goles=...)` (valor distinto por liga: den 6 → 7x7); clases ya [L,E,V] **solo si** el dataset codifica resultado 0=local (tur:385, den:379): el adaptador debe fijar esto y no reordenar. `metricas`: no hay `xgb` (omitir en la comparativa). `pesos_stacking`: `weights_opt` o `weights_stacking`. `importancia`: `M["features"]` y `pipe_rf` (los pipes lasso/xgb existen y se pueden exponer). `tabla`: ya trae `dg/gf` (`gc`, `pj`, `puntos`), añadir `pg/pe/pp` si existe `g/e/p`. `proyeccion`: `simular_campeonato(M, n_sims, modelo=m)`; **cache key** con `len(partidos)+fecha.max()`. `validar`: `(df_val, met, df_evol)` con `Prediccion` entero 0=local → string y `evol` con columnas `partidos, logloss, accuracy` (hay que construir `logloss_acum` y `baseline`; hoy solo trae `logloss` acumulado parcial y accuracy). `cuota` y `mercados` implementados en `comun`. Fijar `temporada` desde `_temporada_actual()`.
- **E4 ucl**: A1-like con `predecir_match(..., is_neutral, is_knockout)`, `matriz_marcador_exacto`, `weights_stacking`; proyección con `P_top8, P_playoffs, P_eliminado` + bracket; tabla `pj, puntos, dg, gf`. **unl**: adaptador propio (`calcular_mercados(la,lb)`, `simular_campeonato(modelo_tipo=)` dict, `alpha_stack`, SVM).
- **nba**: adaptador propio (`simular_partido_mc`, `simular_temporada`, `validacion_en_vivo(M, temporada=2027)` con `acierto` 0-100 y `target_win_local`); `Pred1X2` no aplica (usa `p_local`, `p_visita`, spread, totals).
- **lab**: adaptador propio (`grilla(M,a,b,cancha,modelo)`, `monte_carlo`, `simular_bracket`, `validacion_en_vivo(M, resultados, modelo)`).

### 4.4 Cambios mínimos recomendados en motores (opcionales, reducen adaptadores)
- Unificar kwarg a `modelo=` (aceptando `modelo_tipo=` como alias, como ya hacen eng y ucl).
- `predecir_match` → siempre `(p, la, lb)` con orden [L,E,V].
- `mercados` plano común (mover a `comun/mercados.py`).
- `validacion_en_vivo(M, temporada_val, modelo)` en C con `modelo` real (usar `pipe_*` y OOS) y DataFrames vacíos en vez de `None`.
- `obtener_tabla_actual` con columnas canónicas; `simular_*` con nombre canónico `simular_campeonato`.

## 5. Plan de migración
Principio: nada se borra hasta que la página nueva pasa AppTest. Cada paso = una liga migrada en una rama/commit propio.

0. **Preparación**: crear `comun/` (estilos, escudos, componentes, `ui_liga`, `adapters/`), sin tocar páginas. Mantener `use_container_width → width="stretch"` como primer cambio mecánico (mex/bra/chile/esp/eng/bund/ita/arg/nba/ucl), una sola PR; verificar que no cambia nada visible.
1. **Piloto: eng** (adaptador A1 más simple, orden de columnas conocido). Migrar `app_eng.py` a `run_liga_app`. Verificar: AppTest abre y no lanza excepción; mismas 4 pestañas; mismos widgets (`at.tabs`, `at.sidebar.selectbox`, número de `dataframe/metric`) que la versión anterior (comparar con una captura previa con el `probe.py`); cambiar modelo y confirmar que `at.exception` está vacío; equipos a==b muestra error.
2. **Resto A1/A2**: esp, bund, ita, fra, ned, por, bel (cada una con su `LigaConfig`). En fra/ned/por/bel se **añade** la comparativa de modelos y se reemplaza el año 2026 por temporada actual.
3. **Familia C**: chile → bra → mex → arg. arg se migra al final (añadir mercados, matriz, CSS, quitar `set_page_config`). mex mantiene la extensión liguilla. Verificar la validación con mapeo 0/1/2 → strings y `Validacion` vacía (sin None).
4. **Familia D** (7 ligas): primero tur con adaptador D; luego sco, aut, den (con matriz 6), gre, nor, swe. Aquí cambia la UI visiblemente (nombres de pestañas y títulos en español, mercados, H2H, comparativa); validar `metricas` sin xgb y la nueva `evol`.
5. **ucl** y **unl**: ucl con adaptador E4 + extensiones (bracket/export IA); unl con su adaptador y mantiene Grupos/Exportar.
6. **nba y lab**: solo estilos, sidebar, selector de modelo y validación compartidos; el cuerpo queda como extensión.
7. Ajustes de `app.py` (opcional): `?torneo=` para las 23 páginas (hoy 4, `app.py:121-130`).

### Verificación por paso (AppTest)
- Usar `tests/ui_smoke.py <filtro>` por liga (abre, rerun, falla si hay excepción o no hay contenido) y ampliarlo:
  ```python
  at = AppTest.from_file("app.py", default_timeout=1200); at.run()
  at.sidebar.selectbox[0].select("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Inglaterra)").run()
  assert not at.exception and [t.label for t in at.tabs][:4] == ESPERADAS
  for m in at.sidebar.selectbox[1].options: at.sidebar.selectbox[1].select(m).run(); assert not at.exception
  ```
  Comprobar también: título con `main-title` presente en `at.markdown`, `at.sidebar.caption` con métricas, `at.metric` ≥ N, `at.dataframe` ≥ N, y que el segundo rerun tras `select` de otro torneo y vuelta al primero no mezcla módulos (el enrutador usa `sys.modules`).
- Diff de regresión: serializar (`json`) `at.tabs`, etiquetas de selectbox y conteos por tipo antes/después con `probe.py` (ya guarda esto en `probe.json`).
- Equivalencia numérica: para un partido fijo, `adapter.predecir(...)` debe igualar a `mo.predecir_match` (p suma 1, mismo orden); test unitario por familia para la normalización 0/1/2 y de columnas de tabla/proyección.

### Riesgos
1. **Orden de clases** ([V,E,L] vs [L,E,V]): error silencioso que invierte local/visita; test obligatorio por familia (D codifica 0=local).
2. **Módulos con el mismo nombre** (`motor`, `recolectar`) y el enrutador con lock/`sys.modules` (app.py:48-90): `comun` debe importarse como paquete único estable y **no** depender de `sys.path` de la liga; cuidado con pickle que referencia `motor`.
3. **Cachés**: `st.cache_data` con `_M` y key estática (D, ucl) → proyecciones obsoletas; usar key por `len(partidos)+fecha.max()`. Cache de disco `simulacion_mc.pkl` y modelos `.pkl` hacen que el primer arranque re-entrene (~1 min/liga: ver probe) y escriban en el repo.
4. **Tiempo**: D pasa de 1.000 a 50.000 simulaciones por defecto en UI (ya lo hace la UI); validar tiempos de la proyección por liga y que `n_sims` mostrado coincida con el real.
5. **Datos faltantes por familia**: D no tiene `get_advanced_features`/`pi_tracker`/`recent_*` ni `mercados`; `detalle_variables` debe degradar (solo métricas de `get_features_for_match`). `metricas` sin `xgb`, `equipos` sin `abbreviation`, fixture sin `temporada`.
6. **Validación C/arg no depende del modelo** (y NBA histórico es in-sample): decidir si se cambia el motor o se avisa en la UI; no vender "modelo activo" en esa pestaña hasta resolverlo.
7. **Textos "históricos" y heurísticas** (ita, nba): moverlos a configuración con evidencia o eliminarlos; mostrar solo lo derivable de `validar`.
8. **Streamlit**: `width="stretch"` requiere ≥1.50 (ya instalado 1.50.0; ver requirements); `ImageColumn` y `Styler.map` dependen de pandas reciente; `st.set_page_config` duplicado en arg puede romper según versión.
9. **Cambios visibles para usuarios** (títulos, nombres de pestañas en D, defaults del modelo): avisar y mantener `?torneo=` y keys de widgets (`sel_a_eng`, ...) al cambiar sólo cuando se migre cada página.
10. **Cobertura de pruebas**: no hay tests de los motores; el smoke test solo detecta excepciones, no errores semánticos (orden de clases, columnas intercambiadas).
