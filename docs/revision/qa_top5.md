# REPORTE QA (parcial, verificado con AppTest) - eng, esp, bund, ita, fra
Scripts: _agente/h.py, audit.py, datos.py, datos2.py, t1.py, t2.py, t3.py, t4.py, t_names.py

## Errores
E1 (medio) eng, esp, bund, ita?, fra? - Pestana "Importancia de Variables" con Stacking o XGBoost muestra la tabla/titulo "Random Forest (Gini)" (eng/app_eng.py ~l.560-585: else -> pipe_rf). Reproducir: `_agente/t2.py eng`; salida: XGBoost y Stacking -> titulo "Importancia de Caracteristicas: Random Forest", columnas ['Variable','Importancia (Gini)']. Verificado en eng; ita/fra/esp/bund usan el mismo patron (sin confirmar). Arreglo: rama propia para xgb (pipe_xgb.feature_importances_) y para stacking (mostrar RF/XGB/LASSO con aviso) o rotular "del RF dentro del ensamble".
E2 (medio) esp - Escudo faltante de "Deportivo": partidos.csv usa "Deportivo", equipos.csv tiene norm_name "Deportivo La Coruna" -> logo_url None; en tablas sale "⚽ Deportivo" y sin escudo (`_agente/t3.py esp`). Arreglo: alias de nombres en motor/_cargar_equipos o corregir norm_name.
E3 (bajo) esp - esp/app_esp.py TEAM_DETAILS: claves no coinciden con nombres reales ("Barcelona" vs "FC Barcelona", "Atletico Madrid" vs "Atletico de Madrid", Osasuna/CA Osasuna, Sevilla/Sevilla FC, Valencia/Valencia CF, Villarreal/Villarreal CF, Celta Vigo/Celta de Vigo, Elche; faltan Deportivo, Malaga, Racing Santander). 11 de 20 equipos 2026 caen al emoji generico ⚽ (`_agente/t_names.py`). Arreglo: usar nombres de partidos.csv.
E4 (medio) esp - fixture.csv incompleto: falta Rayo Vallecano vs Athletic Club (379 de 380 partidos; Rayo PJ=10). Monte Carlo simula 37 partidos para esos dos (`_agente/datos2.py`). Tambien huecos historicos: esp 2022 (-5), fra 2025 (Nantes-Toulouse), ita 2022 Spezia-Verona duplicado (ocurre 2 veces). Arreglo: recolectar rellena partidos aplazados / dedup por (local,visita,temporada); sin confirmar impacto en entrenamiento.
E5 (bajo) eng - app_eng.py ~l.246: caption "Expectativa de clasificacion en eliminacion directa neutral" en una liga (texto copiado de torneos de copa; sin sentido en liga). Presente en copias de las otras ligas (sin confirmar). Arreglo: quitar.
E6 (bajo) eng/esp/bund - subtitulo "LASSO + RF + Simulacion..." omite XGBoost/Stacking (ita/fra si dicen XGBoost); sidebar caption global "LASSO, Random Forest y Stacking" omite XGBoost. Cambiar texto.
E7 (bajo) eng - motor.py ordenar_tabla: comentario "LaLiga desempata..." (motor copiado). Tambien la tabla de MC desempata por DG/GF; LaLiga real usa head-to-head primero, Bundesliga/Serie A/Ligue 1 distintos (mejora).
E8 (bajo) todas - Etiqueta "Metricas Out-of-Sample (2025+)" y baseline "temporadas <= 2024" hardcodeados (motor.validacion_en_vivo); se desfasaran al avanzar los anios.
E9 (bajo) bund (y probablemente todas) - selector Local/Visitante lista TODOS los equipos del dataset (bund: 27 opciones vs 18 de la temporada: Hertha, Schalke... aparecen). Arreglo: filtrar a equipos de la temporada actual o rotular.
E10 (bajo) todas - Goles esperados mostrados (la, lb Poisson crudo) no corresponden a la matriz "ajustada al 1X2" (grilla reescalada) -> marcadores top y lambdas inconsistentes (app_eng.py ~l.275 vs motor.grilla_goles).

## Verificado OK (sin hallazgos)
- eng/esp/bund/ita/fra, los 4 modelos: sin excepciones; MC: P(Campeon)=1.0, P(Copas)= cupos (8/7/6/7/6), P(Descenso)= 3/3/2/3/3, P(Champions)=4 (ita, fra); tabla actual = 20/20/18/20/18 equipos igual a proyeccion. (t3.py)
- eng: 1X2 suma 1, cuotas=1/p, Over/Under, BTTS, DNB, doble oportunidad coherentes con 4 modelos (t2.py). bund: 9 pares + 5 fixtures cargan ambos equipos, sin fallas (t4.py).
- Fixture futuro sin partidos ya jugados ni solapes en las 5 ligas; sin duplicados en temporada actual.

## Pendiente de revisar (no alcanzado)
t4.py (pares/fixtures) en eng, esp, ita, fra; validacion por temporada (2025 y 2026) y % acierto por equipo en esp/bund/ita/fra; matriz de goles suma; modelos LASSO en equipos recien ascendidos.

## Mejoras propuestas
- Selector de fixture queda desfasado si el usuario cambia equipos a mano; no re-dispara si se elige el mismo partido.
- Advertencia "por debajo del baseline" en validacion solo menciona pocos partidos.
- Eliminar warnings use_container_width (deprecado) en dataframes de app_*.py.
