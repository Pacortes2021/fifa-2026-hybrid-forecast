# REPORTE QA (parcial: NBA y UCL revisadas; resto pendiente por presupuesto)
Scripts: _agente/common.py, t_nba*.py, t_ucl*.py (ejecutar con el venv: `python _agente/t_nba2.py`).

## ERRORES
### B1 (ALTO, NBA) Probabilidades de handicap (spread) invertidas
nba/motor.py:~786 (`cubierto = np.mean((pts_h_sims - pts_a_sims) > line)`); el comentario dice `> -line`. Etiqueta en nba/app_nba.py:~350.
Repro: t_nba2.py. Washington (19.4% de ganar) vs OKC: "Washington -11.5 = 50.1%", "+11.5 = 6.2%" (debería ser lo contrario). Boston vs Lakers: "Boston -11.5 = 86.9%", "+3.5 = 54.6%".
Arreglo: `> -line` (cubre si margen + línea > 0). Revisar la cuota justa derivada.

### B2 (ALTO, NBA) El selector de modelo no afecta Clasificación/Playoffs ni Validación
nba/motor.py:807 `simular_temporada(M, n_sims)` llama `predecir_match(M, eq1, eq2)` sin `modelo`; app_nba.py:80 solo usa `modelo` en la clave de caché de Streamlit. La caché de disco `simulacion_mc.pkl` tampoco lo incluye. `validacion_en_vivo` usa siempre `pipe_lasso` (motor.py:~1025).
Repro: t_nba2.py: las tablas Este/Oeste son idénticas con los 4 modelos (salida "True" x4) y las métricas de validación no cambian (135 partidos, 63.7%, LL 0.6128).
Arreglo: pasar `modelo` a simular_temporada y validacion_en_vivo, e incluirlo en las claves de caché; o rotular que es fijo.

### B3 (MEDIO, NBA) "P_Playoff_Directo" es en realidad P(clasificar a playoffs)
motor.py: `playoff_counts` suma puestos 1-6 y además los 2 ganadores del play-in por conferencia. La suma por conferencia es 8.00, y P_Playoff_Directo + P_PlayIn llega a 1.42-1.46 para un equipo.
Arreglo: contar directo solo 1-6 (suma 6 por conferencia) y añadir "P_Playoffs" total; o renombrar la columna.

### B4 (MEDIO, NBA) Desempate alfabético en el Monte Carlo NBA
motor.py:~865: `sorted(..., key=wins, reverse=True)` es estable sobre `todos_equipos` ordenado alfabéticamente, así que a igualdad de victorias gana el primero por orden alfabético. El arreglo ya aplicado a otros torneos no se hizo aquí. También afecta posición proyectada y localía en las Finales.
Arreglo: añadir ruido aleatorio < 1 al desempate, o usar diferencial.

### B5 (MEDIO, UCL) Cuadro eliminatorio aleatorio, no el cuadro fijo de la UEFA
ucl/motor.py:~1262-1285: `np.random.shuffle(po_winners)` antes de octavos y `np.random.shuffle(qf_teams)` antes de cuartos. En la UEFA 2026-27 el cuadro es fijo (el cruce 1/2 con los playoffs 15-18 / 7-10 etc. condiciona el camino). Con shuffle el puesto 1-8 no da ventaja de camino y se pueden cruzar equipos que no deberían (p. ej. dos del mismo lado). Además no se simula la localía de ida/vuelta en cuartos/semis (equipos arbitrarios son team_a/team_b).
Arreglo: definir tabla de cruces fija (cuadro UEFA) con seeds 1-8 contra ganadores de playoff por pareja, y árbol fijo hacia la final.

### B6 (MEDIO, UCL) Rotulado de jornadas incorrecto
ucl/app_ucl.py (radio "Por Jornada Oficial UCL"): "Jornada 1 (24/11 - 25/11/2026)", "Jornada 2 (08/12...)" etc. Cada equipo ya tiene PJ=4 en la tabla, así que las jornadas restantes son la 5 a 8. Además "Fixture Completo (36 Clubes)... Incluye los 72 partidos de la Fase de Liga" son solo los 72 pendientes (otros 72 ya se jugaron).
Repro: t_ucl3.py. Arreglo: numerar según jornadas ya disputadas (partidos jugados por equipo + 1) o leer el número de jornada de ESPN; cambiar el texto a "72 partidos restantes".

### B7 (BAJO, UCL) Caché de simulación inválida en cada despliegue
ucl/motor.py:~1150: la clave usa `fixture.csv` st_mtime y tamaño (no un hash del contenido ni de partidos.csv / código del motor). Tras un git clone el mtime cambia, así que el pkl versionado nunca vale y se recalculan 10.000 simulaciones. Al revés, un cambio del motor con el mismo fixture sirve resultados viejos.
Arreglo: hash de contenido de fixture.csv + partidos.csv + motor.py.

### B8 (BAJO, UCL) Goles simulados independientes del resultado 1X2
ucl/motor.py `_simular_fixture_vec`: el resultado sale de las probabilidades, pero los goles (DG/GF para desempates) salen de Poisson independientes. Un "gana local" puede tener DG negativo. Además `eq2idx.get(l, 0)` asigna silenciosamente al equipo 0 los equipos desconocidos.
Arreglo: muestrear goles condicionados al resultado o derivar el resultado de la matriz de marcadores.

### B9 (BAJO, NBA) Textos/valores fijos
- app_nba.py:~120 y ~443: "Test Out-of-Sample (1.394 partidos)" fijo; el test real (temporada>=2026) tiene 1461 partidos (t_nba3.py). El sidebar dice "Test 2025-26+".
- Alertas con "Cumplimiento Histórico: 90.5% / 88.8% - 95.7% / 77.1% / 237.1 pts / 217.8 pts" fijos y calculados sobre todo el histórico, incluida la muestra de entrenamiento y calibración (in-sample). Fuera de muestra: P(visita)>=80% da 92% con n=50 (similar, pero n distinto del "n=126" que se muestra). Valores desactualizados cuando entran datos nuevos.
- Radio "Histórico Completo (2022-2027)": mezcla entrenamiento/calibración (in-sample) con test; da 65.7% / LL 0.6161 como si fuera validación. Rotular "incluye datos de entrenamiento".
- Jugadores estrella fijos en `NBA_FRANCHISE_STARS` (p. ej. "Jayson Tatum, Jaylen Brown", "LeBron James, Anthony Davis") posiblemente desactualizados para 2026-27: sin confirmar.

## Verificado OK
- NBA: sin excepciones con los 4 modelos, local=visita (aviso correcto), equipos extremos. Campeón suma 1.0 (East .492 + West .508), finalistas de conferencia 2.0 por conferencia, play-in 4 por conferencia, 15 equipos por conferencia; sin NaN.
- UCL: sin excepciones con 4 modelos (sidebar y pestaña), 4 radios de fixture, checkboxes de eliminatoria y sede neutral, local=visita (aviso). Campeón 1.0, final 2.0, Top8 8.0, Playoffs 16.0, Eliminado 12.0, 36 equipos, PJ=4 para todos, sin NaN. Mercados complementarios (O/U, BTTS, DC, DNB) coherentes.

## MEJORAS PROPUESTAS
- M1 (NBA): la probabilidad de victoria del clasificador y la del Monte Carlo gaussiano (win_pct_local_mc) son modelos distintos; mostrar una sola o ajustar mu del margen a p_local. Además sigma_spread guardado es 14.7 pero se muestra "±11.5" (MAE).
- M2 (UCL): la simulación no usa tiempo extra/penales en cruces (empate = moneda).
- M3: el warning `use_container_width` (deprecado) se emite decenas de veces; migrar a `width="stretch"`.

## PENDIENTE DE REVISAR (no alcanzó el presupuesto)
Copa Mundial (lab), UEFA Nations League (unl), Brasileirão (bra), Liga MX (mex; causa de 19 equipos), Chile, Argentina (zonas/playoffs/promedios), validación de ucl por modelo en detalle, formatos de liguilla/play-offs.
