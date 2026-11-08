# Reporte QA (parcial)
## E1 [alto] mex - 19 equipos en simulacion
mex/motor.py:55-73 ALTITUDES contiene 19 claves (Mazatlán FC con 0.010 y Atlante). fixture.csv/partidos.csv del torneo vigente tienen 18 equipos (sin Mazatlán FC). Se usa como lista de equipos en motor.py:881, 1123 (eqs=sorted(ALTITUDES)), 1253, 1375 -> Mazatlán entra con 0 pts y dato fantasma. Arreglo: derivar equipos del fixture+partidos del torneo vigente (_torneo_actual) o quitar Mazatlán FC de ALTITUDES.
Evidencia: _agente/mex2.py -> tabla "Actual" 19 filas (incluye "⚓ Mazatlán FC" 0 pts, Pts esperados 0.0); selectores Local/Visita con 19 opciones; pestaña validación "Por equipo" 19 filas (Mazatlán aparece por el Clausura 2026, válido) pero en tab 2 sobra. Sumas OK: P(Top6)=6.0, P(Liguilla)=8.0, P(Camp)=1.0.
## E2 [medio] mex - play-in: empate se resuelve al azar
mex/motor.py ~1176-1190 (np.where(gl==gv, np.where(rnd>0.5,...))): en el play-in real (7v8, 9v10, perdedor 7v8 vs ganador 9v10) un empate favorece al mejor ubicado en la tabla (7º / 9º / perdedor del 7v8); el código usa moneda al aire. Sesga P_Liguilla de 7º-10º (~hacia 50/50). Lo mismo en la final (f1 vs f2, ~1240) donde el empate global lo decide la tabla en la liguilla real.
Nota: reproducir con mex2.py (tabla 19 filas), mex3.py (Mazatlán FC seleccionable como Local/Visita: sel_a_mex/sel_b_mex tienen 19 opciones, TEAM_DETAILS en mex/app_mex.py:53 incluye Mazatlán). Con Local=Visita muestra "Selecciona dos equipos distintos." (OK). 4 modelos OK sin excepciones, 1X2 suman 100%. Sumas MC OK (6/8/1).
Mejora [bajo] mex: tabla actual sin columna PJ.
## E3 [alto] arg - proyección usa una sola tabla de 30 equipos; P(campeón)=100% y descenso por puntos del torneo
arg/motor.py:~830-890 (_simular_fixture_vec + monte_carlo): campeon_c = order[0] de la tabla única de 30 equipos (sin zonas A/B ni playoffs), copas_c = order[:4], descenso_c = order[-2:] (últimos 2 por puntos del torneo, no por promedios). Con solo 1 partido restante (Central Córdoba-Lanús 08/11, fixture.csv) la salida es degenerada: Rosario Central P_campeon=100%, Instituto 0% (mismo 34 pts, desempata DG), Estudiantes de Río Cuarto descenso 100%, Central Córdoba/Aldosivi 56%/44%. Repro: _agente/arg1.py (DF 2). Sumas: campeón 1.0, copas 4.0, descenso 2.0 (internamente coherentes, pero el formato no es el real).
Arreglo: modelar zonas + playoffs (o, si no hay datos de zona, no mostrar P(campeón) y rotularlo "líder de la tabla general"); descenso con tabla de promedios (necesita 3 temporadas); mostrar mensaje "fase regular terminada" cuando fixture<=1 partido.
## E4 [medio] arg - texto "4.000 iteraciones" vs 50.000 reales
arg/app_arg.py:394 dice "4.000 iteraciones del torneo completo"; arg/app_arg.py:99 llama simular_campeonato(n_sims=50000). Además 'torneo completo' falso: casi todo está jugado.
## E5 [medio] arg - tab 2 sin tabla real, columnas crudas
arg/app_arg.py:409-417: sin "Tabla actual (real)" (mex sí la tiene), columnas P_campeon/P_copas/P_descenso sin explicar qué copas (top-4) ni qué descenso. El tab muestra solo proyección.
Arg pares/modelos (arg2.py): 4 modelos x 5 pares OK, sin excepciones, 1X2 suma 100% (±0.1 redondeo), local=visita da "Selecciona dos equipos distintos.".
## M1 [bajo] arg/mex - tarjeta principal vs tabla comparativa difieren ~1-2 pt
arg2.py: Lanús vs Central Córdoba, XGB: tarjeta 47.0% vs tabla de modelos 48.2%; mex América-Guadalajara RF tarjeta 49.4% vs tabla 49.1%. Sin confirmar causa (tarjeta vía grilla_goles/Poisson, tabla vía predict_proba). Si es mezcla intencional, rotularlo.
## Bra / Chile (bra1.py, bc2.py)
Sin excepciones; 4 modelos x 3 pares (incl. local=visita -> mensaje correcto) OK; selector de fixture carga equipos OK; sin NaN. Sumas MC: bra campeón 1, Lib G.G. 4, Lib total 6, Sud 6, descenso 4 (coherente con formato 2026 de 20 equipos; desempate PTS,PG,DG,GF correcto). chile: campeón 1, Lib GG 2, Lib total 3, Sud 4, descenso 2 (chile/motor.py:913-917; cupos reales 2027 SIN CONFIRMAR, no verificables offline).
## M2 [bajo] bra/chile/mex - tabla "Actual (Real)" sin columna PJ ni posición; texto "Pts Esperados" con 5 decimales en raw (formateado en UI, OK).
## M3 [bajo] chile/motor.py:860 desempate DG>PG>GF; reglamento ANFP usa DG, luego GF (sin PG) - SIN CONFIRMAR.
## UNL (bra1.py con "Nations")
## E6 [medio] unl - cifras fijas incorrectas en sidebar
unl/app_unl.py:131: "54 Selecciones · 670 Partidos Históricos · 148 en Fixture" hardcodeado. Real: partidos.csv=766 filas (2018:142,2020:166,2022:164,2024:186,2026:108), fixture.csv=52 partidos (jornadas 5-6, 12-17 nov). Arreglo: calcular con len(M["partidos"]) y len(fixture).
## E7 [bajo] unl - texto "Poisson Dixon-Coles" falso
unl/app_unl.py:399 (prompt exportar para IA) dice "Poisson Dixon-Coles"; unl/motor.py:351 aclara "sin corrección Dixon-Coles".
## M4 [bajo] unl/motor.py:560-575 desempate de grupo = Pts, DG, GF (sin enfrentamiento directo UEFA); semifinales Final Four emparejan A1-A2 y A3-A4 fijo (sorteo real) y sin empate/prórroga (p_win/(p_win+p_loss)). Sumas OK: P(1°..3°, Desc)=1.0 por grupo; P_FinalFour total 4.
Extensión E6: el radio 'Seleccionar Partidos a Exportar' (key unl_filtro_ia) dice "🌐 Fixture Completo (148 Partidos)" pero fixture.csv tiene 52 (grep "148" en unl/app_unl.py). Sumas por grupo OK (Ligas A-D, 5 modelos, sin excepciones).
## E8 [medio] unl - 1X2 incoherente con xG en goleadas (Selección Libre)
unl3.py: Spain vs San Marino -> Victoria 84.1% / Empate 9.3% / San Marino 6.6% pero xG 5.00-0.11 y Over 2.5 = 81.2%; San Marino vs Spain (local débil) -> Spain 57.6%, xG 0.20-5.00. El 1X2 del clasificador ML se subestima en extremos (mientras Poisson satura en el tope 5.00 -> ver clip en unl/motor.py). Un 5.00 xG exacto indica clipeo. Arreglo: calibrar/mezclar 1X2 con Poisson en extremos o rotular.
## Mundial (lab1-4.py)
Torneo terminado (104/104 partidos; final España 1-0 Argentina 19-jul). Sin excepciones, 3 modelos x canchas x pares OK; local=visita da "Elige dos selecciones distintas.".
## E9 [alto] lab - "Probabilidades del Torneo (48)" y portada muestran la simulación PRE-torneo
lab/app_lab.py ~464-465 (métricas "Máximo/Segundo/Tercer Favorito/Cuarto Candidato": España/Argentina/Francia/Inglaterra) y tab "Probabilidades del Torneo" (DF: España P(Campeón)=29.2%, Argentina 18.9%, P(Avanzar) Uruguay 35%, Rep. Checa 31%...) ignoran los 104 resultados reales cargados. Contradice el tab "Cuadro de eliminatorias" (España 100%) y la sidebar ("104 partidos reales"). Pronóstico de 72 partidos de grupos también muestra solo P(local/empate/visita) de partidos ya jugados. Arreglo: si ESPN_DF tiene 104 partidos / final jugada, mostrar campeón real (España), subcampeón (Argentina) y archivar las probabilidades como "pre-torneo".
## E10 [medio] lab - podio 🥈/🥉 incorrecto en "Campeón según la simulación"
lab/app_lab.py:610-613 toma tabla.head(3) por P_campeon: salen "🥇 España 100.0%, 🥈 Argelia 0.0%, 🥉 Argentina 0.0%" (Argelia eliminada y subcampeona real Argentina; empate en 0.0 resuelto arbitrariamente). Arreglo: usar ganadores_ko (campeón/finalista/3er puesto reales) cuando el torneo terminó; no rotular 🥈🥉 si P=0.
## E11 [medio] lab - textos de torneo en curso
lab/app_lab.py: subtítulo "Simulación de campeonato en vivo", "Modelo vivo: resultados reales → re-simulación", botón "🔄 Actualizar y re-simular (4.000 mundiales)", "Cuadro real, ya definido: los 16 cruces de dieciseisavos... 15.000 torneos jugando solo las eliminatorias" -> el torneo ya acabó; la simulación es trivial (100%/0%).
## E12 [medio] lab - llamadas a ESPN en vivo en cada carga
Con socket.connect bloqueado (lab3.py): intentos a ESPN desde cargar_espn (app_lab.py:123), cargar_envivo (142), cargar_bracket (150), sim_bracket->ganadores_ko (163) -> espn_live._fetch_events (espn_live.py:46) x4 por run (TTL 120 s; sim_bracket sin ttl pero llama ganadores_ko). Con torneo terminado deberían leer solo el respaldo local data/espn_mundial_2026_events.json.
## M5 [bajo] lab - mensajes "traídos de ESPN"/"ESPN: 104 partidos reales cargados" (app_lab.py:306, 638) aparecen aunque la red falle y se use el respaldo local (CSV/JSON); engañoso.
## M6 [bajo] lab - tab "Torneo en vivo": tabla de partidos usa nombres en inglés (Mexico, South Africa) mientras el resto usa español.
