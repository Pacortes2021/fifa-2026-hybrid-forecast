# Revision de modelos de futbol (20 ligas) - medido sobre test (ucl: train<=2024, cal 2025, test>=2026; resto train<=2023, cal 2024, test>=2025)
Scripts: common.py, lib.py, exp1.py (modelos/baselines/fugas/calibracion), an1-an3.py (tablas), leak_cov.py (cobertura de features avanzadas), exp2.py (K/HOME_ADV de Elo). Resultados: res1.csv, res1.pkl, res_elo.pkl, leak_cov.csv.
Protocolo: se re-entrenan los modelos con los hiperparametros de cada motor solo con train; pesos de stacking y calibradores sobre cal; metricas en test. Tambien se mide el modelo desplegado (train+cal).

## 1. Log-loss test por liga (menor es mejor)
| liga | n | freq | elo-LR | lasso | rf | xgb | stack | core-LR |
|---|---|---|---|---|---|---|---|---|
| eng|475|1.088|1.038|1.045|1.034|1.030|1.033|1.039|
| esp|494|1.056|0.985|0.957|0.969|0.971|0.958|0.959|
| bund|384|1.066|0.977|0.981|0.966|0.982|0.977|0.968|
| ita|485|1.079|0.982|0.987|0.982|0.996|0.982|0.987|
| fra|389|1.065|1.000|0.989|0.994|1.017|0.989|1.010|
| ned|412|1.084|1.006|0.988|0.981|0.997|0.984|0.996|
| por|399|1.082|0.922|0.925|0.922|0.937|0.925|0.932|
| bel|408|1.071|1.009|1.012|1.020|1.043|1.013|1.017|
| tur|401|1.076|0.994|0.997|1.016|1.066|1.016|1.002|
| sco|301|1.074|0.999|0.992|1.031|1.091|0.996|1.007|
| aut|264|1.075|1.065|1.041|1.064|1.108|1.044|1.072|
| den|273|1.076|1.037|1.032|1.037|1.066|1.032|1.031|
| gre|300|1.077|0.946|0.939|0.955|0.966|0.939|0.931|
| nor|441|1.036|0.960|0.967|0.965|1.032|0.974|0.958|
| swe|459|1.079|1.022|1.030|1.035|1.075|1.030|1.034|
| bra|719|1.045|1.005|1.003|1.006|1.013|1.006|0.998|
| mex|647|1.066|1.011|1.003|1.009|1.016|1.003|1.008|
| chile|450|1.051|1.025|1.026|1.024|1.028|1.025|1.025|
| arg|1004|1.084|1.070|1.068|1.069|1.072|1.068|1.083|
| ucl|261|1.007|0.909|0.907|0.916|0.975|0.917|0.930|
| **pooled (por partido)** | 8966 | 1.0678 | 1.0039 | 1.0010 | 1.0050 | 1.0251 | 1.0017 | 1.0053 |
Brier medio (20 ligas): freq .645, elo .595, lasso .593, rf .595, xgb .609, stack .594. Accuracy media: freq 44.7%, elo 51.1%, lasso 51.9%, rf 51.6%, xgb 50.1%, stack 51.8%.
Desplegado (train+cal) pooled: stack 1.0049, rf 1.0013, lasso 1.0007.

## Hallazgos medidos
1. **Todo el valor esta en Elo; el ML aporta ~0.** Elo-LR vs frecuencias: -0.0638 [-0.072,-0.056] (IC95 bootstrap por partido). ML vs Elo-LR (delta log-loss pooled): lasso -0.0030 [-0.0073,+0.0017]; stack -0.0022 [-0.0065,+0.0021]; rf +0.0010; xgb +0.0212 [+0.015,+0.027] (peor). Ninguno significativo. Por liga, stack vs Elo significativamente mejor solo en esp (-0.026) y ned (-0.021); significativamente peor en tur (+0.023). Stack < Elo en 12/20 ligas (coin flip). Una LR de 6 variables (core: elo_diff, pi_diff, pi_overall_diff, form_diff, ppg_diff, squad_value_diff) rinde igual que el ML de 30-70 variables (1.0053).
2. **XGBoost esta subajustado** (100 arboles, lr 0.02-0.05, depth 3): casi prior, peor que Elo en 17/20 ligas, ECE mayor. Temperature scaling lo mejora -0.0087 [-0.012,-0.006], confirmando subconfianza. Tiene peso ~0 en el stacking casi siempre.
3. **Stacking degenerado:** los pesos optimizados (Nelder-Mead sobre 125-380 partidos de cal) caen en esquinas: [1,0,0] en den, gre, swe, mex, arg; [0,1,0] en tur, bra; [0.04,0.96,0] ita. Stack vs mejor individual por liga ~0; stack vs lasso +0.0008 (n.s.); stack vs rf -0.0032 [-0.0055,-0.0008]; promedio simple 1/3 es peor que el stack (+0.0024 [0.0007,0.0041]), por arrastre de xgb. En la practica el stacking = elegir un modelo con ruido (en tur eligio rf, 1.016, cuando lasso daba 0.997).
4. **Calibracion:** global esta bien (pooled 8966 partidos: empate predicho .256 vs real .250; local .403 vs .409; ECE por clase ~.05 en todos, incluido Elo). Curva pooled muy cercana a la diagonal salvo cola de empate: P(empate) predicha 0.337 -> real 0.289 (sobreconfianza en el rango alto; rango predicho hasta 0.46). Desvios por liga (eng empate .227 vs .274; chile .270 vs .216; ucl .204 vs .157; fra local .41 vs .47; swe local .31 vs .37) son comparables al error de muestreo (SE ~0.02) salvo chile/ucl en empates. El argmax casi nunca predice empate (0-8%): accuracy no es metrica util aqui.
   Calibrar sobre cal 2024: temperature scaling +0.0013 [-0.0004,+0.0031] (n.s., no ayuda; T medio .93); ajuste de sesgo de empate +0.0002 (nulo); **isotonica empeora +0.0715 [+0.058,+0.085]** (muestra de cal chica). No calibrar el stack/lasso/rf; solo tendria sentido T para xgb.
5. **Fugas de variables avanzadas (squad_value, edad, tamano plantilla, % extranjeros, estadio, asistencia):** cobertura medida: en test casi todo equipo-temporada tiene fila exacta (no-exacto: esp 2.4%, por 1.3%, bra 12.9% [5% sin equipo -> constantes por defecto], chile 6.7%, resto 0%). Fallback a temporada FUTURA solo en train: bra 1.9%, chile 0.9% (el resto 0%): la fuga por "temporada mas cercana" es marginal. En ucl, get_advanced_features devuelve la PRIMERA temporada del equipo en el dict si no hay exacta (no es la mas cercana; no ocurre en test por cobertura exacta). Impacto de re-entrenar sin las ~14 variables avanzadas: stack +0.0020 [-0.0010,+0.0050], lasso +0.0013, rf +0.0020 (n.s.); sin ellas el stack sigue ~= Elo (-0.0002). Quitar solo asistencia/ocupacion (agregados de temporada completa, posible fuga intra-temporada): stack -0.0006 [-0.0013,+0.0002] (mejora ligeramente o neutro; ganancia en 14/20 ligas). Conclusion: no hay evidencia de que la fuga infle el test; las variables avanzadas aportan <=0.002. Que squad_value sea valor a inicio de temporada o fin NO se puede verificar con estos datos (hipotesis).
6. **Elo (K, HOME_ADV):** se reprodujo el Elo del motor exactamente (corr 1.00; por/bel 0.995). Grid K in {10,20,30,45,60} x HOME in {30,55,80} (LR sobre elo_diff, 19 ligas): log-loss test medio K=30: 1.0020; K=20: 1.0029; K=35 actual (HOME por liga): 1.0025; K=45: 1.0038; K=60: 1.0064; K=10: 1.0117. HOME_ADV dentro del update es irrelevante (+-0.0002; la LR reajusta el sesgo local). Elegir K por liga con cal da 1.0027 (nada). Conclusion: K~25-35 unico para todas esta bien; no tunear por liga. (Nota: la definicion de elo_diff difiere entre motores: eng/esp/bra... sin HOME_ADV, den/tur/bel/ucl... incluyen HOME_ADV en elo_local/elo_visita.)

## Inconsistencias entre motores (medido leyendo el codigo)
- RF: eng max_depth5/500 arboles/msplit30; esp depth3/500/30; bund depth3/100/5; ita-fra-ned-por-bel depth5/200/15; tur-gre... depth5/200/15; bra depth3/500/30; mex depth5/500/30; chile depth5/250/5; arg depth3/100/30; nba otra. XGB: lr 0.02 (eng, esp, bund, bra, mex, chile, arg) vs 0.05 (ita, fra, ned, por, bel, tur, sco, aut, den, gre...) vs 0.03 (unl); sin n_jobs en algunos. Lasso: max_iter 2000-4000, algunos sin random_state (bra, mex, chile, arg); grids de C distintos.
- Variables: eng 51 col (incluye elo_home/away, 10 stats x total/sede), esp 37, bel 30, bra 73, mex 76 (box score extendido, es_liguilla), den/gre/nor 31; ppg_diff solo en algunas; distancia como travel_dist_log_km vs distance_km/log; elo_diff con o sin HOME_ADV.
- Constantes: K_LIGA 35 en todas excepto chile 30 y ucl 40; HOME_ADV 45-60 variable (irrelevante segun lo medido); DESCIENDEN/CUPOS_COPA hardcodeados por liga (sco=1, aut=1, tur=4...): verificar contra reglamento vigente (pendiente); mex y bra sin DESCIENDEN/CUPOS en el scan (otro mecanismo).
- Split: ucl usa train<=2024/cal 2025/test>=2026; unl usa train<=2022, cal 2024 (hueco 2023), test>=2026; nba <=2024/2025/2026.
- Los modelos desplegados se reentrenan con train+cal pero los pesos de stacking se fijaron con modelos train-only (aceptable); el stacking de "den/gre/nor/swe" usa weights_stacking/otro nombre de clave (M["stack_w"] no existe en todos: tur 'weights_opt', den 'weights_stacking', ucl 'weights_stacking').

## Recomendaciones priorizadas
| # | Accion | Impacto esperado | Esfuerzo | Donde | Validacion |
|---|---|---|---|---|---|
| 1 | Reemplazar stacking por un modelo unico simple: LR (L2 o lasso) con ~6-10 variables (elo_diff, pi_diff, pi_overall_diff, form_diff, ppg_diff, squad_value_diff) o lasso actual; descartar XGB o subirlo (300+ arboles, lr .05) | log-loss ~igual (+-0.003) con menos varianza y mantenimiento; evita casos como tur (+0.02) | bajo | `cargar_y_entrenar` de cada motor (bloques lasso/rf/xgb/_stk3) | mismo split; comparar con exp1.py (res1.csv), IC bootstrap |
| 2 | Quitar stadium_occupation/avg_attendance (y por defecto los *_diff de estadio/edad/tamano plantilla) | -0.0006 a +0.002 (neutro), elimina riesgo de fuga intra-temporada | bajo | lista `cols_feat`/`features` | re-correr exp1 variante no_att/no_adv |
| 3 | Entrenar un modelo agrupado multi-liga (elo_diff + variables core + intercepts por liga) para ligas chicas (sco, aut, den, gre: ~600-900 partidos de train) | HIPOTESIS (no medido): mejora por mas datos | medio | nuevo modulo comun | test por liga vs elo-LR |
| 4 | No usar calibracion isotonica; no calibrar stack/lasso/rf; si se mantiene xgb, temperature scaling | evita +0.07 de perdida | nulo | ninguna | ya medido |
| 5 | Limitar P(empate) (p. ej. clip superior ~0.32) o shrink hacia la media de liga; ucl/chile sobreestiman empates | pequeno, calibracion de cola | bajo | `predecir_match` | curva de calibracion de empate pooled |
| 6 | Estandarizar K=30 (todas), HOME_ADV unico/definicion unica de elo_diff, hiperparametros unicos en un modulo compartido | consistencia; Elo K=30 -0.0005 vs actual (n.s.) | medio | constantes y pipelines de los 22 motores | exp2.py |
| 7 | Cobertura de adv features: bra 5% de test sin equipo (constantes por defecto) y 13% con temporada no exacta; fallback ucl devuelve primera temporada, no la mas cercana pasada | pequeno | bajo | `get_advanced_features` bra/chile/ucl | leak_cov.py |

## Pendiente (no medido por limite de presupuesto)
- (4) Modelo de goles Poisson: goles medios, Over 2.5 y BTTS predichos vs reales, efecto del reescalado al 1X2.
- (5) Backtest Monte Carlo (puntos finales simulados desde mitad de temporada vs reales).
- Verificar constantes de cupos/descensos contra reglamento; unl y nba; calibracion por clase con IC por liga; ECE tiene sesgo con n~300-1000 (8 bins por cuantiles).
- Caveat: test incluye la temporada en curso (parcial); IC por partido ignoran correlacion por equipo/temporada (IC reales algo mas anchos).
