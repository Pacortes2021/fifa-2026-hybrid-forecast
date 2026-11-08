# Plan de trabajo autónomo (noche del 8-nov-2026)

Pablo (@Pacortes2021) autorizó trabajar de forma autónoma mientras duerme: usar la app con agentes, buscar errores y mejoras, arreglarlos, estandarizar las interfaces de las ligas y revisar los modelos. **Regla: todo va en ramas y PRs; no mergear a `main` sin su visto bueno** (la app pública se despliega desde `main`).

Este archivo es el estado compartido entre sesiones. **Actualízalo al terminar cada bloque de trabajo** (commit + push a `wip/revision-nocturna`).

## Hecho antes de esta noche
- PR #1 (mergeado): recolectores ESPN con consulta anual, temporada completa en ligas jul–jun, tabla de arg/mex.
- PR #2 (mergeado): validación pre-partido honesta, enrutador con lock, CSS en cada rerun, botón de re-entrenar oculto (`PORTAL_ADMIN=1` o `?admin=<clave>`), box scores diarios, textos (sin aciertos inventados, sin "Dixon-Coles" falso), desempate aleatorio del Monte Carlo.
- Workflow manual 8-nov 04:31 UTC: OK, box scores al día en `main`.

## Agentes lanzados (8-nov ~04:40 UTC)
Cada uno deja su reporte en `docs/revision/` cuando termina:
| Archivo | Alcance |
|---|---|
| `qa_top5.md` | QA con AppTest: eng, esp, bund, ita, fra |
| `qa_europa.md` | QA con AppTest: ned, por, bel, tur, sco, aut, den, gre, nor, swe |
| `qa_especiales.md` | QA con AppTest: lab (Mundial), ucl, unl, nba, bra, mex, chile, arg |
| `modelos.md` | Revisión metodológica de los modelos con métricas en test ≥2025 |
| `ui_estandar.md` | Comparación de las 23 interfaces + propuesta de estándar + adaptador de motores |

## Pendiente (en orden) — backlog consolidado de los 5 reportes
1. [x] CI en PRs + chequeo de frescura → **PR #3** (abierto).
2. [~] **PR #4 `fix/qa-nba-ucl`** (abierto o por abrir): NBA hándicap invertido (B1), desempate alfabético (B4), P_Playoff_Directo separado de P_Playoffs (B3), nº de partidos de test fijo (B9); ned "Schaal" → "Campeón". Verificado: spread monótono (débil local −11.5 = 6.2%, +11.5 = 50.1%), sumas por conferencia 6/8/4.
3. [ ] **Errores pendientes de los QA** (verificar cada uno antes de arreglar):
   - NBA B2: el selector de modelo no afecta `simular_temporada` ni `validacion_en_vivo` (pasar `modelo`, incluirlo en la clave de caché). NBA B9: textos "Cumplimiento Histórico" in-sample; "Histórico Completo" mezcla train/test → rotular.
   - UCL B5: cuadro eliminatorio con `np.random.shuffle` en vez del árbol fijo UEFA (`ucl/motor.py` ~1262-1285). B6: "Jornada 1…" debería ser 5–8 (`ucl/app_ucl.py:546-557`). B7: clave de caché por mtime. B8: goles simulados independientes del 1X2; `eq2idx.get(l, 0)` silencioso.
   - Top-5 E1: "Importancia de Variables" muestra RF con XGB/Stacking (eng/esp/bund ~l.555-585). E2/E3: esp "Deportivo" sin escudo (norm_name "Deportivo La Coruña"), claves de TEAM_DETAILS no coinciden. E4: fixture esp sin Rayo–Athletic (379/380); fra 2025 sin Nantes–Toulouse; ita 2022 Spezia–Verona duplicado → revisar recolector (aplazados, dedupe). E5: caption "eliminación directa neutral" en ligas (eng/esp/bund/ita). E9: selector de equipos lista equipos de temporadas pasadas.
   - Europa E1: "Puntos esperados" con nº de partidos fijo y 1.35 pts/partido fijo (bel `motor.py`:750, tur 789, sco 766, aut 769, den 787, gre 784) → usar PJ + restantes del fixture y las probabilidades del modelo. E2: sco/aut/den/gre(/bel) ignoran el split/playoffs → al menos rotular "tras fase regular". E3/E4: cupos (DESCIENDEN ned/por = 3, P_champions ned suma 3) → verificar reglamento 2026-27 y parametrizar. E6: por Sporting–Benfica xG incoherente con 1X2 (sin causa confirmada). E7: nor/swe stacking = solo XGB (el peor) rotulado "óptimo". E8: sco temporada 2023-24 con 52 partidos; bel 16→18 equipos.
   - QA sin terminar: lab, unl, bra, mex (causa de 19 equipos), chile, arg; y en europa/top-5 la pestaña de validación por temporada, pares extremos y textos.
4. [ ] **Estandarizar la interfaz** según `ui_estandar.md` (familias A eng/esp/bund/ita, B fra/ned/por/bel, C mex/bra/chile/arg, D tur/sco/aut/den/gre/nor/swe, E ucl/unl; nba/lab aparte). Empezar por el cambio mecánico `use_container_width` → `width="stretch"`, luego `comun/` con `MotorAdapter` normalizado (p siempre [Local, Empate, Visita]) y piloto eng; migrar familia por familia verificando con `tests/ui_smoke.py`. Cuidado con el orden de clases (familia D codifica 0 = local).
5. [ ] **Modelos** según `modelos.md`: el ML no supera a un Elo-LR (Δ log-loss pooled −0.003, no significativo); stacking degenerado; XGB subajustado; isotónica empeora (+0.07). Recomendación 1: modelo único simple (lasso o LR con 6–10 variables) en vez del stacking, o XGB con ≥300 árboles; 2: quitar `stadium_occupation`/`avg_attendance`; 4: tope/encogimiento de P(empate); 5: K≈30 común. Validar con el mismo split y bootstrap (scripts en `docs/revision/scripts_modelos/`). Pendiente medir: Poisson (goles, O/U, BTTS) y backtest Monte Carlo.
6. [ ] Pendientes conocidos: tasas reales de las alertas; `predecir_match` sin fecha; Liga MX 19 equipos; unificar `motor.py`.

## Cómo trabajar (para una sesión nueva)
```bash
git clone https://github.com/Pacortes2021/fifa-2026-hybrid-forecast.git && cd fifa-2026-hybrid-forecast
git checkout wip/revision-nocturna
python3.11 -m venv ../venv && ../venv/bin/pip install -q -r requirements.txt
../venv/bin/python tests/smoke.py          # 20 ligas, ~3 min
```
Recorrido de la UI headless: `streamlit.testing.v1.AppTest.from_file("app.py", default_timeout=900)`; antes `os.chdir(repo); sys.path.insert(0, repo)`; el selector de torneo es `at.sidebar.selectbox[0]`.

## Registro
- 04:40 UTC — agentes lanzados; rama `wip/revision-nocturna` creada; continuación programada para después del reinicio de la ventana (07:50 UTC).
- 04:55 UTC — PR #3 (CI + frescura) abierto; su primer run de CI es la verificación en GitHub.
- ~05:00 UTC — los 5 agentes terminaron (QA parciales por presupuesto); reportes en `docs/revision/`; backlog consolidado arriba. Ventana de 5 h al 81%.
