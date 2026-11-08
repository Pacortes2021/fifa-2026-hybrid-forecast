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

## Pendiente (en orden)
1. [ ] CI en PRs: `.github/workflows/ci.yml` con smoke tests + recorrido AppTest de los 23 torneos (`tests/ui_smoke.py`).
2. [ ] Arreglar los errores verificados de los reportes `qa_*.md` → PR "fix: hallazgos de QA".
3. [ ] Estandarizar la interfaz de las 19 ligas domésticas según `ui_estandar.md` → módulo compartido + migración incremental, verificada liga por liga con AppTest → PR propio.
4. [ ] Mejoras de modelos según `modelos.md` (solo las que mejoren log-loss en test de forma medible) → PR propio.
5. [ ] Pendientes conocidos: tasas reales de las alertas desde la validación; `predecir_match` sin fecha (descanso/congestión = 0 en vivo); Liga MX simula 19 equipos (datos tienen 18); unificar los `motor.py` duplicados.

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
