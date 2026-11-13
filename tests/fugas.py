"""
Suite de pruebas de integridad científica y prevención de fugas de información (data leakage):
  1. Guardia estática AST sobre todos los `*/motor.py`: prohíbe usar `X_test` o `y_test` dentro de
     bucles de búsqueda de hiperparámetros (`C`, `C_cand`, `c_val`).
  2. Guardia de invalidación de caché: todo motor con `.pkl` debe incluir `hash_modulos_comunes()`.
  3. Prueba de invariancia causal en NBA (`_precomputar_contexto_causal_equipos`): alterar de forma
     extrema los puntos, rebotes o ausencias en partidos futuros (`>= t_corte`) NO modifica ni un
     solo bit de las estadísticas de equipo, rotación titular ni bajas conocidas en partidos `< t_corte`.
  4. Verificación de terminología honesta en el escáner NBA (sin `ROI (-110)` ni `SELLO DE ORO`).
  5. Verificación del registro inmutable de predicciones (`generado_utc < fecha_partido`, append-only).
"""
import ast
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from comun.motor_comun import (
    excluir_variables_estimadas,
    hash_modulos_comunes,
    seleccionar_c_lasso,
)
from comun.registro_predicciones import cargar_registro, registrar_predicciones


def test_sin_fuga_hiperparametros_en_test():
    """Ningún motor puede usar X_test o y_test dentro de un bucle de selección de hiperparámetros."""
    motores = sorted(REPO.glob("*/motor.py"))
    assert len(motores) >= 20, f"Se esperaban >=20 motores, encontrados {len(motores)}"

    violaciones = []
    for path in motores:
        liga = path.parent.name
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                if node.target.id in ("C", "C_cand", "c_val", "alpha"):
                    nombres_en_bucle = {
                         sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)
                    }
                    fugas = nombres_en_bucle.intersection({"X_test", "y_test", "X_te", "y_te"})
                    if fugas:
                        violaciones.append(f"{liga}/motor.py:{node.lineno} usa {sorted(fugas)} al elegir {node.target.id}")

        # Si el motor guarda caché de modelo en disco (.model_cache.pkl), debe incluir hash_modulos_comunes
        if (".model_cache.pkl" in src or "def _cache_key" in src) and "hash_modulos_comunes" not in src:
            violaciones.append(f"{liga}/motor.py no incorpora hash_modulos_comunes() en su clave de caché")

    assert not violaciones, "Violaciones de higiene detectadas:\n  - " + "\n  - ".join(violaciones)
    print(f"✅ [1/5] Guardia AST sin fugas de test en {len(motores)} motores y claves de caché verificadas.")


def test_helpers_comun():
    """Verifica seleccionar_c_lasso y excluir_variables_estimadas."""
    cols = [
        "elo_diff", "form_diff", "squad_value_diff", "squad_value_home_log",
        "avg_age_diff", "pct_foreigners_diff", "avg_attendance_home",
        "stadium_occupation_diff", "stadium_capacity_home",
    ]
    limpias = excluir_variables_estimadas(cols)
    assert limpias == ["elo_diff", "form_diff", "stadium_capacity_home"], f"Columnas inesperadas: {limpias}"

    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(240, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(rng.integers(0, 3, size=240))
    c1, l1 = seleccionar_c_lasso(X, y, grid=(0.02, 0.1))
    c2, l2 = seleccionar_c_lasso(X, y, grid=(0.02, 0.1))
    assert c1 == c2 and np.isclose(l1, l2), "seleccionar_c_lasso debe ser determinista"
    assert len(hash_modulos_comunes()) == 64
    print("✅ [2/5] Helpers compartidos (seleccionar_c_lasso, excluir_variables_estimadas, hash_modulos_comunes) OK.")


def test_causalidad_backtest_nba():
    """Demuestra que alterar partidos futuros (>= t_corte) no cambia el contexto causal (< t_corte)."""
    sys.path.insert(0, str(REPO / "nba"))
    import motor_jugadores as mj

    df_b, df_r = mj.cargar_dataset_jugadores()
    equipos_sub = {"Golden State Warriors", "Denver Nuggets", "Boston Celtics", "Los Angeles Lakers"}
    sub = df_b[df_b["equipo"].isin(equipos_sub) & df_b["rival"].isin(equipos_sub)].copy()
    sub = sub.sort_values(["fecha_dt", "event_id"]).reset_index(drop=True)
    fechas_unicas = sorted(sub["fecha_dt"].unique())
    t_corte = fechas_unicas[int(len(fechas_unicas) * 0.70)]

    ts_orig, rot_orig, aus_orig, _ = mj._precomputar_contexto_causal_equipos(sub)

    # Perturbar drásticamente todos los partidos en fecha_dt >= t_corte
    sub_mut = sub.copy()
    mask_fut = sub_mut["fecha_dt"] >= t_corte
    sub_mut.loc[mask_fut, "pts_num"] = 999.0
    sub_mut.loc[mask_fut, "fga"] = 250.0
    sub_mut.loc[mask_fut, "min_num"] = 1.0  # simular que todos los titulares faltaron en el futuro

    ts_mut, rot_mut, aus_mut, _ = mj._precomputar_contexto_causal_equipos(sub_mut)

    eids_pasado = set(sub[sub["fecha_dt"] < t_corte]["event_id"].unique())
    assert len(eids_pasado) >= 5
    for (eid, eq), val in ts_orig.items():
        if eid in eids_pasado:
            assert np.isclose(val["pace"], ts_mut[(eid, eq)]["pace"]), f"Fuga en pace para {eid} {eq}"
            assert np.isclose(val["drtg"], ts_mut[(eid, eq)]["drtg"]), f"Fuga en drtg para {eid} {eq}"
            assert rot_orig[(eid, eq)] == rot_mut[(eid, eq)], f"Fuga en rotación para {eid} {eq}"
            assert aus_orig[(eid, eq)] == aus_mut[(eid, eq)], f"Fuga en ausentes previos para {eid} {eq}"

    # Verificar que el escáner no contenga etiquetas de ROI ficticio ni Sello de Oro
    dvp = mj.calcular_defensa_vs_posicion(sub)
    _, df_jug = mj.escanear_oportunidades_partido(
        sub, df_r, dvp, "Golden State Warriors", "Denver Nuggets"
    )
    if not df_jug.empty:
        assert "ROI (-110)" not in df_jug.columns, "El escáner no debe presentar ROI sobre cuotas hipotéticas"
        niveles = set(df_jug["Magnitud"].astype(str))
        assert not any("SELLO DE ORO" in x or "+EV" in x for x in niveles), f"Etiquetas no permitidas: {niveles}"
    print("✅ [3/5] Invariancia causal del backtest NBA y terminología honesta del escáner verificadas.")


def test_registro_inmutable():
    """Verifica que el registro rechace pronósticos post-partido y no sobrescriba filas."""
    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / "predicciones_log.csv"
        filas = [
            {
                "generado_utc": "2026-11-09T12:00:00Z",
                "competicion": "eng",
                "partido_id": "test_1",
                "fecha_partido": "2026-11-10T20:00:00Z",  # Futuro -> VÁLIDA
                "local": "Arsenal",
                "visita": "Chelsea",
                "modelo": "stacking",
                "p_local": 0.50,
                "p_empate": 0.25,
                "p_visita": 0.25,
            },
            {
                "generado_utc": "2026-11-10T21:00:00Z",
                "competicion": "eng",
                "partido_id": "test_pasado",
                "fecha_partido": "2026-11-10T20:00:00Z",  # Pasado -> DEBE RECHAZARSE
                "local": "Liverpool",
                "visita": "Everton",
                "modelo": "stacking",
                "p_local": 0.65,
                "p_empate": 0.20,
                "p_visita": 0.15,
            },
        ]
        n1 = registrar_predicciones(filas, log_path=log_file)
        assert n1 == 1, f"Solo debía aceptarse 1 predicción pre-partido, se aceptaron {n1}"

        # Intentar sobrescribir test_1 con otras probabilidades
        filas_dup = [{
            "generado_utc": "2026-11-09T15:00:00Z",
            "competicion": "eng",
            "partido_id": "test_1",
            "fecha_partido": "2026-11-10T20:00:00Z",
            "local": "Arsenal",
            "visita": "Chelsea",
            "modelo": "stacking",
            "p_local": 0.90,
            "p_empate": 0.05,
            "p_visita": 0.05,
        }]
        n2 = registrar_predicciones(filas_dup, log_path=log_file)
        assert n2 == 0, "El registro inmutable jamás debe sobrescribir un partido ya archivado"
        df_saved = cargar_registro(log_file)
        assert len(df_saved) == 1 and np.isclose(float(df_saved.iloc[0]["p_local"]), 0.50)

        # Verificar evaluador pareado vs Solo Elo incluyendo columnas puntos_local/puntos_visita (NBA)
        nba_p = REPO / "nba" / "data" / "partidos.csv"
        if nba_p.exists():
            import pandas as pd
            from scripts.evaluar_archivo import evaluar_registro

            df_nba_p = pd.read_csv(nba_p).dropna(subset=["puntos_local", "puntos_visita"]).tail(1)
            if not df_nba_p.empty:
                r0 = df_nba_p.iloc[0]
                f_part = pd.to_datetime(r0["fecha"], utc=True)
                gen_s = (f_part - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
                f_s = f_part.strftime("%Y-%m-%dT%H:%M:%SZ")
                log_nba = Path(tmp) / "log_nba.csv"
                registrar_predicciones([
                    {
                        "generado_utc": gen_s, "competicion": "nba", "partido_id": str(r0["event_id"]),
                        "fecha_partido": f_s, "local": str(r0["local"]), "visita": str(r0["visita"]),
                        "modelo": "stacking", "p_local": 0.62, "p_empate": 0.0, "p_visita": 0.38,
                    },
                    {
                        "generado_utc": gen_s, "competicion": "nba", "partido_id": str(r0["event_id"]),
                        "fecha_partido": f_s, "local": str(r0["local"]), "visita": str(r0["visita"]),
                        "modelo": "elo", "p_local": 0.55, "p_empate": 0.0, "p_visita": 0.45,
                    },
                ], log_path=log_nba)
                df_eval, _ = evaluar_registro(log_path=log_nba)
                assert not df_eval.empty and set(df_eval["modelo"]) == {"stacking", "elo"}
                row_stk = df_eval[df_eval["modelo"] == "stacking"].iloc[0]
                assert pd.notna(row_stk["delta_elo_logloss"]), "Falta delta_elo_logloss pareado en evaluar_registro"
    print("✅ [4/5] Registro inmutable de predicciones pre-partido (append-only) OK.")


def test_corte_mundial_y_ucl():
    """Verifica el corte estricto pre-Mundial en lab/motor.py y temporada dinámica en ucl/motor.py."""
    src_lab = (REPO / "lab" / "motor.py").read_text()
    assert 'df["fecha"] < "2026-06-11"' in src_lab, "Falta filtro de corte pre-Mundial en lab/motor.py"
    assert 'assert (df["fecha"] < "2026-06-11").all()' in src_lab, "Falta assert pre-Mundial en lab/motor.py"
    src_ucl = (REPO / "ucl" / "motor.py").read_text()
    assert "return 2027" not in src_ucl, "_temporada_actual() sigue fija en 2027 en ucl/motor.py"
    print("✅ [5/5] Corte pre-Mundial en lab/motor.py y temporada dinámica en ucl/motor.py OK.")


def main():
    test_sin_fuga_hiperparametros_en_test()
    test_helpers_comun()
    test_causalidad_backtest_nba()
    test_registro_inmutable()
    test_corte_mundial_y_ucl()
    print("\n🎉 Todas las pruebas de integridad científica y anti-fugas pasaron exitosamente.")


if __name__ == "__main__":
    main()
