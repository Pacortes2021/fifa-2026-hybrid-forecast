"""
Pruebas unitarias e invariantes científicas del motor Monte Carlo dinámico (`comun/monte_carlo.py`):
  1. Determinismo exacto por semilla (`seed`).
  2. Coherencia estricta marcador-resultado en muestreos Poisson condicionales.
  3. Calibración de sensibilidad sin fuga del conjunto de test (`temporada >= test_desde`).
  4. Propagación causal dinámica: un resultado forzado temprano (`fijos`) altera el Elo/forma y
     modifica la esperanza de resultados en partidos posteriores contra terceros rivales.
  5. Conservación de Elo y evolución dinámica en el paso binario FiveThirtyEight (NBA) y Best-of-7.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from comun.monte_carlo import (
    calibrar_sensibilidad_dinamica,
    muestrear_goles_coherentes,
    paso_binario_dinamico_vec,
    simular_fixture_dinamico,
    simular_serie_best_of_n_vec,
)


def _df_sintetico_1x2(corrupcion_test: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    filas = []
    for temp in (2022, 2023, 2024, 2025, 2026):
        for _ in range(80):
            elo_diff = float(rng.normal(0.0, 140.0))
            form_diff = float(rng.normal(0.0, 0.45))
            es_test = temp >= 2025
            if es_test and corrupcion_test:
                # Etiqueta invertida adrede en el bloque test para comprobar aislamiento
                gl, gv = (0, 3) if elo_diff > 0 else (3, 0)
            else:
                z = 0.004 * elo_diff + 0.35 * form_diff
                p_l = 1.0 / (1.0 + np.exp(-z)) * 0.72
                p_e = 0.25
                u = rng.random()
                if u < p_l:
                    gl, gv = 2, 0
                elif u < p_l + p_e:
                    gl, gv = 1, 1
                else:
                    gl, gv = 0, 2
            res = 0 if gl > gv else (1 if gl == gv else 2)
            filas.append({
                "temporada": temp,
                "elo_diff": elo_diff,
                "form_diff": form_diff,
                "goles_local": gl,
                "goles_visita": gv,
                "resultado": res,
            })
    return pd.DataFrame(filas)


def test_determinismo():
    eqs = ["A", "B", "C", "D"]
    fix = pd.DataFrame([
        {"fecha": "2026-04-01", "local": "A", "visita": "B"},
        {"fecha": "2026-04-05", "local": "C", "visita": "D"},
        {"fecha": "2026-04-10", "local": "A", "visita": "C"},
        {"fecha": "2026-04-12", "local": "B", "visita": "D"},
    ])
    M = {"elos": {e: 1500.0 for e in eqs}, "df_dataset": _df_sintetico_1x2()}
    preds = {(r.local, r.visita): (np.array([0.45, 0.28, 0.27]), 1.4, 1.05) for r in fix.itertuples(index=False)}
    z = np.zeros(len(eqs), dtype=np.float32)

    r1 = simular_fixture_dinamico(
        M, fix, eqs, z, z, z, preds, n_sims=600, liga="eng", seed=42, track_terminal_states=True
    )
    r2 = simular_fixture_dinamico(
        M, fix, eqs, z, z, z, preds, n_sims=600, liga="eng", seed=42, track_terminal_states=True
    )
    for key in ("PTS", "GF", "GC", "order", "ELO", "DFORM"):
        assert np.array_equal(r1[key], r2[key]), f"Falta determinismo en {key} con seed=42"


def test_coherencia_goles_resultado():
    rng = np.random.default_rng(7)
    n = 10000
    u = rng.random(n)
    w_l = u < 0.45
    w_e = (u >= 0.45) & (u < 0.72)
    w_v = u >= 0.72
    la = rng.uniform(0.35, 3.2, size=n)
    lb = rng.uniform(0.35, 3.2, size=n)
    gl, gv = muestrear_goles_coherentes(rng, la, lb, w_l, w_e, w_v)
    assert not np.any(w_l & (gl <= gv)), "Incoherencia: triunfo local con gl <= gv"
    assert not np.any(w_e & (gl != gv)), "Incoherencia: empate con gl != gv"
    assert not np.any(w_v & (gl >= gv)), "Incoherencia: triunfo visita con gl >= gv"
    assert (gl >= 0).all() and (gv >= 0).all(), "Goles negativos detectados"


def test_calibracion_sin_fuga():
    M_clean = {"df_dataset": _df_sintetico_1x2(False), "g_d": 0.0014}
    M_corr = {"df_dataset": _df_sintetico_1x2(True), "g_d": 0.0014}
    s_clean = calibrar_sensibilidad_dinamica(M_clean, liga="eng")
    s_corr = calibrar_sensibilidad_dinamica(M_corr, liga="eng")

    assert s_clean["b_elo"][0] > s_clean["b_elo"][2], "Sensibilidad Elo debe favorecer victoria local"
    assert np.allclose(s_clean["b_elo"], s_corr["b_elo"]), "Fuga detectada: el bloque test afectó b_elo"
    assert np.allclose(s_clean["b_form"], s_corr["b_form"]), "Fuga detectada: el bloque test afectó b_form"
    assert s_clean["b_gol"] == s_corr["b_gol"], "Fuga detectada en b_gol"


def test_propagacion_dinamica_causal():
    eqs = ["A", "B", "C", "D"]
    # Jornada 1: A vs B; Jornada 2: A vs C y B vs D (rivales que NO jugaron en J1)
    fix = pd.DataFrame([
        {"fecha": "2026-05-01", "local": "A", "visita": "B"},
        {"fecha": "2026-05-06", "local": "A", "visita": "C"},
        {"fecha": "2026-05-08", "local": "B", "visita": "D"},
    ])
    M = {"elos": {e: 1500.0 for e in eqs}, "df_dataset": _df_sintetico_1x2()}
    preds = {(r.local, r.visita): (np.array([0.40, 0.28, 0.32]), 1.35, 1.15) for r in fix.itertuples(index=False)}
    z = np.zeros(len(eqs), dtype=np.float32)

    out_a_win = simular_fixture_dinamico(
        M, fix, eqs, z, z, z, preds, fijos={("A", "B"): (5, 0)},
        n_sims=6000, k_liga=45.0, liga="eng", seed=99,
    )
    out_b_win = simular_fixture_dinamico(
        M, fix, eqs, z, z, z, preds, fijos={("A", "B"): (0, 5)},
        n_sims=6000, k_liga=45.0, liga="eng", seed=99,
    )

    pts_c_when_a_hot = float(out_a_win["PTS"][2].mean())
    pts_c_when_a_cold = float(out_b_win["PTS"][2].mean())
    pts_d_when_b_cold = float(out_a_win["PTS"][3].mean())
    pts_d_when_b_hot = float(out_b_win["PTS"][3].mean())

    assert pts_c_when_a_hot < pts_c_when_a_cold - 0.15, (
        f"Elo/forma de A no se propagó a A vs C: {pts_c_when_a_hot:.3f} vs {pts_c_when_a_cold:.3f}"
    )
    assert pts_d_when_b_hot < pts_d_when_b_cold - 0.15, (
        f"Elo/forma de B no se propagó a B vs D: {pts_d_when_b_hot:.3f} vs {pts_d_when_b_cold:.3f}"
    )


def test_nba_dinamico_y_best_of_7():
    rng = np.random.default_rng(11)
    n_eq, n_sims = 4, 2000
    elo0 = np.array([1620.0, 1540.0, 1480.0, 1400.0], dtype=np.float32)
    ELO = np.tile(elo0[:, None], (1, n_sims))
    DFORM = np.zeros((n_eq, n_sims), dtype=np.float32)
    P0 = np.full((n_eq, n_eq), 0.50, dtype=np.float32)
    for i in range(n_eq):
        for j in range(n_eq):
            if i != j:
                dr = (elo0[i] + 100.0) - elo0[j]
                P0[i, j] = np.float32(1.0 / (1.0 + 10.0 ** (-dr / 400.0)))

    sims = np.arange(n_sims)
    idx0 = np.zeros(n_sims, dtype=np.int32)
    idx3 = np.full(n_sims, 3, dtype=np.int32)
    elo_sum_antes = ELO.sum(axis=0).copy()
    _ = paso_binario_dinamico_vec(
        rng, ELO, DFORM, elo0, idx0, idx3, sims, P0,
        b_elo_bin=0.00575, b_form_bin=0.25, home_adv=100.0, sigma_margin=12.0,
    )
    assert np.allclose(ELO.sum(axis=0), elo_sum_antes, atol=1e-2), "Elo total debe conservarse suma cero"
    assert not np.allclose(ELO[0], elo0[0]), "ELO debe evolucionar tras partido NBA"

    ganador = simular_serie_best_of_n_vec(
        rng, ELO, DFORM, elo0, idx0, idx3, P0, b_elo_bin=0.00575,
    )
    p_fav = float(np.mean(ganador == 0))
    assert 0.65 < p_fav < 0.99, f"Probabilidad serie Best-of-7 anómala para 1620 vs 1400: {p_fav}"


def run_all():
    test_determinismo()
    test_coherencia_goles_resultado()
    test_calibracion_sin_fuga()
    test_propagacion_dinamica_causal()
    test_nba_dinamico_y_best_of_7()
    print("✅ Pruebas unitarias e invariantes de Monte Carlo dinámico OK.")


if __name__ == "__main__":
    run_all()
