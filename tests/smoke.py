"""
Smoke test de humo: valida que los 5 motores de liga cargan, predicen y simulan.

Uso:
    python3 tests/smoke.py            # todas las ligas
    python3 tests/smoke.py mex esp    # solo las indicadas
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

LIGAS = {
    "mex":   dict(sim="monte_carlo",   n_sims=5,  modelo="stacking"),
    "arg":   dict(sim="simular_campeonato", n_sims=5, modelo="stacking"),
    "bra":   dict(sim="monte_carlo",   n_sims=5,  modelo="stacking"),
    "chile": dict(sim="simular_campeonato", n_sims=5, modelo="stacking"),
    "esp":   dict(sim="simular_campeonato", n_sims=5, modelo_tipo="stacking"),
}


def check_league(liga, cfg):
    # Purga el módulo "motor" del caché de Python: cada liga tiene su propio
    # motor.py y Python los cachea por nombre en sys.modules (ver limpiar_cache_importacion en app.py)
    for m in [k for k in list(sys.modules) if k == "motor" or k.startswith("motor.")]:
        del sys.modules[m]
    sys.path = [str(REPO / liga)] + [p for p in sys.path if p not in [str(REPO / l) for l in LIGAS]]
    motor = __import__("motor")

    M = motor.cargar()
    if not isinstance(M, dict) or "tracker" not in M:
        raise AssertionError("cargar() no devolvió el dict esperado")

    fix_path = REPO / liga / "data" / "fixture.csv"
    fix = pd.read_csv(fix_path)
    if len(fix) == 0:
        raise AssertionError("fixture vacío")
    local, visita = fix.iloc[0]["local"], fix.iloc[0]["visita"]

    if "modelo_tipo" in cfg:
        p = motor.predecir_match(M, local, visita, modelo_tipo=cfg["modelo_tipo"])
    else:
        p, la, lb = motor.predecir_match(M, local, visita, modelo=cfg.get("modelo", "rf"))
        assert np.isfinite(la) and np.isfinite(lb) and la > 0 and lb > 0, f"lambdas inválidas: {la}, {lb}"
    p = np.asarray(p, dtype=float)
    assert p.shape == (3,), f"predecir_match devolvió {p.shape}"
    assert np.isfinite(p).all() and 0.99 < p.sum() < 1.01, f"probabilidades inválidas: {p}"

    fn = getattr(motor, cfg["sim"])
    sim_kwargs = {"n_sims": cfg["n_sims"]}
    if "modelo_tipo" in cfg:
        sim_kwargs["modelo_tipo"] = cfg["modelo_tipo"]
    else:
        sim_kwargs["modelo"] = cfg.get("modelo", "rf")
    df = fn(M, **sim_kwargs)
    assert isinstance(df, pd.DataFrame) and len(df) > 0, "simulación vacía"
    num_cols = df.select_dtypes(include=[np.number]).columns
    assert num_cols.size > 0, "sin columnas numéricas en la simulación"
    assert df[num_cols].notna().all().all(), "NaN en la simulación"
    assert np.isfinite(df[num_cols].to_numpy()).all(), "valores no finitos en la simulación"

    return M, df


def main():
    ligas = sys.argv[1:] or list(LIGAS)
    fallos = 0
    for liga in ligas:
        cfg = LIGAS[liga]
        print(f"== {liga} ==")
        try:
            M, df = check_league(liga, cfg)
            top = df.iloc[0].iloc[0] if df.shape[1] >= 1 else "?"
            print(f"  OK: {len(df)} filas | top: {top}")
        except Exception as ex:
            fallos += 1
            print(f"  FALLO: {type(ex).__name__}: {ex}")
    if fallos:
        print(f"\n{fallos} liga(s) con fallos")
        sys.exit(1)
    print("\nTodo OK")


if __name__ == "__main__":
    main()
