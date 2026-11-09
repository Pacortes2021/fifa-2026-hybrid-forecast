"""
Evaluador universal de motores: carga cada competición y vuelca sus métricas de test
(log-loss, accuracy, Brier, baseline) a un CSV comparable entre versiones del código.

Uso:
    python3 scripts/evaluar_ligas.py --salida outputs/evaluacion_antes.csv
    python3 scripts/evaluar_ligas.py --salida outputs/evaluacion_despues.csv fra ucl
    python3 scripts/evaluar_ligas.py --comparar outputs/evaluacion_antes.csv outputs/evaluacion_despues.csv

Las métricas son las que cada motor calcula sobre su bloque de TEST temporal. Sirven para
comparar versiones del código, no como prueba prospectiva: para eso está el registro
inmutable de predicciones (comun/registro_predicciones.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LIGAS = ["mex", "arg", "bra", "chile", "esp", "eng", "bund", "ita", "fra", "ned", "por",
         "bel", "tur", "sco", "aut", "den", "gre", "nor", "swe", "ucl", "unl", "nba"]
CLAVES = ("logloss", "accuracy", "acierto", "brier", "logloss_base", "n")


def _importar_motor(liga):
    for m in [k for k in list(sys.modules) if k == "motor" or k.startswith("motor.")]:
        del sys.modules[m]
    sys.path = [str(REPO / liga), str(REPO)] + [p for p in sys.path if p not in [str(REPO / l) for l in LIGAS]]
    return __import__("motor")


def _aplanar_metricas(liga, met):
    """Normaliza los distintos formatos de M['metricas'] a filas (liga, modelo, métricas)."""
    filas = []
    if not isinstance(met, dict):
        return filas
    if any(k in met for k in ("logloss", "log_loss")):
        met = {"modelo": met}
    base_global = met.get("logloss_base") if isinstance(met.get("logloss_base"), (int, float)) else None
    for modelo, m in met.items():
        if not isinstance(m, dict):
            continue
        ll = m.get("logloss", m.get("log_loss"))
        if ll is None:
            continue
        fila = {"liga": liga, "modelo": modelo, "logloss": float(ll)}
        for k in CLAVES[1:]:
            v = m.get(k)
            if isinstance(v, (int, float, np.floating, np.integer)):
                fila[k] = float(v)
        if "logloss_base" not in fila and base_global is not None:
            fila["logloss_base"] = float(base_global)
        filas.append(fila)
    return filas


def evaluar(ligas):
    filas = []
    for liga in ligas:
        print(f"== {liga} ==", flush=True)
        try:
            mo = _importar_motor(liga)
            M = mo.cargar()
            f = _aplanar_metricas(liga, M.get("metricas", {}))
            if not f:
                print("  (sin métricas reconocibles)")
            filas.extend(f)
            for r in f:
                print(f"  {r['modelo']:<10} logloss={r['logloss']:.4f}")
        except Exception as ex:
            print(f"  FALLO: {type(ex).__name__}: {ex}")
            filas.append({"liga": liga, "modelo": "ERROR", "logloss": np.nan})
    return pd.DataFrame(filas)


def comparar(antes, despues):
    a = pd.read_csv(antes)
    d = pd.read_csv(despues)
    m = a.merge(d, on=["liga", "modelo"], suffixes=("_antes", "_despues"))
    m["delta_logloss"] = m["logloss_despues"] - m["logloss_antes"]
    cols = ["liga", "modelo", "logloss_antes", "logloss_despues", "delta_logloss"]
    print(m[cols].round(4).to_string(index=False))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ligas", nargs="*")
    ap.add_argument("--salida", default=str(REPO / "outputs" / "evaluacion_ligas.csv"))
    ap.add_argument("--comparar", nargs=2, metavar=("ANTES", "DESPUES"))
    args = ap.parse_args()
    if args.comparar:
        comparar(*args.comparar)
        return
    df = evaluar(args.ligas or LIGAS)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False)
    print(f"\nGuardado en {args.salida} ({len(df)} filas)")


if __name__ == "__main__":
    main()
