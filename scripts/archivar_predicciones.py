"""
Archivador prospectivo de predicciones pre-partido (`outputs/predicciones_log.csv`).

Recorre las competiciones del repositorio, identifica los partidos programados en los próximos
`--dias` días cuya hora de inicio sea estrictamente futura (`fecha > ahora_utc`) y guarda de
forma inmutable las probabilidades pre-partido del modelo principal (`stacking`) y del
baseline de referencia (`elo`) en `outputs/predicciones_log.csv`.

Uso:
    python3 scripts/archivar_predicciones.py --dias 7
    python3 scripts/archivar_predicciones.py --dias 7 eng esp ucl nba
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from comun.motor_comun import entrenar_baseline_elo, predecir_elo
from comun.registro_predicciones import (
    LOG_PATH,
    calcular_hash_datos,
    obtener_git_sha,
    registrar_predicciones,
)

LIGAS = [
    "mex", "arg", "bra", "chile", "esp", "eng", "bund", "ita", "fra", "ned", "por",
    "bel", "tur", "sco", "aut", "den", "gre", "nor", "swe", "ucl", "unl", "nba"
]


def _importar_motor(liga):
    for m in [k for k in list(sys.modules) if k == "motor" or k.startswith("motor.")]:
        del sys.modules[m]
    sys.path = [str(REPO / liga), str(REPO)] + [p for p in sys.path if p not in [str(REPO / l) for l in LIGAS]]
    return __import__("motor")


def _crear_fila_log(gen_str, liga, eid, f_utc, local, visita, modelo, git_sha, h_datos, p_vec):
    return {
        "generado_utc": gen_str,
        "competicion": liga,
        "partido_id": eid,
        "fecha_partido": f_utc,
        "local": local,
        "visita": visita,
        "modelo": modelo,
        "version_codigo": git_sha,
        "hash_datos": h_datos,
        "p_local": float(p_vec[0]),
        "p_empate": float(p_vec[1]),
        "p_visita": float(p_vec[2]),
    }


def archivar_competicion(liga, dias=7, log_path=LOG_PATH):
    data_dir = REPO / liga / "data"
    fix_path = data_dir / "fixture.csv"
    if not fix_path.exists():
        return 0

    df_fix = pd.read_csv(fix_path)
    if df_fix.empty or "fecha" not in df_fix.columns:
        return 0

    ahora = pd.Timestamp.now(tz="UTC")
    horizonte = ahora + pd.Timedelta(days=dias)
    df_fix["fecha_utc"] = pd.to_datetime(df_fix["fecha"], utc=True, errors="coerce")
    proximos = df_fix[(df_fix["fecha_utc"] > ahora) & (df_fix["fecha_utc"] <= horizonte)].sort_values("fecha_utc")
    if proximos.empty:
        return 0

    mo = _importar_motor(liga)
    M = mo.cargar()
    elo_m = None if liga == "nba" else entrenar_baseline_elo(M, liga=liga)
    git_sha = obtener_git_sha()
    h_datos = calcular_hash_datos(data_dir)
    gen_str = ahora.strftime("%Y-%m-%dT%H:%M:%SZ")

    filas = []
    for r in proximos.itertuples(index=False):
        local = str(r.local)
        visita = str(r.visita)
        f_utc = r.fecha_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        eid = str(getattr(r, "event_id", "") or f"{liga}_{r.fecha_utc.strftime('%Y%m%d')}_{local}_{visita}")
        temp = int(r.temporada) if hasattr(r, "temporada") and pd.notna(r.temporada) else None
        fecha_naive = r.fecha_utc.tz_localize(None) if getattr(r.fecha_utc, "tzinfo", None) is not None else r.fecha_utc

        if liga == "nba":
            for mod in ("stacking", "elo"):
                pred = mo.predecir_match(M, local, visita, fecha=fecha_naive, temporada=temp, modelo=mod)
                p_vec = [float(pred["p_local"]), 0.0, float(pred["p_visita"])]
                filas.append(_crear_fila_log(gen_str, liga, eid, f_utc, local, visita, mod, git_sha, h_datos, p_vec))
        else:
            res = mo.predecir_match(M, local, visita, modelo="stacking")
            p_stk = res[0] if isinstance(res, tuple) else res
            filas.append(_crear_fila_log(gen_str, liga, eid, f_utc, local, visita, "stacking", git_sha, h_datos, p_stk))
            p_elo = predecir_elo(M, elo_m, local, visita, temporada=temp, fecha=fecha_naive)
            filas.append(_crear_fila_log(gen_str, liga, eid, f_utc, local, visita, "elo", git_sha, h_datos, p_elo))

    return registrar_predicciones(filas, log_path=log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ligas", nargs="*")
    ap.add_argument("--dias", type=int, default=7)
    args = ap.parse_args()

    ligas = args.ligas or LIGAS
    total = 0
    for liga in ligas:
        try:
            n = archivar_competicion(liga, dias=args.dias)
            if n > 0:
                print(f"  [{liga}] +{n} predicciones pre-partido archivadas")
            total += n
        except Exception as ex:
            print(f"  [{liga}] Aviso al archivar: {ex}")
    print(f"Total nuevas predicciones archivadas en {LOG_PATH}: {total}")


if __name__ == "__main__":
    main()
