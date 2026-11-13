"""
Registro inmutable (append-only) de predicciones pre-partido (`outputs/predicciones_log.csv`).

Propósito científico:
  - Garantizar que toda evaluación prospectiva utilice únicamente pronósticos emitidos y fechados
    ANTES del inicio de cada encuentro (`generado_utc < fecha_partido`).
  - Cada fila registra el hash SHA-256 de los datos y el commit de código vigente al momento de
    emitir la predicción. Si un `(competicion, partido_id, modelo)` ya existe en el archivo,
    jamás se sobrescribe.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOG_PATH = REPO / "outputs" / "predicciones_log.csv"

COLUMNAS_LOG = [
    "generado_utc",
    "competicion",
    "partido_id",
    "fecha_partido",
    "local",
    "visita",
    "modelo",
    "version_codigo",
    "hash_datos",
    "p_local",
    "p_empate",
    "p_visita",
]


def obtener_git_sha():
    """Devuelve el SHA corto del commit actual del repositorio."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def calcular_hash_datos(data_dir):
    """Calcula un hash SHA-256 corto (12 hex) sobre los CSV de entrada de la competición."""
    h = hashlib.sha256()
    d = Path(data_dir)
    if d.exists():
        for f in sorted(d.glob("*.csv")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def cargar_registro(log_path=LOG_PATH):
    """Carga el registro histórico de predicciones o devuelve un DataFrame vacío con el esquema oficial."""
    p = Path(log_path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNAS_LOG)
    df = pd.read_csv(p, dtype={"partido_id": str, "competicion": str, "modelo": str})
    for c in COLUMNAS_LOG:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNAS_LOG]


def registrar_predicciones(nuevas_filas, log_path=LOG_PATH):
    """
    Agrega de forma inmutable (`append-only`) nuevas predicciones pre-partido al registro:
      - Rechaza cualquier fila cuya `fecha_partido` sea anterior o igual a `generado_utc`.
      - Ignora cualquier `(competicion, partido_id, modelo)` que ya figure en el registro para
        impedir reescrituras retroactivas.
    Devuelve el número de filas efectivamente añadidas.
    """
    if not nuevas_filas:
        return 0

    df_actual = cargar_registro(log_path)
    existentes = set(
        zip(
            df_actual["competicion"].astype(str),
            df_actual["partido_id"].astype(str),
            df_actual["modelo"].astype(str),
        )
    )

    filas_validas = []
    for r in nuevas_filas:
        comp = str(r["competicion"])
        pid = str(r["partido_id"])
        mod = str(r["modelo"])
        if (comp, pid, mod) in existentes:
            continue

        gen_dt = pd.to_datetime(r["generado_utc"], utc=True)
        f_part = pd.to_datetime(r["fecha_partido"], utc=True)
        # Regla estricta: solo archivar pronósticos emitidos ANTES de la hora programada del partido
        if pd.notna(f_part) and gen_dt >= f_part:
            continue

        filas_validas.append({
            "generado_utc": gen_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "competicion": comp,
            "partido_id": pid,
            "fecha_partido": f_part.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(f_part) else str(r["fecha_partido"]),
            "local": str(r["local"]),
            "visita": str(r["visita"]),
            "modelo": mod,
            "version_codigo": str(r.get("version_codigo") or obtener_git_sha()),
            "hash_datos": str(r.get("hash_datos") or ""),
            "p_local": round(float(r["p_local"]), 5),
            "p_empate": round(float(r.get("p_empate", 0.0)), 5),
            "p_visita": round(float(r["p_visita"]), 5),
        })
        existentes.add((comp, pid, mod))

    if not filas_validas:
        return 0

    df_nuevas = pd.DataFrame(filas_validas, columns=COLUMNAS_LOG)
    if df_actual.empty:
        df_out = df_nuevas
    else:
        df_out = pd.concat([df_actual, df_nuevas], ignore_index=True)
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(p, index=False)
    return len(filas_validas)
