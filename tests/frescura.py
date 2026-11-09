"""
Chequeo de frescura de los datos: si un fixture tiene partidos "por jugar" con fecha de hace
varios días, el recolector dejó de actualizar esa liga (así pasó en sep-2026, cuando ESPN
empezó a rechazar las consultas por rango y 13 ligas quedaron congeladas sin que nada fallara).

Uso:
    python3 tests/frescura.py            # falla (exit 1) si alguna liga está desactualizada
    python3 tests/frescura.py --dias 5   # tolerancia en días (por defecto 3)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=3, help="días de tolerancia para un partido pendiente")
    ap.add_argument("--max-vencidos", type=int, default=2,
                    help="partidos vencidos tolerados por liga (suspensiones sueltas)")
    args = ap.parse_args()

    limite = pd.Timestamp.now() - pd.Timedelta(days=args.dias)
    desactualizadas = []
    for fix_path in sorted(REPO.glob("*/data/fixture.csv")):
        liga = fix_path.parent.parent.name
        fix = pd.read_csv(fix_path)
        if fix.empty or "fecha" not in fix.columns:
            continue
        fechas = pd.to_datetime(fix["fecha"], errors="coerce", format="mixed")
        # los aplazados/suspendidos conservan su fecha original hasta que ESPN los reprograma
        aplazado = fix["aplazado"].fillna(False).astype(bool) if "aplazado" in fix.columns else pd.Series(False, index=fix.index)
        vencidos = int(((fechas < limite) & ~aplazado).sum())
        partidos = fix_path.parent / "partidos.csv"
        ultimo = pd.to_datetime(pd.read_csv(partidos)["fecha"], format="mixed").max() if partidos.exists() else None
        estado = "OK" if vencidos <= args.max_vencidos else "DESACTUALIZADA"
        print(f"{estado:15} {liga:6} último partido: {str(ultimo)[:10]} | pendientes con fecha pasada: {vencidos}")
        if vencidos > args.max_vencidos:
            desactualizadas.append(liga)

    # Verificar cobertura de box scores individuales de jugadores NBA para la temporada actual (>= 2026)
    nba_partidos = REPO / "nba" / "data" / "partidos.csv"
    nba_jugadores = REPO / "nba" / "data" / "jugadores_boxscore.csv"
    if nba_partidos.exists() and nba_jugadores.exists():
        df_np = pd.read_csv(nba_partidos)
        df_nj = pd.read_csv(nba_jugadores, usecols=["event_id"])
        df_np["fecha_dt"] = pd.to_datetime(df_np["fecha"], utc=True, errors="coerce")
        corte_nba = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
        recientes = df_np[(df_np["temporada"] >= 2026) & (df_np["fecha_dt"] < corte_nba)]
        eids_part = set(recientes["event_id"].dropna().astype(str))
        eids_jug = set(df_nj["event_id"].dropna().astype(str))
        faltantes_jug = len(eids_part - eids_jug)
        estado_j = "OK" if faltantes_jug <= 5 else "DESACTUALIZADA"
        print(f"{estado_j:15} nba_jug cobertura boxscore jugadores (>=2026): {len(eids_part) - faltantes_jug}/{len(eids_part)} | faltantes (>48h): {faltantes_jug}")
        if faltantes_jug > 5:
            desactualizadas.append("nba_jugadores")

    if desactualizadas:
        print(f"\n⚠️  Ligas sin actualizar: {', '.join(desactualizadas)}")
        sys.exit(1)
    print("\nTodas las ligas al día")


if __name__ == "__main__":
    main()
