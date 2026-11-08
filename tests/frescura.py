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

    if desactualizadas:
        print(f"\n⚠️  Ligas sin actualizar: {', '.join(desactualizadas)}")
        sys.exit(1)
    print("\nTodas las ligas al día")


if __name__ == "__main__":
    main()
