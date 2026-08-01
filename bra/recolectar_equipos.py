"""
Recolector de equipos de la Serie A de Brasil (Brasileirão) desde el endpoint /teams de ESPN (bra.1).
Guarda data/equipos.csv con el ID estable de cada club, su nombre canónico
(con el mismo NORM_MAP de recolectar.py), abreviatura, colores y logo.

Uso:  python3 recolectar_equipos.py   -> data/equipos.csv
"""
import requests
import pandas as pd
from recolectar import DATA, TEAMS_URL, norm_team


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    r = requests.get(TEAMS_URL, timeout=40)
    r.raise_for_status()
    data = r.json()
    teams = data["sports"][0]["leagues"][0].get("teams", [])

    filas = []
    for t in teams:
        team = t["team"]
        logos = team.get("logos") or []
        logo = next((lg["href"] for lg in logos if "dark" in lg.get("rel", [])), None) or (logos[0]["href"] if logos else None)
        filas.append({
            "id": str(team["id"]),
            "name": team.get("displayName", ""),
            "norm_name": norm_team(team.get("displayName", "")),
            "short_name": team.get("shortDisplayName", ""),
            "abbreviation": team.get("abbreviation", ""),
            "slug": team.get("slug", ""),
            "color": team.get("color", ""),
            "alternate_color": team.get("alternateColor", ""),
            "logo": logo,
            "is_active": team.get("isActive", False)
        })

    df = pd.DataFrame(filas)
    df.to_csv(DATA / "equipos.csv", index=False)
    print(f"Guardado {DATA / 'equipos.csv'} ({len(df)} equipos)")
    return df


if __name__ == "__main__":
    recolectar()
