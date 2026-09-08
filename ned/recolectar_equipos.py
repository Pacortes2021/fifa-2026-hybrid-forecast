"""
Recolector de equipos de la Eredivisie desde el endpoint /teams de ESPN (ned.1 y ned.2 para clubes históricos).
Guarda data/equipos.csv con el ID estable de cada club, su nombre canónico,
abreviatura, colores y logo.
"""
import requests
import pandas as pd
from recolectar import DATA, norm_team


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    filas = []

    for slug in ["ned.1", "ned.2"]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                teams = data["sports"][0]["leagues"][0].get("teams", [])
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
        except Exception as ex:
            print(f"Error consultando {slug}: {ex}")

    df = pd.DataFrame(filas).drop_duplicates(subset=["norm_name"]).reset_index(drop=True)
    out_p = DATA / "equipos.csv"
    df.to_csv(out_p, index=False)
    print(f"Guardado {out_p} ({len(df)} equipos)")
    return df


if __name__ == "__main__":
    recolectar()
