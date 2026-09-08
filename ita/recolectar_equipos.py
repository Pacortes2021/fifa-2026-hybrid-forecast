"""
Recolector de equipos de la Serie A desde el endpoint /teams de ESPN.
Guarda data/equipos.csv con el ID estable de cada club, su nombre canónico,
abreviatura, colores y logo.

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

    # 2. Equipos adicionales de Serie A que pasaron recientemente por la liga
    extra_ids = [119, 2574, 4050, 3956, 2734, 4059, 3240, 4056, 3173]
    for tid in extra_ids:
        try:
            r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/teams/{tid}", timeout=15)
            if r.status_code != 200:
                r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/ita.2/teams/{tid}", timeout=15)
            if r.status_code == 200:
                team = r.json().get("team", {})
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
        except Exception:
            pass

    df = pd.DataFrame(filas).drop_duplicates(subset=["norm_name"]).reset_index(drop=True)
    out_p = DATA / "equipos.csv"
    df.to_csv(out_p, index=False)
    print(f"Guardado {out_p} ({len(df)} equipos)")
    return df


if __name__ == "__main__":
    recolectar()
