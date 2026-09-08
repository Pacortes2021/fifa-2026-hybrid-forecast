"""
Recolector de equipos de la Scottish Premiership (Escocia) desde ESPN.
Guarda data/equipos.csv con el ID estable de cada club, nombre normalizado,
abreviatura, colores y logos oficiales en HD.
"""
from pathlib import Path
import requests
import pandas as pd
from recolectar import DATA, norm_team

def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    equipos_map = {}

    # 1. Endpoint /teams
    url_teams = "https://site.api.espn.com/apis/site/v2/sports/soccer/sco.1/teams"
    try:
        r = requests.get(url_teams, timeout=30)
        if r.status_code == 200:
            data = r.json()
            teams = data["sports"][0]["leagues"][0].get("teams", [])
            for t in teams:
                team = t["team"]
                logos = team.get("logos") or []
                logo = next((lg["href"] for lg in logos if "dark" in lg.get("rel", [])), None) or (logos[0]["href"] if logos else None)
                norm_nm = norm_team(team.get("displayName", ""))
                equipos_map[norm_nm] = {
                    "id": str(team["id"]),
                    "name": team.get("displayName", ""),
                    "norm_name": norm_nm,
                    "short_name": team.get("shortDisplayName", ""),
                    "abbreviation": team.get("abbreviation", ""),
                    "slug": team.get("slug", ""),
                    "color": team.get("color", "002B7F"),
                    "alternate_color": team.get("alternateColor", "FFFFFF"),
                    "logo": logo or f"https://a.espncdn.com/i/teamlogos/soccer/500-dark/{team['id']}.png",
                    "is_active": team.get("isActive", True)
                }
    except Exception as ex:
        print(f"Error consultando /teams sco.1: {ex}")

    # 2. Escanear partidos para capturar equipos descendidos/históricos
    for anio in range(2021, 2027):
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/sco.1/scoreboard?dates={anio}0101-{anio}1231&limit=400"
        try:
            r = requests.get(url, timeout=30).json()
            for e in r.get("events", []):
                for c in e.get("competitions", [{}])[0].get("competitors", []):
                    tm = c.get("team", {})
                    if not tm: continue
                    norm_nm = norm_team(tm.get("displayName", ""))
                    if norm_nm not in equipos_map:
                        tid = str(tm.get("id"))
                        logo = tm.get("logo") or f"https://a.espncdn.com/i/teamlogos/soccer/500-dark/{tid}.png"
                        equipos_map[norm_nm] = {
                            "id": tid,
                            "name": tm.get("displayName", norm_nm),
                            "norm_name": norm_nm,
                            "short_name": tm.get("shortDisplayName", norm_nm[:10]),
                            "abbreviation": tm.get("abbreviation", norm_nm[:3].upper()),
                            "slug": tm.get("slug", ""),
                            "color": tm.get("color", "002B7F"),
                            "alternate_color": tm.get("alternateColor", "FFFFFF"),
                            "logo": logo,
                            "is_active": True
                        }
        except Exception:
            pass

    df = pd.DataFrame(list(equipos_map.values())).sort_values("norm_name").reset_index(drop=True)
    out_p = DATA / "equipos.csv"
    df.to_csv(out_p, index=False)
    print(f"Guardado {out_p} ({len(df)} equipos escoceses)")
    return df

if __name__ == "__main__":
    recolectar()
