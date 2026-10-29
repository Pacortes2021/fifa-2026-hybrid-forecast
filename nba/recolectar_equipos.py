"""
Recolector de franquicias de la NBA desde la API de ESPN (/teams y /standings).
Genera nba/data/equipos.csv con ID estable de ESPN, nombre canónico,
abreviatura oficial, conferencia, división, colores y logos en alta definición.
"""
from pathlib import Path
import requests
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"

DIVISIONS_MAP = {
    # Atlantic
    "Boston Celtics": "Atlantic", "Brooklyn Nets": "Atlantic", "New York Knicks": "Atlantic",
    "Philadelphia 76ers": "Atlantic", "Toronto Raptors": "Atlantic",
    # Central
    "Chicago Bulls": "Central", "Cleveland Cavaliers": "Central", "Detroit Pistons": "Central",
    "Indiana Pacers": "Central", "Milwaukee Bucks": "Central",
    # Southeast
    "Atlanta Hawks": "Southeast", "Charlotte Hornets": "Southeast", "Miami Heat": "Southeast",
    "Orlando Magic": "Southeast", "Washington Wizards": "Southeast",
    # Northwest
    "Denver Nuggets": "Northwest", "Minnesota Timberwolves": "Northwest", "Oklahoma City Thunder": "Northwest",
    "Portland Trail Blazers": "Northwest", "Utah Jazz": "Northwest",
    # Pacific
    "Golden State Warriors": "Pacific", "LA Clippers": "Pacific", "Los Angeles Lakers": "Pacific",
    "Phoenix Suns": "Pacific", "Sacramento Kings": "Pacific",
    # Southwest
    "Dallas Mavericks": "Southwest", "Houston Rockets": "Southwest", "Memphis Grizzlies": "Southwest",
    "New Orleans Pelicans": "Southwest", "San Antonio Spurs": "Southwest"
}

NORM_MAP = {
    "LA Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers"
}


def norm_team(name):
    return NORM_MAP.get(name, name)


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    
    # 1. Obtener equipos
    r = requests.get(TEAMS_URL, timeout=40)
    r.raise_for_status()
    teams = r.json()["sports"][0]["leagues"][0].get("teams", [])
    
    # 2. Obtener conferencias desde Standings
    conf_map = {}
    try:
        sr = requests.get(STANDINGS_URL, timeout=40).json()
        for conf in sr.get("children", []):
            cname = conf.get("name", "")
            for entry in conf.get("standings", {}).get("entries", []):
                tid = str(entry.get("team", {}).get("id"))
                conf_map[tid] = cname
    except Exception as ex:
        print(f"Advertencia al consultar standings: {ex}")

    filas = []
    for t in teams:
        team = t["team"]
        tid = str(team["id"])
        name = team.get("displayName", "")
        norm = norm_team(name)
        logos = team.get("logos") or []
        logo = next((lg["href"] for lg in logos if "dark" in lg.get("rel", [])), None) or (logos[0]["href"] if logos else None)
        
        conf = conf_map.get(tid, "Eastern Conference" if norm in [
            "Boston Celtics", "Brooklyn Nets", "New York Knicks", "Philadelphia 76ers", "Toronto Raptors",
            "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Milwaukee Bucks",
            "Atlanta Hawks", "Charlotte Hornets", "Miami Heat", "Orlando Magic", "Washington Wizards"
        ] else "Western Conference")
        
        div = DIVISIONS_MAP.get(norm, "General")
        
        filas.append({
            "id": tid,
            "name": name,
            "norm_name": norm,
            "short_name": team.get("shortDisplayName", ""),
            "abbreviation": team.get("abbreviation", ""),
            "slug": team.get("slug", ""),
            "conference": conf,
            "division": div,
            "color": team.get("color", "1d428a"),
            "alternate_color": team.get("alternateColor", "c8102e"),
            "logo": logo,
            "is_active": team.get("isActive", True)
        })

    df = pd.DataFrame(filas).sort_values("name").reset_index(drop=True)
    df.to_csv(DATA / "equipos.csv", index=False)
    print(f"Guardado {DATA / 'equipos.csv'} ({len(df)} franquicias NBA)")
    return df


if __name__ == "__main__":
    recolectar()
