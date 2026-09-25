"""
Recolector de equipos y grupos oficiales de la UEFA Nations League desde ESPN.
Guarda data/equipos.csv con el ID de ESPN, nombre normalizado, abreviatura,
liga (A, B, C, D), grupo asignado y escudo oficial en HD.
"""
from pathlib import Path
import requests
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

NORM_MAP = {
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

FLAGS = {
    "Albania": "🇦🇱", "Andorra": "🇦🇩", "Armenia": "🇦🇲", "Austria": "🇦🇹", "Azerbaijan": "🇦🇿",
    "Belarus": "🇧🇾", "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Bulgaria": "🇧🇬",
    "Croatia": "🇭🇷", "Cyprus": "🇨🇾", "Czech Republic": "🇨🇿", "Denmark": "🇩🇰", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Estonia": "🇪🇪", "Faroe Islands": "🇫🇴", "Finland": "🇫🇮", "France": "🇫🇷", "Georgia": "🇬🇪",
    "Germany": "🇩🇪", "Gibraltar": "🇬🇮", "Greece": "🇬🇷", "Hungary": "🇭🇺", "Iceland": "🇮🇸",
    "Israel": "🇮🇱", "Italy": "🇮🇹", "Kazakhstan": "🇰🇿", "Kosovo": "🇽🇰", "Latvia": "🇱🇻",
    "Liechtenstein": "🇱🇮", "Lithuania": "🇱🇹", "Luxembourg": "🇱🇺", "Malta": "🇲🇹", "Moldova": "🇲🇩",
    "Montenegro": "🇲🇪", "Netherlands": "🇳🇱", "North Macedonia": "🇲🇰", "Northern Ireland": "🏴󠁧󠁢󠁮󠁩󠁲󠁿",
    "Norway": "🇳🇴", "Poland": "🇵🇱", "Portugal": "🇵🇹", "Republic of Ireland": "🇮🇪", "Romania": "🇷🇴",
    "San Marino": "🇸🇲", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Serbia": "🇷🇸", "Slovakia": "🇸🇰", "Slovenia": "🇸🇮",
    "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Turkey": "🇹🇷", "Ukraine": "🇺🇦",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿"
}


def norm_team(name):
    return NORM_MAP.get(name, name)


def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    equipos_map = {}

    url_standings = "https://site.api.espn.com/apis/v2/sports/soccer/uefa.nations/standings"
    try:
        r = requests.get(url_standings, timeout=20)
        if r.status_code == 200:
            data = r.json()
            children = data.get("children", [])
            for c in children:
                group_name = c.get("name", "")  # e.g., 'Group A1'
                league = group_name.split()[1][0] if len(group_name.split()) > 1 else "A"
                entries = c.get("standings", {}).get("entries", [])
                for e in entries:
                    team = e.get("team", {})
                    tid = str(team.get("id"))
                    raw_name = team.get("displayName", "")
                    norm_nm = norm_team(raw_name)
                    abbr = team.get("abbreviation", norm_nm[:3].upper())
                    logos = team.get("logos", [])
                    logo = logos[0].get("href") if logos else f"https://a.espncdn.com/i/teamlogos/countries/500/{abbr.lower()}.png"
                    
                    equipos_map[norm_nm] = {
                        "id": tid,
                        "name": raw_name,
                        "norm_name": norm_nm,
                        "flag": FLAGS.get(norm_nm, "⚽"),
                        "abbreviation": abbr,
                        "group": group_name,
                        "league": league,
                        "color": team.get("color", "001438"),
                        "alternate_color": team.get("alternateColor", "00E5FF"),
                        "logo": logo,
                        "is_active": True
                    }
    except Exception as ex:
        print(f"Error consultando /standings: {ex}")

    df = pd.DataFrame(list(equipos_map.values())).sort_values(["league", "group", "norm_name"]).reset_index(drop=True)
    out_p = DATA / "equipos.csv"
    df.to_csv(out_p, index=False)
    print(f"Guardado {out_p} ({len(df)} selecciones de UEFA Nations League)")
    return df


if __name__ == "__main__":
    recolectar()
