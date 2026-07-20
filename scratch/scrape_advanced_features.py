import os
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEAGUES = {
    "esp": {
        "tm_id": "ES1",
        "espn_partidos": ROOT / "esp" / "data" / "partidos.csv",
        "out_csv": ROOT / "esp" / "data" / "advanced_features_historical.csv",
        "manual": {
            "FC Barcelona": "Barcelona",
            "Athletic Bilbao": "Athletic Club",
            "Atlético de Madrid": "Atlético Madrid",
            "Celta de Vigo": "Celta Vigo",
            "UD Las Palmas": "Las Palmas",
            "Deportivo Alavés": "Alavés",
            "RCD Mallorca": "Mallorca",
            "Real Valladolid CF": "Real Valladolid",
            "CD Leganés": "Leganés",
            "RCD Espanyol Barcelona": "Espanyol",
            "RCD Espanyol": "Espanyol",
            "CA Osasuna": "Osasuna",
            "Getafe CF": "Getafe",
            "Sevilla FC": "Sevilla",
            "Villarreal CF": "Villarreal",
            "Real Betis Balompié": "Real Betis",
            "Valencia CF": "Valencia",
            "Girona FC": "Girona",
            "Levante UD": "Levante",
            "UD Almería": "Almería",
            "Cádiz CF": "Cádiz",
            "Granada CF": "Granada",
            "Elche CF": "Elche",
            "SD Eibar": "Eibar",
            "SD Huesca": "Huesca"
        }
    },
    "mex": {
        "tm_id": "MEX1",
        "espn_partidos": ROOT / "mex" / "data" / "partidos.csv",
        "out_csv": ROOT / "mex" / "data" / "advanced_features_historical.csv",
        "manual": {
            "Club América": "América",
            "CF América": "América",
            "Deportivo Guadalajara": "Guadalajara",
            "Club Deportivo Guadalajara": "Guadalajara",
            "UNAM Pumas": "Pumas UNAM",
            "Club Universidad Nacional": "Pumas UNAM",
            "Atlas Guadalajara": "Atlas",
            "Atlas FC": "Atlas",
            "Querétaro FC": "Querétaro",
            "CF Monterrey": "Monterrey",
            "Tigres UANL": "Tigres UANL",
            "CF Pachuca": "Pachuca",
            "Santos Laguna": "Santos",
            "Club Santos Laguna": "Santos",
            "Club Tijuana": "Tijuana",
            "Atlético de San Luis": "Atlético de San Luis",
            "FC Juárez": "FC Juarez"
        }
    },
    "bra": {
        "tm_id": "BRA1",
        "espn_partidos": ROOT / "bra" / "data" / "partidos.csv",
        "out_csv": ROOT / "bra" / "data" / "advanced_features_historical.csv",
        "manual": {
            "Club Athletico Paranaense": "Athletico-PR",
            "Athletico Paranaense": "Athletico-PR",
            "Atlético Mineiro": "Atlético-MG",
            "Clube Atlético Mineiro": "Atlético-MG",
            "Atlético Goianiense": "Atlético Goianiense",
            "Clube Atlético Goianiense": "Atlético Goianiense",
            "America Futebol Clube (MG)": "América Mineiro",
            "América Futebol Clube (MG)": "América Mineiro",
            "América FC": "América Mineiro",
            "Red Bull Bragantino": "Red Bull Bragantino",
            "Bragantino": "Red Bull Bragantino",
            "Cuiabá Esporte Clube": "Cuiabá",
            "Cuiabá EC": "Cuiabá",
            "Goiás Esporte Clube": "Goiás",
            "Avaí FC": "Avaí",
            "Ceará Sporting Club": "Ceará",
            "Chapecoense": "Chapecoense",
            "Associação Chapecoense de Futebol": "Chapecoense",
            "Coritiba FC": "Coritiba",
            "Criciúma Esporte Clube": "Criciúma",
            "Vasco da Gama": "Vasco da Gama",
            "Club de Regatas Vasco da Gama": "Vasco da Gama"
        }
    },
    "chile": {
        "tm_id": "CLPD",
        "espn_partidos": ROOT / "chile" / "data" / "partidos.csv",
        "out_csv": ROOT / "chile" / "data" / "advanced_features_historical.csv",
        "manual": {
            "CD Universidad Católica": "Universidad Católica",
            "Universidad Católica": "Universidad Católica",
            "Universidad de Chile": "Universidad de Chile",
            "Unión Española": "Unión Española",
            "CD Palestino": "Palestino",
            "Palestino": "Palestino",
            "Huachipato FC": "Huachipato",
            "Unión La Calera": "Unión La Calera",
            "Audax Italiano": "Audax Italiano",
            "Coquimbo Unido": "Coquimbo Unido",
            "Everton de Viña del Mar": "Everton CD",
            "Everton CD": "Everton CD",
            "CD O'Higgins": "O'Higgins",
            "O'Higgins": "O'Higgins",
            "Ñublense": "Ñublense",
            "CD Ñublense": "Ñublense",
            "Curicó Unido": "Curicó Unido",
            "Deportes Copiapó": "Copiapó",
            "Cobresal": "Cobresal",
            "CD Cobresal": "Cobresal",
            "Cobreloa": "Cobreloa",
            "Santiago Wanderers": "Santiago Wanderers",
            "Deportes Iquique": "Deportes Iquique",
            "Deportes Limache": "Deportes Limache",
            "Unión Wanderers": "Unión Wanderers",
            "Deportes Concepcion": "Deportes Concepcion",
            "Universidad de Concepción": "Universidad de Concepción"
        }
    }
}

def limpiar_valor(v_str):
    if not v_str:
        return 0.0
    v_str = v_str.replace("€", "").strip()
    if "bn" in v_str:
        try:
            return float(v_str.replace("bn", "").strip()) * 1000.0
        except ValueError:
            return 0.0
    elif "m" in v_str:
        try:
            return float(v_str.replace("m", "").strip())
        except ValueError:
            return 0.0
    elif "k" in v_str:
        try:
            return float(v_str.replace("k", "").strip()) / 1000.0
        except ValueError:
            return 0.0
    try:
        return float(v_str)
    except ValueError:
        return 0.0

def clean_name(n):
    n = n.lower()
    n = n.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    for word in ["club", "deportivo", "futbol", "cf", "fc", "cd", "rcd", "ud", "sd", "sc", "sport", "de", "athletic", "atletico", "esporte", "union", "clube", "social"]:
        n = n.replace(f" {word} ", " ").replace(f" {word}", "").replace(f"{word} ", "")
    return n.strip()

def match_team(tm_name, espn_names, manual_map):
    if tm_name in manual_map:
        return manual_map[tm_name]
    if tm_name in espn_names:
        return tm_name
    cleaned_tm = clean_name(tm_name)
    for en in espn_names:
        if clean_name(en) == cleaned_tm:
            return en
    for en in espn_names:
        ce = clean_name(en)
        if ce in cleaned_tm or cleaned_tm in ce:
            return en
    matches = difflib.get_close_matches(tm_name, espn_names, n=1, cutoff=0.5)
    if matches:
        return matches[0]
    return tm_name

def parse_int(v_str):
    if not v_str:
        return 0
    # Remover puntos, espacios, comas
    clean = re.sub(r'[^\d]', '', v_str)
    try:
        return int(clean)
    except ValueError:
        return 0

def parse_float(v_str):
    if not v_str:
        return 0.0
    # En TM se suele usar formato 24.4
    try:
        return float(v_str.replace(",", "."))
    except ValueError:
        return 0.0

import re

def scrape_advanced():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    
    for lkey, info in LEAGUES.items():
        print(f"\n=======================================================")
        print(f"Scrapeando datos avanzados para {lkey.upper()} ({info['tm_id']})")
        print(f"=======================================================")
        
        if not info["espn_partidos"].exists():
            print(f"  Error: No existe partidos.csv en {info['espn_partidos']}")
            continue
            
        p_df = pd.read_csv(info["espn_partidos"])
        espn_names = sorted(list(set(p_df.local.unique()).union(set(p_df.visita.unique()))))
        
        # Guardaremos los diccionarios por (temporada, equipo)
        data_squad = {}
        
        # 1. Scrapear Startseite (squad_size, avg_age, foreigners, squad_value)
        for s in seasons:
            url = f"https://www.transfermarkt.us/campeonato/startseite/wettbewerb/{info['tm_id']}/plus/?saison_id={s}"
            if lkey == "esp":
                url = f"https://www.transfermarkt.us/laliga/startseite/wettbewerb/ES1/plus/?saison_id={s}"
            elif lkey == "mex":
                url = f"https://www.transfermarkt.us/liga-mx-apertura/startseite/wettbewerb/MEX1/plus/?saison_id={s}"
            elif lkey == "bra":
                url = f"https://www.transfermarkt.us/campeonato-brasileiro-serie-a/startseite/wettbewerb/BRA1/plus/?saison_id={s}"
            elif lkey == "chile":
                url = f"https://www.transfermarkt.us/primera-division-de-chile/startseite/wettbewerb/CLPD/plus/?saison_id={s}"
                
            print(f"  [Startseite] Temporada {s} -> {url}")
            try:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                tabla = soup.find("table", class_="items") or soup.find(id="yw1")
                if not tabla:
                    print(f"    Advertencia: No se encontró tabla de equipos para {s}")
                    continue
                    
                tbody = tabla.find("tbody")
                rows = tbody.find_all("tr", recursive=False) if tbody else tabla.find_all("tr", recursive=False)
                
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 6:
                        continue
                    
                    link_eq = row.find("td", class_="hauptlink")
                    tm_name = link_eq.text.strip() if link_eq else None
                    if not tm_name:
                        continue
                        
                    squad_size = parse_int(cells[2].text.strip())
                    avg_age = parse_float(cells[3].text.strip())
                    foreigners = parse_int(cells[4].text.strip())
                    
                    rechts_cells = row.find_all("td", class_="rechts")
                    valor_raw = None
                    for rc in rechts_cells:
                        txt = rc.text.strip()
                        if "m" in txt or "bn" in txt or "k" in txt or "€" in txt:
                            valor_raw = txt
                    squad_value = limpiar_valor(valor_raw)
                    
                    espn_name = match_team(tm_name, espn_names, info["manual"])
                    
                    data_squad[(s, espn_name)] = {
                        "squad_size": squad_size,
                        "avg_age": avg_age,
                        "foreigners": foreigners,
                        "pct_foreigners": round(foreigners / squad_size, 4) if squad_size > 0 else 0.0,
                        "squad_value": squad_value
                    }
                time.sleep(2)
            except Exception as ex:
                print(f"    Error scraping startseite {s}: {ex}")
                
        # 2. Scrapear Besucherzahlen (Stadium capacity & Average attendance)
        for s in seasons:
            url = f"https://www.transfermarkt.us/campeonato/besucherzahlen/wettbewerb/{info['tm_id']}/saison_id/{s}"
            if lkey == "esp":
                url = f"https://www.transfermarkt.us/laliga/besucherzahlen/wettbewerb/ES1/saison_id/{s}"
            elif lkey == "mex":
                url = f"https://www.transfermarkt.us/liga-mx-apertura/besucherzahlen/wettbewerb/MEX1/saison_id/{s}"
            elif lkey == "bra":
                url = f"https://www.transfermarkt.us/campeonato-brasileiro-serie-a/besucherzahlen/wettbewerb/BRA1/saison_id/{s}"
            elif lkey == "chile":
                url = f"https://www.transfermarkt.us/primera-division-de-chile/besucherzahlen/wettbewerb/CLPD/saison_id/{s}"
                
            print(f"  [Asistencia] Temporada {s} -> {url}")
            try:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                tabla = soup.find("table", class_="items") or soup.find(id="yw1")
                if not tabla:
                    print(f"    Advertencia: No se encontró tabla de asistencia para {s}")
                    continue
                    
                tbody = tabla.find("tbody")
                rows = tbody.find_all("tr", recursive=False) if tbody else tabla.find_all("tr", recursive=False)
                
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 7:
                        continue
                    
                    # Col 4 es el nombre del club en la tabla de asistencia
                    club_cell = cells[4]
                    tm_name = club_cell.text.strip()
                    if not tm_name:
                        continue
                        
                    capacity = parse_int(cells[5].text.strip())
                    avg_attendance = parse_int(cells[7].text.strip())
                    
                    espn_name = match_team(tm_name, espn_names, info["manual"])
                    
                    # Fusionar con los datos anteriores
                    key = (s, espn_name)
                    if key in data_squad:
                        data_squad[key]["stadium_capacity"] = capacity
                        data_squad[key]["avg_attendance"] = avg_attendance
                        data_squad[key]["stadium_occupation"] = round(avg_attendance / capacity, 4) if capacity > 0 else 0.0
                    else:
                        # Si por alguna razón no estaba en la startseite (ej. ascendió/descendió raro)
                        data_squad[key] = {
                            "squad_size": 25,
                            "avg_age": 25.0,
                            "foreigners": 0,
                            "pct_foreigners": 0.0,
                            "squad_value": 10.0,
                            "stadium_capacity": capacity,
                            "avg_attendance": avg_attendance,
                            "stadium_occupation": round(avg_attendance / capacity, 4) if capacity > 0 else 0.0
                        }
                time.sleep(2)
            except Exception as ex:
                print(f"    Error scraping attendance {s}: {ex}")
                
        # 3. Guardar CSV unificado para esta liga
        filas_unificadas = []
        for (s, espn_name), metrics in data_squad.items():
            # Rellenar nulos
            capacity = metrics.get("stadium_capacity", 15000)
            avg_att = metrics.get("avg_attendance", 5000)
            
            filas_unificadas.append({
                "temporada": s,
                "equipo": espn_name,
                "squad_size": metrics["squad_size"],
                "avg_age": metrics["avg_age"],
                "foreigners": metrics["foreigners"],
                "pct_foreigners": metrics["pct_foreigners"],
                "squad_value": metrics["squad_value"],
                "stadium_capacity": capacity,
                "avg_attendance": avg_att,
                "stadium_occupation": metrics.get("stadium_occupation", round(avg_att / capacity, 4) if capacity > 0 else 0.0)
            })
            
        if filas_unificadas:
            df = pd.DataFrame(filas_unificadas)
            df.to_csv(info["out_csv"], index=False)
            print(f"  ✓ Exito: Guardado {info['out_csv']} con {len(df)} registros.")
        else:
            print(f"  ✗ Error: No se guardaron registros para {lkey.upper()}")

if __name__ == "__main__":
    scrape_advanced()
