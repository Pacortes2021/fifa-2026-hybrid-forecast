import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

MAPEO_NOMBRES = {
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

def recolectar():
    DATA.mkdir(parents=True, exist_ok=True)
    csv_path = DATA / "squad_values_historical.csv"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    filas = []
    # Scrapear desde 2020 a 2025
    seasons = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    
    for s in seasons:
        url = f"https://www.transfermarkt.us/laliga/startseite/wettbewerb/ES1/plus/?saison_id={s}"
        print(f"Scrapeando temporada {s} (LaLiga) de TM...")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            tabla = soup.find("table", class_="items") or soup.find(id="yw1")
            if not tabla:
                print(f"  Advertencia: No se encontró tabla de equipos para temporada {s}")
                continue
                
            tbody = tabla.find("tbody")
            rows = tbody.find_all("tr", recursive=False) if tbody else tabla.find_all("tr", recursive=False)
            
            count = 0
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                
                link_eq = row.find("td", class_="hauptlink")
                tm_name = link_eq.text.strip() if link_eq else None
                if not tm_name:
                    continue
                    
                # Buscar el valor
                rechts_cells = row.find_all("td", class_="rechts")
                valor_raw = None
                for rc in rechts_cells:
                    txt = rc.text.strip()
                    if "m" in txt or "bn" in txt or "k" in txt or "€" in txt:
                        valor_raw = txt
                        
                valor_m = limpiar_valor(valor_raw)
                
                # Mapear nombre de equipo TM -> ESPN
                espn_name = MAPEO_NOMBRES.get(tm_name, tm_name)
                
                filas.append({
                    "temporada": s,
                    "equipo": espn_name,
                    "squad_value": valor_m
                })
                count += 1
                
            print(f"  Completado: {count} equipos procesados para temporada {s}.")
            time.sleep(2)  # Delay amigable
            
        except Exception as ex:
            print(f"  Error en temporada {s}: {ex}")
            
    if filas:
        df = pd.DataFrame(filas)
        df.to_csv(csv_path, index=False)
        print(f"Éxito: guardado squad_values_historical.csv con {len(df)} registros.")
    else:
        print("No se recopilaron datos de plantilla.")

if __name__ == "__main__":
    recolectar()
