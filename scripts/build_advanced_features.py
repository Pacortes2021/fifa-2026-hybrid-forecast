"""
Genera squad_values_historical.csv y advanced_features_historical.csv
para Francia (fra), Países Bajos (ned) y Portugal (por).
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent

# --- FRANCIA ---
FRA_TEAMS = {
    "Paris Saint-Germain": {"val": 950.0, "cap": 48583, "att": 47000, "age": 25.4, "pct_for": 0.68},
    "Monaco": {"val": 340.0, "cap": 18523, "att": 10000, "age": 24.8, "pct_for": 0.65},
    "Marseille": {"val": 320.0, "cap": 67394, "att": 63000, "age": 26.1, "pct_for": 0.64},
    "Lyon": {"val": 260.0, "cap": 59186, "att": 48000, "age": 25.5, "pct_for": 0.55},
    "Lille": {"val": 250.0, "cap": 50186, "att": 40000, "age": 25.2, "pct_for": 0.60},
    "Rennes": {"val": 210.0, "cap": 29778, "att": 27500, "age": 25.0, "pct_for": 0.52},
    "Nice": {"val": 200.0, "cap": 36178, "att": 25000, "age": 25.6, "pct_for": 0.58},
    "Lens": {"val": 180.0, "cap": 38223, "att": 37500, "age": 26.0, "pct_for": 0.50},
    "Strasbourg": {"val": 130.0, "cap": 26109, "att": 25000, "age": 23.8, "pct_for": 0.55},
    "Brest": {"val": 120.0, "cap": 15220, "att": 14500, "age": 26.2, "pct_for": 0.45},
    "Reims": {"val": 110.0, "cap": 21029, "att": 16000, "age": 24.5, "pct_for": 0.60},
    "Nantes": {"val": 100.0, "cap": 35322, "att": 28000, "age": 26.5, "pct_for": 0.48},
    "Toulouse": {"val": 95.0, "cap": 33150, "att": 24000, "age": 25.1, "pct_for": 0.62},
    "Montpellier": {"val": 85.0, "cap": 22000, "att": 14000, "age": 26.8, "pct_for": 0.42},
    "Saint-Etienne": {"val": 70.0, "cap": 41965, "att": 33000, "age": 25.8, "pct_for": 0.40},
    "Le Havre": {"val": 65.0, "cap": 25178, "att": 21000, "age": 25.5, "pct_for": 0.48},
    "Lorient": {"val": 65.0, "cap": 18890, "att": 15000, "age": 25.3, "pct_for": 0.50},
    "Auxerre": {"val": 60.0, "cap": 18541, "att": 16000, "age": 25.9, "pct_for": 0.45},
    "Metz": {"val": 55.0, "cap": 28786, "att": 20000, "age": 25.2, "pct_for": 0.52},
    "Angers": {"val": 55.0, "cap": 18752, "att": 13000, "age": 26.0, "pct_for": 0.40},
    "Clermont": {"val": 45.0, "cap": 13000, "att": 10500, "age": 26.2, "pct_for": 0.45},
    "Paris FC": {"val": 45.0, "cap": 20000, "att": 12000, "age": 25.0, "pct_for": 0.42},
    "Troyes": {"val": 40.0, "cap": 20400, "att": 10000, "age": 24.8, "pct_for": 0.48},
    "Bordeaux": {"val": 35.0, "cap": 42115, "att": 25000, "age": 24.5, "pct_for": 0.45},
    "AC Ajaccio": {"val": 20.0, "cap": 10660, "att": 7000, "age": 27.2, "pct_for": 0.35},
    "Le Mans": {"val": 20.0, "cap": 25064, "att": 10000, "age": 25.5, "pct_for": 0.30},
    "Nimes": {"val": 15.0, "cap": 18482, "att": 7000, "age": 26.5, "pct_for": 0.35},
}

# --- PAÍSES BAJOS ---
NED_TEAMS = {
    "PSV": {"val": 330.0, "cap": 35000, "att": 34500, "age": 24.8, "pct_for": 0.58},
    "Feyenoord": {"val": 290.0, "cap": 47500, "att": 47000, "age": 24.5, "pct_for": 0.60},
    "Ajax": {"val": 230.0, "cap": 55865, "att": 53500, "age": 24.2, "pct_for": 0.55},
    "AZ Alkmaar": {"val": 160.0, "cap": 19478, "att": 18500, "age": 24.0, "pct_for": 0.48},
    "Twente": {"val": 110.0, "cap": 30205, "att": 29500, "age": 25.2, "pct_for": 0.42},
    "Utrecht": {"val": 75.0, "cap": 23750, "att": 21000, "age": 25.5, "pct_for": 0.40},
    "NEC": {"val": 50.0, "cap": 12500, "att": 12500, "age": 25.0, "pct_for": 0.50},
    "Heerenveen": {"val": 45.0, "cap": 27224, "att": 23000, "age": 24.6, "pct_for": 0.45},
    "Sparta Rotterdam": {"val": 40.0, "cap": 10606, "att": 10200, "age": 25.8, "pct_for": 0.38},
    "Go Ahead Eagles": {"val": 38.0, "cap": 10000, "att": 9800, "age": 25.1, "pct_for": 0.45},
    "Groningen": {"val": 35.0, "cap": 22550, "att": 21500, "age": 24.2, "pct_for": 0.42},
    "Fortuna Sittard": {"val": 30.0, "cap": 12800, "att": 10500, "age": 26.0, "pct_for": 0.65},
    "PEC Zwolle": {"val": 28.0, "cap": 14000, "att": 13500, "age": 25.3, "pct_for": 0.40},
    "Heracles Almelo": {"val": 25.0, "cap": 12080, "att": 11500, "age": 25.6, "pct_for": 0.42},
    "NAC Breda": {"val": 25.0, "cap": 19000, "att": 18000, "age": 25.4, "pct_for": 0.48},
    "Willem II": {"val": 22.0, "cap": 14700, "att": 13800, "age": 25.7, "pct_for": 0.38},
    "Almere City": {"val": 22.0, "cap": 4501, "att": 4300, "age": 25.5, "pct_for": 0.40},
    "Vitesse": {"val": 20.0, "cap": 21248, "att": 15000, "age": 24.8, "pct_for": 0.45},
    "Excelsior": {"val": 18.0, "cap": 4500, "att": 4400, "age": 24.5, "pct_for": 0.42},
    "FC Volendam": {"val": 18.0, "cap": 7384, "att": 6800, "age": 24.2, "pct_for": 0.35},
    "RKC Waalwijk": {"val": 16.0, "cap": 7500, "att": 6500, "age": 26.5, "pct_for": 0.38},
    "ADO Den Haag": {"val": 16.0, "cap": 15000, "att": 12000, "age": 25.8, "pct_for": 0.35},
    "SC Cambuur": {"val": 15.0, "cap": 10250, "att": 9800, "age": 25.5, "pct_for": 0.36},
    "FC Emmen": {"val": 14.0, "cap": 8600, "att": 8200, "age": 26.0, "pct_for": 0.38},
    "De Graafschap": {"val": 12.0, "cap": 12600, "att": 11000, "age": 25.2, "pct_for": 0.30},
    "Telstar": {"val": 10.0, "cap": 3625, "att": 3000, "age": 24.8, "pct_for": 0.28},
}

# --- PORTUGAL ---
POR_TEAMS = {
    "Sporting CP": {"val": 410.0, "cap": 50095, "att": 40000, "age": 25.2, "pct_for": 0.62},
    "Benfica": {"val": 360.0, "cap": 64642, "att": 56000, "age": 25.6, "pct_for": 0.65},
    "Porto": {"val": 340.0, "cap": 50033, "att": 42000, "age": 25.5, "pct_for": 0.60},
    "Braga": {"val": 140.0, "cap": 30286, "att": 17000, "age": 25.8, "pct_for": 0.58},
    "Vitoria de Guimaraes": {"val": 65.0, "cap": 30000, "att": 18000, "age": 25.0, "pct_for": 0.52},
    "Famalicao": {"val": 50.0, "cap": 5186, "att": 4200, "age": 24.5, "pct_for": 0.60},
    "Arouca": {"val": 38.0, "cap": 5000, "att": 2500, "age": 26.2, "pct_for": 0.55},
    "Rio Ave": {"val": 35.0, "cap": 9065, "att": 3800, "age": 25.7, "pct_for": 0.52},
    "Gil Vicente": {"val": 32.0, "cap": 12504, "att": 4500, "age": 25.4, "pct_for": 0.54},
    "Moreirense": {"val": 32.0, "cap": 6153, "att": 3000, "age": 25.9, "pct_for": 0.50},
    "Estoril": {"val": 30.0, "cap": 8015, "att": 3200, "age": 24.8, "pct_for": 0.58},
    "Casa Pia": {"val": 28.0, "cap": 7000, "att": 2500, "age": 26.5, "pct_for": 0.50},
    "Santa Clara": {"val": 28.0, "cap": 13277, "att": 5500, "age": 25.8, "pct_for": 0.62},
    "Boavista": {"val": 26.0, "cap": 28263, "att": 8000, "age": 26.8, "pct_for": 0.50},
    "Farense": {"val": 24.0, "cap": 7000, "att": 5000, "age": 27.0, "pct_for": 0.48},
    "Portimonense": {"val": 22.0, "cap": 6000, "att": 3000, "age": 26.0, "pct_for": 0.65},
    "Estrela": {"val": 22.0, "cap": 9288, "att": 4000, "age": 26.2, "pct_for": 0.52},
    "Nacional": {"val": 22.0, "cap": 5132, "att": 2500, "age": 25.5, "pct_for": 0.55},
    "Vizela": {"val": 20.0, "cap": 6000, "att": 3500, "age": 25.8, "pct_for": 0.52},
    "AVS": {"val": 20.0, "cap": 5000, "att": 2500, "age": 26.8, "pct_for": 0.45},
    "Maritimo": {"val": 20.0, "cap": 10600, "att": 6000, "age": 26.2, "pct_for": 0.58},
    "Chaves": {"val": 18.0, "cap": 8400, "att": 3000, "age": 26.5, "pct_for": 0.50},
    "GD Chaves": {"val": 18.0, "cap": 8400, "att": 3000, "age": 26.5, "pct_for": 0.50},
    "Pacos de Ferreira": {"val": 18.0, "cap": 9076, "att": 3500, "age": 26.4, "pct_for": 0.45},
    "Belenenses": {"val": 15.0, "cap": 19856, "att": 4000, "age": 26.0, "pct_for": 0.45},
    "Tondela": {"val": 14.0, "cap": 5000, "att": 2500, "age": 26.1, "pct_for": 0.42},
    "Academico de Viseu": {"val": 12.0, "cap": 7744, "att": 2500, "age": 25.5, "pct_for": 0.40},
    "Alverca": {"val": 10.0, "cap": 7705, "att": 2000, "age": 25.2, "pct_for": 0.35},
}

# --- BÉLGICA ---
BEL_TEAMS = {
    "Club Brugge": {"val": 140.0, "cap": 29062, "att": 24000, "age": 24.8, "pct_for": 0.65},
    "Anderlecht": {"val": 110.0, "cap": 22500, "att": 20500, "age": 24.6, "pct_for": 0.58},
    "Genk": {"val": 100.0, "cap": 23718, "att": 19000, "age": 24.2, "pct_for": 0.60},
    "Union Saint-Gilloise": {"val": 90.0, "cap": 9400, "att": 8500, "age": 25.4, "pct_for": 0.70},
    "Gent": {"val": 85.0, "cap": 20000, "att": 17500, "age": 25.2, "pct_for": 0.58},
    "Antwerp": {"val": 75.0, "cap": 16144, "att": 14500, "age": 25.8, "pct_for": 0.55},
    "Cercle Brugge": {"val": 55.0, "cap": 29062, "att": 6500, "age": 23.9, "pct_for": 0.62},
    "Standard Liège": {"val": 50.0, "cap": 30023, "att": 21000, "age": 25.5, "pct_for": 0.52},
    "KV Mechelen": {"val": 40.0, "cap": 16672, "att": 13500, "age": 25.6, "pct_for": 0.45},
    "KVC Westerlo": {"val": 38.0, "cap": 8035, "att": 6500, "age": 24.5, "pct_for": 0.58},
    "Sint-Truiden": {"val": 35.0, "cap": 14600, "att": 7500, "age": 25.2, "pct_for": 0.55},
    "Charleroi": {"val": 35.0, "cap": 15000, "att": 9000, "age": 25.9, "pct_for": 0.60},
    "OH Leuven": {"val": 32.0, "cap": 10020, "att": 7200, "age": 25.4, "pct_for": 0.52},
    "KV Kortrijk": {"val": 22.0, "cap": 9399, "att": 7000, "age": 25.8, "pct_for": 0.50},
    "Dender": {"val": 18.0, "cap": 6429, "att": 5000, "age": 26.2, "pct_for": 0.40},
    "Beerschot": {"val": 15.0, "cap": 12771, "att": 7500, "age": 25.5, "pct_for": 0.45},
    "Eupen": {"val": 14.0, "cap": 8363, "att": 4000, "age": 25.0, "pct_for": 0.65},
    "RWDM": {"val": 14.0, "cap": 12266, "att": 6500, "age": 25.2, "pct_for": 0.60},
    "Zulte-Waregem": {"val": 12.0, "cap": 12250, "att": 8000, "age": 26.0, "pct_for": 0.42},
    "KV Oostende": {"val": 10.0, "cap": 8400, "att": 4500, "age": 25.6, "pct_for": 0.48},
    "Seraing": {"val": 8.0, "cap": 8207, "att": 2500, "age": 24.8, "pct_for": 0.55},
    "Waasland-Beveren": {"val": 8.0, "cap": 8100, "att": 3500, "age": 25.5, "pct_for": 0.45},
    "Lommel SK": {"val": 8.0, "cap": 8000, "att": 2500, "age": 24.0, "pct_for": 0.55},
    "Mouscron": {"val": 6.0, "cap": 10800, "att": 3500, "age": 25.5, "pct_for": 0.50},
    "RAAL La Louvière": {"val": 6.0, "cap": 12500, "att": 3000, "age": 25.0, "pct_for": 0.35}
}


def build_for_league(liga, team_dict):
    out_dir = BASE / liga / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    temporadas = list(range(2020, 2027))
    filas_adv = []
    filas_sq = []
    
    for t in temporadas:
        factor = 1.0 + (t - 2026) * 0.035
        for eq, d in team_dict.items():
            val = round(max(5.0, d["val"] * factor), 1)
            cap = d["cap"]
            att = min(cap, d["att"])
            occ = round(att / cap if cap > 0 else 0.8, 4)
            size = 28
            age = d["age"]
            for_cnt = int(round(size * d["pct_for"]))
            pct_for = round(for_cnt / size, 2)
            
            filas_adv.append({
                "temporada": t,
                "equipo": eq,
                "squad_size": size,
                "avg_age": age,
                "foreigners": for_cnt,
                "pct_foreigners": pct_for,
                "squad_value": val,
                "stadium_capacity": cap,
                "avg_attendance": att,
                "stadium_occupation": occ
            })
            filas_sq.append({
                "temporada": t,
                "equipo": eq,
                "squad_value": val
            })
            
    df_adv = pd.DataFrame(filas_adv)
    df_sq = pd.DataFrame(filas_sq)
    
    df_adv.to_csv(out_dir / "advanced_features_historical.csv", index=False)
    df_sq.to_csv(out_dir / "squad_values_historical.csv", index=False)
    print(f"{liga}: Generado advanced_features ({len(df_adv)} filas) y squad_values ({len(df_sq)} filas)")


def main():
    build_for_league("fra", FRA_TEAMS)
    build_for_league("ned", NED_TEAMS)
    build_for_league("por", POR_TEAMS)
    build_for_league("bel", BEL_TEAMS)


if __name__ == "__main__":
    main()
