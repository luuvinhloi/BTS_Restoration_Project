import pandas as pd
import os

# CONFIGURATION (STATIC ATTRIBUTES OF COW TYPES)
COVERAGE_RADIUS = {"heavy": 4000, "standard": 2000, "small": 1000}
POWER_KW = {"heavy": 40, "standard": 20, "small": 10}
COST_VND = {"heavy": 2e7, "standard": 1.5e7, "small": 1e7}
SPEED_KMH = {"heavy": 50, "standard": 55, "small": 60}
ENDURANCE_HR = {"heavy": 72, "standard": 48, "small": 48}

# BASE STATIONS OF 9 DISTRICTS/TOWNS OF HUE CITY (with coordinates)
BASES = [
    {"id": "BASE_1", "name": "Phu Xuan District", "lat": 16.4832781, "lon": 107.5715416},
    {"id": "BASE_2", "name": "Thuan Hoa District", "lat": 16.4626288, "lon": 107.6022722},
    {"id": "BASE_3", "name": "Huong Tra Town", "lat": 16.5254677, "lon": 107.3312759},
    {"id": "BASE_4", "name": "Huong Thuy Town", "lat": 16.4233011, "lon": 107.515289},
    {"id": "BASE_5", "name": "Phong Dien Town", "lat": 16.5820939, "lon": 107.3626442},
    {"id": "BASE_6", "name": "Phu Loc District", "lat": 16.2893818, "lon": 107.832841},
    {"id": "BASE_7", "name": "Phu Vang District", "lat": 16.4401105, "lon": 107.7120704},
    {"id": "BASE_8", "name": "Quang Dien District", "lat": 16.5743603, "lon": 107.508071},
    {"id": "BASE_9", "name": "A Luoi District", "lat": 16.2726766, "lon": 107.2307293}
]

# COW DISTRIBUTION BASED ON POPULATION + AREA ANALYSIS (TOTAL = 36 COWS)
# Heavy: 13
# Standard: 13
# Small: 10

BASE_COW_DISTRIBUTION = {
    "Phu Xuan District": {"heavy": 1, "standard": 1, "small": 2},   # 4
    "Thuan Hoa District": {"heavy": 1, "standard": 2, "small": 3},  # 6
    "Huong Tra Town": {"heavy": 1, "standard": 1, "small": 0},      # 2
    "Huong Thuy Town": {"heavy": 1, "standard": 1, "small": 1},     # 3
    "Phong Dien Town": {"heavy": 2, "standard": 1, "small": 1},     # 4
    "Phu Loc District": {"heavy": 3, "standard": 3, "small": 2},    # 8
    "Phu Vang District": {"heavy": 1, "standard": 2, "small": 0},   # 3
    "Quang Dien District": {"heavy": 1, "standard": 1, "small": 0}, # 2
    "A Luoi District": {"heavy": 2, "standard": 1, "small": 1}      # 4
}

# FUNCTION: ASSIGN ALL COWs TO THEIR BASES (RETURN A CLEAN DATAFRAME)
def assign_cows_to_bases():
    all_cows = []
    cow_id = 1

    for base in BASES:
        base_name = base["name"]
        lat, lon = base["lat"], base["lon"]

        if base_name not in BASE_COW_DISTRIBUTION:
            raise KeyError(f"Missing COW distribution for base '{base_name}'")

        cow_types = BASE_COW_DISTRIBUTION[base_name]

        for cow_type, count in cow_types.items():
            for _ in range(count):
                all_cows.append({
                    "cow_id": f"COW_{cow_id:03d}",
                    "base_id": base["id"],
                    "base_name": base_name,
                    "type": cow_type,
                    "lat": lat,
                    "lon": lon,
                    "coverage_radius_m": COVERAGE_RADIUS[cow_type],
                    "power_kw": POWER_KW[cow_type],
                    "speed_kmh": SPEED_KMH[cow_type],
                    "endurance_hr": ENDURANCE_HR[cow_type],
                    "cost_vnd": COST_VND[cow_type],
                    "assigned_region": base_name
                })
                cow_id += 1

    return pd.DataFrame(all_cows)

# FUNCTION: EXPORT DATASET TO CSV (CALLED FROM PIPELINE)
def generate_cow_dataset(output_folder: str):
    df = assign_cows_to_bases()

    os.makedirs(output_folder, exist_ok=True)
    out_csv = os.path.join(output_folder, "cow_dataset.csv")

    df.to_csv(out_csv, index=False)

    print(f"[COW] Generated {len(df)} COW records")
    print(f"[COW] Saved CSV: {out_csv}")

    return df
