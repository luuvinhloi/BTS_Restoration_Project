"""
Generate Backup Power Dataset for BTS sites experiencing power outage.
Outputs exactly ONE file: backup_power.csv
Each backup power unit is assigned to its nearest BASE.

Updated:
- Reads failed_bts.csv and filters only status="power_outage"
- Computes 24-hour operation cost for GENSET/BATTERY
- Creates backup unit per power-outage BTS and assigns it to nearest BASE
- Output file contains exactly required columns only
"""

from pathlib import Path
import pandas as pd
import numpy as np
import math
import random

# ========================================
# CONFIGURATION
# ========================================
DIESEL_PRICE = 20_000
ELECTRICITY_PRICE = 3_000
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Typical load per BTS type
TYPICAL_LOAD = {
    "5G_macro": 10.0,
    "4G_macro": 5.0,
    "4G_remote": 1.2,
    "5G_small": 0.25
}

# Device specifications
DEVICES = {
    "GENS_L": {
        "type": "GENSET",
        "model": "KPS KP14000Q-3D-10KW",
        "power_kW": 10,
        "consumption_L_kWh": 0.35
    },
    "GENS_S": {
        "type": "GENSET",
        "model": "KPS KP7000Q-5.0KW",
        "power_kW": 5,
        "consumption_L_kWh": 0.35
    },
    "BAT_10": {
        "type": "BATTERY",
        "model": "48V 200Ah LiFePO4 (10 kWh) rack",
        "energy_kWh": 10,
        "usable_fraction": 0.8,
        "roundtrip_efficiency": 0.98
    },
    "BAT_5_EV48100_15S": {
        "type": "BATTERY",
        "model": "EV48100-T(15S) 48V 100Ah (4.8 kWh)",
        "energy_kWh": 4.8,
        "usable_fraction": 0.8,
        "roundtrip_efficiency": 0.98
    },
    "BAT_5_EV48100_16S": {
        "type": "BATTERY",
        "model": "EV48100-T(16S) 51.2V 100Ah (5.12 kWh)",
        "energy_kWh": 5.12,
        "usable_fraction": 0.8,
        "roundtrip_efficiency": 0.98
    }
}

# Choosing recommended device per BTS type
def choose_device(bts_type):
    if bts_type == "5G_macro":
        return "GENS_L"
    if bts_type == "4G_macro":
        return "GENS_L"
    if bts_type == "4G_remote":
        return "BAT_10"
    if bts_type == "5G_small":
        return "BAT_5_EV48100_16S"
    return "BAT_10"


# Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


# Compute runtime and cost for 24 hours operation
def compute_runtime_and_cost_24h(device_key, bts_type):
    dev = DEVICES[device_key]
    load_kw = TYPICAL_LOAD[bts_type]

    # GENSET calculation
    if dev["type"] == "GENSET":
        energy_24h = load_kw * 24                    # kWh
        fuel_liters = energy_24h * dev["consumption_L_kWh"]
        cost_24h = fuel_liters * DIESEL_PRICE

        # runtime (for reference)
        runtime_h = 24

        return runtime_h, fuel_liters, cost_24h

    # BATTERY calculation
    usable = dev["energy_kWh"] * dev["usable_fraction"]
    cost_per_cycle = (usable / dev["roundtrip_efficiency"]) * ELECTRICITY_PRICE
    energy_needed = load_kw * 24
    num_cycles = energy_needed / usable
    cost_24h = num_cycles * cost_per_cycle

    runtime_h = usable / load_kw  # one full discharge cycle

    return runtime_h, num_cycles, cost_24h


# BASE stations
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

def find_nearest_base(lat, lon):
    best = None
    best_dist = 9999999
    for b in BASES:
        d = haversine(lat, lon, b["lat"], b["lon"])
        if d < best_dist:
            best_dist = d
            best = b
    return best


# ================================================================
# MAIN FUNCTION
# ================================================================
def generate_backup_power_dataset(outage_csv_path, output_csv_path):
    df = pd.read_csv(outage_csv_path)

    # Filter only power-outage stations
    df = df[df["status"] == "power_outage"].copy()

    if df.empty:
        print("[WARNING] No power_outage stations found.")
        pd.DataFrame().to_csv(output_csv_path, index=False)
        return

    results = []
    unit_counter = 1

    for _, row in df.iterrows():
        latitude = row["latitude"]
        longitude = row["longitude"]
        bts_type = row["bts_type"]

        # Select device
        device_key = choose_device(bts_type)
        dev = DEVICES[device_key]

        # Compute runtime + 24h cost
        runtime_h, resource_amount, cost_24h = compute_runtime_and_cost_24h(device_key, bts_type)

        # Determine staging BASE
        base = find_nearest_base(latitude, longitude)

        results.append({
            "base_id": base["id"],
            "power_id": f"POWER_{unit_counter:04d}",
            "lat": base["lat"],
            "lon": base["lon"],
            "base_name": base["name"],
            "type": dev["type"],
            "model": dev["model"],
            "runtime_h": round(runtime_h, 2),
            "cost_vnd_24h": int(cost_24h),
            "resource_amount": round(resource_amount, 2)
        })

        unit_counter += 1

    df_out = pd.DataFrame(results)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)

    print(f"Generated backup_power.csv → {output_csv_path}")
