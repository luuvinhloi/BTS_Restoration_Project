# Python code to generate a backup-power dataset and assignment for outage BTS sites.
# This code is intended to run in a Jupyter-like environment.
# It reads `power_outage_bts.csv` (if present), otherwise generates a fallback sample.
# It produces:
#  - backup_catalog.csv : list of backup units (one line per physical unit)
#  - backup_assignment.csv : assignment of each outage site to one backup unit and base
#  - base_summary.csv : counts of units to stage at each base
#
# The script follows the user's specification and the device models selected:
#   - GENS-L: KPS KP14000Q-3D-10KW (10 kW)
#   - GENS-S: KPS KP7000Q-5.0KW (5 kW)
#   - BAT_10: 48V 200Ah LiFePO4 (10 kWh)
#   - BAT_5: 48V 100Ah LiFePO4 (4.8–5.12 kWh) (two variants supported EV48100-T 15S/16S)
#
# Cost assumptions:
#   - diesel price = 20,000 VND / L
#   - electricity price = 3,000 VND / kWh
#
# Notes:
#   - The script adds a 10% spare margin in quantity (ceiling).
#   - Runtime calculations use conservative usable capacity (80% DOD for batteries)
#     and an approximate fuel burn rate for gensets (0.35 L per kWh produced).
#
# Output files are written to working directory. DataFrames are displayed below.
from pathlib import Path
import math
import pandas as pd
import numpy as np

# Configuration & device specs
DIESEL_PRICE_VND_PER_L = 20_000
ELECTRICITY_PRICE_VND_PER_KWH = 3_000

# Typical load per station type (kW) - based on user's earlier dataset
TYPICAL_LOAD_KW = {
    "5G_macro": 10.0,   # kW
    "4G_macro": 5.0,
    "4G_remote": 1.2,
    "5G_small": 0.25
}

# Backup device catalog templates (models selected by user)
DEVICES = {
    "GENS_L": {
        "type": "GENSET",
        "model": "KPS KP14000Q-3D-10KW",
        "power_kW": 10.0,
        "energy_kWh": None,
        "fuel_type": "diesel",
        "fuel_tank_L": 25.0,            # from spec image
        "consumption_L_per_kWh": 0.35,  # approx liters per kWh produced
        "voltage": "AC",
        "weight_kg": 210,
        "deployment_complexity": "high"
    },
    "GENS_S": {
        "type": "GENSET",
        "model": "KPS KP7000Q-5.0KW",
        "power_kW": 5.0,
        "energy_kWh": None,
        "fuel_type": "diesel",
        "fuel_tank_L": 17.0,            # from spec image (used conservatively)
        "consumption_L_per_kWh": 0.35,
        "voltage": "AC",
        "weight_kg": 153,
        "deployment_complexity": "medium"
    },
    "BAT_10": {
        "type": "BATTERY",
        "model": "48V 200Ah LiFePO4 (10 kWh) rack",
        "power_kW": None,   # dependant on inverter; use max discharge current to compute power if needed
        "energy_kWh": 10.0,
        "usable_fraction": 0.8,  # assume 80% usable DoD for longevity (can be adjusted)
        "roundtrip_efficiency": 0.98,
        "fuel_type": "—",
        "voltage": "48V DC",
        "weight_kg": 90,
        "deployment_complexity": "medium"
    },
    "BAT_5_EV48100_15S": {
        "type": "BATTERY",
        "model": "EV48100-T(15S) 48V 100Ah (4.8 kWh)",
        "power_kW": None,
        "energy_kWh": 4.8,
        "usable_fraction": 0.8,
        "roundtrip_efficiency": 0.98,
        "fuel_type": "—",
        "voltage": "48V DC",
        "weight_kg": 45,
        "deployment_complexity": "low"
    },
    "BAT_5_EV48100_16S": {
        "type": "BATTERY",
        "model": "EV48100-T(16S) 51.2V 100Ah (5.12 kWh)",
        "power_kW": None,
        "energy_kWh": 5.12,
        "usable_fraction": 0.8,
        "roundtrip_efficiency": 0.98,
        "fuel_type": "—",
        "voltage": "51.2V DC",
        "weight_kg": 48,
        "deployment_complexity": "low"
    }
}

# Bases (static)
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

# Helper functions
def haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    dphi = math.radians(lat2 - lat1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.asin(math.sqrt(a))

def choose_device_for_station(bts_type):
    # Allocation logic per user's specification
    if bts_type == "5G_macro":
        return "GENS_L"
    if bts_type == "4G_macro":
        return "GENS_L"   # prefer genset for macro to provide long runtime
    if bts_type == "4G_remote":
        return "BAT_10"
    if bts_type == "5G_small":
        # small cells can use smaller battery; choose BAT_5_EV48100_16S first
        return "BAT_5_EV48100_16S"
    # fallback
    return "BAT_10"

def compute_runtime_and_cost(device_key, load_kw):
    """Return runtime_hours_estimate, fuel_or_energy_needed, recharge_or_fuel_cost_vnd"""
    dev = DEVICES[device_key]
    if dev["type"] == "GENSET":
        # approximate liters per hour = load_kw * consumption_L_per_kWh
        lph = load_kw * dev["consumption_L_per_kWh"]
        tank = dev["fuel_tank_L"]
        runtime_h = tank / lph if lph > 0 else 0.0
        liters_needed_full_tank = tank
        cost_vnd = liters_needed_full_tank * DIESEL_PRICE_VND_PER_L
        return runtime_h, liters_needed_full_tank, cost_vnd
    else:
        # battery
        energy_kwh = dev["energy_kWh"]
        usable = energy_kwh * dev.get("usable_fraction", 0.8)
        # runtime hours at given load
        runtime_h = usable / load_kw if load_kw > 0 else float('inf')
        # energy used to deliver that usable energy (account for roundtrip efficiency)
        energy_to_charge_kwh = usable / dev.get("roundtrip_efficiency", 0.98)
        cost_vnd = energy_to_charge_kwh * ELECTRICITY_PRICE_VND_PER_KWH
        return runtime_h, usable, cost_vnd

# Load input BTS outage file (or synthesize sample)
infile = Path("power_outage_bts.csv")
if infile.exists():
    df_sites = pd.read_csv(infile)
else:
    # Create fallback sample per user's earlier counts (10,20,40,30)
    rows = []
    def add_sites(prefix, count, btype):
        for i in range(count):
            rows.append({
                "site_id": f"{prefix}_{i+1:03d}",
                "latitude": 16.48 + (i % 5) * 0.001,
                "longitude": 107.50 + (i % 7) * 0.0015,
                "status": "POWER_OUTAGE",
                "bts_type": btype
            })
    add_sites("B5G", 10, "5G_macro")
    add_sites("B4M", 20, "4G_macro")
    add_sites("B4R", 40, "4G_remote")
    add_sites("B5S", 30, "5G_small")
    df_sites = pd.DataFrame(rows)

# Ensure coordinate columns exist (fallback)
if "latitude" not in df_sites.columns or "longitude" not in df_sites.columns:
    # put all sites near center if coords missing
    df_sites["latitude"] = 16.48
    df_sites["longitude"] = 107.55

# Count by type
counts = df_sites["bts_type"].value_counts().to_dict()

# Add 10% spare margin
counts_with_spare = {k: math.ceil(v * 1.1) for k, v in counts.items()}

# Create list of backup units to satisfy counts_with_spare
backup_units = []
unit_id_counter = 1

for btype, qty in counts_with_spare.items():
    # For each site type, choose device model mapping
    device_key = choose_device_for_station(btype)
    # For allocation simplicity we create exactly qty units of that device_key
    for _ in range(qty):
        dev = DEVICES[device_key]
        load_kw = TYPICAL_LOAD_KW.get(btype, 1.0)
        runtime_h, resource_amount, cost_vnd = compute_runtime_and_cost(device_key, load_kw)
        backup_units.append({
            "backup_id": f"BK_{unit_id_counter:04d}",
            "type": dev["type"],
            "model": dev["model"],
            "device_key": device_key,
            "power_kW": dev.get("power_kW"),
            "energy_kWh": dev.get("energy_kWh"),
            "runtime_hours_at_typical_load": round(runtime_h, 2) if runtime_h is not None else None,
            "suitable_station_types": btype,
            "fuel_type": dev.get("fuel_type"),
            "voltage": dev.get("voltage"),
            "weight_kg": dev.get("weight_kg"),
            "deployment_complexity": dev.get("deployment_complexity"),
            "resource_amount_for_runtime": round(resource_amount, 2),
            "deployment_cost_VND_estimate": int(cost_vnd)
        })
        unit_id_counter += 1

df_backup_catalog = pd.DataFrame(backup_units)

# Assign each outage site to nearest base and to one backup unit
# Build numpy arrays for bases
base_lats = np.array([b["lat"] for b in BASES])
base_lons = np.array([b["lon"] for b in BASES])
base_ids = [b["id"] for b in BASES]

def find_nearest_base(lat, lon):
    dists = [haversine_km(lat, lon, bl, bo) for bl, bo in zip(base_lats, base_lons)]
    idx = int(np.argmin(dists))
    return base_ids[idx], dists[idx]

# We'll assign units to sites in order: for each site choose an unassigned unit matching its bts_type
assignments = []
# make mapping of available units by suitable_station_types (device_key may match multiple types in general)
available_units = df_backup_catalog.copy()
for idx, row in df_sites.iterrows():
    btype = row.get("bts_type", "")
    # find first available unit that has suitable_station_types == btype
    match_idx = available_units[available_units["suitable_station_types"] == btype].index
    if len(match_idx) == 0:
        # fallback: choose any available unit
        if len(available_units) == 0:
            raise RuntimeError("Not enough backup units generated.")
        chosen = available_units.iloc[0]
        unit_index = chosen.name
    else:
        chosen = available_units.loc[match_idx[0]]
        unit_index = match_idx[0]
    base_id, dist_km = find_nearest_base(row["latitude"], row["longitude"])
    assignments.append({
        "site_id": row.get("site_id"),
        "site_lat": row.get("latitude"),
        "site_lon": row.get("longitude"),
        "bts_type": btype,
        "assigned_backup_id": chosen["backup_id"],
        "assigned_model": chosen["model"],
        "base_id": base_id,
        "dist_to_base_km": round(dist_km, 3),
        "estimated_runtime_h": chosen["runtime_hours_at_typical_load"],
        "estimated_deployment_cost_VND": chosen["deployment_cost_VND_estimate"]
    })
    # remove chosen unit from available_units
    available_units = available_units.drop(unit_index)

df_assign = pd.DataFrame(assignments)

# Summarize allocation per base
base_summary = df_assign.groupby("base_id").agg(
    sites_assigned=("site_id", "count"),
    total_est_cost_VND=("estimated_deployment_cost_VND", "sum")
).reset_index()

# Also compute units to stock at each base by counting assigned units grouped by base and model
units_per_base = df_assign.groupby(["base_id", "assigned_model"]).size().reset_index(name="units_count")

# Save outputs
out_dir = Path(".")
df_backup_catalog.to_csv(out_dir / "backup_catalog.csv", index=False)
df_assign.to_csv(out_dir / "backup_assignment.csv", index=False)
base_summary.to_csv(out_dir / "base_summary.csv", index=False)
units_per_base.to_csv(out_dir / "units_per_base.csv", index=False)

# Display results
import ace_tools as tools  # display helper provided by the notebook environment for DataFrames
tools.display_dataframe_to_user("Backup catalog (sample)", df_backup_catalog.head(50))
tools.display_dataframe_to_user("Assignments (sample)", df_assign.head(50))
tools.display_dataframe_to_user("Base summary", base_summary)
tools.display_dataframe_to_user("Units per base", units_per_base)

# Print concise textual summary
print("SUMMARY")
print("-------")
print(f"Total outage sites read: {len(df_sites)}")
print("Counts by BTS type (original):")
print(counts)
print("Counts after +10% spare margin:")
print(counts_with_spare)
print(f"Total backup units generated: {len(df_backup_catalog)}")
print(f"Output files written: backup_catalog.csv, backup_assignment.csv, base_summary.csv, units_per_base.csv")
