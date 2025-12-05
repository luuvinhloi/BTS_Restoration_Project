# FILE: src/optimization/GA_PSO/cover_travel.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .utils import ensure_numeric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_inputs(processed_dir: Path, params: dict):
    processed_dir = Path(processed_dir)
    # paths
    I_path = processed_dir / "position_I_J" / "I_points.csv"
    J_path = processed_dir / "position_I_J" / "J_sites.csv"
    cow_path = processed_dir / "cow" / "cow_dataset.csv"
    power_path = processed_dir / "backup_power" / "backup_power.csv"
    failed_bts_path = processed_dir / "damage_bts" / "failed_bts.csv"

    I_df = pd.read_csv(I_path)
    J_df = pd.read_csv(J_path)
    cow_df = pd.read_csv(cow_path)
    power_df = pd.read_csv(power_path)
    failed_bts_df = pd.read_csv(failed_bts_path)

    # ensure numeric columns exist
    ensure_numeric(I_df, ["pop"])
    ensure_numeric(J_df, ["pop"])
    ensure_numeric(cow_df, ["coverage_radius_m", "speed_kmh", "cost_vnd"])
    ensure_numeric(power_df, ["runtime_h", "cost_vnd_24h", "resource_amount"])
    ensure_numeric(failed_bts_df, ["power_W"])

    return I_df, J_df, cow_df, power_df, failed_bts_df


def build_cover_indicator_array(I_df, J_df, cow_df):
    """
    Build boolean array cover_arr[cow_idx, i_idx, j_idx] = True if cow p placed at j covers I point i.
    Use Euclidean/haversine with cow coverage_radius_m.
    """
    n_cows = len(cow_df)
    n_I = len(I_df)
    n_J = len(J_df)
    cover = np.zeros((n_cows, n_I, n_J), dtype=bool)
    I_lats = I_df["latitude"].astype(float).values
    I_lons = I_df["longitude"].astype(float).values
    J_lats = J_df["latitude"].astype(float).values
    J_lons = J_df["longitude"].astype(float).values
    cow_radii = cow_df["coverage_radius_m"].astype(float).values

    # vectorized approx: for each cow p and each J, compute distance to each I and compare
    # We'll compute for each cow p:
    for p in range(n_cows):
        r = cow_radii[p]
        # distances from J positions to I points: shape (n_I, n_J)
        for j in range(n_J):
            # compute haversine between I points and J[j]
            jl = J_lats[j]; jo = J_lons[j]
            # simple loop over I is fine (n_I few thousands)
            for i in range(n_I):
                # use haversine
                lat_i = I_lats[i]; lon_i = I_lons[i]
                # approximate distance (km->m)
                # to avoid dependency, use simple haversine:
                from .utils import haversine_m
                d = haversine_m(lat_i, lon_i, jl, jo)
                if d <= r:
                    cover[p, i, j] = True
    return cover


def build_travel_dicts(processed_dir: Path):
    """
    Read cow_to_J_sites.csv and backup_to_failed_bts.csv, return two dicts keyed by (cow_id, site_id) and (power_id, bts_id)
    """
    processed_dir = Path(processed_dir)
    cow_travel_path = processed_dir / "travel_cost" / "cow_to_J_sites.csv"
    power_travel_path = processed_dir / "travel_cost" / "backup_to_failed_bts.csv"
    cow_df = pd.read_csv(cow_travel_path)
    power_df = pd.read_csv(power_travel_path)

    cow_travel = {}
    for _, r in cow_df.iterrows():
        key = (str(r["cow_id"]), str(r["site_id"]))
        cow_travel[key] = {
            "distance_km": float(r.get("distance_km", 0.0)),
            "travel_time_hr": float(r.get("travel_time_hr", 0.0)),
            "travel_cost_vnd": float(r.get("travel_cost_vnd", 0.0))
        }
    power_travel = {}
    for _, r in power_df.iterrows():
        key = (str(r["power_id"]), str(r["bts_id"]))
        power_travel[key] = {
            "distance_km": float(r.get("distance_km", 0.0)),
            "total_time_hr": float(r.get("total_time_hr", 0.0)),
            "total_cost_vnd": float(r.get("total_cost_vnd", 0.0)),
            "note": r.get("note", "")
        }
    return cow_travel, power_travel
