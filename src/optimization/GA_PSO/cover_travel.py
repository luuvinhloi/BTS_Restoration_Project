# FILE: src/optimization/GA_PSO/cover_travel.py
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import haversine_m, ensure_numeric, to_float, logging

def load_inputs(processed_dir: str, raw_dir: str, params: dict):
    """
    Loads I_points, J_sites, cow_dataset, travel_matrix (travel_cost_matrix_A.csv).
    Returns I_df, J_df, cow_df, travel_dict.
    travel_dict keyed by (cow_id, site_id) -> {distance_km, travel_time_hr, travel_cost_vnd}
    """
    processed = Path(processed_dir)
    raw = Path(raw_dir)
    # file names allow overrides from params
    I_name = params.get("I_points_file", "I_points.csv")
    J_name = params.get("J_sites_file", "J_sites.csv")
    travel_name = params.get("travel_matrix_file", "travel_cost_matrix_A.csv")
    I_path = processed / I_name
    J_path = processed / J_name
    cow_path = raw / "cow_dataset.csv"
    travel_path = processed / travel_name

    if not I_path.exists():
        raise FileNotFoundError(f"I points missing: {I_path}")
    if not J_path.exists():
        raise FileNotFoundError(f"J sites missing: {J_path}")
    if not cow_path.exists():
        raise FileNotFoundError(f"COW dataset missing: {cow_path}")
    if not travel_path.exists():
        raise FileNotFoundError(f"travel matrix missing: {travel_path}")

    I_df = pd.read_csv(I_path).fillna(0)
    J_df = pd.read_csv(J_path).fillna(0)
    cow_df = pd.read_csv(cow_path).fillna(0)
    travel_df = pd.read_csv(travel_path).fillna(0)

    # normalize numeric columns
    I_df = ensure_numeric(I_df, ["latitude", "longitude", "pop"])
    J_df = ensure_numeric(J_df, ["latitude", "longitude", "priority_weight", "pop", "slope", "dist_to_road_m"])
    cow_df = ensure_numeric(cow_df, ["coverage_radius_m", "speed_kmh", "endurance_hr", "cost_vnd", "lat", "lon"])

    # build travel dict
    travel = {}
    for _, r in travel_df.iterrows():
        k = str(r["cow_id"])
        j = str(r["site_id"])
        travel[(k, j)] = {
            "distance_km": to_float(r.get("distance_km", 0.0)),
            "travel_time_hr": to_float(r.get("travel_time_hr", 0.0)),
            "travel_cost_vnd": to_float(r.get("travel_cost_vnd", 0.0))
        }

    return I_df, J_df, cow_df, travel

def build_cover_indicator_array(I_df, J_df, cow_df, max_workers=1):
    """
    Build boolean array cover[cow_idx, i_idx, j_idx]
    Uses exact per-cow coverage_radius_m
    """
    n_cows = len(cow_df)
    n_I = len(I_df)
    n_J = len(J_df)
    cover = np.zeros((n_cows, n_I, n_J), dtype=bool)

    I_lats = I_df["latitude"].values
    I_lons = I_df["longitude"].values
    J_lats = J_df["latitude"].values
    J_lons = J_df["longitude"].values

    for p in range(n_cows):
        radius_m = to_float(cow_df.iloc[p].get("coverage_radius_m", 0.0))
        if radius_m <= 0:
            continue
        for j in range(n_J):
            latj = float(J_lats[j])
            lonj = float(J_lons[j])
            # vectorized computation for distances I->j
            # small vectorized loop to avoid huge memory when n_I large
            dists = np.array([haversine_m(I_lats[i], I_lons[i], latj, lonj) for i in range(n_I)])
            cover[p, :, j] = dists <= radius_m
    return cover
