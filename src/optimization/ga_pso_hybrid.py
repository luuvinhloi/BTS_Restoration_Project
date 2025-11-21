# src/optimization/ga_pso_hybrid.py
"""
GA–PSO Hybrid solver for COW deployment (real-data version)

This implementation:
    - Uses actual input tables in data/processed: I_points.csv, J_sites.csv, cow_dataset.csv
    - Builds a road network graph from roads_hue.geojson and computes network shortest-path distances
    - Precomputes travel_time and travel_cost matrices for each COW -> each candidate site
    - Builds coverage matrices per COW type (using haversine distance + simple filters)
    - Runs a hybrid GA-PSO metaheuristic with discrete operators (mutation, crossover with pbest, crossover with gbest)
    - Uses repair operators to enforce depot capacities and budget constraint
    - Saves best solution + fitness history for later visualization and analysis
"""
import math
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, List
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Geospatial helper functions
def haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    Haversine great-circle distance (meters).
    Inputs in decimal degrees.
    Note: signature uses lat1, lon1, lat2, lon2 (consistent with calls in file).
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def build_road_graph(roads_gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build a NetworkX graph from a roads GeoDataFrame.
    Each LineString segment is split to edges connecting its vertices.
    Nodes are (x, y) coordinate tuples (lon, lat).
    Edge weight is length in meters (geodesic approx using haversine on segment endpoints).
    """
    G = nx.Graph()
    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        # Some roads may be MultiLineString; normalize to LineStrings
        if geom is None:
            continue
        if geom.geom_type == "MultiLineString":
            lines = list(geom)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            # ignore other geometry types
            continue

        for line in lines:
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                lon1, lat1 = a[0], a[1]
                lon2, lat2 = b[0], b[1]
                u = (lon1, lat1)
                v = (lon2, lat2)
                if not G.has_node(u):
                    G.add_node(u, x=lon1, y=lat1)
                if not G.has_node(v):
                    G.add_node(v, x=lon2, y=lat2)
                length = haversine_distance_m(lat1, lon1, lat2, lon2)
                if G.has_edge(u, v):
                    # keep the minimum if multiple edges
                    if length < G[u][v]["weight"]:
                        G[u][v]["weight"] = length
                else:
                    G.add_edge(u, v, weight=length)
    logging.info(f"Built road graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def nearest_graph_node(G: nx.Graph, lon: float, lat: float, kd_tree=None, nodes_list=None):
    """
    Snap coordinate (lon, lat) to nearest graph node (lon, lat).
    If kd_tree and nodes_list provided, use them for speed.
    Returns node tuple (lon, lat).
    """
    if kd_tree is None or nodes_list is None:
        nodes_list = list(G.nodes)
        coords = [(n[0], n[1]) for n in nodes_list]
        kd_tree = cKDTree(coords)
    dist, idx = kd_tree.query([(lon, lat)], k=1)
    return nodes_list[int(idx[0])]

def network_distance_m(G: nx.Graph, src_coord: Tuple[float, float], dst_coord: Tuple[float, float],
                       kd_tree=None, nodes_list=None) -> float:
    """
    Compute network shortest-path distance (meters) between two coordinates by:
      - snapping each coordinate to nearest graph node
      - running Dijkstra shortest path (weight='weight')
    Returns path length in meters. If no path found, falls back to haversine distance.
    """
    try:
        if kd_tree is None or nodes_list is None:
            nodes_list = list(G.nodes)
            coords = [(n[0], n[1]) for n in nodes_list]
            kd_tree = cKDTree(coords)
        src_node = nearest_graph_node(G, src_coord[0], src_coord[1], kd_tree, nodes_list)
        dst_node = nearest_graph_node(G, dst_coord[0], dst_coord[1], kd_tree, nodes_list)
        length = nx.shortest_path_length(G, source=src_node, target=dst_node, weight="weight")
        return float(length)
    except Exception as e:
        # fallback: use haversine if graph path not found
        logging.debug(f"Network path fail ({e}), fallback to haversine.")
        return haversine_distance_m(src_coord[1], src_coord[0], dst_coord[1], dst_coord[0])

# Coverage matrices
def build_cover_matrices(I_df: pd.DataFrame, J_df: pd.DataFrame, cow_types: List[str]) -> Dict[str, np.ndarray]:
    """
    Build boolean coverage matrices per cow type.
    Returns dict: type -> array shape (|I|, |J|) of booleans.
    Coverage condition: haversine_distance(i, j) <= coverage_radius_for_type
    Additional filters: if J site 'in_water' or slope high, mark False.
    NOTE: This function returns placeholders; compute_cover_for_type builds actual matrices later.
    """
    cover_mats = {}
    for t in cow_types:
        cover_mats[t] = np.zeros((len(I_df), len(J_df)), dtype=bool)
    return cover_mats

def compute_cover_for_type(I_df: pd.DataFrame, J_df: pd.DataFrame, radius_m: float) -> np.ndarray:
    """
    Compute boolean cover matrix for a single radius (meters).
    Returns array shape (|I|, |J|).
    """
    nI = len(I_df)
    nJ = len(J_df)
    M = np.zeros((nI, nJ), dtype=bool)
    I_lats = I_df["latitude"].values
    I_lons = I_df["longitude"].values
    J_lats = J_df["latitude"].values
    J_lons = J_df["longitude"].values
    for j in range(nJ):
        latj = J_lats[j]
        lonj = J_lons[j]
        # loop over I points
        for i in range(nI):
            dist = haversine_distance_m(I_lats[i], I_lons[i], latj, lonj)
            if dist <= radius_m:
                M[i, j] = True
    return M

# Travel matrices
def precompute_travel_matrices(G, kd_tree, nodes_list, cow_df, J_df, cost_per_km_factor=0.1):
    n_cows = len(cow_df)
    n_sites = len(J_df)
    travel_time = np.zeros((n_cows, n_sites))
    travel_cost = np.zeros((n_cows, n_sites))
    J_coords = list(zip(J_df["longitude"].values, J_df["latitude"].values))
    # Snap all J to nearest nodes once
    J_nodes = [nearest_graph_node(G, lon, lat, kd_tree, nodes_list) for lon, lat in J_coords]
    for p in range(n_cows):
        row = cow_df.iloc[p]
        src = (row["lon"], row["lat"])
        src_node = nearest_graph_node(G, src[0], src[1], kd_tree, nodes_list)
        # Dijkstra 1 lần: trả về dict node->distance
        dist_dict = single_source_dijkstra_path_length(G, source=src_node, weight="weight")
        speed_kmh = max(row.get("speed_kmh", 40), 1.0)
        fixed_cost = row.get("cost_vnd", 0.0)
        for j, jnode in enumerate(J_nodes):
            dist_m = dist_dict.get(jnode, np.inf)
            if np.isinf(dist_m):
                dist_m = haversine_distance_m(row["lat"], row["lon"], J_df.iloc[j]["latitude"], J_df.iloc[j]["longitude"])
            dist_km = dist_m / 1000.0
            t_hours = dist_km / speed_kmh
            variable_cost = dist_km * cost_per_km_factor
            travel_time[p, j] = t_hours
            travel_cost[p, j] = fixed_cost + variable_cost
    return travel_cost, travel_time

# Fitness / evaluation helpers
def compute_coverage_from_solution(X: np.ndarray, cover_per_type: Dict[str, np.ndarray],
                                   cow_df: pd.DataFrame, pop_vector: np.ndarray) -> float:
    """
    Given solution X (length n_cows) where X[p] in {0,1,..,|J|} (0 means idle),
    compute population-weighted coverage ratio.

    cover_per_type: dict type -> (|I| x |J|) boolean array
    cow_df: DataFrame containing 'type' column
    pop_vector: array length |I|
    """
    nI = len(pop_vector)
    covered = np.zeros(nI, dtype=bool)

    for p, site in enumerate(X):
        if site == 0:
            continue
        typ = cow_df.iloc[p]["type"]
        j_idx = int(site) - 1
        if j_idx < 0:
            continue
        cov_mat = cover_per_type.get(typ)
        if cov_mat is None:
            continue
        covered = np.logical_or(covered, cov_mat[:, j_idx])

    served_pop = (pop_vector * covered).sum()
    total_pop = pop_vector.sum()
    return float(served_pop / total_pop) if total_pop > 0 else 0.0

def compute_cost_from_solution(X: np.ndarray, cow_df: pd.DataFrame, travel_cost_mat: np.ndarray) -> float:
    """
    Compute deployment cost: sum of fixed cost per deployed COW + travel cost to site.
    """
    total = 0.0
    for p, site in enumerate(X):
        if site == 0:
            continue
        site_idx = int(site) - 1
        fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
        travel = float(travel_cost_mat[p, site_idx]) if (site_idx >= 0 and site_idx < travel_cost_mat.shape[1]) else 0.0
        total += fixed + travel
    return float(total)

def compute_makespan_from_solution(X: np.ndarray, cow_df: pd.DataFrame, travel_time_mat: np.ndarray) -> float:
    """
    Compute makespan (hours) = max over all deployed COWs of (travel_time + setup_time).
    """
    times = []
    for p, site in enumerate(X):
        if site == 0:
            continue
        site_idx = int(site) - 1
        t_move = float(travel_time_mat[p, site_idx]) if (site_idx >= 0 and site_idx < travel_time_mat.shape[1]) else 0.0
        t_setup = float(cow_df.iloc[p].get("setup_time", 0.0))
        times.append(t_move + t_setup)
    return float(max(times)) if len(times) > 0 else 0.0

# Repair operators (updated to use cover_per_type and I_df)
def repair_solution_budget_and_depot(X: np.ndarray, cow_df: pd.DataFrame,
                                     travel_cost_mat: np.ndarray,
                                     depot_capacity_map: Dict[int, int],
                                     budget_max: float,
                                     cover_per_type: Dict[str, np.ndarray],
                                     I_df: pd.DataFrame) -> np.ndarray:
    """
    Repair X to satisfy:
     - depot capacity per base_id (base_id column in cow_df)
     - budget constraint (total cost <= budget_max)

    Strategy changes:
     - Enforce depot capacities by removing excess from the same depot if necessary (least beneficial first)
     - For budget enforcement: compute benefit = pop_gain / cost, keep those with highest benefit until budget satisfied
    """
    X = X.copy()
    # enforce depot capacities
    deployed_idxs = np.where(X > 0)[0].tolist()
    base_map = {}
    for p in deployed_idxs:
        base_id = str(cow_df.iloc[p].get("base_id", ""))
        base_map.setdefault(base_id, []).append(p)
    for base_id, indices in base_map.items():
        cap = depot_capacity_map.get(str(base_id), len(indices))
        if len(indices) > cap:
            # compute benefit per cow and remove worst ones
            benefits = []
            for idx in indices:
                site = int(X[idx])
                if site == 0:
                    benefit = 0.0
                else:
                    typ = cow_df.iloc[idx]["type"]
                    cov_mat = cover_per_type.get(typ)
                    if cov_mat is None:
                        benefit = 0.0
                    else:
                        j_idx = site - 1
                        cov_idxs = np.where(cov_mat[:, j_idx])[0]
                        benefit = I_df["pop"].values[cov_idxs].sum()
                fixed_cost = float(cow_df.iloc[idx].get("cost_vnd", 0.0))
                benefits.append((idx, benefit, fixed_cost))
            # sort by benefit ascending (remove lowest benefit first). tie-break by higher cost
            benefits_sorted = sorted(benefits, key=lambda x: (x[1], -x[2]))
            for idx_to_remove, _, _ in benefits_sorted[:max(0, len(indices) - cap)]:
                X[idx_to_remove] = 0

    # enforce budget: greedy keep highest pop_gain / cost ratio
    total_cost = compute_cost_from_solution(X, cow_df, travel_cost_mat)
    if total_cost <= budget_max:
        return X

    # compute benefit score for each deployed cow
    scored = []
    for p in range(len(cow_df)):
        site = int(X[p])
        if site == 0:
            continue
        typ = cow_df.iloc[p]["type"]
        cov_mat = cover_per_type.get(typ)
        j_idx = site - 1
        if cov_mat is None or j_idx < 0:
            pop_gain = 0.0
        else:
            cov_idxs = np.where(cov_mat[:, j_idx])[0]
            pop_gain = I_df["pop"].values[cov_idxs].sum() if len(cov_idxs) > 0 else 0.0
        fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
        travel = float(travel_cost_mat[p, j_idx]) if (j_idx >= 0 and j_idx < travel_cost_mat.shape[1]) else 0.0
        cost = fixed + travel
        score = pop_gain / (cost + 1e-9)
        scored.append((p, score, pop_gain, cost))

    # sort by score descending (best first), and keep adding until budget reached
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    X_new = np.zeros_like(X)
    total_cost = 0.0
    for p, score, pop_gain, cost in scored_sorted:
        if total_cost + cost <= budget_max:
            X_new[p] = int(X[p])  # keep assignment
            total_cost += cost
        else:
            X_new[p] = 0

    # nếu còn dư ngân sách > 10%, bổ sung thêm xe có hiệu quả cao nhưng chưa dùng
    if total_cost < 0.9 * budget_max:
        remaining = budget_max - total_cost
        candidates = [p for p in range(len(cow_df)) if X_new[p] == 0]
        gains = []
        for p in candidates:
            for j_idx in range(travel_cost_mat.shape[1]):
                typ = cow_df.iloc[p]["type"]
                cov_mat = cover_per_type.get(typ)
                if cov_mat is None:
                    continue
                cov_idxs = np.where(cov_mat[:, j_idx])[0]
                pop_gain = I_df["pop"].values[cov_idxs].sum()
                cost = cow_df.iloc[p]["cost_vnd"] + np.min(travel_cost_mat[p])
                if cost < remaining:
                    gains.append((p, j_idx, pop_gain / cost))
        gains_sorted = sorted(gains, key=lambda x: x[2], reverse=True)
        for p, j_idx, score in gains_sorted:
            cost = cow_df.iloc[p]["cost_vnd"] + np.min(travel_cost_mat[p])
            if total_cost + cost <= budget_max:
                X_new[p] = j_idx + 1
                total_cost += cost

    return X_new

# Genetic / PSO operators (discrete)
def mutation_operator_discrete(X: np.ndarray, num_sites: int, mutation_rate: float) -> np.ndarray:
    """
    Discrete mutation: for each gene p, with probability mutation_rate, assign a new random site in [0..num_sites].
    """
    X_new = X.copy()
    for i in range(len(X_new)):
        if np.random.rand() < mutation_rate:
            X_new[i] = np.random.randint(0, num_sites + 1)
    return X_new

def uniform_crossover(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Uniform crossover between two parent arrays (element-wise choose from X1 or X2)."""
    mask = np.random.rand(len(X1)) < 0.5
    child = X1.copy()
    child[~mask] = X2[~mask]
    return child

def overlap_penalty(X, cover_per_type, cow_df):
    nI = next(iter(cover_per_type.values())).shape[0]
    cover_count = np.zeros(nI, dtype=int)
    for p, site in enumerate(X):
        if site == 0:
            continue
        typ = cow_df.iloc[p]["type"]
        j_idx = site - 1
        cov_mat = cover_per_type.get(typ)
        if cov_mat is None:
            continue
        cover_count += cov_mat[:, j_idx].astype(int)
    overlap_ratio = (cover_count > 1).sum() / nI
    return overlap_ratio

# Core hybrid GA–PSO algorithm that uses real matrices
def ga_pso_hybrid_main(processed_data_dir: str,
                       outputs_dir: str,
                       config: Dict):
    """
    Main entry for GA–PSO hybrid run using real data files.

    processed_data_dir: folder path to data/processed
    outputs_dir: folder path to write outputs/results
    config: dictionary with algorithm parameters and assumptions (budget, cost factors, pop_size, etc.)
    """

    t0 = time.time()
    processed = Path(processed_data_dir)
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    # 1) Read input tables (real data)
    I_df = pd.read_csv(processed / "I_points_B.csv")      # columns: site_id, latitude, longitude, pop, ...
    J_df = pd.read_csv(processed / "J_sites_B.csv")       # columns: site_id, latitude, longitude, slope, dist_to_road_m, in_water, ...
    cow_df_full = pd.read_csv(processed.parent / "raw" / "cow_dataset.csv")  # using raw location of each COW

    # defensive checks
    assert {"latitude", "longitude"}.issubset(set(I_df.columns)), "I_points.csv missing lat/lon"
    assert {"latitude", "longitude"}.issubset(set(J_df.columns)), "J_sites.csv missing lat/lon"
    assert {"lat", "lon", "type"}.issubset(set(cow_df_full.columns)), "cow_dataset missing lat/lon/type columns"
    if "pop" not in I_df.columns:
        logging.error("I_points.csv missing 'pop' column. Required.")
        raise AssertionError("I_points.csv missing 'pop' column")

    # Convert types
    I_df["latitude"] = I_df["latitude"].astype(float)
    I_df["longitude"] = I_df["longitude"].astype(float)
    J_df["latitude"] = J_df["latitude"].astype(float)
    J_df["longitude"] = J_df["longitude"].astype(float)
    cow_df_full["lat"] = cow_df_full["lat"].astype(float)
    cow_df_full["lon"] = cow_df_full["lon"].astype(float)

    # 1b) Configuration / parameters (normalize & defaults)
    # Use config dict values with defaults
    M_max = int(config.get("M_max", 25))
    pop_size = int(config.get("pop_size", 100))
    max_iter = int(config.get("max_iter", 200))
    mutation_rate = float(config.get("mutation_rate", 0.08))
    ga_period = int(config.get("ga_period", 10))
    elitism = float(config.get("elitism", 0.10))
    budget_max = float(config.get("budget_max", 5e8))
    cost_per_km_factor = float(config.get("cost_per_km_factor", 0.2))
    # w1 = float(config.get("w1", 0.95))
    # w2 = float(config.get("w2", 0.03))
    # w3 = float(config.get("w3", 0.02))
    w_time = float(config.get("w_time", 0.2))  # hệ số phạt cho thời gian
    w_cost = float(config.get("w_cost", 0.4))  # hệ số phạt cho chi phí
    max_slope_deg = float(config.get("max_slope_deg", 15.0))
    seed = int(config.get("seed", 42))

    np.random.seed(seed)

    logging.info(f"GA-PSO config: M_max={M_max}, pop_size={pop_size}, max_iter={max_iter}, budget_max={budget_max}, w_time={w_time}, w_cost={w_cost}")

    # 1c) Limit COW fleet to M_max (use first M_max vehicles)
    if M_max <= 0:
        raise ValueError("M_max must be > 0")
    cow_df = cow_df_full.copy().reset_index(drop=True)
    if len(cow_df) > M_max:
        cow_df = cow_df.iloc[:M_max].reset_index(drop=True)
        logging.info(f"Truncated cow_df to M_max={M_max} entries (use first M_max rows).")
    n_cows = len(cow_df)
    n_sites = len(J_df)

    # build depot capacities from cow_df grouping by base_id
    depot_capacity_map = {}
    for base_id, group in cow_df.groupby("base_id"):
        depot_capacity_map[str(base_id)] = int(group.shape[0])  # total vehicles from that base
    logging.info(f"Depot capacities: {depot_capacity_map}")

    # 2) Build road graph from roads_hue.geojson (real network)
    raw_dir = processed.parent / "raw"
    roads_path = raw_dir / "roads_hue.geojson"
    if not roads_path.exists():
        raise FileNotFoundError(f"{roads_path} not found. Road network required for routing.")
    roads_gdf = gpd.read_file(roads_path)
    G = build_road_graph(roads_gdf)
    nodes_list = list(G.nodes)
    coords = [(n[0], n[1]) for n in nodes_list]
    kd_tree = cKDTree(coords)

    # 3) Precompute coverage matrices per cow type
    #    (use coverage_radius_m from cow_dataset per type)
    cow_types = cow_df["type"].unique().tolist()
    type_radius_map = {}
    for t in cow_types:
        rad_vals = cow_df.loc[cow_df["type"] == t, "coverage_radius_m"].dropna().values
        if len(rad_vals) == 0:
            logging.warning(f"No radius data for type {t}, default to 1000 m")
            type_radius_map[t] = 1000.0
        else:
            type_radius_map[t] = float(np.median(rad_vals))

    logging.info(f"Coverage radii per type: {type_radius_map}")

    cover_per_type = {}
    for t, r in type_radius_map.items():
        logging.info(f"Computing cover matrix for type={t}, radius={r}m ...")
        cover_per_type[t] = compute_cover_for_type(I_df, J_df, radius_m=r)
        logging.info(f"Cover matrix for {t} computed: shape {cover_per_type[t].shape}")

    # 4) Precompute travel_time and travel_cost matrices (real network distances)
    cache_file = Path(outputs) / "travel_matrices.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        travel_cost_mat = data["travel_cost"]
        travel_time_mat = data["travel_time"]
        logging.info("Loaded cached travel matrices.")
    else:
        logging.info("Precomputing travel_time and travel_cost matrices using network distances...")
        travel_cost_mat, travel_time_mat = precompute_travel_matrices(
            G, kd_tree, nodes_list, cow_df, J_df, cost_per_km_factor=cost_per_km_factor
        )
        np.savez(cache_file, travel_cost=travel_cost_mat, travel_time=travel_time_mat)
        logging.info("Travel matrices computed and cached.")

    # 5) Prepare site priority ranking (based on population covered and optional 'priority' column)
    if "priority" in J_df.columns:
        J_df["priority_norm"] = J_df["priority"] / (J_df["priority"].max() + 1e-9)
    else:
        J_df["priority_norm"] = 1.0

    # compute approximate covered weighted pop per site (across types)
    pop_vector = I_df["pop"].values
    pop_cover_by_site = []
    for j in range(n_sites):
        covered_pop = 0.0
        for t in cover_per_type.keys():
            idxs = np.where(cover_per_type[t][:, j])[0]
            if len(idxs) > 0:
                covered_pop += pop_vector[idxs].sum()
        # weight by priority
        covered_pop = covered_pop * float(J_df.loc[j, "priority_norm"])
        pop_cover_by_site.append(covered_pop)
    J_df["covered_pop_weighted"] = pop_cover_by_site
    J_rank = J_df.sort_values("covered_pop_weighted", ascending=False).index.tolist()

    # 6) GA–PSO parameters (already read above)
    logging.info(f"GA–PSO start: pop_size={pop_size}, max_iter={max_iter}, mutation_rate={mutation_rate}")

    # 7) Initialize population (improved heuristics)
    population = []
    # Heuristic A: assign each cow to best ranked site within its radius (prioritize using J_rank)
    for _ in range(int(pop_size * 0.5)):
        X = np.zeros(n_cows, dtype=int)
        for p in range(n_cows):
            typ = cow_df.iloc[p]["type"]
            radius = type_radius_map.get(typ, 1000.0)
            assigned = 0
            for j in J_rank:
                # respect slope/in_water constraints
                if J_df.iloc[j].get("in_water", 0) == 1:
                    continue
                if "slope" in J_df.columns and J_df.iloc[j]["slope"] > max_slope_deg:
                    continue
                dist = haversine_distance_m(cow_df.iloc[p]["lat"], cow_df.iloc[p]["lon"],
                                            J_df.iloc[j]["latitude"], J_df.iloc[j]["longitude"])
                if dist <= radius:
                    assigned = j + 1
                    break
            # If none found within radius, assign nearest feasible (to encourage more deployments)
            if assigned == 0:
                # nearest feasible site
                min_d = float("inf")
                min_j = -1
                for j in range(n_sites):
                    if J_df.iloc[j].get("in_water", 0) == 1:
                        continue
                    if "slope" in J_df.columns and J_df.iloc[j]["slope"] > max_slope_deg:
                        continue
                    d = haversine_distance_m(cow_df.iloc[p]["lat"], cow_df.iloc[p]["lon"],
                                             J_df.iloc[j]["latitude"], J_df.iloc[j]["longitude"])
                    if d < min_d:
                        min_d = d
                        min_j = j
                assigned = (min_j + 1) if min_j >= 0 else 0
            X[p] = assigned
        population.append(X)

    # Heuristic B: greedy coverage seed (assign COWs to maximize marginal pop gain)
    for _ in range(int(pop_size * 0.2)):
        X = np.zeros(n_cows, dtype=int)
        covered_mask = np.zeros(len(I_df), dtype=bool)
        for p in range(n_cows):
            typ = cow_df.iloc[p]["type"]
            best_j = 0
            best_gain = 0.0
            for j in J_rank:
                if J_df.iloc[j].get("in_water", 0) == 1:
                    continue
                if "slope" in J_df.columns and J_df.iloc[j]["slope"] > max_slope_deg:
                    continue
                cov_idxs = np.where(cover_per_type[typ][:, j])[0]
                if len(cov_idxs) == 0:
                    continue
                marginal = I_df["pop"].values[cov_idxs][~covered_mask[cov_idxs]].sum()
                marginal *= float(J_df.loc[j, "priority_norm"])
                if marginal > best_gain:
                    best_gain = marginal
                    best_j = j + 1
            X[p] = best_j
            if X[p] > 0:
                covered_mask = covered_mask | cover_per_type[typ][:, X[p]-1]
        population.append(X)

    # Random feasible fill for remaining members
    for _ in range(pop_size - len(population)):
        X = np.zeros(n_cows, dtype=int)
        for p in range(n_cows):
            feasible_js = []
            for j in range(n_sites):
                if J_df.iloc[j].get("in_water", 0) == 1:
                    continue
                if "slope" in J_df.columns and J_df.iloc[j]["slope"] > max_slope_deg:
                    continue
                feasible_js.append(j + 1)
            if len(feasible_js) == 0:
                X[p] = 0
            else:
                X[p] = int(np.random.choice(feasible_js))
        population.append(X)

    # pbest init
    pbest = [ind.copy() for ind in population]
    pbest_fit = [float("inf")] * pop_size

    # compute normalization baselines (cost_max/time_max)
    baseline = population[0]
    cost_baseline = compute_cost_from_solution(baseline, cow_df, travel_cost_mat)
    time_baseline = compute_makespan_from_solution(baseline, cow_df, travel_time_mat)
    cost_max = max(cost_baseline, 1.0)
    time_max = max(time_baseline, 1.0)

    # algorithm main loop
    best_global = None
    best_global_fit = float("inf")
    fitness_history = []

    pop_array = np.array(population)  # shape (pop_size, n_cows)

    logging.info("Starting GA–PSO hybrid optimization...")
    for it in range(max_iter):
        iter_best_fit = float("inf")
        iter_best_idx = -1

        # Evaluate current population
        for i in range(pop_size):
            X = pop_array[i].copy()
            # repair (depot capacity + budget) using coverage-aware repair
            X = repair_solution_budget_and_depot(X, cow_df, travel_cost_mat, depot_capacity_map, budget_max,
                                                cover_per_type, I_df)
            # compute fitness components
            R_cov = compute_coverage_from_solution(X, cover_per_type, cow_df, I_df["pop"].values)
            C = compute_cost_from_solution(X, cow_df, travel_cost_mat)
            T = compute_makespan_from_solution(X, cow_df, travel_time_mat)
            # normalized fitness (priority to coverage)
            # f = w1 * (1 - R_cov) + w2 * (C / cost_max) + w3 * (T / time_max)
            penalty = max(0, (C - budget_max) / budget_max)
            f = (1 - R_cov) + w_time * (T / time_max) + w_cost * penalty
            # O = overlap_penalty(X, cover_per_type, cow_df)
            # f = (1 - R_cov) + w_time * (T / time_max) + w_cost * penalty + 0.3 * O
            # update personal best
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = X.copy()
            if f < iter_best_fit:
                iter_best_fit = f
                iter_best_idx = i

        # update global best
        if iter_best_idx >= 0:
            candidate = pop_array[iter_best_idx].copy()
            if iter_best_fit < best_global_fit:
                best_global_fit = iter_best_fit
                best_global = candidate.copy()

        # generate new population using mutation/crossover with pbest & gbest
        new_pop = []
        for i in range(pop_size):
            Xi = pop_array[i].copy()
            # F1 - mutation
            X_mut = mutation_operator_discrete(Xi, n_sites, mutation_rate)
            # F2 - crossover with pbest_i
            X_p = uniform_crossover(X_mut, pbest[i])
            # F3 - crossover with gbest (if exists)
            if best_global is not None:
                X_g = uniform_crossover(X_p, best_global)
            else:
                X_g = X_p
            # repair
            X_rep = repair_solution_budget_and_depot(X_g, cow_df, travel_cost_mat, depot_capacity_map, budget_max,
                                                    cover_per_type, I_df)
            new_pop.append(X_rep)
        new_pop = np.array(new_pop)

        # elitism: keep top elites from pbest
        elite_n = max(1, int(elitism * pop_size))
        elite_indices = np.argsort(pbest_fit)[:elite_n]
        for k, idx in enumerate(elite_indices):
            new_pop[k] = pbest[idx].copy()

        pop_array = new_pop

        # periodic GA: crossover random pair -> replace worst
        if (it % ga_period) == 0:
            # pick parents
            parents_idx = np.random.choice(pop_size, size=2, replace=False)
            child = uniform_crossover(pop_array[parents_idx[0]], pop_array[parents_idx[1]])
            child = mutation_operator_discrete(child, n_sites, mutation_rate)
            # repair child
            child = repair_solution_budget_and_depot(child, cow_df, travel_cost_mat, depot_capacity_map, budget_max,
                                                    cover_per_type, I_df)
            # replace worst based on pbest_fit
            worst_idx = int(np.argmax(pbest_fit))
            pop_array[worst_idx] = child

        fitness_history.append(best_global_fit)
        if it % 10 == 0:
            logging.info(f"Iter {it}/{max_iter} - best_fit={best_global_fit:.6f}")

    # done
    runtime = time.time() - t0
    logging.info(f"GA–PSO finished. Best fitness={best_global_fit:.6f} time(s)={runtime:.1f}s")

    # save outputs
    np.save(outputs / "ga_pso_best_solution.npy", best_global)
    np.save(outputs / "ga_pso_fitness_history.npy", np.array(fitness_history))

    # decode assignments for CSV
    assign_list = []
    if best_global is not None:
        for p_idx, site in enumerate(best_global):
            site_idx = int(site)
            if site_idx > 0 and site_idx <= len(J_df):
                site_id_str = str(J_df.iloc[site_idx - 1]["site_id"])
            else:
                site_id_str = None
            assign_list.append({
                "cow_id": cow_df.iloc[p_idx]["cow_id"],
                "base_id": cow_df.iloc[p_idx]["base_id"],
                "assigned_site_index": site_idx if site_idx > 0 else None,
                "assigned_site_id": site_id_str
            })
    assign_df = pd.DataFrame(assign_list)
    assign_df.to_csv(outputs / "ga_pso_assignments.csv", index=False)

    # summary metrics
    if best_global is not None:
        R_cov_best = compute_coverage_from_solution(best_global, cover_per_type, cow_df, I_df["pop"].values)
        C_best = compute_cost_from_solution(best_global, cow_df, travel_cost_mat)
        T_best = compute_makespan_from_solution(best_global, cow_df, travel_time_mat)
        used_cows = int((best_global > 0).sum())
    else:
        R_cov_best, C_best, T_best, used_cows = 0.0, 0.0, 0.0, 0

    summary = {
        "best_fitness": float(best_global_fit),
        "coverage_ratio": float(R_cov_best),
        "total_cost_vnd": float(C_best),
        "makespan_hours": float(T_best),
        "runtime_s": float(runtime),
        "used_cows": int(used_cows),
        "M_max": int(M_max),
        "pop_size": pop_size,
        "max_iter": max_iter
    }
    with open(outputs / "ga_pso_summary.json", "w") as f:
        import json
        json.dump(summary, f, indent=2)

    logging.info(f"Outputs saved to {outputs}")

    # ensure central results folder exists (copy key outputs)
    project_root = Path(__file__).resolve().parents[2]
    central_results = Path(project_root) / "outputs" / "results"
    central_results.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copyfile(str(outputs / "ga_pso_assignments.csv"), str(central_results / "ga_pso_assignments.csv"))
        shutil.copyfile(str(outputs / "ga_pso_summary.json"), str(central_results / "ga_pso_summary.json"))
        shutil.copyfile(str(outputs / "ga_pso_best_solution.npy"), str(central_results / "ga_pso_best_solution.npy"))
        shutil.copyfile(str(outputs / "ga_pso_fitness_history.npy"), str(central_results / "ga_pso_fitness_history.npy"))
    except Exception as e:
        logging.warning(f"Failed to copy some outputs to central results: {e}")

    return summary


# Allow running this module standalone
if __name__ == "__main__":
    import yaml
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "config" / "params.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(open(cfg_path))
        # expect top-level ga_pso key
        ga_cfg = cfg.get("ga_pso", {})
        # merge top-level defaults if present (e.g., M_max at top)
        # flatten a few top-level parameters if present
        merged_cfg = {}
        # M_max could be at top-level or under ga_pso
        if "M_max" in cfg:
            merged_cfg["M_max"] = cfg["M_max"]
        merged_cfg.update(ga_cfg)
        # also include some top-level params if present
        for key in ["M_max", "budget", "default_R", "seed"]:
            if key in cfg and key not in merged_cfg:
                merged_cfg[key] = cfg[key]
        cfg_to_use = merged_cfg
    else:
        # default config
        cfg_to_use = {
            "M_max": 25,
            "pop_size": 100,
            "max_iter": 200,
            "mutation_rate": 0.08,
            "ga_period": 10,
            "elitism": 0.1,
            "budget_max": 5e8,
            "cost_per_km_factor": 0.2,
            "max_slope_deg": 15.0,
            "seed": 42
        }

    processed_dir = project_root / "data" / "processed"
    out_dir = project_root / "outputs" / "results"
    s = ga_pso_hybrid_main(str(processed_dir), str(out_dir), cfg_to_use)
    logging.info(s)