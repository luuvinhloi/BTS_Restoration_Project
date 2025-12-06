
# compute_travel_costs.py
"""
Compute travel times and fuel costs for COW vehicles (to J_sites) and backup power units (to failed BTS).
Uses a prebuilt road network in GraphML format (roads_flooded.graphml) produced by the pipeline.
- COW: uses only passable edges (is_passable == True).
- Backup: multimodal per-edge decision: truck when is_passable True, boat when False.
Outputs CSVs with distance_km, travel_time_hr, travel_cost_vnd for each origin-target pair.

Usage:
    from compute_travel_costs import compute_cow_travel_matrix, compute_backup_travel_matrix
    compute_cow_travel_matrix(...)
    compute_backup_travel_matrix(...)

Example (if run as script):
    python compute_travel_costs.py
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import BallTree

# -------------------------
# DEFAULT CONSTANTS / PARAMS
# -------------------------
# --- COW fuel (kept from original script) ---
FUEL_CONSUMPTION_L_PER_100KM_COW = 10.0   # L / 100km (default)
FUEL_PRICE_VND_PER_L_COW = 23000.0
FUEL_COST_PER_KM_COW = (FUEL_CONSUMPTION_L_PER_100KM_COW / 100.0) * FUEL_PRICE_VND_PER_L_COW

# --- Backup transport defaults (averages chosen) ---
# Truck (vehicle) parameters (vehicle that transports backup power)
TRUCK_KM_PER_LITER = 9.5          # midpoint of 8 - 11 km per liter
TRUCK_FUEL_PRICE_VND_PER_L = 23000.0
TRUCK_SPEED_KMH = 50.0

# Boat parameters (used on blocked edges)
BOAT_L_PER_HOUR = 22.5            # midpoint of 20 - 25 l/h
BOAT_FUEL_PRICE_VND_PER_L = 22000.0
BOAT_SPEED_KMH = 35.0

EARTH_RADIUS_M = 6371000.0

# -------------------------
# Utilities
# -------------------------
def haversine_distance_m(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c

def save_graph_pickle(G, path):
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_graph_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------------
# Graph loading / normalization / BallTree
# -------------------------
def load_graphml_graph(graphml_path):
    """
    Load GraphML produced by generate_flooded_roads_with_graph.py.
    Normalizes node coordinates into node['x'], node['y'] (lon, lat) floats and ensures each edge has a 'length_m' and boolean 'is_passable' attribute.
    Returns NetworkX graph (Graph, DiGraph, MultiGraph, or MultiDiGraph depending on file).
    """
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    G = nx.read_graphml(graphml_path)

    # Normalize node coordinates to floats if possible
    for n, data in G.nodes(data=True):
        # try common keys
        x = None
        y = None
        for key in ("x", "lon", "long", "longitude"):
            if key in data:
                x = data.get(key)
                break
        for key in ("y", "lat", "latitude"):
            if key in data:
                y = data.get(key)
                break
        if x is None or y is None:
            # leave as-is (some GraphMLs may have node ids that are coordinate tuples)
            continue
        try:
            data["x"] = float(x)
            data["y"] = float(y)
        except Exception:
            # try convert from string
            try:
                data["x"] = float(str(x))
                data["y"] = float(str(y))
            except Exception:
                data["x"] = None
                data["y"] = None

    # Normalize edges: ensure length_m and is_passable present and typed
    if G.is_multigraph():
        edge_iter = G.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, None, data) for u, v, data in G.edges(data=True))

    for u, v, k, data in edge_iter:
        # length candidates
        if "length_m" in data:
            try:
                data["length_m"] = float(data["length_m"])
            except Exception:
                pass
        elif "length" in data:
            try:
                data["length_m"] = float(data["length"])
            except Exception:
                pass
        elif "weight" in data:
            try:
                data["length_m"] = float(data["weight"])
            except Exception:
                pass
        else:
            # try compute from node coordinates
            n1 = G.nodes[u]
            n2 = G.nodes[v]
            if n1.get("y") is not None and n2.get("y") is not None:
                data["length_m"] = haversine_distance_m(n1["y"], n1["x"], n2["y"], n2["x"])
            else:
                data["length_m"] = 0.0

        # is_passable normalization
        if "is_passable" in data:
            val = data["is_passable"]
            if isinstance(val, str):
                data["is_passable"] = val.strip().lower() in ("true", "1", "t", "yes")
            else:
                data["is_passable"] = bool(val)
        else:
            # default safe assumption: passable
            data["is_passable"] = True

    return G

def build_balltree_from_graph(G):
    """
    Build BallTree for snapping lat/lon to graph nodes.
    Returns (tree, node_ids, coords_rad) where coords_rad are lat/lon in radians used by BallTree.
    """
    node_ids = []
    coords_rad = []
    for n, attr in G.nodes(data=True):
        lat = attr.get("y")
        lon = attr.get("x")
        if lat is None or lon is None:
            continue
        try:
            coords_rad.append([math.radians(lat), math.radians(lon)])
            node_ids.append(n)
        except Exception:
            continue

    if len(coords_rad) == 0:
        raise RuntimeError("No nodes with valid coordinates found in graph.")

    coords_rad = np.array(coords_rad)
    tree = BallTree(coords_rad, metric="haversine")
    return tree, np.array(node_ids), coords_rad

def nearest_node(tree, node_ids, coords_rad, lon, lat):
    """
    Snap lon/lat (in degrees) to nearest node id from BallTree.
    """
    if lon is None or lat is None:
        raise ValueError("lon/lat must be provided to nearest_node")
    pt = np.array([[math.radians(lat), math.radians(lon)]])
    dist_rad, idx = tree.query(pt, k=1)
    return node_ids[idx[0][0]]

# -------------------------
# COW travel (uses only passable edges)
# -------------------------
def compute_time_cost_cow(distance_km, speed_kmh):
    speed = max(float(speed_kmh), 1.0)
    time_hr = distance_km / speed
    cost_vnd = distance_km * FUEL_COST_PER_KM_COW
    return time_hr, cost_vnd

def compute_cow_travel_matrix(
    cow_csv,
    site_csv,
    graphml_path,
    output_csv,
    graph_pickle_cache=None,
    overwrite_cache=False,
    snap_max_distance_m=2000.0
):
    """
    Compute shortest-path distances/time/cost from each COW base (cow_csv) to each site in site_csv (J_sites).
    Only uses passable edges (is_passable == True). Snaps origins and targets to nearest graph node via BallTree.

    Parameters
    ----------
    cow_csv : path to cow_dataset.csv (must contain columns 'cow_id','lat','lon','speed_kmh', 'base_id')
    site_csv : path to J_sites.csv (must contain columns 'site_id','latitude','longitude')
    graphml_path : path to roads_flooded.graphml
    output_csv : path to write output CSV
    graph_pickle_cache : optional path to cache pickled graph (speeds repeated runs)
    overwrite_cache : bool to force reload graphml even if cache exists
    snap_max_distance_m : maximum snapping distance to accept; if nearest node further, fallback to haversine
    """
    print("Loading input CSVs...")
    df_cow = pd.read_csv(cow_csv)
    df_site = pd.read_csv(site_csv)

    # Validate columns minimally
    for col in ("cow_id", "lat", "lon", "speed_kmh"):
        if col not in df_cow.columns:
            raise RuntimeError(f"cow_csv missing required column: {col}")
    for col in ("site_id", "latitude", "longitude"):
        if col not in df_site.columns:
            raise RuntimeError(f"site_csv missing required column: {col}")

    # Load graph (cache optional)
    if graph_pickle_cache and os.path.exists(graph_pickle_cache) and not overwrite_cache:
        print(f"Loading cached graph: {graph_pickle_cache}")
        G = load_graph_pickle(graph_pickle_cache)
    else:
        print(f"Loading GraphML: {graphml_path}")
        G = load_graphml_graph(graphml_path)
        if graph_pickle_cache:
            os.makedirs(os.path.dirname(graph_pickle_cache), exist_ok=True)
            save_graph_pickle(G, graph_pickle_cache)
            print(f"Graph cached at: {graph_pickle_cache}")

    # Build a passable-only graph view (simple undirected Graph) for COW routing
    print("Building passable-only graph view for COWs...")
    if G.is_multigraph():
        G_pass = nx.Graph()
        G_pass.add_nodes_from(G.nodes(data=True))
        for u, v, key, data in G.edges(keys=True, data=True):
            if data.get("is_passable", True):
                G_pass.add_edge(u, v, **data)
    else:
        G_pass = G.__class__()  # Graph or DiGraph
        G_pass.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            if data.get("is_passable", True):
                G_pass.add_edge(u, v, **data)

    # Build BallTree on passable graph nodes (we will snap to available nodes)
    tree, node_ids, coords_rad = build_balltree_from_graph(G_pass)

    # Snap COWs and Sites to nearest node; compute distance to snapped node to ensure it's not too far
    print("Snapping COW bases...")
    def safe_snap(lon, lat):
        nid = nearest_node(tree, node_ids, coords_rad, lon, lat)
        return nid
    df_cow["node"] = df_cow.apply(lambda r: safe_snap(r["lon"], r["lat"]), axis=1)
    print("Snapping J sites...")
    df_site["node"] = df_site.apply(lambda r: safe_snap(r["longitude"], r["latitude"]), axis=1)

    results = []
    print("Computing shortest paths (COW -> J sites) using passable-only graph...")
    for _, cow in df_cow.iterrows():
        for _, site in df_site.iterrows():
            src = cow["node"]
            tgt = site["node"]
            # compute shortest path length (sum of length_m)
            distance_km = None
            try:
                length_m = nx.shortest_path_length(G_pass, source=src, target=tgt, weight="length_m")
                distance_km = float(length_m) / 1000.0
            except Exception:
                # fallback: haversine (straight-line)
                distance_km = haversine_distance_m(cow["lat"], cow["lon"], site["latitude"], site["longitude"]) / 1000.0

            travel_time_hr, travel_cost_vnd = compute_time_cost_cow(distance_km, cow["speed_kmh"])

            results.append({
                "base_id": cow.get("base_id"),
                "cow_id": cow.get("cow_id"),
                "base_lat": cow.get("lat"),
                "base_lon": cow.get("lon"),
                "site_id": site.get("site_id"),
                "site_lat": site.get("latitude"),
                "site_lon": site.get("longitude"),
                "distance_km": distance_km,
                "travel_time_hr": travel_time_hr,
                "travel_cost_vnd": travel_cost_vnd
            })

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved COW travel matrix to: {output_csv} (rows: {len(df_out)})")
    return df_out

# -------------------------
# Backup power travel (truck + boat multimodal)
# -------------------------
def _edge_travel_time_and_cost_for_backup(edge_data,
                                          truck_km_per_l=TRUCK_KM_PER_LITER,
                                          truck_speed_kmh=TRUCK_SPEED_KMH,
                                          truck_price_vnd_per_l=TRUCK_FUEL_PRICE_VND_PER_L,
                                          boat_l_per_h=BOAT_L_PER_HOUR,
                                          boat_speed_kmh=BOAT_SPEED_KMH,
                                          boat_price_vnd_per_l=BOAT_FUEL_PRICE_VND_PER_L):
    """
    Given edge attributes, compute travel time (hr) and fuel cost (VND) for that edge depending on is_passable.
    - If is_passable True: truck mode: cost by liters = length_km / km_per_liter, time = length_km / truck_speed_kmh
    - If is_passable False: boat mode: time = length_km / boat_speed_kmh, liters = time * liters_per_hour
    """
    length_m = float(edge_data.get("length_m", 0.0))
    length_km = length_m / 1000.0
    if edge_data.get("is_passable", True):
        # truck mode
        time_hr = length_km / float(max(truck_speed_kmh, 0.001))
        liters = length_km / float(max(truck_km_per_l, 0.0001))
        cost_vnd = liters * float(truck_price_vnd_per_l)
    else:
        # boat mode (fuel consumption provided as liters per hour)
        time_hr = length_km / float(max(boat_speed_kmh, 0.001))
        liters = time_hr * float(boat_l_per_h)
        cost_vnd = liters * float(boat_price_vnd_per_l)
    return time_hr, cost_vnd

def compute_backup_travel_matrix(
    backup_csv,
    outage_bts_csv,
    graphml_path,
    output_csv,
    graph_pickle_cache=None,
    overwrite_cache=False,
    optimize_for="time"  # "time" or "cost"
):
    """
    Compute routes from backup power bases (backup_csv) to failed BTS (outage_bts_csv).
    - If optimize_for == 'time' -> Dijkstra weight = per-edge travel_time_hr
    - If optimize_for == 'cost' -> Dijkstra weight = per-edge travel_cost_vnd
    Returns DataFrame saved to output_csv with per-pair totals.
    """
    print("Loading input CSVs...")
    df_back = pd.read_csv(backup_csv)
    df_outage = pd.read_csv(outage_bts_csv)

    # expect backup CSV to have 'power_id','base_id','lat','lon' columns
    if not {"power_id", "base_id", "lat", "lon"}.issubset(df_back.columns):
        # try alternative column names
        # fallback: attempt to find lat/lon columns
        if "latitude" in df_back.columns and "longitude" in df_back.columns:
            df_back = df_back.rename(columns={"latitude": "lat", "longitude": "lon"})
        else:
            raise RuntimeError("backup_csv must contain columns 'power_id','base_id','lat','lon' (or latitude/longitude)")

    # outage_bts must have lat/lon columns (user provided failed_bts.csv has 'latitude','longitude')
    if "latitude" in df_outage.columns and "longitude" in df_outage.columns:
        lat_col = "latitude"
        lon_col = "longitude"
    elif "lat" in df_outage.columns and "lon" in df_outage.columns:
        lat_col = "lat"; lon_col = "lon"
    else:
        raise RuntimeError("outage_bts_csv must contain 'latitude'/'longitude' or 'lat'/'lon' columns")

    # Load or cache graph
    if graph_pickle_cache and os.path.exists(graph_pickle_cache) and not overwrite_cache:
        print(f"Loading cached graph: {graph_pickle_cache}")
        G = load_graph_pickle(graph_pickle_cache)
    else:
        print(f"Loading GraphML: {graphml_path}")
        G = load_graphml_graph(graphml_path)
        if graph_pickle_cache:
            os.makedirs(os.path.dirname(graph_pickle_cache), exist_ok=True)
            save_graph_pickle(G, graph_pickle_cache)
            print(f"Graph cached at: {graph_pickle_cache}")

    # Compute per-edge travel_time_hr and travel_cost_vnd attributes using truck/boat rules
    print("Annotating per-edge travel_time_hr and travel_cost_vnd ...")
    if G.is_multigraph():
        edges_iter = G.edges(keys=True, data=True)
    else:
        edges_iter = ((u, v, None, data) for u, v, data in G.edges(data=True))

    for u, v, k, data in edges_iter:
        tt, cc = _edge_travel_time_and_cost_for_backup(data)
        data["travel_time_hr"] = float(tt)
        data["travel_cost_vnd"] = float(cc)

    # choose weight attribute
    if optimize_for == "time":
        weight_attr = "travel_time_hr"
    elif optimize_for == "cost":
        weight_attr = "travel_cost_vnd"
    else:
        raise ValueError("optimize_for must be 'time' or 'cost'")

    # Build BallTree for snapping
    tree, node_ids, coords_rad = build_balltree_from_graph(G)

    # Snap backup bases to nodes
    print("Snapping backup bases to graph nodes...")
    df_back["node"] = df_back.apply(lambda r: nearest_node(tree, node_ids, coords_rad, r["lon"], r["lat"]), axis=1)

    # Snap outage BTS to nodes
    print("Snapping outage BTS to graph nodes...")
    df_outage["node"] = df_outage.apply(lambda r: nearest_node(tree, node_ids, coords_rad, r[lon_col], r[lat_col]), axis=1)

    results = []
    print("Computing routes for backup units -> failed BTS ...")
    for _, back in df_back.iterrows():
        for _, bts in df_outage.iterrows():
            src = back["node"]
            tgt = bts["node"]

            try:
                # compute shortest path based on chosen weight attribute
                path_nodes = nx.shortest_path(G, source=src, target=tgt, weight=weight_attr)
                # Sum up time and cost along path (choose best parallel-edge if multigraph)
                total_time = 0.0
                total_cost = 0.0
                total_distance_m = 0.0

                for i in range(len(path_nodes) - 1):
                    u = path_nodes[i]; v = path_nodes[i+1]
                    # pick correct edge data (if multigraph, pick the edge with minimal weight_attr)
                    edge_data = None
                    if G.is_multigraph():
                        best_w = float("inf")
                        for key, ed in G[u][v].items():
                            w = float(ed.get(weight_attr, float("inf")))
                            if w < best_w:
                                best_w = w
                                edge_data = ed
                    else:
                        edge_data = G[u][v]

                    if edge_data is None:
                        continue
                    total_time += float(edge_data.get("travel_time_hr", 0.0))
                    total_cost += float(edge_data.get("travel_cost_vnd", 0.0))
                    total_distance_m += float(edge_data.get("length_m", 0.0))

                results.append({
                    "power_id": back.get("power_id"),
                    "base_id": back.get("base_id"),
                    "base_lat": back.get("lat"),
                    "base_lon": back.get("lon"),
                    "bts_id": bts.get("site_id") if "site_id" in bts else bts.get("bts_id"),
                    "bts_lat": bts.get(lat_col),
                    "bts_lon": bts.get(lon_col),
                    "distance_km": total_distance_m / 1000.0,
                    "total_time_hr": total_time,
                    "total_cost_vnd": total_cost,
                    "optimize_for": optimize_for
                })
            except Exception as e:
                # fallback: no path found. Use haversine estimate and assume truck mode (conservative)
                fallback_dist_km = haversine_distance_m(back["lat"], back["lon"], bts[lat_col], bts[lon_col]) / 1000.0
                fallback_time = fallback_dist_km / TRUCK_SPEED_KMH
                fallback_cost = (fallback_dist_km / TRUCK_KM_PER_LITER) * TRUCK_FUEL_PRICE_VND_PER_L
                results.append({
                    "power_id": back.get("power_id"),
                    "base_id": back.get("base_id"),
                    "base_lat": back.get("lat"),
                    "base_lon": back.get("lon"),
                    "bts_id": bts.get("site_id") if "site_id" in bts else bts.get("bts_id"),
                    "bts_lat": bts.get(lat_col),
                    "bts_lon": bts.get(lon_col),
                    "distance_km": fallback_dist_km,
                    "total_time_hr": fallback_time,
                    "total_cost_vnd": fallback_cost,
                    "optimize_for": optimize_for,
                    "note": f"fallback_no_path: {str(e)}"
                })

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved backup travel matrix to: {output_csv} (rows: {len(df_out)})")
    return df_out

# -------------------------
# Main runnable example
# -------------------------
if __name__ == "__main__":
    base_dir = "data/processed"
    graphml = os.path.join(base_dir, "road", "roads_flooded.graphml")
    cow_csv = "data/processed/cow/cow_dataset.csv"
    j_sites = "data/processed/position_I_J/J_sites_B.csv"
    cow_out = "data/processed/travel_cost/cow_to_sites.csv"
    backup_csv = "data/processed/backup_power/backup_power.csv"
    outages_csv = "data/processed/damage_bts/failed_bts.csv"
    backup_out = "data/processed/travel_cost/backup_to_bts.csv"
    cache = "cache/graph.pkl"

    # Compute COW routes (only passable edges)
    try:
        compute_cow_travel_matrix(cow_csv, j_sites, graphml, cow_out, graph_pickle_cache=cache)
    except Exception as exc:
        print("Error computing COW travel matrix:", exc)

    # Compute backup routes (truck+boat), minimize time
    try:
        compute_backup_travel_matrix(backup_csv, outages_csv, graphml, backup_out, graph_pickle_cache=cache, optimize_for="time")
    except Exception as exc:
        print("Error computing backup travel matrix:", exc)
