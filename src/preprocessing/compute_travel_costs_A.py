# src/preprocessing/compute_travel_costs.py

import os
import math
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from sklearn.neighbors import BallTree

# CONSTANTS (fuel)
FUEL_CONSUMPTION_L_PER_100KM = 10.0
FUEL_PRICE_VND_PER_L = 23000.0
FUEL_COST_PER_KM = (FUEL_CONSUMPTION_L_PER_100KM / 100.0) * FUEL_PRICE_VND_PER_L

EARTH_RADIUS_M = 6371000.0


def haversine_distance_m(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def save_graph_pickle(G, path):
    """Save graph safely using pickle."""
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_pickle(path):
    """Load graph safely using pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_graph_from_geojson(roads_geojson_path, cache_path=None, overwrite_cache=False):
    """
    Build graph using only local roads (roads_hue.geojson).
    This version:
       - DOES NOT use OSM
       - DOES NOT snap
       - Creates graph from LineStrings
       - Adds length via haversine
       - Caches graph using pickle
    """
    # Load cache if available
    if cache_path and os.path.exists(cache_path) and not overwrite_cache:
        print(f"     Loading graph cache: {cache_path}")
        return load_graph_pickle(cache_path)

    print("     Reading local road dataset...")
    gdf = gpd.read_file(roads_geojson_path).to_crs(epsg=4326)

    print("     Building graph nodes + edges...")
    coord_to_id = {}
    next_id = 0
    nodes = []
    edges = []

    def get_node_id(coord):
        nonlocal next_id
        key = (round(coord[0], 7), round(coord[1], 7))
        if key not in coord_to_id:
            coord_to_id[key] = next_id
            nodes.append((next_id, {"x": key[0], "y": key[1]}))
            next_id += 1
        return coord_to_id[key]

    for _, row in gdf.iterrows():
        geom = row.geometry

        if geom is None:
            continue

        if isinstance(geom, LineString):
            lines = [geom]
        else:
            try:
                lines = list(geom)
            except:
                continue

        for line in lines:
            coords = list(line.coords)

            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i + 1]

                u = get_node_id((lon1, lat1))
                v = get_node_id((lon2, lat2))
                length_m = haversine_distance_m(lat1, lon1, lat2, lon2)

                edges.append((u, v, {"length": length_m}))
                edges.append((v, u, {"length": length_m}))

    # Build graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    print(f"     Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"     Saving graph cache to {cache_path}")
        save_graph_pickle(G, cache_path)

    return G


def build_balltree(G):
    node_ids = []
    coords_rad = []

    for nid, attr in G.nodes(data=True):
        lat = attr["y"]
        lon = attr["x"]
        coords_rad.append([math.radians(lat), math.radians(lon)])
        node_ids.append(nid)

    coords_rad = np.array(coords_rad)
    tree = BallTree(coords_rad, metric="haversine")

    return tree, np.array(node_ids), coords_rad


def nearest_node(tree, node_ids, lon, lat):
    point = np.array([[math.radians(lat), math.radians(lon)]])
    dist_rad, idx = tree.query(point, k=1)
    nid = node_ids[idx[0][0]]
    return int(nid)


def compute_time_cost(distance_km, speed_kmh):
    speed = max(float(speed_kmh), 1.0)
    time_hr = distance_km / speed
    cost = distance_km * FUEL_COST_PER_KM
    return time_hr, cost


def compute_travel_matrix(
    cow_csv,
    site_csv,
    roads_path,
    output_csv,
    cache_graph_path="data/processed/roads_graph.gpickle",
    overwrite_cache=False
):
    print("     Loading CSVs...")
    df_cow = pd.read_csv(cow_csv)
    df_site = pd.read_csv(site_csv)

    # Build graph
    print("     Building/loading local road graph...")
    G = build_graph_from_geojson(
        roads_path,
        cache_path=cache_graph_path,
        overwrite_cache=overwrite_cache
    )

    # Build BallTree
    print("     Building BallTree for nearest node snapping...")
    tree, node_ids, coords_rad = build_balltree(G)

    # Snap COWs
    print("     Snapping COWs to graph...")
    df_cow["node"] = df_cow.apply(
        lambda r: nearest_node(tree, node_ids, r["lon"], r["lat"]), axis=1
    )

    # Snap Sites
    print("     Snapping Sites to graph...")
    df_site["node"] = df_site.apply(
        lambda r: nearest_node(tree, node_ids, r["longitude"], r["latitude"]), axis=1
    )

    results = []

    print("     Computing COW–Site distances...")

    for _, cow in df_cow.iterrows():
        for _, site in df_site.iterrows():

            cow_node = int(cow["node"])
            site_node = int(site["node"])

            # Shortest path
            try:
                length_m = nx.shortest_path_length(G, cow_node, site_node, weight="length")
                distance_km = length_m / 1000.0
            except:
                # fallback haversine
                distance_km = haversine_distance_m(
                    cow["lat"], cow["lon"], site["latitude"], site["longitude"]
                ) / 1000.0

            travel_time, travel_cost = compute_time_cost(distance_km, cow["speed_kmh"])

            results.append({
                "base_lat": cow["lat"],
                "base_lon": cow["lon"],
                "site_lat": site["latitude"],
                "site_lon": site["longitude"],
                "base_id": cow["base_id"],
                "cow_id": cow["cow_id"],
                "site_id": site["site_id"],
                "priority_category": site["priority_category"],
                "priority_weight": site["priority_weight"],
                "distance_km": distance_km,
                "travel_time_hr": travel_time,
                "travel_cost_vnd": travel_cost
            })

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print(f"    DONE! Saved: {output_csv}")
    return df_out
