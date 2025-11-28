# src/preprocessing/compute_travel_costs.py

import os
import math
import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Point, LineString
import geopandas as gpd

# CONSTANTS
FUEL_CONSUMPTION_L_PER_100KM = 10
FUEL_PRICE_VND_PER_L = 23000
FUEL_COST_PER_KM = (FUEL_CONSUMPTION_L_PER_100KM / 100) * FUEL_PRICE_VND_PER_L


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    Return distance in meters between two lat/lon points using haversine formula.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_road_network(roads_geojson_path, dist_buff=0.03):
    """
    Build a routing graph by merging local roads (roads_hue.geojson) and OSM.
    Implementation detail:
      - Load local roads (GeoJSON)
      - Download OSM driving graph for buffered bounding box
      - For each segment in local roads, snap its coordinates to nearest OSM node(s)
        and add edges between those OSM nodes (with proper 'length' attribute).
    This avoids creating nodes without 'x'/'y' attributes that break osmnx.distance.
    """
    print("     Loading local road dataset...")
    roads_local = gpd.read_file(roads_geojson_path)
    roads_local = roads_local.to_crs(epsg=4326)

    # Build bounding box for OSM (buffer in degrees, ~0.03 ~= 3km)
    minx, miny, maxx, maxy = roads_local.total_bounds
    print("     Downloading OSM roads (buffer)...")
    bbox = (minx - dist_buff, miny - dist_buff, maxx + dist_buff, maxy + dist_buff)

    # download OSM drive graph
    G = ox.graph_from_bbox(bbox=bbox, network_type="drive", simplify=True, retain_all=False)
    print(f"     OSM graph nodes: {len(G.nodes)} edges: {len(G.edges)}")

    # Ensure node attributes 'x' and 'y' exist (OSM nodes normally have them)
    # For each local LineString, snap coordinates to nearest OSM nodes and add edges between them.
    print("     Integrating local roads into OSM graph by snapping to nearest OSM nodes...")
    added_edges = 0
    for idx, row in roads_local.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            # skip non-lines (MultiLineString etc.). If MultiLineString present, user may preprocess.
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        # map coordinates -> nearest OSM nodes
        snapped_nodes = []
        for (lon, lat) in coords:
            try:
                # nearest_nodes expects (G, X, Y) = (graph, lon, lat)
                n = ox.distance.nearest_nodes(G, lon, lat)
            except Exception:
                # fallback: use BallTree via ox.distance but if it fails, skip this coord
                n = None
            if n is not None:
                snapped_nodes.append((n, lat, lon))
        # add edges between consecutive snapped OSM nodes
        for i in range(len(snapped_nodes) - 1):
            u, lat_u, lon_u = snapped_nodes[i]
            v, lat_v, lon_v = snapped_nodes[i + 1]
            if u == v:
                continue
            # compute geodesic length between original coords (more accurate than straight node-to-node maybe)
            seg_len_m = haversine_distance_m(lat_u, lon_u, lat_v, lon_v)
            # Add directed edges both ways (OSM graph is directed; adding both ensures travel both directions)
            # If edge already exists in G, we keep existing attributes; else we add/overwrite length.
            try:
                G.add_edge(u, v, length=seg_len_m, highway="local_snapped")
                G.add_edge(v, u, length=seg_len_m, highway="local_snapped")
                added_edges += 1
            except Exception:
                # On rare occasions adding edges may fail; ignore and continue
                continue

    print(f"     Integration done. Added ~{added_edges} snapped edges. Final graph nodes: {len(G.nodes)} edges: {len(G.edges)}")

    # By design, edges from OSM already include 'length' attribute. For the edges we added we also set 'length'.
    return G


def compute_time_cost(distance_km, speed_kmh):
    """
    Compute travel time in hours and cost in VND for a given distance (km) and speed (km/h).
    """
    # protect against speed 0 or NaN
    try:
        speed_kmh = float(speed_kmh)
        if speed_kmh <= 0:
            # set a reasonable default low speed (e.g., 5 km/h) to avoid ZeroDivisionError
            speed_kmh = 5.0
    except Exception:
        speed_kmh = 5.0

    travel_time_hr = distance_km / speed_kmh
    travel_cost_vnd = distance_km * FUEL_COST_PER_KM
    return travel_time_hr, travel_cost_vnd


def compute_travel_matrix(cow_csv, site_csv, roads_path, output_csv):
    """
    Main entry: compute travel distance/time/cost between each COW (depot) and each site.
    Steps:
      - read cow and site CSVs
      - validate lat/lon present; drop invalid rows but log counts
      - build routing graph
      - find nearest graph node for each cow and site
      - for each pair, compute shortest path length (weight='length'); fallback to haversine
      - compute time & cost and save results to CSV
    """
    print("     Loading datasets...")
    df_cow = pd.read_csv(cow_csv)
    df_site = pd.read_csv(site_csv)

    # Ensure expected columns exist
    required_cow_cols = {"cow_id", "base_id", "lat", "lon", "speed_kmh"}
    required_site_cols = {"site_id", "latitude", "longitude", "priority_category", "priority_weight"}
    missing_cow = required_cow_cols - set(df_cow.columns)
    missing_site = required_site_cols - set(df_site.columns)
    if missing_cow:
        raise ValueError(f"Missing required columns in cow CSV: {missing_cow}")
    if missing_site:
        raise ValueError(f"Missing required columns in site CSV: {missing_site}")

    # drop rows with NaN coordinates and warn user
    before_cow = len(df_cow)
    df_cow = df_cow.dropna(subset=["lat", "lon"])
    dropped_cow = before_cow - len(df_cow)
    if dropped_cow:
        print(f"     WARNING: Dropped {dropped_cow} COW rows with missing lat/lon.")

    before_site = len(df_site)
    df_site = df_site.dropna(subset=["latitude", "longitude"])
    dropped_site = before_site - len(df_site)
    if dropped_site:
        print(f"     WARNING: Dropped {dropped_site} site rows with missing latitude/longitude.")

    print("     Building routing graph...")
    G = build_road_network(roads_path)

    # Prepare nearest node lookup
    print("     Preparing nearest nodes for COWs...")
    cow_nodes = []
    for _, r in df_cow.iterrows():
        lon = r["lon"]
        lat = r["lat"]
        try:
            n = ox.distance.nearest_nodes(G, lon, lat)
            cow_nodes.append(n)
        except Exception as e:
            # fallback: None
            cow_nodes.append(None)

    df_cow = df_cow.reset_index(drop=True)
    df_cow["node"] = cow_nodes

    print("     Preparing nearest nodes for Sites...")
    site_nodes = []
    for _, r in df_site.iterrows():
        lon = r["longitude"]
        lat = r["latitude"]
        try:
            n = ox.distance.nearest_nodes(G, lon, lat)
            site_nodes.append(n)
        except Exception:
            site_nodes.append(None)

    df_site = df_site.reset_index(drop=True)
    df_site["node"] = site_nodes

    # Remove any rows where nearest node resolution failed
    invalid_cows = df_cow["node"].isna().sum()
    invalid_sites = df_site["node"].isna().sum()
    if invalid_cows > 0 or invalid_sites > 0:
        print(f"     WARNING: {invalid_cows} cows and {invalid_sites} sites could not be snapped to graph nodes. These will be skipped.")
    df_cow_valid = df_cow.dropna(subset=["node"]).copy()
    df_site_valid = df_site.dropna(subset=["node"]).copy()

    results = []
    print("     Computing travel distances for all COW–Site pairs...")
    total_pairs = len(df_cow_valid) * len(df_site_valid)
    print(f"     Pairs to compute: {total_pairs} (COWs: {len(df_cow_valid)}, Sites: {len(df_site_valid)})")

    pair_counter = 0
    for _, cow in df_cow_valid.iterrows():
        for _, site in df_site_valid.iterrows():
            pair_counter += 1
            if pair_counter % 5000 == 0:
                print(f"       processed {pair_counter}/{total_pairs} pairs...")

            cow_node = int(cow["node"])
            site_node = int(site["node"])

            distance_km = None
            try:
                # compute shortest path length using NetworkX on the graph, weight by 'length' (meters)
                length_m = nx.shortest_path_length(G, cow_node, site_node, weight="length")
                distance_km = float(length_m) / 1000.0
            except Exception:
                # fallback: great-circle/haversine between locations
                try:
                    distance_m = haversine_distance_m(float(cow["lat"]), float(cow["lon"]),
                                                      float(site["latitude"]), float(site["longitude"]))
                    distance_km = distance_m / 1000.0
                except Exception:
                    distance_km = float("nan")

            travel_time_hr, travel_cost_vnd = compute_time_cost(distance_km if not pd.isna(distance_km) else 0.0,
                                                                cow.get("speed_kmh", 5.0))

            results.append({
                "base_lat": cow["lat"],
                "base_lon": cow["lon"],
                "site_lat": site["latitude"],
                "site_lon": site["longitude"],
                "base_id": cow.get("base_id", ""),
                "cow_id": cow.get("cow_id", ""),
                "site_id": site.get("site_id", ""),
                "priority_category": site.get("priority_category", ""),
                "priority_weight": site.get("priority_weight", ""),
                "distance_km": distance_km,
                "travel_time_hr": travel_time_hr,
                "travel_cost_vnd": travel_cost_vnd
            })

    df_out = pd.DataFrame(results)
    # ensure output directory exists
    outdir = os.path.dirname(output_csv)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    df_out.to_csv(output_csv, index=False)
    print(f"    DONE! Output saved to: {output_csv} (rows: {len(df_out)})")

    return df_out
