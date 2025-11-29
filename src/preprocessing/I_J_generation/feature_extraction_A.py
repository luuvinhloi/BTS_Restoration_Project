# src/preprocessing/feature_extraction.py
"""
Feature Extraction (DBSCAN clustering, realistic I/J generation)
- Build I (demand points) from pop raster excluding area covered by active BTS.
- Cluster population points with DBSCAN (project to metric CRS).
- Create I centroids, compute pop within 3km (non-overlapping) using greedy selection.
- Generate J candidate sites around I and along roads, filter by slope/water, distance-to-road.
- Compute pop covered by each J (3km) and save I_points.csv, J_sites.csv, cover.npy.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
import math
import random
import warnings

from src.utils.geo_utils import compute_distance_matrix
from src.utils.io_utils import read_geojson

# PATHS
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
DAMAGE_BTS_DIR = DATA_DIR / "processed" / "damage_bts"

# Helper: read CSV for active/failed BTS
def read_bts_files():
    active_path = DAMAGE_BTS_DIR / "active_bts.csv"
    failed_path = DAMAGE_BTS_DIR / "failed_bts.csv"
    active = pd.read_csv(active_path) if active_path.exists() else pd.DataFrame()
    failed = pd.read_csv(failed_path) if failed_path.exists() else pd.DataFrame()
    return active, failed

# 1) extract population raster cells above threshold -> GeoDataFrame (EPSG:4326)
def extract_population_cells(pop_tif, threshold=10):
    pts = []
    with rasterio.open(pop_tif) as src:
        arr = src.read(1)
        arr = np.nan_to_num(arr, nan=0.0)
        transform = src.transform
        rows, cols = np.where(arr > threshold)
        for r, c in zip(rows, cols):
            val = float(arr[r, c])
            lon, lat = transform * (int(c) + 0.5, int(r) + 0.5)
            pts.append((lon, lat, val))
    if not pts:
        return gpd.GeoDataFrame(columns=["longitude", "latitude", "pop", "geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(
        [(x, y, float(v)) for x, y, v in pts],
        columns=["longitude", "latitude", "pop"],
        geometry=[Point(x, y) for x, y, _ in pts],
        crs="EPSG:4326"
    )
    return gdf

# 2) remove pop points inside coverage of active BTS
def remove_covered_by_active_bts(pop_gdf, active_bts_df):
    if active_bts_df is None or active_bts_df.empty:
        return pop_gdf
    # create buffers in metric CRS
    pop_proj = pop_gdf.to_crs(epsg=3857)
    bts_gdf = gpd.GeoDataFrame(active_bts_df.copy(), geometry=gpd.points_from_xy(active_bts_df.longitude, active_bts_df.latitude), crs="EPSG:4326")
    bts_proj = bts_gdf.to_crs(epsg=3857)
    # ensure coverage_radius_m column exists
    if "coverage_radius_m" not in bts_proj.columns:
        bts_proj["coverage_radius_m"] = 3000.0
    buffers = [pt.buffer(float(r)) for pt, r in zip(bts_proj.geometry, bts_proj.coverage_radius_m)]
    union_buf = unary_union(buffers) if buffers else None
    if union_buf is None or union_buf.is_empty:
        return pop_gdf
    # mask out points within union_buf
    mask = pop_proj.geometry.within(union_buf)
    pop_keep = pop_gdf.loc[~mask.values].reset_index(drop=True)
    return pop_keep

# 3) cluster pop points with DBSCAN (eps in meters). Return labels and centroids projected back to lat/lon.
def cluster_population_dbscan(pop_gdf, eps_m=3000, min_samples=5):
    if len(pop_gdf) == 0:
        return pd.DataFrame()
    pop_proj = pop_gdf.to_crs(epsg=3857)
    coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    db = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit(coords)
    labels = db.labels_
    pop_proj["cluster"] = labels
    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        sub = pop_proj[pop_proj.cluster == lab]
        # centroid in metric CRS
        centroid_x = float(sub.geometry.x.mean())
        centroid_y = float(sub.geometry.y.mean())
        # back to latlon
        centroid_point = gpd.GeoSeries([Point(centroid_x, centroid_y)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        clusters.append({
            "cluster": int(lab),
            "centroid_lon": centroid_point.x,
            "centroid_lat": centroid_point.y,
            "n_points": len(sub),
            "pop_cells_index": sub.index.to_list()
        })
    return pd.DataFrame(clusters), pop_proj

# 4) select clusters greedily ensuring coverage non-overlap (count pop of uncovered cells)
def select_clusters_nonoverlap(clusters_df, pop_proj, radius_m=3000):
    # maintain a boolean mask for covered pop cell indices
    covered_idx = set()
    selected = []
    # precompute mapping from cluster -> pop sum (all cells)
    # for _, row in clusters_df.iterrows():
    #     idxs = row.pop_cells_index
    #     total_pop = float(pop_proj.loc[idxs, "pop"].sum())
    #     row["total_pop"] = total_pop

    clusters_df = clusters_df.copy()
    clusters_df["total_pop"] = 0.0
    for idx, row in clusters_df.iterrows():
        idxs = row["pop_cells_index"]
        clusters_df.at[idx, "total_pop"] = float(pop_proj.loc[idxs, "pop"].sum())

    # sort clusters by total_pop desc
    clusters_sorted = clusters_df.sort_values("total_pop", ascending=False).to_dict("records")

    for c in clusters_sorted:
        idxs = c["pop_cells_index"]
        # compute population among uncovered cells
        uncovered = [i for i in idxs if i not in covered_idx]
        if not uncovered:
            continue
        uncovered_pop = float(pop_proj.loc[uncovered, "pop"].sum())
        if uncovered_pop <= 0:
            continue

        # accept this cluster, mark its uncovered cells covered
        selected.append({
            "site_id": f"I_{len(selected):05d}",
            "latitude": float(c["centroid_lat"]),
            "longitude": float(c["centroid_lon"]),
            "pop": uncovered_pop,
            "covered_cells": uncovered  # keep for later mapping
        })
        for i in uncovered:
            covered_idx.add(i)
    return pd.DataFrame(selected), covered_idx

# 5) add priority weights by proximity to infra layers (if available)
def assign_priority_to_I(I_df, infra_paths, buffer_m=1500):
    """
    infra_paths: dict with keys like 'schools','hospitals',... values are file paths (geojson).
    Returns I_df with priority_category, priority_weight
    """
    I_gdf = gpd.GeoDataFrame(I_df.copy(), geometry=gpd.points_from_xy(I_df.longitude, I_df.latitude), crs="EPSG:4326")
    I_gdf = I_gdf.to_crs(epsg=3857)
    I_gdf["priority_weight"] = 1.0
    I_gdf["priority_category"] = "normal"

    for name, path in infra_paths.items():
        if not path:
            continue
        try:
            infra = read_geojson(path)
        except Exception:
            infra = None
        if infra is None or infra.empty:
            continue
        try:
            infra_proj = infra.to_crs(epsg=3857)
            infra_union = infra_proj.unary_union
        except Exception:
            infra_proj = infra
            infra_union = infra.unary_union
        for idx, row in I_gdf.iterrows():
            try:
                d = float(row.geometry.distance(infra_union))
            except Exception:
                d = np.inf
            if np.isfinite(d) and d < buffer_m:
                add = {
                    'schools': 2.0,
                    'hospitals': 3.0,
                    'medical_centers': 2.5,
                    'industrial': 1.5,
                    'residential': 1.2,
                    'command_centers': 3.0
                }.get(name, 1.0)
                I_gdf.at[idx, "priority_weight"] = float(I_gdf.at[idx, "priority_weight"]) + add
                I_gdf.at[idx, "priority_category"] = name
    # back to lat/lon
    I_df_out = pd.DataFrame({
        "site_id": I_gdf["site_id"].astype(str),
        "latitude": I_gdf.to_crs(epsg=4326).geometry.y.astype(float),
        "longitude": I_gdf.to_crs(epsg=4326).geometry.x.astype(float),
        "pop": I_gdf["pop"].astype(float),
        "priority_category": I_gdf["priority_category"].astype(str),
        "priority_weight": I_gdf["priority_weight"].astype(float)
    })
    return I_df_out

# 6) Generate candidate sites J: around each selected I and sampling along roads
def generate_J_candidates(I_df, roads_gdf, water_gdf, slope_tif,
                          candidate_per_cluster=5, jitter_m=300, slope_threshold=15, road_buffer_m=2000, n_samples_global=2000, seed=42):
    """
    For each I, create candidate_per_cluster points by jitter on road or near I. Also global random sampling near roads.
    Returns DataFrame J with lat/lon, slope, dist_to_road_m, in_water
    """
    random.seed(seed)
    np.random.seed(seed)
    J = []
    # prepare roads and water in metric for distance calculation
    try:
        roads_proj = roads_gdf.to_crs(epsg=3857)
    except Exception:
        roads_proj = roads_gdf.copy()
    try:
        water_proj = water_gdf.to_crs(epsg=3857)
    except Exception:
        water_proj = water_gdf.copy()

    # load slope raster once
    slope_src = rasterio.open(slope_tif)

    # helper: get distance to roads
    def dist_to_roads_m(lon, lat):
        try:
            pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            return float(roads_proj.distance(pt).min())
        except Exception:
            return float(np.inf)

    def point_in_water(lon, lat):
        try:
            pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            return bool(water_proj.contains(pt).any())
        except Exception:
            return False

    def slope_at(lon, lat):
        try:
            val = list(slope_src.sample([(lon, lat)]))[0][0]
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    # per cluster candidates: jitter around I prefer near roads
    for _, row in I_df.iterrows():
        base_lon, base_lat = float(row.longitude), float(row.latitude)
        # attempt to find points on nearby roads: sample nearest road geometry if available
        candidates_found = 0
        # try to find points along roads within 2km by buffering roads and sampling intersection
        if len(roads_proj) > 0:
            try:
                # buffer roads by 2000m and intersect with circle around I
                pt3857 = gpd.GeoSeries([Point(base_lon, base_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
                buf = pt3857.buffer(2000)
                # find roads within that buffer
                rsel = roads_proj[roads_proj.intersects(buf)]
                # sample points along these road geometries
                for geom in rsel.geometry:
                    # break geometry into many points and pick some
                    xs = []
                    try:
                        xs = list(geom.coords)
                    except Exception:
                        try:
                            xs = list(geom[0].coords)
                        except Exception:
                            xs = []
                    if not xs:
                        continue
                    # choose up to 2 points from coords
                    for _ in range(2):
                        c = xs[np.random.randint(0, len(xs))]
                        # transform back to lon/lat
                        p_geo = gpd.GeoSeries([Point(c)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
                        lon_c, lat_c = p_geo.x, p_geo.y
                        s_val = slope_at(lon_c, lat_c)
                        droad = dist_to_roads_m(lon_c, lat_c)
                        in_w = point_in_water(lon_c, lat_c)
                        if s_val <= slope_threshold and droad <= road_buffer_m and (not in_w):
                            J.append({"latitude": float(lat_c), "longitude": float(lon_c),
                                      "slope": float(s_val), "dist_to_road_m": float(droad), "in_water": bool(in_w)})
                            candidates_found += 1
                            if candidates_found >= candidate_per_cluster:
                                break
                    if candidates_found >= candidate_per_cluster:
                        break
            except Exception:
                pass

        # if not enough candidates found, jitter around base point (on ground)
        attempts = 0
        while candidates_found < candidate_per_cluster and attempts < candidate_per_cluster * 6:
            # jitter in meters -> convert to degrees approx (works small distances)
            dx = np.random.normal(scale=jitter_m)  # meters
            dy = np.random.normal(scale=jitter_m)
            # transform base to metric coords, add dx/dy, transform back
            try:
                base_pt_3857 = gpd.GeoSeries([Point(base_lon, base_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
                new_x = base_pt_3857.x + dx
                new_y = base_pt_3857.y + dy
                p_geo = gpd.GeoSeries([Point(new_x, new_y)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
                lon_c, lat_c = float(p_geo.x), float(p_geo.y)
            except Exception:
                lon_c = base_lon + (dx / 111000.0)
                lat_c = base_lat + (dy / 111000.0)
            s_val = slope_at(lon_c, lat_c)
            droad = dist_to_roads_m(lon_c, lat_c)
            in_w = point_in_water(lon_c, lat_c)
            if s_val <= slope_threshold and droad <= road_buffer_m and (not in_w):
                J.append({"latitude": float(lat_c), "longitude": float(lon_c),
                          "slope": float(s_val), "dist_to_road_m": float(droad), "in_water": bool(in_w)})
                candidates_found += 1
            attempts += 1

    # global sampling near roads to ensure spread
    # sample equally along roads bounding box
    if len(roads_proj) > 0 and len(J) < n_samples_global:
        # sample many points from roads geometries
        road_geoms = roads_proj.geometry.tolist()
        for geom in road_geoms:
            # get coordinates list and sample randomly a few
            try:
                coords = list(geom.coords)
            except Exception:
                try:
                    coords = list(geom[0].coords)
                except Exception:
                    coords = []
            for _ in range(2):
                if not coords:
                    break
                c = coords[np.random.randint(0, len(coords))]
                p_geo = gpd.GeoSeries([Point(c)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
                lon_c, lat_c = float(p_geo.x), float(p_geo.y)
                s_val = slope_at(lon_c, lat_c)
                droad = dist_to_roads_m(lon_c, lat_c)
                in_w = point_in_water(lon_c, lat_c)
                if s_val <= slope_threshold and (not in_w):
                    J.append({"latitude": float(lat_c), "longitude": float(lon_c),
                              "slope": float(s_val), "dist_to_road_m": float(droad), "in_water": bool(in_w)})
                if len(J) >= n_samples_global:
                    break
            if len(J) >= n_samples_global:
                break

    slope_src.close()
    # deduplicate points by rounding coords
    if not J:
        return pd.DataFrame()
    J_df = pd.DataFrame(J)
    J_df["lat_round"] = J_df["latitude"].round(6)
    J_df["lon_round"] = J_df["longitude"].round(6)
    J_df = J_df.drop_duplicates(subset=["lat_round", "lon_round"]).drop(columns=["lat_round", "lon_round"])
    return J_df.reset_index(drop=True)

###

# # ============================================================
# # 6) Generate candidate sites J: optimized with STRtree
# # ============================================================
# from shapely.strtree import STRtree
#
# def generate_J_candidates(I_df, roads_gdf, water_gdf, slope_tif,
#                           candidate_per_cluster=5, jitter_m=300,
#                           slope_threshold=15, road_buffer_m=2000,
#                           n_samples_global=1000, seed=42):
#     """
#     Optimized version:
#     - Uses STRtree spatial index for fast road proximity queries.
#     - Minimizes geometry intersections.
#     - Preserves realistic distribution of J (near I and along roads).
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#
#     J = []
#     if I_df.empty or roads_gdf.empty:
#         return pd.DataFrame()
#
#     # Prepare data
#     roads_proj = roads_gdf.to_crs(epsg=3857)
#     water_proj = water_gdf.to_crs(epsg=3857)
#     slope_src = rasterio.open(slope_tif)
#
#     # Spatial index for fast querying
#     road_geoms = list(roads_proj.geometry)
#     road_tree = STRtree(road_geoms)
#
#     def slope_at(lon, lat):
#         try:
#             val = list(slope_src.sample([(lon, lat)]))[0][0]
#             return float(val) if val is not None else 0.0
#         except Exception:
#             return 0.0
#
#     def dist_to_road(pt_3857):
#         try:
#             dmin = np.min([pt_3857.distance(g) for g in road_tree.query(pt_3857.buffer(road_buffer_m))])
#             return float(dmin)
#         except Exception:
#             return float('inf')
#
#     def in_water(pt_3857):
#         try:
#             return water_proj.intersects(pt_3857).any()
#         except Exception:
#             return False
#
#     # Loop over demand centroids I
#     for _, row in I_df.iterrows():
#         base_lon, base_lat = float(row.longitude), float(row.latitude)
#         pt3857 = gpd.GeoSeries([Point(base_lon, base_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
#
#         # Find nearby roads quickly using spatial index
#         nearby_roads = road_tree.query(pt3857.buffer(road_buffer_m))
#         road_candidates = []
#         for geom in nearby_roads:
#             try:
#                 coords = list(geom.coords)
#             except Exception:
#                 coords = []
#             if not coords:
#                 continue
#             # sample 1–2 points from each nearby road
#             for _ in range(min(2, len(coords))):
#                 cx, cy = coords[np.random.randint(0, len(coords))]
#                 p_geo = gpd.GeoSeries([Point(cx, cy)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
#                 road_candidates.append((p_geo.x, p_geo.y))
#
#         # Filter road candidates
#         road_candidates_checked = 0
#         for lon_c, lat_c in road_candidates:
#             s_val = slope_at(lon_c, lat_c)
#             pt_c_3857 = gpd.GeoSeries([Point(lon_c, lat_c)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
#             if s_val <= slope_threshold and not in_water(pt_c_3857):
#                 J.append({
#                     "latitude": lat_c, "longitude": lon_c,
#                     "slope": s_val, "dist_to_road_m": dist_to_road(pt_c_3857),
#                     "in_water": False
#                 })
#                 road_candidates_checked += 1
#                 if road_candidates_checked >= candidate_per_cluster:
#                     break
#
#         # If not enough points, add jitter points near base
#         while road_candidates_checked < candidate_per_cluster:
#             dx = np.random.normal(scale=jitter_m)
#             dy = np.random.normal(scale=jitter_m)
#             new_x = pt3857.x + dx
#             new_y = pt3857.y + dy
#             p_geo = gpd.GeoSeries([Point(new_x, new_y)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
#             lon_c, lat_c = p_geo.x, p_geo.y
#             s_val = slope_at(lon_c, lat_c)
#             pt_c_3857 = Point(new_x, new_y)
#             if s_val <= slope_threshold and not in_water(pt_c_3857):
#                 J.append({
#                     "latitude": lat_c, "longitude": lon_c,
#                     "slope": s_val, "dist_to_road_m": dist_to_road(pt_c_3857),
#                     "in_water": False
#                 })
#                 road_candidates_checked += 1
#
#     # Add small global sample (optional)
#     if len(J) < n_samples_global:
#         print(f"ℹ️ Adding {n_samples_global - len(J)} global road samples for balance...")
#         all_coords = []
#         for geom in road_geoms:
#             try:
#                 coords = list(geom.coords)
#             except Exception:
#                 coords = []
#             if coords:
#                 all_coords.extend(random.sample(coords, min(3, len(coords))))
#             if len(all_coords) >= n_samples_global:
#                 break
#
#         for cx, cy in all_coords[:n_samples_global]:
#             p_geo = gpd.GeoSeries([Point(cx, cy)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
#             lon_c, lat_c = p_geo.x, p_geo.y
#             s_val = slope_at(lon_c, lat_c)
#             pt_c_3857 = Point(cx, cy)
#             if s_val <= slope_threshold and not in_water(pt_c_3857):
#                 J.append({
#                     "latitude": lat_c, "longitude": lon_c,
#                     "slope": s_val, "dist_to_road_m": dist_to_road(pt_c_3857),
#                     "in_water": False
#                 })
#
#     slope_src.close()
#     if not J:
#         return pd.DataFrame()
#
#     # Deduplicate and finalize
#     J_df = pd.DataFrame(J).drop_duplicates(subset=["latitude", "longitude"])
#     J_df.reset_index(drop=True, inplace=True)
#     return J_df

###

# 7) evaluate candidate pop coverage for each J (non-overlapping greedy selection)
def evaluate_candidates_and_reduce_overlap(I_df, J_df, pop_gdf, radius_m=3000):
    """
    For each J candidate compute pop within radius_m (from pop_gdf). Then select J greedily to maximize uncovered pop.
    Returns J_selected dataframe with fields and assigned i_ref (nearest I).
    """
    if J_df is None or J_df.empty:
        return pd.DataFrame()
    if pop_gdf is None or pop_gdf.empty:
        # still return J with zeros
        J_df["pop"] = 0.0
        J_df["priority_weight"] = 1.0
        J_df["priority_category"] = "normal"
        J_df["site_id"] = [f"J_{i:05d}" for i in range(len(J_df))]
        J_df["i_ref"] = np.random.choice(I_df["site_id"].values, size=len(J_df))
        return J_df

    # project pop_gdf to metric for distance calculation
    pop_proj = pop_gdf.to_crs(epsg=3857)
    J_proj_pts = gpd.GeoDataFrame(J_df.copy(), geometry=gpd.points_from_xy(J_df.longitude, J_df.latitude), crs="EPSG:4326").to_crs(epsg=3857)
    # precompute pop cell coords and pop values
    pop_coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    pop_vals = pop_proj["pop"].values
    n_pop = len(pop_proj)
    covered_mask = np.zeros(n_pop, dtype=bool)
    # compute for each J the indices of pop cells within radius
    J_pop_idx = []
    for _, row in J_proj_pts.iterrows():
        cx, cy = row.geometry.x, row.geometry.y
        dists = np.sqrt((pop_coords[:, 0] - cx)**2 + (pop_coords[:, 1] - cy)**2)
        idxs = np.where(dists <= float(radius_m))[0].tolist()
        J_pop_idx.append(idxs)
    # greedy: order J by total pop descending
    total_pops = [float(pop_vals[idxs].sum()) if idxs else 0.0 for idxs in J_pop_idx]
    order = np.argsort(total_pops)[::-1]
    selected_idxs = []
    for idx in order:
        idxs = J_pop_idx[idx]
        if not idxs:
            continue
        # uncovered pop amount
        uncovered = [i for i in idxs if not covered_mask[i]]
        if not uncovered:
            continue
        unc_pop = float(pop_vals[uncovered].sum())
        if unc_pop <= 0:
            continue
        # select this J
        selected_idxs.append(idx)
        for i in uncovered:
            covered_mask[i] = True

    # Build selected J_df
    if not selected_idxs:
        # no selection -> still return top N by total_pops
        selected_idxs = list(np.argsort(total_pops)[::-1][:min(50, len(J_df))])

    sel = J_df.iloc[selected_idxs].copy().reset_index(drop=True)
    # assign site ids and map nearest I
    sel["site_id"] = [f"J_{i:05d}" for i in range(len(sel))]
    # nearest I mapping by haversine
    I_coords = I_df[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
    J_coords = sel[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
    if len(I_coords) > 0 and len(J_coords) > 0:
        dist_IJ = compute_distance_matrix(I_coords, J_coords, metric='haversine')
        nearest = np.argmin(dist_IJ, axis=0)
        sel["i_ref"] = [str(I_df.loc[int(i), "site_id"]) for i in nearest]
        sel["pop"] = [float(pop_vals[J_pop_idx[selected_idxs[k]]].sum()) if J_pop_idx[selected_idxs[k]] else 0.0 for k in range(len(selected_idxs))]
        # inherit priority from assigned I
        sel["priority_weight"] = [float(I_df.loc[int(i), "priority_weight"]) for i in nearest]
        sel["priority_category"] = [str(I_df.loc[int(i), "priority_category"]) for i in nearest]
    else:
        sel["i_ref"] = [None]*len(sel)
        sel["pop"] = 0.0
        sel["priority_weight"] = 1.0
        sel["priority_category"] = "normal"

    # ensure columns expected downstream
    sel = sel[['site_id','i_ref','latitude','longitude','pop','priority_category','priority_weight','slope','dist_to_road_m','in_water']]

    return sel

# MAIN
def main(config, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parameters (use config when available)
    pop_threshold = float(config.get('pop_threshold', 10))
    eps_dbscan_m = float(config.get('eps_dbscan_m', 3000))
    db_min_samples = int(config.get('db_min_samples', 5))
    radius_I_m = float(config.get('default_R', 3000))
    candidate_per_cluster = int(config.get('candidate_per_cluster', 5))
    jitter_m = float(config.get('candidate_jitter_m', 300))
    slope_threshold = float(config.get('slope_threshold_deg', 15))
    road_buffer_m = float(config.get('road_buffer_m', 2000))
    candidate_samples_global = int(config.get('candidate_samples_global', 1000))

    # load base datasets
    boundary = gpd.read_file(CLEANED_DIR / "hue_boundary_clean.geojson")
    pop_tif = str(CLEANED_DIR / "pop_hue_clean.tif")
    slope_tif = str(CLEANED_DIR / "slope_hue_clean.tif")
    roads = read_geojson(str(CLEANED_DIR / "roads_hue_clean.geojson"))
    water = read_geojson(str(CLEANED_DIR / "water_hue_clean.geojson"))
    active_bts, failed_bts = read_bts_files()

    print("Extract population cells from raster...")
    pop_gdf = extract_population_cells(pop_tif, threshold=pop_threshold)
    if pop_gdf.empty:
        print("No population cells found above threshold.")
    else:
        # clip to boundary
        try:
            pop_gdf = gpd.clip(pop_gdf, boundary)
        except Exception:
            pass

    print("Remove pop points inside coverage of active BTS...")
    pop_gdf_uncovered = remove_covered_by_active_bts(pop_gdf, active_bts)

    print(f"   population cells before: {len(pop_gdf)}, after removing active BTS coverage: {len(pop_gdf_uncovered)}")

    # cluster uncovered pop
    print("Cluster population points with DBSCAN (eps_m=%s)..." % eps_dbscan_m)
    clusters_df, pop_proj = cluster_population_dbscan(pop_gdf_uncovered, eps_m=eps_dbscan_m, min_samples=db_min_samples)
    if clusters_df is None or clusters_df.empty:
        print("No clusters found; fallback to forming simple grid of demand points.")
        # fallback simple grid sampling of remaining pop
        # pick top N pop cells as I
        pop_sorted = pop_gdf_uncovered.sort_values("pop", ascending=False).head(500)
        I_list = []
        for i, row in pop_sorted.iterrows():
            I_list.append({
                "site_id": f"I_{len(I_list):05d}",
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "pop": float(row.pop),
                "covered_cells": [i]
            })
        I_df = pd.DataFrame(I_list)
    else:
        # select clusters greedily for non-overlap using radius_I_m
        print("Select clusters greedily to avoid overlap and compute I pop (radius_m=%s)..." % radius_I_m)
        # cluster selection uses pop_proj (projected pop cells)
        selected_clusters, covered_idx = select_clusters_nonoverlap(clusters_df, pop_proj, radius_m=radius_I_m)
        # build I_df
        I_df = pd.DataFrame([{
            "site_id": row["site_id"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "pop": float(row["pop"]),
            "covered_cells": row["covered_cells"]
        } for _, row in selected_clusters.reset_index(drop=True).iterrows()])

    # add priority by infra (prefer local raw OSM files if present)
    print("Attempt to assign priority categories using available infra layers (local or OSM)...")
    # build infra sources dict from raw folder if exist
    infra_files = {}
    for name in ["schools_clean", "hospitals_clean", "medical_centers_clean", "industrial_clean", "residential_clean",
                 "command_centers_clean"]:
        p = CLEANED_DIR / f"{name}.geojson"
        if p.exists():
            infra_files[name] = str(p)
    # if not present, try to use osm via read_geojson wrapper or skip (read_geojson may call gpd.read_file)
    # assign weights
    I_df = assign_priority_to_I(I_df, infra_files, buffer_m=config.get('infra_buffer_m', 1500))

    # 6) Generate J candidates
    print("Generating J candidate sites around I and along roads...")
    J_raw = generate_J_candidates(I_df, roads, water, slope_tif,
                                  candidate_per_cluster=candidate_per_cluster,
                                  jitter_m=jitter_m,
                                  slope_threshold=slope_threshold,
                                  road_buffer_m=road_buffer_m,
                                  n_samples_global=candidate_samples_global,
                                  seed=int(config.get('seed', 42)))
    if J_raw is None or J_raw.empty:
        print("No J candidates produced. Exiting feature extraction with current I only.")
        J_df = pd.DataFrame()
    else:
        # evaluate candidate coverage and select J subset greedily non-overlap
        print("Evaluate J candidates and select set to maximize uncovered pop (non-overlap)...")
        J_df = evaluate_candidates_and_reduce_overlap(I_df, J_raw, pop_gdf_uncovered, radius_m=radius_I_m)

    # ensure columns/order expected downstream
    # I_points.csv: site_id, latitude, longitude, pop, priority_category, priority_weight
    I_out = pd.DataFrame({
        "site_id": I_df["site_id"].astype(str),
        "latitude": I_df["latitude"].astype(float),
        "longitude": I_df["longitude"].astype(float),
        "pop": I_df["pop"].astype(float),
    })
    # try to preserve priority info from earlier step (assign_priority_to_I returned such)
    if "priority_category" in I_df.columns:
        I_out["priority_category"] = I_df["priority_category"].astype(str)
    else:
        I_out["priority_category"] = "normal"
    if "priority_weight" in I_df.columns:
        I_out["priority_weight"] = I_df["priority_weight"].astype(float)
    else:
        I_out["priority_weight"] = 1.0

    # J_sites.csv expected columns: site_id_ref or i_ref, latitude, longitude, pop, priority_category, priority_weight, slope, dist_to_road_m, in_water
    if J_df is None or J_df.empty:
        J_out = pd.DataFrame(columns=["site_id","i_ref","latitude","longitude","pop","priority_category","priority_weight","slope","dist_to_road_m","in_water"])
    else:
        J_out = J_df.copy()
        # ensure types
        for c in ["latitude","longitude","pop","priority_weight","slope","dist_to_road_m"]:
            if c in J_out.columns:
                J_out[c] = J_out[c].astype(float)
        # fill missing
        if "priority_category" not in J_out.columns:
            J_out["priority_category"] = "normal"
        if "priority_weight" not in J_out.columns:
            J_out["priority_weight"] = 1.0
        if "i_ref" not in J_out.columns:
            J_out["i_ref"] = np.nan
        # reorder columns
        cols = ["site_id","i_ref","latitude","longitude","pop","priority_category","priority_weight","slope","dist_to_road_m","in_water"]
        J_out = J_out[cols]

    # save outputs
    I_out.to_csv(Path(out_dir) / "I_points_A.csv", index=False)
    J_out.to_csv(Path(out_dir) / "J_sites_A.csv", index=False)

    # compute cover matrix (I x J) for downstream pipeline; must be shape (N, M, K) in solver_milp expects K (COWs) later,
    # But earlier pipeline used cover.npy shape (N,M) or (N,M,K) depending - original expected cover.npy of shape (N,M,K)
    # We'll save (N,M) for now to match later reading code that recomputes cover internally.
    if len(J_out) > 0 and len(I_out) > 0:
        I_coords = I_out[['latitude','longitude']].rename(columns={'latitude':'y','longitude':'x'}).to_dict('records')
        J_coords = J_out[['latitude','longitude']].rename(columns={'latitude':'y','longitude':'x'}).to_dict('records')
        dist = compute_distance_matrix(I_coords, J_coords, metric='haversine')
        cover = (dist <= float(radius_I_m)).astype(int)
    else:
        cover = np.zeros((len(I_out), len(J_out)), dtype=int)
    np.save(Path(out_dir) / "cover.npy", cover)

    print(f"[Feature Extraction] Saved {len(I_out)} I_points, {len(J_out)} J_sites, and cover.npy to {out_dir}")
