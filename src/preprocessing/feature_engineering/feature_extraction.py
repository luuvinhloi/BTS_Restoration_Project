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
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
import math
import random
import warnings
from scipy.spatial import cKDTree

from src.utils.geo_utils import compute_distance_matrix
from src.utils.io_utils import read_geojson

# PATHS
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# Helper: read CSV for active/failed BTS
def read_bts_files():
    active_path = PROCESSED_DIR / "active_bts.csv"
    failed_path = PROCESSED_DIR / "failed_bts.csv"
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
    bts_gdf = gpd.GeoDataFrame(active_bts_df.copy(),
                               geometry=gpd.points_from_xy(active_bts_df.longitude, active_bts_df.latitude),
                               crs="EPSG:4326")
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
        return pd.DataFrame(), pop_gdf
    pop_proj = pop_gdf.to_crs(epsg=3857)
    coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    if len(coords) == 0:
        return pd.DataFrame(), pop_proj
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

    if clusters_df is None or clusters_df.empty:
        return pd.DataFrame(), covered_idx

    # precompute mapping from cluster -> pop sum (all cells)
    clusters_df = clusters_df.copy()
    clusters_df["total_pop"] = 0.0
    for idx, row in clusters_df.iterrows():
        idxs = row["pop_cells_index"]
        clusters_df.at[idx, "total_pop"] = float(pop_proj.loc[idxs, "pop"].sum()) if len(idxs) > 0 else 0.0

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
    if I_df is None or I_df.empty:
        return I_df

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
                          candidate_per_cluster=15, jitter_m=300, slope_threshold=25,
                          road_buffer_m=3000, n_samples_global=5000, seed=42,
                          dedup_round=5):
    """
    For each I, create candidate_per_cluster points by:
      - sampling along roads (interpolation along line segments) within buffer
      - jitter around I if road-based points insufficient
    Also global random sampling near roads.
    Returns DataFrame J with lat/lon, slope, dist_to_road_m, in_water
    dedup_round: decimal places to round lat/lon for deduplication (None to disable)
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

    # build KDTree of road vertices in projected coordinates for quick dist-to-road
    road_pts = []
    road_pts_map = []  # keep mapping to original coords in 3857
    for geom in roads_proj.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            for c in coords:
                road_pts.append((c[0], c[1]))
                road_pts_map.append(c)
        elif geom.geom_type == "MultiLineString":
            for line in geom:
                coords = list(line.coords)
                for c in coords:
                    road_pts.append((c[0], c[1]))
                    road_pts_map.append(c)
    if len(road_pts) > 0:
        road_kdtree = cKDTree(np.array(road_pts))
    else:
        road_kdtree = None

    # load slope raster once
    slope_src = rasterio.open(slope_tif)

    # helper: get distance to roads using kdtree in meters (both are in EPSG:3857)
    def dist_to_roads_m(lon, lat):
        try:
            # lon,lat are in WGS84, transform to 3857
            pt3857 = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            if road_kdtree is None:
                # fallback to measuring distance to whole roads_proj unary_union
                return float(roads_proj.distance(pt3857).min()) if len(roads_proj) > 0 else float(np.inf)
            dist, idx = road_kdtree.query([(pt3857.x, pt3857.y)], k=1)
            return float(dist[0])
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
            pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(slope_src.crs)
            val = list(slope_src.sample([(float(pt.geometry.x.iloc[0]), float(pt.geometry.y.iloc[0]))]))[0][0]
            return float(max(0, min(90, val))) if np.isfinite(val) else 0.0
        except Exception:
            return 0.0

    # function: sample a random point along a LineString (by linear interpolation)
    def sample_point_along_linestring(line: LineString):
        coords = list(line.coords)
        if len(coords) < 2:
            return None
        # pick random segment, then random fraction along it
        seg_idx = np.random.randint(0, len(coords) - 1)
        x1, y1 = coords[seg_idx]
        x2, y2 = coords[seg_idx + 1]
        frac = np.random.rand()
        xi = x1 + frac * (x2 - x1)
        yi = y1 + frac * (y2 - y1)
        return (xi, yi)  # in projected coords

    # per cluster candidates: sample along roads inside buffer (via interpolation) and jitter if needed
    for _, row in I_df.iterrows():
        base_lon, base_lat = float(row.longitude), float(row.latitude)
        candidates_found = 0
        # try to find points along roads within 2km by buffering roads and sampling
        if len(roads_proj) > 0:
            try:
                pt3857 = gpd.GeoSeries([Point(base_lon, base_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
                buf = pt3857.buffer(2000)
                rsel = roads_proj[roads_proj.intersects(buf)]
                # sample by interpolation along selected LineStrings
                for geom in rsel.geometry:
                    lines = [geom] if geom.geom_type == "LineString" else list(geom)
                    for line in lines:
                        for _ in range(5):
                            sampled = sample_point_along_linestring(line)
                            if sampled is None:
                                continue
                            # sampled is in 3857; convert to latlon
                            try:
                                p_geo = gpd.GeoSeries([Point(sampled)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
                                lon_c, lat_c = p_geo.x, p_geo.y
                            except Exception:
                                continue
                            s_val = slope_at(lon_c, lat_c)
                            droad = dist_to_roads_m(lon_c, lat_c)
                            in_w = point_in_water(lon_c, lat_c)
                            if s_val <= slope_threshold and droad <= road_buffer_m and (not in_w):
                                J.append({
                                    "latitude": float(lat_c),
                                    "longitude": float(lon_c),
                                    "slope": float(s_val),
                                    "dist_to_road_m": float(droad),
                                    "in_water": bool(in_w)
                                })
                                candidates_found += 1
                                if candidates_found >= candidate_per_cluster:
                                    break
                        if candidates_found >= candidate_per_cluster:
                            break
                    if candidates_found >= candidate_per_cluster:
                        break
            except Exception:
                pass

        # if not enough candidates found, jitter around base point (on ground)
        attempts = 0
        while candidates_found < candidate_per_cluster and attempts < candidate_per_cluster * 8:
            dx = np.random.normal(scale=jitter_m)
            dy = np.random.normal(scale=jitter_m)
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
                J.append({
                    "latitude": float(lat_c),
                    "longitude": float(lon_c),
                    "slope": float(s_val),
                    "dist_to_road_m": float(droad),
                    "in_water": bool(in_w)
                })
                candidates_found += 1
            attempts += 1

    # global sampling near roads (interpolate on random road segments)
    if len(roads_proj) > 0 and len(J) < n_samples_global:
        road_geoms = roads_proj.geometry.tolist()
        for geom in road_geoms:
            lines = [geom] if geom.geom_type == "LineString" else list(geom)
            for line in lines:
                for _ in range(3):
                    sampled = sample_point_along_linestring(line)
                    if sampled is None:
                        continue
                    try:
                        p_geo = gpd.GeoSeries([Point(sampled)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
                        lon_c, lat_c = float(p_geo.x), float(p_geo.y)
                    except Exception:
                        continue
                    s_val = slope_at(lon_c, lat_c)
                    droad = dist_to_roads_m(lon_c, lat_c)
                    in_w = point_in_water(lon_c, lat_c)
                    if s_val <= slope_threshold and (not in_w):
                        J.append({
                            "latitude": float(lat_c),
                            "longitude": float(lon_c),
                            "slope": float(s_val),
                            "dist_to_road_m": float(droad),
                            "in_water": bool(in_w)
                        })
                    if len(J) >= n_samples_global:
                        break
                if len(J) >= n_samples_global:
                    break
            if len(J) >= n_samples_global:
                break

    slope_src.close()
    if not J:
        return pd.DataFrame()
    J_df = pd.DataFrame(J)

    # Deduplicate: reduce rounding sensitivity (configurable)
    if dedup_round is None:
        # drop exact duplicates only
        J_df = J_df.drop_duplicates()
    else:
        J_df["lat_round"] = J_df["latitude"].round(dedup_round)
        J_df["lon_round"] = J_df["longitude"].round(dedup_round)
        J_df = J_df.drop_duplicates(subset=["lat_round", "lon_round"]).drop(columns=["lat_round", "lon_round"])
    return J_df.reset_index(drop=True)


# 7) evaluate candidate pop coverage for each J (non-overlapping greedy selection)
def evaluate_candidates_and_reduce_overlap(I_df, J_df, pop_gdf, radius_m=3000, overlap_keep_threshold=0.1, max_J=1200):
    """
    For each J candidate compute pop within radius_m (from pop_gdf).
    Then select J greedily to maximize uncovered pop.
    overlap_keep_threshold: fraction of uncovered/total that we require to accept a J (smaller -> more J kept)
    max_J: upper limit to select top J (used for fallback)
    Returns J_selected dataframe with fields and assigned i_ref (nearest I).
    """
    if J_df is None or J_df.empty:
        return pd.DataFrame()
    if pop_gdf is None or pop_gdf.empty:
        J_df["pop"] = 0.0
        J_df["priority_weight"] = 1.0
        J_df["priority_category"] = "normal"
        J_df["site_id"] = [f"J_{i:05d}" for i in range(len(J_df))]
        J_df["i_ref"] = np.random.choice(I_df["site_id"].values, size=len(J_df)) if (I_df is not None and not I_df.empty) else [None] * len(J_df)
        return J_df

    pop_proj = pop_gdf.to_crs(epsg=3857)
    J_proj_pts = gpd.GeoDataFrame(J_df.copy(),
                                  geometry=gpd.points_from_xy(J_df.longitude, J_df.latitude),
                                  crs="EPSG:4326").to_crs(epsg=3857)
    pop_coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    pop_vals = pop_proj["pop"].values
    n_pop = len(pop_proj)
    covered_mask = np.zeros(n_pop, dtype=bool)

    J_pop_idx = []
    for _, row in J_proj_pts.iterrows():
        cx, cy = row.geometry.x, row.geometry.y
        dists = np.sqrt((pop_coords[:, 0] - cx)**2 + (pop_coords[:, 1] - cy)**2)
        idxs = np.where(dists <= float(radius_m))[0].tolist()
        J_pop_idx.append(idxs)

    total_pops = [float(pop_vals[idxs].sum()) if idxs else 0.0 for idxs in J_pop_idx]
    order = np.argsort(total_pops)[::-1]
    selected_idxs = []
    for idx in order:
        idxs = J_pop_idx[idx]
        if not idxs:
            continue
        uncovered = [i for i in idxs if not covered_mask[i]]
        if not uncovered:
            continue
        unc_pop = float(pop_vals[uncovered].sum())
        # accept if uncovered fraction is larger than threshold (looser threshold => more J)
        frac = (len(uncovered) / len(idxs)) if len(idxs) > 0 else 0.0
        if frac < overlap_keep_threshold and unc_pop < 0.1 * pop_vals.sum():
            continue
        if unc_pop <= 0:
            continue
        selected_idxs.append(idx)
        for i in uncovered:
            covered_mask[i] = True
        # optional early stop if large enough
        if len(selected_idxs) >= max_J:
            break

    # fallback: if none selected because threshold too strict, pick top max_J by total_pops
    if not selected_idxs:
        topn = min(max_J, len(J_df))
        selected_idxs = list(np.argsort(total_pops)[::-1][:topn])

    sel = J_df.iloc[selected_idxs].copy().reset_index(drop=True)
    sel["site_id"] = [f"J_{i:05d}" for i in range(len(sel))]

    # compute nearest I for each selected J
    if I_df is None or I_df.empty:
        sel["i_ref"] = [None] * len(sel)
        sel["pop"] = 0.0
        sel["priority_weight"] = 1.0
        sel["priority_category"] = "normal"
    else:
        I_coords = I_df[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
        J_coords = sel[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
        dist_IJ = compute_distance_matrix(I_coords, J_coords, metric='haversine')
        nearest = np.argmin(dist_IJ, axis=0)
        # guard nearest indices
        i_refs = []
        pweights = []
        pcats = []
        pops = []
        for k, nidx in enumerate(nearest):
            try:
                i_ref = str(I_df.loc[int(nidx), "site_id"])
                pweight = float(I_df.loc[int(nidx), "priority_weight"]) if "priority_weight" in I_df.columns else 1.0
                pcat = str(I_df.loc[int(nidx), "priority_category"]) if "priority_category" in I_df.columns else "normal"
            except Exception:
                i_ref = None
                pweight = 1.0
                pcat = "normal"
            i_refs.append(i_ref)
            pweights.append(pweight)
            pcats.append(pcat)
            pops.append(float(pop_vals[J_pop_idx[selected_idxs[k]]].sum()) if J_pop_idx[selected_idxs[k]] else 0.0)
        sel["i_ref"] = i_refs
        sel["pop"] = pops
        sel["priority_weight"] = pweights
        sel["priority_category"] = pcats

    sel = sel[['site_id', 'i_ref', 'latitude', 'longitude', 'pop',
               'priority_category', 'priority_weight', 'slope', 'dist_to_road_m', 'in_water']]
    return sel


# MAIN
def main(config, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # defaults (tuned to produce more I/J)
    pop_threshold = float(config.get('pop_threshold', 3))            # reduce -> more raster cells
    eps_dbscan_m = float(config.get('eps_dbscan_m', 1000))           # reduce -> more clusters
    db_min_samples = int(config.get('db_min_samples', 3))
    radius_I_m = float(config.get('default_R', 3000))
    candidate_per_cluster = int(config.get('candidate_per_cluster', 15))
    jitter_m = float(config.get('candidate_jitter_m', 300))
    slope_threshold = float(config.get('slope_threshold_deg', 15))
    road_buffer_m = float(config.get('road_buffer_m', 2000))
    candidate_samples_global = int(config.get('candidate_samples_global', 5000))
    max_I = int(config.get('max_I', 1000))
    max_J = int(config.get('max_J', 1200))
    dedup_round = config.get('dedup_round', 7)
    overlap_keep_threshold = float(config.get('overlap_keep_threshold', 0.1))  # smaller -> keep more J
    seed = int(config.get('seed', 42))

    boundary = gpd.read_file(RAW_DIR / "hue_boundary.geojson")
    pop_tif = str(RAW_DIR / "pop_hue.tif")
    slope_tif = str(RAW_DIR / "slope_hue.tif")
    roads = read_geojson(str(RAW_DIR / "roads_hue.geojson"))
    water = read_geojson(str(RAW_DIR / "water_hue.geojson"))
    active_bts, failed_bts = read_bts_files()

    np.random.seed(seed)
    random.seed(seed)

    print(" Extract population cells from raster...")
    pop_gdf = extract_population_cells(pop_tif, threshold=pop_threshold)
    if not pop_gdf.empty:
        try:
            pop_gdf = gpd.clip(pop_gdf, boundary)
        except Exception:
            pass

    print(" Remove pop points inside coverage of active BTS...")
    pop_gdf_uncovered = remove_covered_by_active_bts(pop_gdf, active_bts)
    print(f" population cells before: {len(pop_gdf)}, after removing active BTS coverage: {len(pop_gdf_uncovered)}")

    print(" Cluster population points with DBSCAN (eps_m=%s)..." % eps_dbscan_m)
    clusters_df, pop_proj = cluster_population_dbscan(pop_gdf_uncovered, eps_m=eps_dbscan_m, min_samples=db_min_samples)
    if clusters_df is None or clusters_df.empty:
        print("No clusters found; fallback to forming simple grid of demand points.")
        pop_sorted = pop_gdf_uncovered.sort_values("pop", ascending=False).head(max_I)
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
        print(" Select clusters greedily to avoid overlap and compute I pop (radius_m=%s)..." % radius_I_m)
        selected_clusters, covered_idx = select_clusters_nonoverlap(clusters_df, pop_proj, radius_m=radius_I_m)
        I_df = pd.DataFrame([{
            "site_id": row["site_id"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "pop": float(row["pop"]),
            "covered_cells": row["covered_cells"]
        } for _, row in selected_clusters.reset_index(drop=True).iterrows()])

    # If resulting I is smaller than desired, expand by jittering around existing I points
    if I_df is None or I_df.empty:
        I_df = pd.DataFrame(columns=["site_id", "latitude", "longitude", "pop", "covered_cells"])
    if len(I_df) < max_I and len(I_df) > 0:
        extra_needed = max_I - len(I_df)
        jittered = []
        for k in range(extra_needed):
            base = I_df.sample(1, random_state=seed + k).iloc[0]
            # jitter in degrees approx (0.001 deg ~ 111m) use small sigma to keep near cluster
            lat_j = base["latitude"] + np.random.normal(scale=0.001)
            lon_j = base["longitude"] + np.random.normal(scale=0.001)
            jittered.append({
                "site_id": f"I_{len(I_df) + len(jittered):05d}",
                "latitude": lat_j,
                "longitude": lon_j,
                "pop": float(base.get("pop", 0.0)) * np.random.uniform(0.6, 1.2),
                "covered_cells": base.get("covered_cells", [])
            })
        I_df = pd.concat([I_df, pd.DataFrame(jittered)], ignore_index=True)

    # limit I to max_I
    if len(I_df) > max_I:
        I_df = I_df.sort_values("pop", ascending=False).head(max_I).reset_index(drop=True)

    print(" Attempt to assign priority categories using available infra layers (local or OSM)...")
    infra_files = {}
    for name in ["schools", "hospitals", "medical_centers", "industrial", "residential", "command_centers"]:
        p = RAW_DIR / f"{name}.geojson"
        if p.exists():
            infra_files[name] = str(p)
    I_df = assign_priority_to_I(I_df, infra_files, buffer_m=config.get('infra_buffer_m', 1500))

    print(" Generating J candidate sites around I and along roads...")
    J_raw = generate_J_candidates(I_df, roads, water, slope_tif,
                                  candidate_per_cluster=candidate_per_cluster,
                                  jitter_m=jitter_m,
                                  slope_threshold=slope_threshold,
                                  road_buffer_m=road_buffer_m,
                                  n_samples_global=candidate_samples_global,
                                  seed=seed,
                                  dedup_round=dedup_round)
    if J_raw is None or J_raw.empty:
        print("No J candidates produced. Exiting feature extraction with current I only.")
        J_df = pd.DataFrame()
    else:
        print(" Evaluate J candidates and select set to maximize uncovered pop (non-overlap)...")
        J_df = evaluate_candidates_and_reduce_overlap(I_df, J_raw, pop_gdf_uncovered, radius_m=radius_I_m,
                                                      overlap_keep_threshold=overlap_keep_threshold, max_J=max_J)

    # Prepare outputs with required fields and defaults
    I_out = pd.DataFrame({
        "site_id": I_df["site_id"].astype(str),
        "latitude": I_df["latitude"].astype(float),
        "longitude": I_df["longitude"].astype(float),
        "pop": I_df["pop"].astype(float),
    })
    if "priority_category" in I_df.columns:
        I_out["priority_category"] = I_df["priority_category"].astype(str)
    else:
        I_out["priority_category"] = "normal"
    if "priority_weight" in I_df.columns:
        I_out["priority_weight"] = I_df["priority_weight"].astype(float)
    else:
        I_out["priority_weight"] = 1.0

    if J_df is None or J_df.empty:
        J_out = pd.DataFrame(columns=[
            "site_id", "i_ref", "latitude", "longitude", "pop", "priority_category",
            "priority_weight", "slope", "dist_to_road_m", "in_water"
        ])
    else:
        J_out = J_df.copy()
        for c in ["latitude", "longitude", "pop", "priority_weight", "slope", "dist_to_road_m"]:
            if c in J_out.columns:
                J_out[c] = J_out[c].astype(float)
        if "priority_category" not in J_out.columns:
            J_out["priority_category"] = "normal"
        if "priority_weight" not in J_out.columns:
            J_out["priority_weight"] = 1.0
        if "i_ref" not in J_out.columns:
            J_out["i_ref"] = np.nan
        if "site_id" not in J_out.columns:
            J_out["site_id"] = [f"J_{i:05d}" for i in range(len(J_out))]
        # limit number of J to max_J to avoid extremely large outputs
        if len(J_out) > max_J:
            J_out = J_out.sort_values("pop", ascending=False).head(max_J).reset_index(drop=True)
        cols = ["site_id", "i_ref", "latitude", "longitude", "pop", "priority_category",
                "priority_weight", "slope", "dist_to_road_m", "in_water"]
        # ensure all columns exist
        for col in cols:
            if col not in J_out.columns:
                J_out[col] = np.nan
        J_out = J_out[cols]

    I_out.to_csv(Path(out_dir) / "I_points.csv", index=False)
    J_out.to_csv(Path(out_dir) / "J_sites.csv", index=False)

    if len(J_out) > 0 and len(I_out) > 0:
        I_coords = I_out[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
        J_coords = J_out[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
        dist = compute_distance_matrix(I_coords, J_coords, metric='haversine')
        cover = (dist <= float(radius_I_m)).astype(int)
    else:
        cover = np.zeros((len(I_out), len(J_out)), dtype=int)
    np.save(Path(out_dir) / "cover.npy", cover)

    print(f"[Feature Extraction] Saved {len(I_out)} I_points, {len(J_out)} J_sites, and cover.npy to {out_dir}")
