# src/preprocessing/feature_extraction_optimize_A.py
"""
Feature Extraction (DBSCAN clustering, realistic I/J generation)
    - Build I (demand points) from pop raster excluding area covered by active BTS.
    - Cluster population points with DBSCAN (project to metric CRS).
    - Create I centroids, compute pop within 3km (non-overlapping) using greedy selection.
    - Generate J candidate sites around I and along roads, filter by slope/water, distance-to-road.
    - Compute pop covered by each J (3km) and save I_points_A.csv, J_sites_A.csv, cover.npy.

Optimized (Phương án A):
 - use cKDTree for road vertex distances and for pop queries
 - read slope raster once and use dataset.index + numpy array access
 - avoid per-point GeoSeries transformations by using pyproj.Transformer
 - keep original logic / outputs but faster
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
from pyproj import Transformer

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
        # vectorized locate indices > threshold
        rows, cols = np.where(arr > threshold)
        if len(rows) == 0:
            return gpd.GeoDataFrame(columns=["longitude", "latitude", "pop", "geometry"], crs="EPSG:4326")
        # compute lon/lat for all indices quickly
        # note: transform * (col + 0.5, row + 0.5)
        xs, ys, vs = [], [], []
        for r, c in zip(rows, cols):
            val = float(arr[r, c])
            x, y = transform * (int(c) + 0.5, int(r) + 0.5)
            xs.append(x)
            ys.append(y)
            vs.append(val)
        gdf = gpd.GeoDataFrame(
            {"longitude": xs, "latitude": ys, "pop": vs},
            geometry=[Point(x, y) for x, y in zip(xs, ys)],
            crs="EPSG:4326"
        )
    return gdf


# 2) remove pop points inside coverage of active BTS
def remove_covered_by_active_bts(pop_gdf, active_bts_df):
    if active_bts_df is None or active_bts_df.empty or pop_gdf is None or pop_gdf.empty:
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
    if pop_gdf is None or len(pop_gdf) == 0:
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
        centroid_x = float(sub.geometry.x.mean())
        centroid_y = float(sub.geometry.y.mean())
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
    covered_idx = set()
    selected = []

    if clusters_df is None or clusters_df.empty:
        return pd.DataFrame(), covered_idx

    clusters_df = clusters_df.copy()
    clusters_df["total_pop"] = 0.0
    for idx, row in clusters_df.iterrows():
        idxs = row["pop_cells_index"]
        clusters_df.at[idx, "total_pop"] = float(pop_proj.loc[idxs, "pop"].sum()) if len(idxs) > 0 else 0.0

    clusters_sorted = clusters_df.sort_values("total_pop", ascending=False).to_dict("records")

    for c in clusters_sorted:
        idxs = c["pop_cells_index"]
        uncovered = [i for i in idxs if i not in covered_idx]
        if not uncovered:
            continue
        uncovered_pop = float(pop_proj.loc[uncovered, "pop"].sum())
        if uncovered_pop <= 0:
            continue
        selected.append({
            "site_id": f"I_{len(selected):05d}",
            "latitude": float(c["centroid_lat"]),
            "longitude": float(c["centroid_lon"]),
            "pop": uncovered_pop,
            "covered_cells": uncovered
        })
        for i in uncovered:
            covered_idx.add(i)
    return pd.DataFrame(selected), covered_idx


# 5) add priority weights by proximity to infra layers (if available)
def assign_priority_to_I(I_df, infra_paths, buffer_m=1500):
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
        # distance check per I point (in projected CRS)
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

    I_df_out = pd.DataFrame({
        "site_id": I_gdf["site_id"].astype(str),
        "latitude": I_gdf.to_crs(epsg=4326).geometry.y.astype(float),
        "longitude": I_gdf.to_crs(epsg=4326).geometry.x.astype(float),
        "pop": I_gdf["pop"].astype(float),
        "priority_category": I_gdf["priority_category"].astype(str),
        "priority_weight": I_gdf["priority_weight"].astype(float)
    })
    return I_df_out


# 6) Generate candidate sites J: optimized (KDTree + raster array)
def generate_J_candidates(I_df, roads_gdf, water_gdf, slope_tif,
                          candidate_per_cluster=15, jitter_m=300, slope_threshold=15,
                          road_buffer_m=2000, n_samples_global=5000, seed=42,
                          dedup_round=7):
    random.seed(seed)
    np.random.seed(seed)
    J = []

    try:
        roads_proj = roads_gdf.to_crs(epsg=3857)
    except Exception:
        roads_proj = roads_gdf.copy() if roads_gdf is not None else gpd.GeoDataFrame()

    try:
        water_proj = water_gdf.to_crs(epsg=3857)
    except Exception:
        water_proj = water_gdf.copy() if water_gdf is not None else gpd.GeoDataFrame()

    # build KDTree of road vertices (projected coords)
    road_pts = []
    for geom in roads_proj.geometry if roads_proj is not None else []:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            for c in geom.coords:
                road_pts.append((c[0], c[1]))
        elif geom.geom_type == "MultiLineString":
            for line in geom:
                for c in line.coords:
                    road_pts.append((c[0], c[1]))
    road_kdtree = cKDTree(np.array(road_pts)) if len(road_pts) > 0 else None

    # prepare water union (projected) for quick contains
    water_union = water_proj.unary_union if (water_proj is not None and len(water_proj) > 0) else None

    # load slope raster once and build transformer
    slope_val_arr = None
    slope_transform = None
    slope_crs = None
    slope_src = None
    transformer_to_slope = None
    transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_from_3857 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    try:
        slope_src = rasterio.open(slope_tif)
        slope_val_arr = slope_src.read(1)
        slope_transform = slope_src.transform
        slope_crs = slope_src.crs
        # transformer from WGS84 -> slope raster CRS for fast coordinate mapping
        transformer_to_slope = Transformer.from_crs("EPSG:4326", slope_src.crs.to_string(), always_xy=True)
    except Exception:
        slope_src = None
        slope_val_arr = None
        transformer_to_slope = None

    def lonlat_to_3857(lon, lat):
        try:
            x, y = transformer_to_3857.transform(lon, lat)
            return x, y
        except Exception:
            # fallback: use approximate conversion (not ideal)
            return lon * 111000.0, lat * 111000.0

    def dist_to_roads_m(lon, lat):
        try:
            x3857, y3857 = lonlat_to_3857(lon, lat)
            if road_kdtree is None:
                return float(np.inf)
            dist, _ = road_kdtree.query([(x3857, y3857)], k=1)
            return float(dist[0])
        except Exception:
            return float(np.inf)

    def point_in_water(lon, lat):
        if water_union is None:
            return False
        try:
            x3857, y3857 = lonlat_to_3857(lon, lat)
            return bool(water_union.contains(Point(x3857, y3857)))
        except Exception:
            return False

    def slope_at(lon, lat):
        # fast lookup via raster index: transform lon/lat -> slope CRS -> slope_src.index -> array access
        if slope_val_arr is None or transformer_to_slope is None:
            return 0.0
        try:
            x_s, y_s = transformer_to_slope.transform(lon, lat)
            row, col = slope_src.index(x_s, y_s)
            # guard bounds
            if (0 <= row < slope_val_arr.shape[0]) and (0 <= col < slope_val_arr.shape[1]):
                val = float(slope_val_arr[row, col])
                if np.isnan(val):
                    return 0.0
                return float(max(0.0, min(90.0, val)))
            else:
                return 0.0
        except Exception:
            return 0.0

    # helper: sample a random point along a LineString (in projected coords)
    def sample_point_along_linestring(line: LineString):
        coords = list(line.coords)
        if len(coords) < 2:
            return None
        seg_idx = np.random.randint(0, len(coords) - 1)
        x1, y1 = coords[seg_idx]
        x2, y2 = coords[seg_idx + 1]
        frac = np.random.rand()
        xi = x1 + frac * (x2 - x1)
        yi = y1 + frac * (y2 - y1)
        # convert to lon/lat quickly using transformer_from_3857
        try:
            lon, lat = transformer_from_3857.transform(xi, yi)
            return lon, lat
        except Exception:
            return None

    # iterate I points and generate candidates
    for _, row in I_df.iterrows() if (I_df is not None and len(I_df) > 0) else []:
        base_lon, base_lat = float(row.longitude), float(row.latitude)
        candidates_found = 0

        # sample along roads within buffer (use projected geometry test)
        if roads_proj is not None and len(roads_proj) > 0:
            try:
                # build buffer in 3857 quickly using transformer
                x3857, y3857 = lonlat_to_3857(base_lon, base_lat)
                buf = Point(x3857, y3857).buffer(2000)
                # select roads that intersect buffer using bounding box filter (slower than spatial index but safe)
                # use .intersects with to_crs result (roads_proj already in 3857)
                # Convert buf to GeoSeries and test intersection
                # small optimization: use bounds to prefilter
                minx, miny, maxx, maxy = buf.bounds
                rsel = roads_proj.cx[minx:maxx, miny:maxy]
                # fallback if that yields nothing
                if rsel is None or len(rsel) == 0:
                    rsel = roads_proj
                for geom in rsel.geometry:
                    if geom is None:
                        continue
                    lines = [geom] if geom.geom_type == "LineString" else list(geom)
                    for line in lines:
                        for _ in range(5):
                            sampled = sample_point_along_linestring(line)
                            if sampled is None:
                                continue
                            lon_c, lat_c = sampled
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

        # jitter around I if not enough
        attempts = 0
        while candidates_found < candidate_per_cluster and attempts < candidate_per_cluster * 8:
            dx = np.random.normal(scale=jitter_m)
            dy = np.random.normal(scale=jitter_m)
            try:
                # jitter in 3857 directly and back transform
                x0, y0 = lonlat_to_3857(base_lon, base_lat)
                nx3857 = x0 + dx
                ny3857 = y0 + dy
                lon_c, lat_c = transformer_from_3857.transform(nx3857, ny3857)
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

    # global sampling near roads using interpolation
    if roads_proj is not None and len(roads_proj) > 0 and len(J) < n_samples_global:
        for geom in roads_proj.geometry:
            if geom is None:
                continue
            lines = [geom] if geom.geom_type == "LineString" else list(geom)
            for line in lines:
                for _ in range(3):
                    sampled = sample_point_along_linestring(line)
                    if sampled is None:
                        continue
                    lon_c, lat_c = sampled
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

    if slope_src is not None:
        slope_src.close()

    if not J:
        return pd.DataFrame()
    J_df = pd.DataFrame(J)

    # Deduplicate
    if dedup_round is None:
        J_df = J_df.drop_duplicates()
    else:
        J_df["lat_round"] = J_df["latitude"].round(int(dedup_round))
        J_df["lon_round"] = J_df["longitude"].round(int(dedup_round))
        J_df = J_df.drop_duplicates(subset=["lat_round", "lon_round"]).drop(columns=["lat_round", "lon_round"])

    return J_df.reset_index(drop=True)


# 7) evaluate candidate pop coverage for each J (non-overlapping greedy selection)
def evaluate_candidates_and_reduce_overlap(I_df, J_df, pop_gdf, radius_m=3000, overlap_keep_threshold=0.1, max_J=1200):
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
    # build KDTree for pop coords in projected CRS (fast radius queries)
    pop_coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    pop_vals = pop_proj["pop"].values
    if len(pop_coords) == 0:
        return pd.DataFrame()
    pop_kdt = cKDTree(pop_coords)

    # prepare J projected coords
    J_proj_pts = gpd.GeoDataFrame(J_df.copy(),
                                  geometry=gpd.points_from_xy(J_df.longitude, J_df.latitude),
                                  crs="EPSG:4326").to_crs(epsg=3857)

    J_pop_idx = []
    for _, row in J_proj_pts.iterrows():
        cx, cy = row.geometry.x, row.geometry.y
        # query ball point
        idxs = pop_kdt.query_ball_point([cx, cy], r=float(radius_m))
        J_pop_idx.append(idxs)

    total_pops = [float(pop_vals[idxs].sum()) if idxs else 0.0 for idxs in J_pop_idx]
    order = np.argsort(total_pops)[::-1]
    selected_idxs = []
    covered_mask = np.zeros(len(pop_vals), dtype=bool)

    for idx in order:
        idxs = J_pop_idx[idx]
        if not idxs:
            continue
        uncovered = [i for i in idxs if not covered_mask[i]]
        if not uncovered:
            continue
        unc_pop = float(pop_vals[uncovered].sum())
        frac = (len(uncovered) / len(idxs)) if len(idxs) > 0 else 0.0
        if frac < overlap_keep_threshold:
            continue
        if unc_pop <= 0:
            continue
        selected_idxs.append(idx)
        for i in uncovered:
            covered_mask[i] = True
        if len(selected_idxs) >= max_J:
            break

    if not selected_idxs:
        topn = min(max_J, len(J_df))
        selected_idxs = list(np.argsort(total_pops)[::-1][:topn])

    sel = J_df.iloc[selected_idxs].copy().reset_index(drop=True)
    sel["site_id"] = [f"J_{i:05d}" for i in range(len(sel))]

    # compute nearest I for each selected J using haversine matrix (I small)
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

    # defaults tuned for more I/J but safe
    pop_threshold = float(config.get('pop_threshold', 3))
    eps_dbscan_m = float(config.get('eps_dbscan_m', 1000))
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
    overlap_keep_threshold = float(config.get('overlap_keep_threshold', 0.1))
    seed = int(config.get('seed', 42))

    boundary = gpd.read_file(CLEANED_DIR / "hue_boundary_clean.geojson")
    pop_tif = str(CLEANED_DIR / "pop_hue_clean.tif")
    slope_tif = str(CLEANED_DIR / "slope_hue_clean.tif")
    roads = read_geojson(str(CLEANED_DIR / "roads_hue_clean.geojson"))
    water = read_geojson(str(CLEANED_DIR / "water_hue_clean.geojson"))
    active_bts, failed_bts = read_bts_files()

    np.random.seed(seed)
    random.seed(seed)

    print(" Extract population cells from raster...")
    pop_gdf = extract_population_cells(pop_tif, threshold=pop_threshold)
    if pop_gdf is None:
        pop_gdf = gpd.GeoDataFrame(columns=["longitude", "latitude", "pop", "geometry"], crs="EPSG:4326")
    if (not pop_gdf.empty) and (boundary is not None):
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

    # expand I by jittering if less than desired (keeps near clusters)
    if I_df is None or I_df.empty:
        I_df = pd.DataFrame(columns=["site_id", "latitude", "longitude", "pop", "covered_cells"])
    if len(I_df) < max_I and len(I_df) > 0:
        extra_needed = max_I - len(I_df)
        jittered = []
        for k in range(extra_needed):
            base = I_df.sample(1, random_state=seed + k).iloc[0]
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
    for name in ["schools_clean", "hospitals_clean", "medical_centers_clean", "industrial_clean", "residential_clean", "command_centers_clean"]:
        p = CLEANED_DIR / f"{name}.geojson"
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

    # Prepare outputs
    I_out = pd.DataFrame({
        "site_id": I_df["site_id"].astype(str),
        "latitude": I_df["latitude"].astype(float),
        "longitude": I_df["longitude"].astype(float),
        "pop": I_df["pop"].astype(float),
    })
    I_out["priority_category"] = I_df["priority_category"].astype(str) if "priority_category" in I_df.columns else "normal"
    I_out["priority_weight"] = I_df["priority_weight"].astype(float) if "priority_weight" in I_df.columns else 1.0

    if J_df is None or J_df.empty:
        J_out = pd.DataFrame(columns=[
            "site_id", "i_ref", "latitude", "longitude", "pop", "priority_category",
            "priority_weight", "slope", "dist_to_road_m", "in_water"
        ])
    else:
        J_out = J_df.copy()
        # ensure numeric types where possible
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
        if len(J_out) > max_J:
            J_out = J_out.sort_values("pop", ascending=False).head(max_J).reset_index(drop=True)
        cols = ["site_id", "i_ref", "latitude", "longitude", "pop", "priority_category",
                "priority_weight", "slope", "dist_to_road_m", "in_water"]
        for col in cols:
            if col not in J_out.columns:
                J_out[col] = np.nan
        J_out = J_out[cols]

    I_out.to_csv(Path(out_dir) / "I_points_OA.csv", index=False)
    J_out.to_csv(Path(out_dir) / "J_sites_OA.csv", index=False)

    # compute cover matrix (haversine) and save
    if len(J_out) > 0 and len(I_out) > 0:
        I_coords = I_out[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
        J_coords = J_out[['latitude', 'longitude']].rename(columns={'latitude': 'y', 'longitude': 'x'}).to_dict('records')
        dist = compute_distance_matrix(I_coords, J_coords, metric='haversine')
        cover = (dist <= float(radius_I_m)).astype(int)
    else:
        cover = np.zeros((len(I_out), len(J_out)), dtype=int)
    np.save(Path(out_dir) / "cover.npy", cover)

    print(f"[Feature Extraction] Saved {len(I_out)} I_points_A, {len(J_out)} J_sites_A, and cover.npy to {out_dir}")