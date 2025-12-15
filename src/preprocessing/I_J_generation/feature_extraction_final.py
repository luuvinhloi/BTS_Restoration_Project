# feature_extraction_final.py
"""
Feature extraction (I_points, J_sites) — refactored, optimized, compact.

Key behaviors:
 - Method A: remove population inside union(active_bts + failed_bts with status='power_outage')
   before clustering I.
 - Only failed_bts with status=='failed' are considered "hard failed" for generating recovery I/J.
 - J candidates removed if:
     * outside boundary
     * inside vector water polygons (river/lake/sea)
     * flood_depth >= flood_depth_threshold_m (default 0.5 m)
 - Outputs unchanged: I_points.csv, J_sites.csv, cover.npy
 - Clean, modular functions for maintainability.
"""
from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from pyproj import Transformer

# Try to import local utilities; provide safe fallbacks if missing
try:
    from src.utils.geo_utils import compute_distance_matrix
    from src.utils.io_utils import read_geojson
except Exception:
    def compute_distance_matrix(A, B, metric='haversine'):
        # A, B: lists of {'y':lat,'x':lon}
        def hav(lat1, lon1, lat2, lon2):
            R = 6371000.0
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
            return 2*R*math.asin(math.sqrt(max(0.0, min(1.0, a))))
        M = np.zeros((len(A), len(B)), dtype=float)
        for i,a in enumerate(A):
            for j,b in enumerate(B):
                M[i,j] = hav(a['y'], a['x'], b['y'], b['x'])
        return M
    def read_geojson(path):
        try:
            return gpd.read_file(path)
        except Exception:
            return gpd.GeoDataFrame()

# ---------------------- PATHS & DEFAULTS ----------------------
_file = Path(__file__).resolve()
_project_root = _file.parents[3] if len(_file.parents) >= 4 else _file.parents[-1]
DATA_DIR = _project_root / "data"
if not DATA_DIR.exists():
    alt = _file.parents[2] / "data"
    if alt.exists():
        DATA_DIR = alt
CLEANED_DIR = DATA_DIR / "cleaned"
DAMAGE_BTS_DIR = DATA_DIR / "processed" / "damage_bts"

# flood tif default location (user-provided in prompt)
FLOOD_TIF_DEFAULT = _project_root / "BTS_Restoration_Project" / "data" / "processed" / "flood" / "flood_depth_combined_B_clean.tif"

# ---------------------- READ BTS & POP ----------------------
def _read_bts_files(damage_bts_dir: Path = DAMAGE_BTS_DIR):
    active_path = damage_bts_dir / "active_bts.csv"
    failed_path = damage_bts_dir / "failed_bts.csv"
    active = pd.read_csv(active_path) if active_path.exists() else pd.DataFrame()
    failed = pd.read_csv(failed_path) if failed_path.exists() else pd.DataFrame()
    return active, failed

def extract_population_cells(pop_tif: str, threshold: float = 3.0) -> gpd.GeoDataFrame:
    """Return GeoDataFrame with columns longitude, latitude, pop, geometry (EPSG:4326)."""
    try:
        with rasterio.open(pop_tif) as src:
            arr = src.read(1)
            arr = np.nan_to_num(arr, nan=0.0)
            transform = src.transform
            rows, cols = np.where(arr > threshold)
            if len(rows) == 0:
                return gpd.GeoDataFrame(columns=["longitude","latitude","pop","geometry"], crs="EPSG:4326")
            xs, ys, vs = [], [], []
            for r,c in zip(rows, cols):
                v = float(arr[r,c])
                x, y = transform * (float(c) + 0.5, float(r) + 0.5)
                xs.append(x); ys.append(y); vs.append(v)
            gdf = gpd.GeoDataFrame({"longitude": xs, "latitude": ys, "pop": vs},
                                   geometry=[Point(x,y) for x,y in zip(xs,ys)], crs=src.crs.to_string())
            try:
                gdf = gdf.to_crs(epsg=4326)
            except Exception:
                gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
            return gdf
    except Exception:
        return gpd.GeoDataFrame(columns=["longitude","latitude","pop","geometry"], crs="EPSG:4326")

# ---------------------- Remove population covered by operational BTS (Method A) ----------------------
def remove_covered_by_operational_bts(pop_gdf: gpd.GeoDataFrame,
                                      active_bts_df: pd.DataFrame,
                                      failed_bts_df: pd.DataFrame,
                                      default_radius_m: float = 3000.0) -> gpd.GeoDataFrame:
    """
    Treat power_outage BTS (in failed_bts_df) as operational for coverage removal.
    Removes population points inside union(active + power_outage buffers).
    """
    if pop_gdf is None or pop_gdf.empty:
        return pop_gdf

    frames = []
    if active_bts_df is not None and not active_bts_df.empty:
        frames.append(active_bts_df.copy())
    if failed_bts_df is not None and not failed_bts_df.empty:
        try:
            status = failed_bts_df.get("status", "").astype(str).str.lower()
            power_df = failed_bts_df[status == "power_outage"].copy()
            if not power_df.empty:
                frames.append(power_df)
        except Exception:
            pass

    if not frames:
        return pop_gdf

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # construct geometry and project to 3857 for meter buffers
    try:
        bts_gdf = gpd.GeoDataFrame(merged.copy(),
                                   geometry=gpd.points_from_xy(merged.longitude, merged.latitude),
                                   crs="EPSG:4326").to_crs(epsg=3857)
    except Exception:
        bts_gdf = pd.DataFrame(merged.copy())
        bts_gdf = gpd.GeoDataFrame(bts_gdf, geometry=[Point(0,0)]*len(bts_gdf), crs="EPSG:3857")

    if "coverage_radius_m" not in bts_gdf.columns:
        bts_gdf["coverage_radius_m"] = default_radius_m
    else:
        bts_gdf["coverage_radius_m"] = bts_gdf["coverage_radius_m"].fillna(default_radius_m).astype(float)

    buffers = [g.buffer(float(r)) for g,r in zip(bts_gdf.geometry, bts_gdf.coverage_radius_m)]
    if not buffers:
        return pop_gdf
    union_buf = unary_union(buffers)

    try:
        pop_proj = pop_gdf.to_crs(epsg=3857)
    except Exception:
        pop_proj = pop_gdf.copy()
    mask = pop_proj.geometry.within(union_buf)
    return pop_gdf.loc[~mask.values].reset_index(drop=True)

# ---------------------- Clustering & selection helpers ----------------------
def cluster_population_dbscan(pop_gdf: gpd.GeoDataFrame, eps_m: float = 1200, min_samples: int = 3):
    """Cluster population points (EPSG:3857) and return clusters metadata and projected pop_gdf."""
    if pop_gdf is None or pop_gdf.empty:
        return pd.DataFrame(), pop_gdf
    try:
        pop_proj = pop_gdf.to_crs(epsg=3857)
    except Exception:
        pop_proj = pop_gdf.copy()
    coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    if coords.shape[0] == 0:
        return pd.DataFrame(), pop_proj
    db = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit(coords)
    pop_proj = pop_proj.copy()
    pop_proj["cluster"] = db.labels_
    clusters = []
    for lab in sorted(set(db.labels_)):
        if lab == -1:
            continue
        sub = pop_proj[pop_proj.cluster == lab]
        centroid_x, centroid_y = float(sub.geometry.x.mean()), float(sub.geometry.y.mean())
        centroid_point = gpd.GeoSeries([Point(centroid_x, centroid_y)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        clusters.append({
            "cluster": int(lab),
            "centroid_lon": centroid_point.x,
            "centroid_lat": centroid_point.y,
            "n_points": len(sub),
            "pop_cells_index": sub.index.to_list()
        })
    return pd.DataFrame(clusters), pop_proj

def select_clusters_nonoverlap(clusters_df: pd.DataFrame, pop_proj: gpd.GeoDataFrame, radius_m: float = 3000):
    """Greedy selection of clusters to avoid overlapping covered population cells."""
    covered_idx = set()
    selected = []
    if clusters_df is None or clusters_df.empty:
        return pd.DataFrame(), covered_idx
    df = clusters_df.copy()
    df["total_pop"] = 0.0
    for idx, row in df.iterrows():
        idxs = row["pop_cells_index"]
        df.at[idx, "total_pop"] = float(pop_proj.loc[idxs, "pop"].sum()) if len(idxs) else 0.0
    for c in df.sort_values("total_pop", ascending=False).to_dict("records"):
        idxs = c["pop_cells_index"]
        uncovered = [i for i in idxs if i not in covered_idx]
        if not uncovered:
            continue
        unc_pop = float(pop_proj.loc[uncovered, "pop"].sum())
        if unc_pop <= 0:
            continue
        selected.append({
            "site_id": f"I_{len(selected):05d}",
            "latitude": float(c["centroid_lat"]),
            "longitude": float(c["centroid_lon"]),
            "pop": unc_pop,
            "covered_cells": uncovered
        })
        for i in uncovered:
            covered_idx.add(i)
    return pd.DataFrame(selected), covered_idx

def assign_priority_to_I(I_df: pd.DataFrame, infra_paths: dict, buffer_m: float = 1500.0) -> pd.DataFrame:
    """Attach priority_category and priority_weight to I points using infra layers (if present)."""
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
            infra_union = infra.to_crs(epsg=3857).unary_union
        except Exception:
            infra_union = infra.unary_union
        for idx, row in I_gdf.iterrows():
            try:
                d = float(row.geometry.distance(infra_union))
            except Exception:
                d = float("inf")
            if np.isfinite(d) and d < buffer_m:
                add = {
                    'schools_clean': 2.0, 'hospitals_clean': 3.0, 'medical_centers_clean': 2.5,
                    'industrial_clean': 1.5, 'residential_clean': 1.2, 'command_centers_clean': 3.0
                }.get(name, 1.0)
                I_gdf.at[idx, "priority_weight"] = float(I_gdf.at[idx, "priority_weight"]) + add
                I_gdf.at[idx, "priority_category"] = name
    out = pd.DataFrame({
        "site_id": I_gdf["site_id"].astype(str),
        "latitude": I_gdf.to_crs(epsg=4326).geometry.y.astype(float),
        "longitude": I_gdf.to_crs(epsg=4326).geometry.x.astype(float),
        "pop": I_gdf["pop"].astype(float),
        "priority_category": I_gdf["priority_category"].astype(str),
        "priority_weight": I_gdf["priority_weight"].astype(float)
    })
    return out

# ---------------------- Sampling helpers ----------------------
def safe_sample_point(line: LineString, trans_3857_to_wgs: Transformer):
    """Sample random point along a LineString provided in 3857 coords -> returns lon,lat (WGS84)."""
    try:
        coords = list(line.coords)
        if len(coords) < 2:
            return None
        seg_idx = np.random.randint(0, len(coords)-1)
        x1, y1 = coords[seg_idx]; x2, y2 = coords[seg_idx+1]
        frac = np.random.rand()
        xi = x1 + frac*(x2 - x1); yi = y1 + frac*(y2 - y1)
        lon, lat = trans_3857_to_wgs.transform(xi, yi)
        return float(lon), float(lat)
    except Exception:
        return None

def grid_sample_within_bounds(boundary_gdf: gpd.GeoDataFrame, spacing_m: float, max_points: int = 2000):
    """Grid sample lon/lat points within boundary (spacing in meters)."""
    if boundary_gdf is None or boundary_gdf.empty:
        return []
    try:
        b3857 = boundary_gdf.to_crs(epsg=3857)
        union = b3857.unary_union
        minx, miny, maxx, maxy = union.bounds
        nx = max(1, int((maxx - minx) // spacing_m) + 1)
        ny = max(1, int((maxy - miny) // spacing_m) + 1)
        pts = []
        trans = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        for ix in range(nx+1):
            x = minx + ix * spacing_m
            for iy in range(ny+1):
                y = miny + iy * spacing_m
                p = Point(x, y)
                if union.contains(p):
                    lon, lat = trans.transform(x, y)
                    pts.append((float(lon), float(lat)))
                    if len(pts) >= max_points:
                        return pts
        return pts
    except Exception:
        return []

# ---------------------- Generate J candidates (flood & water filtered) ----------------------
def generate_J_candidates(I_df: pd.DataFrame,
                          roads_gdf: gpd.GeoDataFrame,
                          water_gdf: gpd.GeoDataFrame,
                          slope_tif: str,
                          flood_tif: str = None,
                          candidate_per_cluster: int = 12,
                          slope_threshold: float = 30.0,
                          road_buffer_m: float = 4000.0,
                          n_global_samples: int = 500,
                          seed: int = 42,
                          dedup_decimals: int = 5,
                          road_samples_per_line: int = 2,
                          extra_jitter_scales: tuple = (100, 300, 800),
                          boundary_gdf: gpd.GeoDataFrame = None,
                          flood_depth_threshold_m: float = 0.5):
    """
    Build candidate J pool using:
      - road sampling near each I,
      - multi-scale jitter around I,
      - global grid sampling,
      - sparse sampling on roads.
    Filters applied:
      - candidate outside boundary -> drop
      - inside vector water polygons -> drop
      - flood raster depth >= flood_depth_threshold_m -> drop
      - slope too steep -> drop
      - deduplicate by rounding lat/lon to dedup_decimals
    Returns DataFrame with columns: latitude, longitude, slope, dist_to_road_m, in_water, flood_depth_m
    """
    random.seed(seed); np.random.seed(seed)
    if I_df is None:
        I_df = pd.DataFrame()

    # Transformers
    trans_wgs_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    trans_3857_to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Prepare water/boundary/roads in 3857
    try:
        roads_3857 = roads_gdf.to_crs(epsg=3857) if (roads_gdf is not None and len(roads_gdf) > 0) else gpd.GeoDataFrame()
    except Exception:
        roads_3857 = gpd.GeoDataFrame()
    try:
        water_3857 = water_gdf.to_crs(epsg=3857) if (water_gdf is not None and len(water_gdf) > 0) else gpd.GeoDataFrame()
    except Exception:
        water_3857 = gpd.GeoDataFrame()
    water_union = water_3857.unary_union if (water_3857 is not None and len(water_3857) > 0) else None

    # Road vertices KDTree (3857)
    road_pts = []
    for geom in roads_3857.geometry if roads_3857 is not None else []:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            road_pts.extend(list(geom.coords))
        else:
            try:
                for seg in geom:
                    road_pts.extend(list(seg.coords))
            except Exception:
                pass
    road_kdtree = cKDTree(np.array(road_pts)) if len(road_pts) > 0 else None

    # slope raster
    slope_src = None; slope_arr = None; trans_to_slope = None
    try:
        slope_src = rasterio.open(slope_tif)
        slope_arr = slope_src.read(1)
        trans_to_slope = Transformer.from_crs("EPSG:4326", slope_src.crs.to_string(), always_xy=True)
    except Exception:
        slope_src = slope_arr = trans_to_slope = None

    # flood raster
    if flood_tif is None:
        ft = str(FLOOD_TIF_DEFAULT) if FLOOD_TIF_DEFAULT.exists() else None
    else:
        ft = flood_tif
    flood_src = None; flood_arr = None; trans_to_flood = None
    try:
        if ft is not None and Path(ft).exists():
            flood_src = rasterio.open(str(ft))
            flood_arr = flood_src.read(1)
            trans_to_flood = Transformer.from_crs("EPSG:4326", flood_src.crs.to_string(), always_xy=True)
        else:
            flood_src = flood_arr = trans_to_flood = None
    except Exception:
        flood_src = flood_arr = trans_to_flood = None

    # boundary check (WGS84)
    if boundary_gdf is None:
        try:
            boundary_gdf = read_geojson(str(CLEANED_DIR / "hue_boundary_clean.geojson"))
        except Exception:
            boundary_gdf = None
    boundary_union = None
    if boundary_gdf is not None and not boundary_gdf.empty:
        try:
            boundary_union = boundary_gdf.unary_union
        except Exception:
            boundary_union = None

    def lonlat_to_3857(lon, lat):
        try:
            return trans_wgs_to_3857.transform(lon, lat)
        except Exception:
            return lon * 111000.0, lat * 111000.0

    def dist_to_roads_m(lon, lat):
        if road_kdtree is None:
            return float("inf")
        try:
            x,y = lonlat_to_3857(lon, lat)
            d, _ = road_kdtree.query([(x,y)], k=1)
            return float(d[0])
        except Exception:
            return float("inf")

    def point_in_vector_water(lon, lat):
        if water_union is None:
            return False
        try:
            x,y = lonlat_to_3857(lon, lat)
            return bool(water_union.contains(Point(x, y)))
        except Exception:
            return False

    def inside_boundary_wgs(lon, lat):
        if boundary_union is None:
            return True
        try:
            return bool(boundary_union.contains(Point(lon, lat)))
        except Exception:
            # fallback: test in projected coords
            try:
                x,y = lonlat_to_3857(lon, lat)
                return bool(boundary_gdf.to_crs(epsg=3857).unary_union.contains(Point(x,y)))
            except Exception:
                return False

    def flood_depth_at(lon, lat):
        if flood_arr is None or flood_src is None or trans_to_flood is None:
            return 0.0
        try:
            fx, fy = trans_to_flood.transform(lon, lat)
            row, col = flood_src.index(fx, fy)
            if 0 <= row < flood_arr.shape[0] and 0 <= col < flood_arr.shape[1]:
                val = flood_arr[row, col]
                return float(val) if not np.isnan(val) else 0.0
            return 0.0
        except Exception:
            return 0.0

    def slope_at(lon, lat):
        if slope_arr is None or slope_src is None or trans_to_slope is None:
            return 0.0
        try:
            sx, sy = trans_to_slope.transform(lon, lat)
            row, col = slope_src.index(sx, sy)
            if 0 <= row < slope_arr.shape[0] and 0 <= col < slope_arr.shape[1]:
                v = slope_arr[row, col]
                return float(max(0.0, min(90.0, v))) if not np.isnan(v) else 0.0
            return 0.0
        except Exception:
            return 0.0

    # Candidate accumulation
    pool = []

    # ---- Per-I road sampling + jitter ----
    if not I_df.empty:
        for _, irow in I_df.iterrows():
            try:
                base_lon = float(irow.get("longitude", np.nan))
                base_lat = float(irow.get("latitude", np.nan))
            except Exception:
                continue
            # sample road points within a buffer around I
            try:
                xb, yb = lonlat_to_3857(base_lon, base_lat)
                buf = Point(xb, yb).buffer(min(road_buffer_m, 4000))
                minx, miny, maxx, maxy = buf.bounds
                try:
                    rsel = roads_3857.cx[minx:maxx, miny:maxy]
                    if rsel is None or len(rsel) == 0:
                        rsel = roads_3857
                except Exception:
                    rsel = roads_3857
                found = 0
                for geom in (rsel.geometry if rsel is not None else []):
                    if geom is None:
                        continue
                    lines = [geom] if geom.geom_type == "LineString" else list(geom) if hasattr(geom, "__iter__") else []
                    for line in lines:
                        for _ in range(road_samples_per_line):
                            s = safe_sample_point(line, trans_3857_to_wgs)
                            if s is None:
                                continue
                            lon_c, lat_c = s
                            # boundary
                            if not inside_boundary_wgs(lon_c, lat_c):
                                continue
                            # vector water
                            if point_in_vector_water(lon_c, lat_c):
                                continue
                            # flood
                            depth = flood_depth_at(lon_c, lat_c)
                            if depth >= flood_depth_threshold_m:
                                continue
                            # slope
                            s_val = slope_at(lon_c, lat_c)
                            if s_val > slope_threshold:
                                continue
                            droad = dist_to_roads_m(lon_c, lat_c)
                            if droad > road_buffer_m:
                                continue
                            pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                                         "slope": float(s_val), "dist_to_road_m": float(droad),
                                         "in_water": False, "flood_depth_m": float(depth)})
                            found += 1
                            if found >= candidate_per_cluster:
                                break
                        if found >= candidate_per_cluster:
                            break
                    if found >= candidate_per_cluster:
                        break
            except Exception:
                pass

            # jitter samples multi-scale
            for scale in extra_jitter_scales:
                n_try = max(1, int(candidate_per_cluster // len(extra_jitter_scales)))
                for _ in range(n_try):
                    try:
                        x0,y0 = lonlat_to_3857(base_lon, base_lat)
                        jx = x0 + np.random.normal(scale=scale)
                        jy = y0 + np.random.normal(scale=scale)
                        lon_c, lat_c = trans_3857_to_wgs.transform(jx, jy)
                    except Exception:
                        lon_c = base_lon + (np.random.normal(scale=scale) / 111000.0)
                        lat_c = base_lat + (np.random.normal(scale=scale) / 111000.0)
                    if not inside_boundary_wgs(lon_c, lat_c):
                        continue
                    if point_in_vector_water(lon_c, lat_c):
                        continue
                    depth = flood_depth_at(lon_c, lat_c)
                    if depth >= flood_depth_threshold_m:
                        continue
                    s_val = slope_at(lon_c, lat_c)
                    if s_val > slope_threshold:
                        continue
                    droad = dist_to_roads_m(lon_c, lat_c)
                    if droad > road_buffer_m:
                        continue
                    pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                                 "slope": float(s_val), "dist_to_road_m": float(droad),
                                 "in_water": False, "flood_depth_m": float(depth)})

    # ---- Global grid sampling ----
    try:
        if boundary_gdf is not None and not boundary_gdf.empty:
            b3857 = boundary_gdf.to_crs(epsg=3857)
            area = b3857.unary_union.area
            spacing_m = max(250.0, math.sqrt(max(1.0, area / float(max(1, n_global_samples)))))
            grid_pts = grid_sample_within_bounds(boundary_gdf, spacing_m, max_points=n_global_samples)
            for lon_c, lat_c in grid_pts:
                if point_in_vector_water(lon_c, lat_c):
                    continue
                depth = flood_depth_at(lon_c, lat_c)
                if depth >= flood_depth_threshold_m:
                    continue
                s_val = slope_at(lon_c, lat_c)
                if s_val > slope_threshold:
                    continue
                droad = dist_to_roads_m(lon_c, lat_c)
                pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                             "slope": float(s_val), "dist_to_road_m": float(droad),
                             "in_water": False, "flood_depth_m": float(depth)})
    except Exception:
        pass

    # ---- Sparse random sampling along roads (global) ----
    if roads_3857 is not None and len(roads_3857) > 0:
        cnt = 0
        for geom in roads_3857.geometry:
            if geom is None:
                continue
            lines = [geom] if geom.geom_type == "LineString" else (list(geom) if hasattr(geom, "__iter__") else [])
            for line in lines:
                s = safe_sample_point(line, trans_3857_to_wgs)
                if s is None:
                    continue
                lon_c, lat_c = s
                if not inside_boundary_wgs(lon_c, lat_c):
                    continue
                if point_in_vector_water(lon_c, lat_c):
                    continue
                depth = flood_depth_at(lon_c, lat_c)
                if depth >= flood_depth_threshold_m:
                    continue
                s_val = slope_at(lon_c, lat_c)
                if s_val > slope_threshold:
                    continue
                droad = dist_to_roads_m(lon_c, lat_c)
                pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                             "slope": float(s_val), "dist_to_road_m": float(droad),
                             "in_water": False, "flood_depth_m": float(depth)})
                cnt += 1
                if cnt >= int(n_global_samples * 0.6):
                    break
            if cnt >= int(n_global_samples * 0.6):
                break

    # cleanup opened sources
    if slope_src is not None:
        slope_src.close()
    if flood_src is not None:
        flood_src.close()

    if not pool:
        return pd.DataFrame()

    J_df = pd.DataFrame(pool)
    # deduplicate by rounding lat/lon
    decimals = max(4, min(7, int(dedup_decimals)))
    J_df["lat_r"] = J_df["latitude"].round(decimals)
    J_df["lon_r"] = J_df["longitude"].round(decimals)
    J_df = J_df.drop_duplicates(subset=["lat_r", "lon_r"]).drop(columns=["lat_r","lon_r"]).reset_index(drop=True)
    return J_df

# ---------------------- Selection: balanced & target-aware ----------------------
def evaluate_and_select_J(I_df: pd.DataFrame,
                          J_pool: pd.DataFrame,
                          pop_gdf: gpd.GeoDataFrame,
                          radius_m: float = 3000,
                          target_J: int = 1000,
                          overlap_keep_threshold: float = 0.05,
                          min_uncov_pop_frac: float = 0.002,
                          strategy: str = "balanced"):
    """
    Select final J sites from J_pool using:
      1) Greedy uncovered-pop prioritization
      2) Farthest-point sampling for spatial diversity
    """
    if J_pool is None or J_pool.empty:
        return pd.DataFrame()

    # No population: fallback spatial selection
    if pop_gdf is None or pop_gdf.empty:
        return farthest_point_sample(J_pool, target_J)

    pop_proj = pop_gdf.to_crs(epsg=3857)
    pop_coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    pop_vals = pop_proj["pop"].values
    if len(pop_coords) == 0:
        return farthest_point_sample(J_pool, target_J)

    kdt = cKDTree(pop_coords)

    J_geo = gpd.GeoDataFrame(J_pool.copy(),
                             geometry=gpd.points_from_xy(J_pool.longitude, J_pool.latitude),
                             crs="EPSG:4326")
    J_proj = J_geo.to_crs(epsg=3857)
    J_coords = np.vstack([J_proj.geometry.x.values, J_proj.geometry.y.values]).T

    radius = float(radius_m)

    # compute pop covered by each candidate
    J_pop_idx = [kdt.query_ball_point([x,y], r=radius) for x,y in J_coords]
    total_pops = [float(pop_vals[idxs].sum()) if idxs else 0.0 for idxs in J_pop_idx]

    order = np.argsort(total_pops)[::-1]  # largest pop first

    selected = []
    covered_mask = np.zeros(len(pop_vals), dtype=bool)

    # ---------- Step 1: Greedy uncovered-pop ----------
    for idx in order:
        idxs = J_pop_idx[idx]
        if not idxs:
            continue
        uncovered = [i for i in idxs if not covered_mask[i]]
        if not uncovered:
            continue

        unc_pop = float(pop_vals[uncovered].sum())
        frac = len(uncovered) / len(idxs) if idxs else 0.0

        if frac < overlap_keep_threshold and unc_pop < min_uncov_pop_frac * pop_vals.sum():
            continue

        selected.append(idx)
        for i in uncovered:
            covered_mask[i] = True

        if len(selected) >= target_J * 0.5:
            break

    # relax if not enough
    if len(selected) < target_J * 0.5:
        needed = int(target_J * 0.5) - len(selected)
        extra = [i for i in order if i not in selected]
        selected.extend(extra[:needed])

    selected_set = set(selected)

    # if reached target
    if len(selected) >= target_J:
        return J_pool.iloc[selected[:target_J]].copy().reset_index(drop=True)

    # ---------- Step 2: spatial diversity ----------
    remain = target_J - len(selected)
    if remain <= 0:
        return J_pool.iloc[selected].copy().reset_index(drop=True)

    cand_indices = [i for i in range(len(J_pool)) if i not in selected_set]
    cand_coords = J_coords[cand_indices]

    seed_coords = J_coords[selected] if selected else np.empty((0,2))

    fps_rel = farthest_point_sampling_indices(cand_coords, remain, seed_coords=seed_coords)
    fps_global_idx = [cand_indices[i] for i in fps_rel]

    final = selected + fps_global_idx

    # if still not enough, fill with top pop
    if len(final) < target_J:
        filler = [i for i in order if i not in final]
        final.extend(filler[:(target_J - len(final))])

    final = final[:target_J]
    return J_pool.iloc[final].copy().reset_index(drop=True)

# ---------------------- FPS helpers ----------------------
def farthest_point_sampling_indices(points, k, seed_coords=None):
    """FPS selection on points array Nx2."""
    N = points.shape[0]
    if N == 0 or k <= 0:
        return []
    if k >= N:
        return list(range(N))

    if seed_coords is None or seed_coords.size == 0:
        idx0 = np.random.randint(0, N)
        selected = [idx0]
        dists = np.linalg.norm(points - points[idx0], axis=1)
    else:
        dists = np.min(np.linalg.norm(points[:, None, :] - seed_coords[None, :, :], axis=2), axis=1)
        selected = []

    while len(selected) < k:
        idx = int(np.argmax(dists))
        selected.append(idx)
        newd = np.linalg.norm(points - points[idx], axis=1)
        dists = np.minimum(dists, newd)
        dists[idx] = -1.0
    return selected

def farthest_point_sample(J_pool: pd.DataFrame, target: int):
    """Fallback spatial-only selection."""
    if J_pool is None or J_pool.empty:
        return pd.DataFrame()
    J_geo = gpd.GeoDataFrame(J_pool.copy(),
                             geometry=gpd.points_from_xy(J_pool.longitude, J_pool.latitude),
                             crs="EPSG:4326")
    J_proj = J_geo.to_crs(epsg=3857)
    coords = np.vstack([J_proj.geometry.x.values, J_proj.geometry.y.values]).T
    sel_rel = farthest_point_sampling_indices(coords, k=min(target, len(coords)))
    return J_pool.iloc[sel_rel].copy().reset_index(drop=True)

# ---------------------- Wrapper ----------------------
def evaluate_candidates_and_reduce_overlap(I_df: pd.DataFrame,
                                           J_df: pd.DataFrame,
                                           pop_gdf: gpd.GeoDataFrame,
                                           radius_m: float = 3000,
                                           overlap_keep_threshold: float = 0.05,
                                           max_J: int = 1200,
                                           config: dict = None):
    if J_df is None or J_df.empty:
        return pd.DataFrame()

    cfg = config or {}
    target_J = int(cfg.get("target_J", max_J))
    target_J = min(max_J, target_J)

    min_uncov = float(cfg.get("min_uncov_pop_frac", 0.002))
    overlap_keep = float(cfg.get("overlap_keep_threshold", overlap_keep_threshold))
    strategy = cfg.get("J_strategy", "balanced")

    sel = evaluate_and_select_J(I_df, J_df, pop_gdf,
                                radius_m=radius_m,
                                target_J=target_J,
                                overlap_keep_threshold=overlap_keep,
                                min_uncov_pop_frac=min_uncov,
                                strategy=strategy)
    if sel is None or sel.empty:
        return pd.DataFrame()
    return sel.reset_index(drop=True)

# ---------------------- MAIN ROUTINE ----------------------
def main(config: dict, out_dir: str or Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- read config ---
    fe = config or {}
    pop_threshold = float(fe.get("pop_threshold", 5))
    eps_dbscan = float(fe.get("eps_dbscan_m", 1500))
    db_min_samples = int(fe.get("db_min_samples", 3))
    R_I = float(fe.get("default_R", 3000))
    candidate_per_cluster = int(fe.get("candidate_per_cluster", 12))
    slope_threshold = float(fe.get("slope_threshold_deg", 30))
    road_buffer_m = float(fe.get("road_buffer_m", 4000))
    max_I = int(fe.get("max_I", 1200))
    max_J = int(fe.get("max_J", 1500))
    seed = int(fe.get("seed", 42))
    dedup_decimals = int(fe.get("dedup_round", 5))
    pop_sample_max = int(fe.get("pop_sample_max", 12000))
    target_J = int(fe.get("target_J", 1000))
    global_sampling = fe.get("global_sampling", "medium")

    if isinstance(global_sampling, str):
        if global_sampling.lower() == "low":
            n_global_samples = int(fe.get("global_samples_low", 250))
        elif global_sampling.lower() == "high":
            n_global_samples = int(fe.get("global_samples_high", 1000))
        else:
            n_global_samples = int(fe.get("global_samples_medium", 500))
    else:
        n_global_samples = int(global_sampling)

    # --- load data ---
    boundary = gpd.read_file(CLEANED_DIR / "hue_boundary_clean.geojson")
    pop_tif = str(CLEANED_DIR / "pop_hue_clean.tif")
    slope_tif = str(CLEANED_DIR / "slope_hue_clean.tif")
    roads = read_geojson(str(CLEANED_DIR / "roads_hue_clean.geojson"))
    water = read_geojson(str(CLEANED_DIR / "water_hue_clean.geojson"))

    flood_tif = fe.get("flood_tif", None)
    if flood_tif is None:
        flood_tif = str(FLOOD_TIF_DEFAULT) if FLOOD_TIF_DEFAULT.exists() else None

    active_bts, failed_bts = _read_bts_files(DAMAGE_BTS_DIR)

    # split failed BTS
    failed_power = pd.DataFrame()
    failed_hard = pd.DataFrame()
    if failed_bts is not None and not failed_bts.empty:
        st = failed_bts.get("status", "").astype(str).str.lower()
        failed_power = failed_bts[st == "power_outage"].copy()
        failed_hard = failed_bts[st == "failed"].copy()

    random.seed(seed); np.random.seed(seed)
    print("1) Extract population")
    pop_gdf = extract_population_cells(pop_tif, threshold=pop_threshold)
    if pop_gdf is None or pop_gdf.empty:
        pop_gdf = gpd.GeoDataFrame(columns=["longitude","latitude","pop","geometry"], crs="EPSG:4326")

    # clip to boundary
    try:
        pop_gdf = gpd.clip(pop_gdf, boundary)
    except Exception:
        pass

    print("2) Remove population covered by operational BTS")
    pop_uncovered = remove_covered_by_operational_bts(pop_gdf, active_bts, failed_bts, default_radius_m=R_I)
    print(f" population before: {len(pop_gdf)}, after filtering: {len(pop_uncovered)}")

    # downsample if too many
    if len(pop_uncovered) > pop_sample_max:
        print(f" downsampling population from {len(pop_uncovered)} to {pop_sample_max}")
        probs = pop_uncovered["pop"].values / pop_uncovered["pop"].sum()
        idxs = np.random.choice(len(pop_uncovered), size=pop_sample_max, replace=False, p=probs)
        pop_sampled = pop_uncovered.iloc[idxs].reset_index(drop=True)
    else:
        pop_sampled = pop_uncovered.copy()

    # ----- build I -----
    print("3) Cluster population")
    clusters_df, pop_proj = cluster_population_dbscan(pop_sampled, eps_m=eps_dbscan, min_samples=db_min_samples)

    if clusters_df is None or clusters_df.empty:
        print(" no clusters found -> fallback top population cells")
        pop_sorted = pop_sampled.sort_values("pop", ascending=False).head(max_I)
        I_list = []
        for i,row in pop_sorted.iterrows():
            I_list.append({
                "site_id": f"I_{len(I_list):05d}",
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "pop": float(row.pop),
                "covered_cells": [i]
            })
        I_df = pd.DataFrame(I_list)
    else:
        print(" select non-overlapping clusters")
        sel_clust, _ = select_clusters_nonoverlap(clusters_df, pop_proj, radius_m=R_I)
        I_df = sel_clust.copy()

    # jitter I if needed
    if not I_df.empty and len(I_df) < max_I:
        extra = max_I - len(I_df)
        jit = []
        for k in range(extra):
            base = I_df.sample(1, random_state=seed+k).iloc[0]
            lat_j = float(base["latitude"]) + np.random.normal(scale=0.001)
            lon_j = float(base["longitude"]) + np.random.normal(scale=0.001)
            pop_j = base["pop"] * float(np.random.uniform(0.6, 1.2))
            jit.append({"site_id": f"I_{len(I_df)+len(jit):05d}",
                        "latitude": lat_j,
                        "longitude": lon_j,
                        "pop": pop_j,
                        "covered_cells": base.get("covered_cells", [])})
        I_df = pd.concat([I_df, pd.DataFrame(jit)], ignore_index=True)

    if len(I_df) > max_I:
        I_df = I_df.sort_values("pop", ascending=False).head(max_I).reset_index(drop=True)

    print("4) Assign priority to I")
    infra_files = {}
    for name in ["schools_clean","hospitals_clean","medical_centers_clean",
                 "industrial_clean","residential_clean","command_centers_clean"]:
        p = CLEANED_DIR / f"{name}.geojson"
        if p.exists():
            infra_files[name] = str(p)
    I_df = assign_priority_to_I(I_df, infra_files, buffer_m=fe.get("infra_buffer_m",1500))

    # ----- build J -----
    print("5) Generate J candidates (flood + water + boundary filtered)")
    boundary_gdf = boundary
    J_pool = generate_J_candidates(I_df, roads, water, slope_tif,
                                   flood_tif=flood_tif,
                                   candidate_per_cluster=candidate_per_cluster,
                                   slope_threshold=slope_threshold,
                                   road_buffer_m=road_buffer_m,
                                   n_global_samples=n_global_samples,
                                   seed=seed,
                                   dedup_decimals=dedup_decimals,
                                   road_samples_per_line=fe.get("road_samples_per_line",2),
                                   extra_jitter_scales=tuple(fe.get("extra_jitter_scales",(100,300,800))),
                                   boundary_gdf=boundary_gdf,
                                   flood_depth_threshold_m=fe.get("flood_depth_threshold_m",0.5))

    if J_pool is None or J_pool.empty:
        print(" no J candidates -> J empty")
        J_df = pd.DataFrame()
    else:
        print(f" candidate pool size: {len(J_pool)}")
        sel = evaluate_candidates_and_reduce_overlap(
            I_df, J_pool, pop_uncovered,
            radius_m=R_I,
            overlap_keep_threshold=fe.get("overlap_keep_threshold",0.05),
            max_J=max_J,
            config={"target_J": target_J,
                    "J_strategy": fe.get("J_strategy","balanced"),
                    "min_uncov_pop_frac": fe.get("min_uncov_pop_frac",0.002),
                    "overlap_keep_threshold": fe.get("overlap_keep_threshold",0.05)}
        )
        if sel is None or sel.empty:
            print(" selection failed -> fallback top candidates")
            J_df = J_pool.head(min(max_J, len(J_pool))).reset_index(drop=True)
        else:
            J_df = sel.copy()

            # map nearest I to each J
            if not I_df.empty:
                Icoords = I_df[["latitude","longitude"]].rename(columns={"latitude":"y","longitude":"x"}).to_dict("records")
                Jcoords = J_df[["latitude","longitude"]].rename(columns={"latitude":"y","longitude":"x"}).to_dict("records")
                distIJ = compute_distance_matrix(Icoords, Jcoords, metric="haversine")
                nearest = np.argmin(distIJ, axis=0)
                i_refs, pw, pc, pops = [], [], [], []
                for k, idxn in enumerate(nearest):
                    try:
                        rec = I_df.iloc[int(idxn)]
                        i_refs.append(str(rec["site_id"]))
                        pw.append(float(rec["priority_weight"]))
                        pc.append(str(rec["priority_category"]))
                        pops.append(0.0)
                    except Exception:
                        i_refs.append(None); pw.append(1.0); pc.append("normal"); pops.append(0.0)
                J_df["i_ref"] = i_refs
                J_df["priority_weight"] = pw
                J_df["priority_category"] = pc
                # approximate pop cover
                try:
                    pop_proj2 = pop_uncovered.to_crs(epsg=3857)
                    pop_coords2 = np.vstack([pop_proj2.geometry.x.values, pop_proj2.geometry.y.values]).T
                    pop_vals2 = pop_proj2["pop"].values
                    kdt2 = cKDTree(pop_coords2)
                    J_geo2 = gpd.GeoDataFrame(J_df.copy(),
                                              geometry=gpd.points_from_xy(J_df.longitude, J_df.latitude),
                                              crs="EPSG:4326").to_crs(epsg=3857)
                    cov_idx = [kdt2.query_ball_point([g.x, g.y], r=R_I) for g in J_geo2.geometry]
                    pops2 = [float(pop_vals2[idxs].sum()) if idxs else 0.0 for idxs in cov_idx]
                    J_df["pop"] = pops2
                except Exception:
                    J_df["pop"] = 0.0
            else:
                J_df["i_ref"] = None
                J_df["priority_weight"] = 1.0
                J_df["priority_category"] = "normal"
                J_df["pop"] = 0.0

    # ----- Standardize output -----
    if J_df is None or J_df.empty:
        J_out = pd.DataFrame(columns=[
            "site_id","i_ref","latitude","longitude","pop","priority_category",
            "priority_weight","slope","dist_to_road_m","in_water"
        ])
    else:
        if len(J_df) > max_J:
            J_df = J_df.head(max_J).reset_index(drop=True)
        if "site_id" not in J_df.columns:
            J_df["site_id"] = [f"J_{i:05d}" for i in range(len(J_df))]
        for c in ["latitude","longitude","pop","priority_weight","slope","dist_to_road_m"]:
            if c in J_df.columns:
                try:
                    J_df[c] = J_df[c].astype(float)
                except:
                    pass
        if "priority_category" not in J_df.columns:
            J_df["priority_category"] = "normal"
        if "priority_weight" not in J_df.columns:
            J_df["priority_weight"] = 1.0
        if "i_ref" not in J_df.columns:
            J_df["i_ref"] = None
        if "in_water" not in J_df.columns:
            J_df["in_water"] = False
        cols = ["site_id","i_ref","latitude","longitude","pop","priority_category",
                "priority_weight","slope","dist_to_road_m","in_water"]
        for col in cols:
            if col not in J_df.columns:
                J_df[col] = np.nan
        J_out = J_df[cols].copy().reset_index(drop=True)

    # ----- Prepare I output -----
    if I_df is None or I_df.empty:
        I_out = pd.DataFrame(columns=["site_id","latitude","longitude","pop","priority_category","priority_weight"])
    else:
        if "priority_category" not in I_df.columns:
            I_df["priority_category"] = "normal"
        if "priority_weight" not in I_df.columns:
            I_df["priority_weight"] = 1.0
        I_out = pd.DataFrame({
            "site_id": I_df["site_id"].astype(str),
            "latitude": I_df["latitude"].astype(float),
            "longitude": I_df["longitude"].astype(float),
            "pop": I_df["pop"].astype(float),
            "priority_category": I_df["priority_category"].astype(str),
            "priority_weight": I_df["priority_weight"].astype(float)
        })

    # ----- Save outputs -----
    I_out.to_csv(out_dir / "I_points.csv", index=False)
    J_out.to_csv(out_dir / "J_sites.csv", index=False)

    # Cover matrix
    if len(I_out) > 0 and len(J_out) > 0:
        Icoords = I_out[["latitude","longitude"]].rename(columns={"latitude":"y","longitude":"x"}).to_dict("records")
        Jcoords = J_out[["latitude","longitude"]].rename(columns={"latitude":"y","longitude":"x"}).to_dict("records")
        dist = compute_distance_matrix(Icoords, Jcoords, metric="haversine")
        cover = (dist <= R_I).astype(int)
    else:
        cover = np.zeros((len(I_out), len(J_out)), dtype=int)
    np.save(out_dir / "cover.npy", cover)

    print(f"[Feature Extraction] Saved {len(I_out)} I-points, {len(J_out)} J-sites")
    return {"I_points": out_dir / "I_points.csv",
            "J_sites": out_dir / "J_sites.csv",
            "cover": out_dir / "cover.npy"}

# ---------------------- Script Entry ----------------------
if __name__ == "__main__":
    import yaml
    proj = Path(__file__).resolve().parents[3]
    cfg_path = proj / "config" / "params.yaml"
    if cfg_path.exists():
        cfg_all = yaml.safe_load(open(cfg_path))
        cfg = cfg_all.get("feature_extraction", {})
    else:
        cfg = {}
    out = proj / "data" / "processed" / "position_I_J"
    main(cfg, out)
