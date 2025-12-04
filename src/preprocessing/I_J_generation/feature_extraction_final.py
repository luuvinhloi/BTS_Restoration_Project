
# feature_extraction_final.py
"""
Feature extraction (I_points, J_sites) — updated with handling for power_outage BTS

Behavior changes:
 - Pop cells covered by active BTS and by BTS with status='power_outage' are removed before clustering I (Method A).
 - Only 'failed' BTS (status='failed') are considered as genuinely failed and drive recovery I/J generation.
 - Flood filtering remains: J candidates with flood_depth > flood_depth_threshold_m are removed.
 - Configuration for flood path and thresholds remains loadable from params.yaml via feature_extraction.flood_tif
 - Output file formats unchanged.
"""
from pathlib import Path
import math
import random
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from pyproj import Transformer

# local utils (assume same package layout)
try:
    from src.utils.geo_utils import compute_distance_matrix
    from src.utils.io_utils import read_geojson
except Exception:
    # fallback simple placeholders if utils not available (to avoid import errors during linting)
    def compute_distance_matrix(A, B, metric='haversine'):
        # A and B are list of {'y':lat,'x':lon} dicts. We'll compute haversine distances (meters).
        def hav(lat1, lon1, lat2, lon2):
            R = 6371000.0
            phi1 = math.radians(lat1); phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
            a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2.0)**2
            return 2*R*math.asin(math.sqrt(max(0.0,min(1.0,a))))
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

# ---------------------- PATHS ----------------------
_file = Path(__file__).resolve()
_project_root = _file.parents[3] if len(_file.parents) >= 4 else _file.parents[-1]
DATA_DIR = _project_root / "data"
if not DATA_DIR.exists():
    alt = _file.parents[2] / "data"
    if alt.exists():
        DATA_DIR = alt
CLEANED_DIR = DATA_DIR / "cleaned"
DAMAGE_BTS_DIR = DATA_DIR / "processed" / "damage_bts"

# default flood tif path (user provided location)
FLOOD_TIF_DEFAULT = _project_root / "BTS_Restoration_Project" / "data" / "processed" / "flood" / "flood_depth_combined_B_clean.tif"

# ---------------------- Helpers ----------------------
def _read_bts_files(damage_bts_dir: Path = DAMAGE_BTS_DIR):
    active_path = damage_bts_dir / "active_bts.csv"
    failed_path = damage_bts_dir / "failed_bts.csv"
    active = pd.read_csv(active_path) if active_path.exists() else pd.DataFrame()
    failed = pd.read_csv(failed_path) if failed_path.exists() else pd.DataFrame()
    return active, failed

def extract_population_cells(pop_tif: str, threshold: float = 3.0) -> gpd.GeoDataFrame:
    """Extract raster cells with value > threshold -> GeoDataFrame (EPSG:4326)."""
    try:
        with rasterio.open(pop_tif) as src:
            arr = src.read(1)
            arr = np.nan_to_num(arr, nan=0.0)
            transform = src.transform
            rows, cols = np.where(arr > threshold)
            if len(rows) == 0:
                return gpd.GeoDataFrame(columns=["longitude","latitude","pop","geometry"], crs="EPSG:4326")
            xs, ys, vs = [], [], []
            for r, c in zip(rows, cols):
                v = float(arr[r, c])
                x, y = transform * (float(c) + 0.5, float(r) + 0.5)
                xs.append(x); ys.append(y); vs.append(v)
            gdf = gpd.GeoDataFrame({"longitude": xs, "latitude": ys, "pop": vs},
                                   geometry=[Point(x,y) for x,y in zip(xs,ys)], crs=src.crs.to_string())
            # ensure WGS84 output
            try:
                gdf = gdf.to_crs(epsg=4326)
            except Exception:
                gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
            return gdf
    except Exception:
        return gpd.GeoDataFrame(columns=["longitude","latitude","pop","geometry"], crs="EPSG:4326")

def remove_covered_by_operational_bts(pop_gdf: gpd.GeoDataFrame,
                                      active_bts_df: pd.DataFrame,
                                      failed_bts_df: pd.DataFrame,
                                      default_radius_m: float = 3000.0) -> gpd.GeoDataFrame:
    """
    Remove population points that lie within union of active BTS buffers AND
    BTS with status='power_outage' (treated as operational for coverage purposes).

    This implements Method A: treat power_outage BTS as still providing coverage,
    therefore no need to create I/J for areas they cover.
    """
    if pop_gdf is None or pop_gdf.empty:
        return pop_gdf

    # build list of operational BTS: active + power_outage from failed_bts_df
    frames = []
    if active_bts_df is not None and not active_bts_df.empty:
        frames.append(active_bts_df.copy())
    if failed_bts_df is not None and not failed_bts_df.empty:
        try:
            power_df = failed_bts_df[failed_bts_df.get("status", "").astype(str).str.lower() == "power_outage"].copy()
            if not power_df.empty:
                frames.append(power_df)
        except Exception:
            pass
    if not frames:
        return pop_gdf

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # ensure geometry
    try:
        bts_gdf = gpd.GeoDataFrame(merged.copy(),
                                   geometry=gpd.points_from_xy(merged.longitude, merged.latitude),
                                   crs="EPSG:4326").to_crs(epsg=3857)
    except Exception:
        bts_gdf = gpd.GeoDataFrame(merged.copy())
        bts_gdf["geometry"] = [Point(0,0)]*len(bts_gdf)
        bts_gdf = bts_gdf.set_crs(epsg=3857, allow_override=True)

    # coverage radius handling: prefer column 'coverage_radius_m' if present, else default_radius_m
    if "coverage_radius_m" not in bts_gdf.columns:
        bts_gdf["coverage_radius_m"] = default_radius_m
    else:
        # fill missing or invalid values
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

def cluster_population_dbscan(pop_gdf: gpd.GeoDataFrame, eps_m: float =1200, min_samples: int =3):
    """Cluster population points in projected CRS (EPSG:3857)."""
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
    pop_proj["cluster"] = db.labels_
    clusters = []
    for lab in sorted(set(db.labels_)):
        if lab == -1:
            continue
        sub = pop_proj[pop_proj.cluster == lab]
        centroid_x = float(sub.geometry.x.mean()); centroid_y = float(sub.geometry.y.mean())
        centroid_point = gpd.GeoSeries([Point(centroid_x, centroid_y)], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        clusters.append({
            "cluster": int(lab),
            "centroid_lon": centroid_point.x,
            "centroid_lat": centroid_point.y,
            "n_points": len(sub),
            "pop_cells_index": sub.index.to_list()
        })
    return pd.DataFrame(clusters), pop_proj

def select_clusters_nonoverlap(clusters_df: pd.DataFrame, pop_proj: gpd.GeoDataFrame, radius_m: float =3000):
    """Greedy cluster selection to avoid overlapping covered pop cells."""
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

def assign_priority_to_I(I_df: pd.DataFrame, infra_paths: dict, buffer_m: float =1500.0) -> pd.DataFrame:
    """Assign priority categories and weights to I points using infra layers (if available)."""
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
                    'schools_clean': 2.0,
                    'hospitals_clean': 3.0,
                    'medical_centers_clean': 2.5,
                    'industrial_clean': 1.5,
                    'residential_clean': 1.2,
                    'command_centers_clean': 3.0
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

# ---------------------- Sampling helpers ----------------------
def safe_sample_point(line: LineString, trans_3857_to_wgs: Transformer):
    """Sample a random point along a LineString in 3857 and return (lon, lat) - safe guarantee."""
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
    """
    Create grid sample points within provided boundary polygon(s).
    spacing_m: approximate grid spacing in meters
    returns list of (lon, lat) tuples.
    """
    if boundary_gdf is None or boundary_gdf.empty:
        return []
    try:
        # work in 3857 for meters spacing
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
                pt = Point(x, y)
                if union.contains(pt):
                    lon, lat = trans.transform(x, y)
                    pts.append((float(lon), float(lat)))
                    if len(pts) >= max_points:
                        return pts
        return pts
    except Exception:
        return []

# ---------------------- J candidate generation (with flood checks) ----------------------
def generate_J_candidates(
        I_df: pd.DataFrame,
        roads_gdf: gpd.GeoDataFrame,
        water_gdf: gpd.GeoDataFrame,
        slope_tif: str,
        flood_tif: str = None,
        candidate_per_cluster: int = 15,
        jitter_m: float = 300,
        slope_threshold: float = 30.0,
        road_buffer_m: float = 4000.0,
        n_global_samples: int = 500,
        seed: int = 42,
        dedup_round: int = 5,
        road_samples_per_line: int = 3,
        extra_jitter_scales: list = (100, 300, 800),
        boundary_gdf: gpd.GeoDataFrame = None,
        flood_depth_threshold_m: float = 1.0
):
    """
    Produce a candidate pool J_pool with flood filtering:
      - discard any candidate where flood_depth > flood_depth_threshold_m
      - discard candidates in water_gdf (river/lake polygon)
    """
    random.seed(seed); np.random.seed(seed)
    J_pool = []

    # transformers
    trans_wgs_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    trans_3857_to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # prepare roads and water in 3857
    try:
        roads_3857 = roads_gdf.to_crs(epsg=3857) if (roads_gdf is not None and len(roads_gdf) > 0) else gpd.GeoDataFrame()
    except Exception:
        roads_3857 = gpd.GeoDataFrame()
    try:
        water_3857 = water_gdf.to_crs(epsg=3857) if (water_gdf is not None and len(water_gdf) > 0) else gpd.GeoDataFrame()
    except Exception:
        water_3857 = gpd.GeoDataFrame()
    water_union = water_3857.unary_union if (water_3857 is not None and len(water_3857) > 0) else None

    # road vertices KDTree (3857)
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
    slope_src = None; slope_arr = None; transformer_to_slope = None
    try:
        slope_src = rasterio.open(slope_tif)
        slope_arr = slope_src.read(1)
        transformer_to_slope = Transformer.from_crs("EPSG:4326", slope_src.crs.to_string(), always_xy=True)
    except Exception:
        slope_src = None; slope_arr = None; transformer_to_slope = None

    # flood raster (optional)
    flood_src = None; flood_arr = None; transformer_to_flood = None
    if flood_tif is None:
        # attempt default
        ft = FLOOD_TIF_DEFAULT if 'FLOOD_TIF_DEFAULT' in globals() else None
    else:
        ft = flood_tif
    try:
        if ft is not None and Path(ft).exists():
            flood_src = rasterio.open(str(ft))
            flood_arr = flood_src.read(1)
            transformer_to_flood = Transformer.from_crs("EPSG:4326", flood_src.crs.to_string(), always_xy=True)
        else:
            flood_src = None; flood_arr = None; transformer_to_flood = None
    except Exception:
        flood_src = None; flood_arr = None; transformer_to_flood = None

    def lonlat_to_3857(lon, lat):
        try:
            return trans_wgs_to_3857.transform(lon, lat)
        except Exception:
            return lon * 111000.0, lat * 111000.0

    def dist_to_roads_m(lon, lat):
        if road_kdtree is None:
            return float("inf")
        try:
            x, y = lonlat_to_3857(lon, lat)
            d, _ = road_kdtree.query([(x, y)], k=1)
            return float(d[0])
        except Exception:
            return float("inf")

    def point_in_water_vector(lon, lat):
        if water_union is None:
            return False
        try:
            x, y = lonlat_to_3857(lon, lat)
            return bool(water_union.contains(Point(x, y)))
        except Exception:
            return False

    def flood_depth_at(lon, lat):
        """Return flood depth in the flood raster at lon/lat in meters. If nodata or outside, return 0.0"""
        if flood_arr is None or flood_src is None or transformer_to_flood is None:
            return 0.0
        try:
            fx, fy = transformer_to_flood.transform(lon, lat)
            row, col = flood_src.index(fx, fy)
            if 0 <= row < flood_arr.shape[0] and 0 <= col < flood_arr.shape[1]:
                val = flood_arr[row, col]
                if np.isnan(val):
                    return 0.0
                return float(val)
            return 0.0
        except Exception:
            return 0.0

    def slope_at(lon, lat):
        if slope_arr is None or slope_src is None or transformer_to_slope is None:
            return 0.0
        try:
            x_s, y_s = transformer_to_slope.transform(lon, lat)
            row, col = slope_src.index(x_s, y_s)
            if 0 <= row < slope_arr.shape[0] and 0 <= col < slope_arr.shape[1]:
                val = slope_arr[row, col]
                if np.isnan(val):
                    return 0.0
                return float(max(0.0, min(90.0, val)))
            return 0.0
        except Exception:
            return 0.0

    # ---------------- per-I road sampling + multi-scale jitter ----------------
    if I_df is not None and len(I_df) > 0:
        for _, row in I_df.iterrows():
            try:
                base_lon = float(row.get("longitude", row.get("lon", np.nan)))
                base_lat = float(row.get("latitude", row.get("lat", np.nan)))
            except Exception:
                continue
            # if I itself is flooded heavily, still allow J nearby if J not flooded? (we will check per candidate)
            found = 0
            try:
                x_b, y_b = lonlat_to_3857(base_lon, base_lat)
                buf = Point(x_b, y_b).buffer(min(road_buffer_m, 4000))
                minx, miny, maxx, maxy = buf.bounds
                try:
                    rsel = roads_3857.cx[minx:maxx, miny:maxy]
                    if rsel is None or len(rsel) == 0:
                        rsel = roads_3857
                except Exception:
                    rsel = roads_3857
                for geom in (rsel.geometry if rsel is not None else []):
                    if geom is None:
                        continue
                    if geom.geom_type == "LineString":
                        candidates_lines = [geom]
                    else:
                        try:
                            candidates_lines = list(geom)
                        except Exception:
                            continue
                    for line in candidates_lines:
                        for _ in range(road_samples_per_line):
                            s = safe_sample_point(line, trans_3857_to_wgs)
                            if s is None:
                                continue
                            lon_c, lat_c = s
                            # flood check
                            depth = flood_depth_at(lon_c, lat_c)
                            if depth > flood_depth_threshold_m:
                                continue
                            # vector water check
                            if point_in_water_vector(lon_c, lat_c):
                                continue
                            s_val = slope_at(lon_c, lat_c)
                            if s_val > slope_threshold:
                                continue
                            droad = dist_to_roads_m(lon_c, lat_c)
                            if droad > road_buffer_m:
                                continue
                            J_pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                                           "slope": float(s_val), "dist_to_road_m": float(droad), "in_water": False,
                                           "flood_depth_m": float(depth)})
                            found += 1
                            if found >= candidate_per_cluster:
                                break
                        if found >= candidate_per_cluster:
                            break
                    if found >= candidate_per_cluster:
                        break
            except Exception:
                pass

            # multi-scale jitter around I to create more candidates
            for scale in extra_jitter_scales:
                for t in range(max(1, int(candidate_per_cluster // len(extra_jitter_scales)))):
                    try:
                        x0, y0 = lonlat_to_3857(base_lon, base_lat)
                        jx = x0 + np.random.normal(scale=scale)
                        jy = y0 + np.random.normal(scale=scale)
                        lon_c, lat_c = trans_3857_to_wgs.transform(jx, jy)
                    except Exception:
                        lon_c = base_lon + (np.random.normal(scale=scale) / 111000.0)
                        lat_c = base_lat + (np.random.normal(scale=scale) / 111000.0)
                    depth = flood_depth_at(lon_c, lat_c)
                    if depth > flood_depth_threshold_m:
                        continue
                    if point_in_water_vector(lon_c, lat_c):
                        continue
                    s_val = slope_at(lon_c, lat_c)
                    if s_val > slope_threshold:
                        continue
                    droad = dist_to_roads_m(lon_c, lat_c)
                    if droad > road_buffer_m:
                        continue
                    J_pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                                   "slope": float(s_val), "dist_to_road_m": float(droad), "in_water": False,
                                   "flood_depth_m": float(depth)})

    # ---------------- global grid sampling (spatial coverage) ----------------
    if boundary_gdf is None:
        try:
            boundary_gdf = read_geojson(str(CLEANED_DIR / "hue_boundary_clean.geojson"))
        except Exception:
            boundary_gdf = None

    if n_global_samples is None:
        n_global_samples = 500
    grid_pts = []
    try:
        if boundary_gdf is not None and not boundary_gdf.empty:
            b3857 = boundary_gdf.to_crs(epsg=3857)
            area = b3857.unary_union.area
            if n_global_samples > 0:
                spacing_m = max(250.0, math.sqrt(max(1.0, area / float(max(1, n_global_samples)))))
            else:
                spacing_m = 1000.0
            grid_pts = grid_sample_within_bounds(boundary_gdf, spacing_m, max_points=n_global_samples)
            for lon_c, lat_c in grid_pts:
                depth = flood_depth_at(lon_c, lat_c)
                if depth > flood_depth_threshold_m:
                    continue
                if point_in_water_vector(lon_c, lat_c):
                    continue
                s_val = slope_at(lon_c, lat_c)
                if s_val > slope_threshold:
                    continue
                droad = dist_to_roads_m(lon_c, lat_c)
                J_pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                               "slope": float(s_val), "dist_to_road_m": float(droad), "in_water": False,
                               "flood_depth_m": float(depth)})
    except Exception:
        pass

    # ------------- sparse random sampling along roads (global) -------------
    if roads_3857 is not None and len(roads_3857) > 0:
        cnt = 0
        for geom in roads_3857.geometry:
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                lines = [geom]
            else:
                try:
                    lines = list(geom)
                except:
                    continue
            for line in lines:
                s = safe_sample_point(line, trans_3857_to_wgs)
                if s is None:
                    continue
                lon_c, lat_c = s
                depth = flood_depth_at(lon_c, lat_c)
                if depth > flood_depth_threshold_m:
                    continue
                if point_in_water_vector(lon_c, lat_c):
                    continue
                s_val = slope_at(lon_c, lat_c)
                if s_val > slope_threshold:
                    continue
                droad = dist_to_roads_m(lon_c, lat_c)
                J_pool.append({"latitude": float(lat_c), "longitude": float(lon_c),
                               "slope": float(s_val), "dist_to_road_m": float(droad), "in_water": False,
                               "flood_depth_m": float(depth)})
                cnt += 1
                if cnt >= int(n_global_samples * 0.6):
                    break
            if cnt >= int(n_global_samples * 0.6):
                break

    # --------------- final pool postprocessing -----------------
    if slope_src is not None:
        slope_src.close()
    if flood_src is not None:
        flood_src.close()

    if not J_pool:
        return pd.DataFrame()

    J_df = pd.DataFrame(J_pool)

    # deduplicate by rounding lat/lon to given decimals (configurable)
    if dedup_round is None:
        J_df = J_df.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)
    else:
        # Use reasonable rounding: dedup_round is decimals -> guard between 4 and 7 decimals
        decimals = max(4, min(7, int(dedup_round)))
        J_df["lat_r"] = J_df["latitude"].round(decimals)
        J_df["lon_r"] = J_df["longitude"].round(decimals)
        J_df = J_df.drop_duplicates(subset=["lat_r", "lon_r"]).drop(columns=["lat_r", "lon_r"]).reset_index(drop=True)

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
    if J_pool is None or J_pool.empty:
        return pd.DataFrame()
    if pop_gdf is None or pop_gdf.empty:
        return farthest_point_sample(J_pool, target_J)

    pop_proj = pop_gdf.to_crs(epsg=3857)
    pop_coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
    pop_vals = pop_proj["pop"].values
    if len(pop_coords) == 0:
        return farthest_point_sample(J_pool, target_J)

    pop_kdt = cKDTree(pop_coords)

    J_geo = gpd.GeoDataFrame(J_pool.copy(), geometry=gpd.points_from_xy(J_pool.longitude, J_pool.latitude), crs="EPSG:4326")
    J_proj = J_geo.to_crs(epsg=3857)
    J_coords_3857 = np.vstack([J_proj.geometry.x.values, J_proj.geometry.y.values]).T

    radius = float(radius_m)
    J_pop_idx = []
    for x, y in J_coords_3857:
        idxs = pop_kdt.query_ball_point([x, y], r=radius)
        J_pop_idx.append(idxs)
    total_pops = [float(pop_vals[idxs].sum()) if len(idxs) else 0.0 for idxs in J_pop_idx]
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
        if frac < overlap_keep_threshold and unc_pop < min_uncov_pop_frac * pop_vals.sum():
            continue
        selected_idxs.append(idx)
        for i in uncovered:
            covered_mask[i] = True
        if len(selected_idxs) >= int(target_J * 0.5):
            break

    if len(selected_idxs) < int(target_J * 0.5):
        needed = int(target_J * 0.5) - len(selected_idxs)
        extra = [i for i in order if i not in selected_idxs]
        selected_idxs.extend(extra[:needed])

    selected_set = set(selected_idxs)

    if len(selected_idxs) >= target_J:
        final_idxs = selected_idxs[:target_J]
        sel = J_pool.iloc[final_idxs].copy().reset_index(drop=True)
        return sel

    remaining_needed = target_J - len(selected_idxs)
    if remaining_needed <= 0:
        sel = J_pool.iloc[selected_idxs].copy().reset_index(drop=True)
        return sel

    candidate_indices = [i for i in range(len(J_pool)) if i not in selected_set]
    cand_coords = J_coords_3857[candidate_indices]
    seed_coords = J_coords_3857[selected_idxs] if selected_idxs else np.empty((0,2))
    fps_selected_rel = farthest_point_sampling_indices(cand_coords, remaining_needed, seed_coords=seed_coords)
    fps_selected = [candidate_indices[i] for i in fps_selected_rel]

    final_idxs = selected_idxs + fps_selected
    if len(final_idxs) < target_J:
        filler = [i for i in order if i not in final_idxs]
        final_idxs.extend(filler[:(target_J - len(final_idxs))])

    final_idxs = final_idxs[:target_J]
    sel = J_pool.iloc[final_idxs].copy().reset_index(drop=True)
    return sel

# ---------------------- Farthest-point sampling helpers ----------------------
def farthest_point_sampling_indices(points, k, seed_coords=None):
    N = points.shape[0]
    if N == 0 or k <= 0:
        return []
    if k >= N:
        return list(range(N))
    if seed_coords is None or seed_coords.size == 0:
        idx0 = np.random.randint(0, N)
        selected = [idx0]
        dists = np.linalg.norm(points - points[idx0:idx0+1], axis=1)
    else:
        dists = np.min(np.linalg.norm(points[:, None, :] - seed_coords[None, :, :], axis=2), axis=1)
        selected = []
    while len(selected) < k:
        idx = int(np.argmax(dists))
        selected.append(idx)
        newd = np.linalg.norm(points - points[idx:idx+1], axis=1)
        dists = np.minimum(dists, newd)
        dists[idx] = -1.0
    return selected

def farthest_point_sample(J_pool: pd.DataFrame, target: int):
    if J_pool is None or J_pool.empty:
        return pd.DataFrame()
    J_geo = gpd.GeoDataFrame(J_pool.copy(), geometry=gpd.points_from_xy(J_pool.longitude, J_pool.latitude), crs="EPSG:4326")
    J_proj = J_geo.to_crs(epsg=3857)
    coords = np.vstack([J_proj.geometry.x.values, J_proj.geometry.y.values]).T
    sel_rel = farthest_point_sampling_indices(coords, k=min(target, len(coords)))
    sel = J_pool.iloc[sel_rel].copy().reset_index(drop=True)
    return sel

# ---------------------- Main selection wrapper ----------------------
def evaluate_candidates_and_reduce_overlap(I_df: pd.DataFrame,
                                           J_df: pd.DataFrame,
                                           pop_gdf: gpd.GeoDataFrame,
                                           radius_m: float = 3000,
                                           overlap_keep_threshold: float = 0.05,
                                           max_J: int = 1200,
                                           approximate_pop_cover: bool = True,
                                           config: dict = None):
    if J_df is None or J_df.empty:
        return pd.DataFrame()

    cfg = config or {}
    target_J = int(cfg.get("target_J", cfg.get("J_target", max_J)) if cfg is not None else max_J)
    target_J = min(max_J, target_J)
    strategy = cfg.get("J_strategy", cfg.get("strategy", "balanced"))
    min_uncov_pop_frac = float(cfg.get("min_uncov_pop_frac", 0.002))
    overlap_keep = float(cfg.get("overlap_keep_threshold", overlap_keep_threshold))

    selected = evaluate_and_select_J(I_df, J_df, pop_gdf, radius_m=radius_m,
                                     target_J=target_J,
                                     overlap_keep_threshold=overlap_keep,
                                     min_uncov_pop_frac=min_uncov_pop_frac,
                                     strategy=strategy)
    if selected is None or selected.empty:
        return pd.DataFrame()
    selected = selected.reset_index(drop=True)
    return selected

# ---------------------- MAIN ROUTINE ----------------------
def main(config: dict, out_dir: str or Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fe_cfg = config or {}
    pop_threshold = float(fe_cfg.get("pop_threshold", 5))
    eps_dbscan_m = float(fe_cfg.get("eps_dbscan_m", 1500))
    db_min_samples = int(fe_cfg.get("db_min_samples", 3))
    radius_I_m = float(fe_cfg.get("default_R", 3000))
    candidate_per_cluster = int(fe_cfg.get("candidate_per_cluster", 15))
    jitter_m = float(fe_cfg.get("candidate_jitter_m", 300))
    slope_threshold = float(fe_cfg.get("slope_threshold_deg", 30))
    road_buffer_m = float(fe_cfg.get("road_buffer_m", 4000))
    candidate_samples_global = int(fe_cfg.get("candidate_samples_global", 5000))
    max_I = int(fe_cfg.get("max_I", 1200))
    max_J = int(fe_cfg.get("max_J", 1500))
    dedup_round = int(fe_cfg.get("dedup_round", 5))
    seed = int(fe_cfg.get("seed", 42))
    pop_sample_max = int(fe_cfg.get("pop_sample_max", 12000))
    target_J = int(fe_cfg.get("target_J", 1000))
    global_sampling_level = fe_cfg.get("global_sampling", "medium")
    strategy = fe_cfg.get("J_strategy", fe_cfg.get("strategy", "balanced"))

    if isinstance(global_sampling_level, str):
        if global_sampling_level.lower() == "low":
            n_global_samples = int(fe_cfg.get("global_samples_low", 250))
        elif global_sampling_level.lower() == "high":
            n_global_samples = int(fe_cfg.get("global_samples_high", 1000))
        else:
            n_global_samples = int(fe_cfg.get("global_samples_medium", 500))
    else:
        n_global_samples = int(global_sampling_level)

    # load inputs
    boundary = gpd.read_file(CLEANED_DIR / "hue_boundary_clean.geojson")
    pop_tif = str(CLEANED_DIR / "pop_hue_clean.tif")
    slope_tif = str(CLEANED_DIR / "slope_hue_clean.tif")
    roads = read_geojson(str(CLEANED_DIR / "roads_hue_clean.geojson"))
    water = read_geojson(str(CLEANED_DIR / "water_hue_clean.geojson"))

    # flood raster override from config or default path
    flood_tif = fe_cfg.get("flood_tif", None)
    if flood_tif is None:
        # try environment default path defined earlier
        flood_tif = str(FLOOD_TIF_DEFAULT) if FLOOD_TIF_DEFAULT.exists() else None

    active_bts, failed_bts = _read_bts_files(DAMAGE_BTS_DIR)

    # split failed_bts into power_outage (treated operational) and hard failed
    failed_power = pd.DataFrame()
    failed_hard = pd.DataFrame()
    if failed_bts is not None and not failed_bts.empty:
        try:
            status_s = failed_bts.get("status", "")
            # normalize to string lower
            failed_power = failed_bts[status_s.astype(str).str.lower() == "power_outage"].copy()
            failed_hard = failed_bts[status_s.astype(str).str.lower() == "failed"].copy()
        except Exception:
            # fallback: treat whole as hard failed if no status column
            failed_hard = failed_bts.copy()

    np.random.seed(seed); random.seed(seed)

    print("1) Extract population cells from raster...")
    pop_gdf = extract_population_cells(pop_tif, threshold=pop_threshold)
    if (pop_gdf is None) or pop_gdf.empty:
        pop_gdf = gpd.GeoDataFrame(columns=["longitude","latitude","pop","geometry"], crs="EPSG:4326")
    if (boundary is not None) and (not pop_gdf.empty):
        try:
            pop_gdf = gpd.clip(pop_gdf, boundary)
        except Exception:
            pass

    print("2) Remove points covered by active BTS and power_outage BTS (treated as operational)...")
    pop_uncovered = remove_covered_by_operational_bts(pop_gdf, active_bts, failed_bts, default_radius_m=radius_I_m)
    print(f"   population cells before: {len(pop_gdf)}, after removing operational BTS coverage: {len(pop_uncovered)}")

    if len(pop_uncovered) > pop_sample_max:
        print(f"   Downsampling pop cells from {len(pop_uncovered)} to {pop_sample_max} for speed...")
        probs = pop_uncovered["pop"].values / pop_uncovered["pop"].sum()
        idxs = np.random.choice(len(pop_uncovered), size=pop_sample_max, replace=False, p=probs)
        pop_sampled = pop_uncovered.iloc[idxs].reset_index(drop=True)
    else:
        pop_sampled = pop_uncovered.copy()

    print("3) Cluster population points with DBSCAN...")
    clusters_df, pop_proj = cluster_population_dbscan(pop_sampled, eps_m=eps_dbscan_m, min_samples=db_min_samples)
    if clusters_df is None or clusters_df.empty:
        print("   No clusters found; fallback to top pop cells as I.")
        pop_sorted = pop_sampled.sort_values("pop", ascending=False).head(max_I)
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
        print("   Selecting clusters greedily to avoid overlap...")
        selected_clusters, covered_idx = select_clusters_nonoverlap(clusters_df, pop_proj, radius_m=radius_I_m)
        I_df = pd.DataFrame([{
            "site_id": row["site_id"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "pop": float(row["pop"]),
            "covered_cells": row["covered_cells"]
        } for _, row in selected_clusters.reset_index(drop=True).iterrows()])

    if I_df is None or I_df.empty:
        I_df = pd.DataFrame(columns=["site_id","latitude","longitude","pop","covered_cells"])
    if len(I_df) < max_I and len(I_df) > 0:
        extra_needed = max_I - len(I_df)
        jittered = []
        for k in range(extra_needed):
            base = I_df.sample(1, random_state=seed + k).iloc[0]
            lat_j = float(base["latitude"]) + np.random.normal(scale=0.001)
            lon_j = float(base["longitude"]) + np.random.normal(scale=0.001)
            base_pop = float(base.get("pop", 0.0))
            jittered.append({
                "site_id": f"I_{len(I_df) + len(jittered):05d}",
                "latitude": lat_j,
                "longitude": lon_j,
                "pop": base_pop * float(np.random.uniform(0.6, 1.2)),
                "covered_cells": base.get("covered_cells", [])
            })
        I_df = pd.concat([I_df, pd.DataFrame(jittered)], ignore_index=True)
    if len(I_df) > max_I:
        I_df = I_df.sort_values("pop", ascending=False).head(max_I).reset_index(drop=True)

    print("4) Assign priorities to I using infra layers if present...")
    infra_files = {}
    for name in ["schools_clean", "hospitals_clean", "medical_centers_clean", "industrial_clean", "residential_clean", "command_centers_clean"]:
        p = CLEANED_DIR / f"{name}.geojson"
        if p.exists():
            infra_files[name] = str(p)
    I_df = assign_priority_to_I(I_df, infra_files, buffer_m=fe_cfg.get('infra_buffer_m', 1500))

    print("5) Generate J candidate sites (upgraded with flood filtering)...")
    boundary_gdf = boundary if boundary is not None else None
    J_pool = generate_J_candidates(I_df, roads, water, slope_tif,
                                   flood_tif=flood_tif,
                                   candidate_per_cluster=candidate_per_cluster,
                                   jitter_m=jitter_m,
                                   slope_threshold=slope_threshold,
                                   road_buffer_m=road_buffer_m,
                                   n_global_samples=n_global_samples,
                                   seed=seed,
                                   dedup_round=dedup_round,
                                   road_samples_per_line=int(fe_cfg.get('road_samples_per_line', 2)),
                                   extra_jitter_scales=tuple(fe_cfg.get('extra_jitter_scales', (100,300,800))),
                                   boundary_gdf=boundary_gdf,
                                   flood_depth_threshold_m=float(fe_cfg.get('flood_depth_threshold_m',1.0))
                                   )

    if J_pool is None or J_pool.empty:
        print("   No J candidates produced. Exiting with I only.")
        J_df = pd.DataFrame()
    else:
        print(f"   Candidate pool generated: {len(J_pool)} candidates")
        sel = evaluate_candidates_and_reduce_overlap(I_df, J_pool, pop_uncovered,
                                                     radius_m=radius_I_m,
                                                     overlap_keep_threshold=fe_cfg.get('overlap_keep_threshold', 0.05),
                                                     max_J=max_J,
                                                     approximate_pop_cover=True,
                                                     config={"target_J": target_J, "J_strategy": strategy,
                                                             "min_uncov_pop_frac": fe_cfg.get('min_uncov_pop_frac', 0.002),
                                                             "overlap_keep_threshold": fe_cfg.get('overlap_keep_threshold', 0.05)})
        if sel is None or sel.empty:
            print("   Selection failed; falling back to top candidates by pop or spatial.")
            J_df = J_pool.head(min(max_J, len(J_pool))).reset_index(drop=True)
        else:
            J_df = sel.copy()
            if (I_df is not None) and (not I_df.empty):
                I_coords = I_df[['latitude','longitude']].rename(columns={'latitude':'y','longitude':'x'}).to_dict('records')
                J_coords = J_df[['latitude','longitude']].rename(columns={'latitude':'y','longitude':'x'}).to_dict('records')
                dist_IJ = compute_distance_matrix(I_coords, J_coords, metric='haversine')
                nearest = np.argmin(dist_IJ, axis=0)
                i_refs, pweights, pcats, pops = [], [], [], []
                for k, nidx in enumerate(nearest):
                    try:
                        nidx = int(nidx)
                        i_refs.append(str(I_df.loc[nidx, "site_id"]))
                        pweights.append(float(I_df.loc[nidx, "priority_weight"]) if "priority_weight" in I_df.columns else 1.0)
                        pcats.append(str(I_df.loc[nidx, "priority_category"]) if "priority_category" in I_df.columns else "normal")
                        pops.append(0.0)
                    except Exception:
                        i_refs.append(None); pweights.append(1.0); pcats.append("normal"); pops.append(0.0)
                J_df["i_ref"] = i_refs
                J_df["priority_weight"] = pweights
                J_df["priority_category"] = pcats
                try:
                    pop_proj = pop_uncovered.to_crs(epsg=3857)
                    pop_coords = np.vstack([pop_proj.geometry.x.values, pop_proj.geometry.y.values]).T
                    pop_vals = pop_proj["pop"].values
                    pop_kdt = cKDTree(pop_coords)
                    J_geo = gpd.GeoDataFrame(J_df.copy(), geometry=gpd.points_from_xy(J_df.longitude, J_df.latitude), crs="EPSG:4326").to_crs(epsg=3857)
                    cover_idx = []
                    for _, r in J_geo.iterrows():
                        cx, cy = r.geometry.x, r.geometry.y
                        idxs = pop_kdt.query_ball_point([cx, cy], r=radius_I_m)
                        cover_idx.append(idxs)
                    pops = [float(pop_vals[idxs].sum()) if len(idx) else 0.0 for idxs in cover_idx]
                    J_df["pop"] = pops
                except Exception:
                    J_df["pop"] = 0.0
            else:
                J_df["i_ref"] = None
                J_df["priority_weight"] = 1.0
                J_df["priority_category"] = "normal"
                J_df["pop"] = 0.0

    # Standardize output columns
    if J_df is None or J_df.empty:
        J_out = pd.DataFrame(columns=[
            "site_id","i_ref","latitude","longitude","pop","priority_category","priority_weight","slope","dist_to_road_m","in_water"
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
                except Exception:
                    pass
        if "priority_category" not in J_df.columns:
            J_df["priority_category"] = "normal"
        if "priority_weight" not in J_df.columns:
            J_df["priority_weight"] = 1.0
        if "i_ref" not in J_df.columns:
            J_df["i_ref"] = np.nan
        if "in_water" not in J_df.columns:
            J_df["in_water"] = False
        # ensure flood_depth_m present (optional), but we will not include it in final columns to preserve output format
        if "flood_depth_m" not in J_df.columns:
            J_df["flood_depth_m"] = 0.0
        cols = ["site_id","i_ref","latitude","longitude","pop","priority_category","priority_weight","slope","dist_to_road_m","in_water"]
        for col in cols:
            if col not in J_df.columns:
                J_df[col] = np.nan
        J_out = J_df[cols].copy().reset_index(drop=True)

    # Prepare I output (augment with priority fields if missing)
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

    # Write outputs
    I_out.to_csv(Path(out_dir) / "I_points.csv", index=False)
    J_out.to_csv(Path(out_dir) / "J_sites.csv", index=False)

    # compute cover matrix (I x J)
    if (len(J_out) > 0) and (len(I_out) > 0):
        I_coords = I_out[['latitude','longitude']].rename(columns={'latitude':'y','longitude':'x'}).to_dict('records')
        J_coords = J_out[['latitude','longitude']].rename(columns={'latitude':'y','longitude':'x'}).to_dict('records')
        dist = compute_distance_matrix(I_coords, J_coords, metric='haversine')
        cover = (dist <= float(radius_I_m)).astype(int)
    else:
        cover = np.zeros((len(I_out), len(J_out)), dtype=int)
    np.save(Path(out_dir) / "cover.npy", cover)

    print(f"[Feature Extraction - FINAL] Saved {len(I_out)} I_points, {len(J_out)} J_sites to {out_dir}")
    return {"I_points": Path(out_dir) / "I_points.csv", "J_sites": Path(out_dir) / "J_sites.csv", "cover": Path(out_dir) / "cover.npy"}


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
