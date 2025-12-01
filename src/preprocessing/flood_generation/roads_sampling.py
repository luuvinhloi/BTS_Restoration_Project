"""
roads_sampling.py
Functions to sample depth raster along road segments and compute segment-level stats and traversability.
"""
from typing import Tuple, Dict, Any
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Point
from shapely.ops import split
import logging
from tqdm import tqdm
from .utils import save_geojson

logger = logging.getLogger(__name__)


def densify_linestring(ls: LineString, max_segment_length: float) -> LineString:
    """
    Return a LineString densified so that consecutive vertices are at most max_segment_length apart.
    """
    if ls.length == 0:
        return ls
    num_vert = max(int(np.ceil(ls.length / max_segment_length)) + 1, 2)
    points = [ls.interpolate(float(i) / (num_vert - 1), normalized=True) for i in range(num_vert)]
    return LineString(points)


def sample_raster_at_points(raster_path: str, points: [Point]) -> np.ndarray:
    """
    Sample raster value at a list of shapely Points (x,y). Returns 1D numpy array of sampled values (float32).
    Uses nearest sampling (rasterio.sample).
    """
    coords = [(p.x, p.y) for p in points]
    with rasterio.open(raster_path) as src:
        vals = list(src.sample(coords))
    arr = np.array([v[0] if v is not None else np.nan for v in vals], dtype='float32')
    return arr


def compute_segment_depth_stats(road_geom: LineString, raster_path: str,
                                sampling_m: float = 5.0) -> Dict[str, Any]:
    """
    Densify a road, sample depth raster along it, and compute statistics.
    Returns: {depth_mean, depth_median, depth_max, depth_std, n_samples}
    """
    if road_geom is None or road_geom.length == 0:
        return {'depth_mean': 0.0, 'depth_median': 0.0, 'depth_max': 0.0, 'depth_std': 0.0, 'n_samples': 0}

    ls_dense = densify_linestring(road_geom, sampling_m)
    sample_points = [Point(x, y) for x, y in list(ls_dense.coords)]
    vals = sample_raster_at_points(raster_path, sample_points)
    # remove nan / nodata
    valid = np.isfinite(vals)
    if not valid.any():
        return {'depth_mean': 0.0, 'depth_median': 0.0, 'depth_max': 0.0, 'depth_std': 0.0, 'n_samples': 0}
    arr = vals[valid]
    return {
        'depth_mean': float(np.nanmean(arr)),
        'depth_median': float(np.nanmedian(arr)),
        'depth_max': float(np.nanmax(arr)),
        'depth_std': float(np.nanstd(arr)),
        'n_samples': int(arr.size)
    }


def assign_traversability(depth_mean: float, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Given mean depth and thresholds dict {'ok':0.2, 'partial':0.5}, return traversability info.
    """
    ok_th = thresholds.get('ok', 0.2)
    part_th = thresholds.get('partial', 0.5)

    if depth_mean >= part_th:
        status = 'BLOCKED'
        speed_factor = 0.0
        cost_mult = float('inf')
    elif depth_mean >= ok_th:
        status = 'PARTIAL'
        speed_factor = 0.4
        cost_mult = 1.6
    elif depth_mean > 0:
        status = 'OK_with_penalty'
        speed_factor = 0.8
        cost_mult = 1.1
    else:
        status = 'DRY'
        speed_factor = 1.0
        cost_mult = 1.0

    return {
        'traversability': status,
        'speed_factor': speed_factor,
        'cost_multiplier': cost_mult
    }


def sample_all_roads(roads_gdf: gpd.GeoDataFrame, depth_raster_path: str, sampling_m: float,
                     thresholds: Dict[str, float]) -> gpd.GeoDataFrame:
    """
    For each road segment in roads_gdf, sample depth and compute fields, return a new GeoDataFrame with added columns.
    """
    out = roads_gdf.copy()
    added = []
    logger.info("Sampling %d road segments with sampling_m=%s", len(roads_gdf), sampling_m)
    for idx, row in tqdm(out.iterrows(), total=len(out)):
        geom = row.geometry
        stats = compute_segment_depth_stats(geom, depth_raster_path, sampling_m=sampling_m)
        trav = assign_traversability(stats['depth_mean'], thresholds)
        added.append({**stats, **trav})
    # attach columns
    added_df = gpd.GeoDataFrame(added, index=out.index)
    result = out.join(added_df)
    # fill expected columns
    result['length_m'] = result.geometry.length
    # export ready
    return result
