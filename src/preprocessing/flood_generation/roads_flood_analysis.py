# roads_flood_analysis.py
"""
Sample combined flood depth along road network.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
import rasterio
from tqdm import tqdm
from .utils import save_geojson
import logging

logger = logging.getLogger(__name__)


def sample_road_depth(road_geom: LineString, depth_raster, sampling_m=5.0):
    """
    Densify road and sample flood depth.
    """
    if not isinstance(road_geom, LineString):
        return 0.0, 0.0

    n = max(2, int(road_geom.length / sampling_m))
    pts = [road_geom.interpolate(i / (n - 1), normalized=True) for i in range(n)]
    coords = [(p.x, p.y) for p in pts]

    with rasterio.open(depth_raster) as src:
        vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")

    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 0.0

    return float(vals.mean()), float(vals.max())


def analyze_roads(roads_path: str, depth_path: str, out_path: str, sampling_m: float):
    roads = gpd.read_file(roads_path)
    added = []

    logger.info(f"[ROADS] Sampling flood depth for {len(roads)} road segments")

    for _, row in tqdm(roads.iterrows(), total=len(roads)):
        mean_d, max_d = sample_road_depth(row.geometry, depth_path, sampling_m)
        added.append({"depth_mean": mean_d, "depth_max": max_d})

    df_add = gpd.GeoDataFrame(added, geometry=roads.geometry, crs=roads.crs)
    save_geojson(df_add, out_path)

    logger.info(f"[ROADS] Saved flooded roads: {out_path}")
    return df_add
