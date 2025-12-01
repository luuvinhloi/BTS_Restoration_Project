# depth_to_polygon.py
"""
Convert flood depth raster → polygon GeoJSON
"""

import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import rasterio
import numpy as np
from .utils import save_geojson
import logging

logger = logging.getLogger(__name__)


def depth_raster_to_polygon(depth_path: str, out_geojson: str, min_depth=0.05):
    """
    Convert depth raster to polygon where depth > min_depth.
    """
    with rasterio.open(depth_path) as src:
        arr = src.read(1)
        mask = arr > min_depth
        crs = src.crs
        transform = src.transform

    geoms = []
    for geom, val in shapes(arr, mask=mask, transform=transform):
        geoms.append(shape(geom))

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["depth_mean"] = min_depth

    save_geojson(gdf, out_geojson)
    logger.info(f"[VECTOR] Saved polygons: {out_geojson}")

    return gdf
