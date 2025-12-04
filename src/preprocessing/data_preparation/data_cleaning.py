# src/preprocessing/data_preparation/data_cleaning.py
"""
DATA CLEANING PIPELINE — FIXED & IMPROVED
-----------------------------------------

Key Fixes:
- Clean DEM, water, polygons, roads, points strictly INSIDE boundary
- Remove buggy 'land-mask' 108.30° — keep real boundary only
- Repair geometries safely
- Clip rasters using exact boundary CRS transform
- Ensure CRS consistency throughout
"""

from pathlib import Path
import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Directories
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# CRS
VECTOR_OUT_CRS = 4326
METRIC_CRS = 3857
DEFAULT_ROAD_SPEED = 40


# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = VECTOR_OUT_CRS):
    if gdf.crs is None:
        logger.warning("CRS missing → assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    if gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg)
    return gdf


def safe_write(gdf, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = ensure_crs(gdf, VECTOR_OUT_CRS)
    gdf.to_file(path, driver="GeoJSON")
    logger.info("Saved %s (%d features)", path.name, len(gdf))


def repair(geom):
    if geom is None:
        return None
    try:
        if geom.is_valid:
            return geom
        g = geom.buffer(0)
        if g.is_valid:
            return g
    except:
        return None
    return None


# ----------------------------------------------------------
# CLEAN BOUNDARY
# ----------------------------------------------------------
def clean_boundary(raw_name="hue_boundary.geojson"):
    src = RAW_DIR / raw_name
    if not src.exists():
        raise FileNotFoundError(f"Boundary missing: {src}")

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf)

    # Fix geometry
    gdf["geometry"] = gdf.geometry.apply(repair)
    gdf = gdf[gdf.geometry.notna()]

    # Dissolve into single polygon
    unified = unary_union(gdf.geometry)
    boundary = gpd.GeoDataFrame([{"geometry": unified}], crs=f"EPSG:{VECTOR_OUT_CRS}")

    out = CLEAN_DIR / "hue_boundary_clean.geojson"
    safe_write(boundary, out)

    logger.info("Boundary cleaned OK → %s", out)
    return boundary


# ----------------------------------------------------------
# CLIP RASTER WITH STRICT BOUNDARY
# ----------------------------------------------------------
def clip_raster_strict(src_path: Path, boundary_gdf: gpd.GeoDataFrame, out_path: Path):
    with rasterio.open(src_path) as src:
        rast_crs = src.crs
        boundary = boundary_gdf.to_crs(rast_crs)

        geo = [mapping(boundary.unary_union)]
        clipped, transform = mask(src, geo, crop=True)

        meta = src.meta.copy()
        meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform,
            "compress": "lzw"
        })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(clipped)
    logger.info("Raster clipped → %s", out_path)
    return out_path


# ----------------------------------------------------------
# CLEAN POLYGON LAYER
# ----------------------------------------------------------
def clean_polygon(raw_name, out_name, boundary):
    src = RAW_DIR / raw_name
    out = CLEAN_DIR / out_name

    if not src.exists():
        safe_write(gpd.GeoDataFrame(columns=["geometry"], crs=VECTOR_OUT_CRS), out)
        return gpd.GeoDataFrame()

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf)
    gdf["geometry"] = gdf.geometry.apply(repair)
    gdf = gdf[gdf.geometry.notna()]

    # INTERSECTION WITH BOUNDARY
    try:
        clipped = gpd.overlay(gdf, boundary, how="intersection")
    except:
        clipped = gdf[gdf.geometry.intersects(boundary.unary_union)]

    clipped = clipped[clipped.geometry.notna()]
    safe_write(clipped, out)
    return clipped


# ----------------------------------------------------------
# CLEAN POINT LAYER
# ----------------------------------------------------------
def clean_points(raw, out_name, boundary):
    src = RAW_DIR / raw
    out = CLEAN_DIR / out_name

    if not src.exists():
        safe_write(gpd.GeoDataFrame(columns=["geometry"], crs=VECTOR_OUT_CRS), out)
        return gpd.GeoDataFrame()

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf)
    gdf["geometry"] = gdf.geometry.apply(repair)
    gdf = gdf[gdf.geometry.notna()]

    try:
        clipped = gpd.overlay(gdf, boundary, how="intersection")
    except:
        clipped = gdf[gdf.geometry.within(boundary.unary_union)]

    clipped = clipped[clipped.geometry.notna()]
    safe_write(clipped, out)
    return clipped


# ----------------------------------------------------------
# CLEAN ROADS
# ----------------------------------------------------------
def clean_roads(boundary, raw_name="roads_hue.geojson"):
    src = RAW_DIR / raw_name
    out = CLEAN_DIR / "roads_hue_clean.geojson"

    if not src.exists():
        safe_write(gpd.GeoDataFrame(columns=["geometry"], crs=VECTOR_OUT_CRS), out)
        return gpd.GeoDataFrame()

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf)
    gdf["geometry"] = gdf.geometry.apply(repair)
    gdf = gdf[gdf.geometry.notna()]

    # CLIP
    try:
        clipped = gpd.overlay(gdf, boundary, how="intersection")
    except:
        clipped = gdf[gdf.geometry.intersects(boundary.unary_union)]

    clipped = clipped[clipped.geometry.notna()]

    # compute length
    metric = clipped.to_crs(METRIC_CRS)
    clipped["length_m"] = metric.length
    clipped["speed_kmh"] = DEFAULT_ROAD_SPEED

    safe_write(clipped, out)
    return clipped


# ----------------------------------------------------------
# CLEAN RASTER (DEM, SLOPE...)
# ----------------------------------------------------------
def clean_raster(name: str, boundary):
    src = RAW_DIR / name
    if not src.exists():
        logger.warning("Missing raster: %s", name)
        return None
    out = CLEAN_DIR / name.replace(".tif", "_clean.tif")
    return clip_raster_strict(src, boundary, out)


# ----------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------
def run_cleaning_pipeline():
    logger.info("=== CLEANING PIPELINE START ===")

    boundary = clean_boundary()

    # ROADS
    clean_roads(boundary)

    # POINTS
    clean_points("schools.geojson", "schools_clean.geojson", boundary)
    clean_points("hospitals.geojson", "hospitals_clean.geojson", boundary)
    clean_points("residential.geojson", "residential_clean.geojson", boundary)
    clean_points("medical_centers.geojson", "medical_centers_clean.geojson", boundary)
    clean_points("command_centers.geojson", "command_centers_clean.geojson", boundary)

    # POLYGONS
    clean_polygon("industrial.geojson", "industrial_clean.geojson", boundary)
    clean_polygon("water_hue.geojson", "water_hue_clean.geojson", boundary)

    # RASTERS (DEM MUST BE CLEANED!!)
    clean_raster("elev_hue.tif", boundary)
    clean_raster("slope_hue.tif", boundary)
    clean_raster("pop_hue.tif", boundary)

    logger.info("=== CLEANING PIPELINE COMPLETE ===")
    logger.info("Clean data saved in: %s", CLEAN_DIR)


if __name__ == "__main__":
    run_cleaning_pipeline()
