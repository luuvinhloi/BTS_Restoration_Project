# src/preprocessing/data_preparation/data_cleaning.py
"""
Data Cleaning & Normalization Pipeline – Stage 1

This module standardizes vector & raster inputs under data/raw and writes cleaned
artifacts to data/cleaned/.

Key behaviours:
- unify CRS (default output: EPSG:4326 for vector GeoJSON)
- repair invalid geometries (buffer(0) fallback)
- normalize road geometries (LineString-only)
- clip rasters to study boundary (transforming geometries to raster CRS if needed)
- produce safe empty templates when source missing
- compute simple attributes (lon/lat, length_m, area_m2)
"""

from pathlib import Path
import logging
from typing import Optional, Dict, Any, List

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import Point, mapping, shape, LineString, Polygon, MultiPolygon, MultiLineString, GeometryCollection
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Project-level directories
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Defaults
VECTOR_OUT_CRS = 4326        # output GeoJSON CRS
METRIC_CRS = 3857            # metric CRS for length/area computations (Web Mercator, fast)
DEFAULT_ROAD_SPEED = 40      # km/h default

# -------------------------
# Helper utility functions
# -------------------------
def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = VECTOR_OUT_CRS) -> gpd.GeoDataFrame:
    """
    Ensure GeoDataFrame has a CRS and is reprojected to target EPSG.
    If gdf.crs is None, assume EPSG:4326 (safe default) then reproject.
    """
    if gdf is None:
        raise ValueError("None GeoDataFrame provided to ensure_crs")
    if gdf.crs is None:
        logger.warning("GeoDataFrame missing CRS; assuming EPSG:4326")
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    try:
        if gdf.crs.to_epsg() != epsg:
            gdf = gdf.to_crs(epsg=epsg)
    except Exception:
        # robust fallback
        gdf = gdf.to_crs(epsg=epsg)
    return gdf


def project_to_metric(gdf: gpd.GeoDataFrame, metric_epsg: int = METRIC_CRS) -> gpd.GeoDataFrame:
    """Return GeoDataFrame projected to metric CRS for accurate length/area."""
    return gdf.to_crs(epsg=metric_epsg)


def safe_write_geojson(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Write GeoDataFrame to GeoJSON, ensuring directory exists and using EPSG:4326."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure output in 4326 for consistent downstream usage
    try:
        gdf_out = ensure_crs(gdf, epsg=VECTOR_OUT_CRS)
    except Exception:
        # unexpected; try writing original
        gdf_out = gdf
    gdf_out.to_file(out_path, driver="GeoJSON")
    logger.info("Wrote GeoJSON to %s (count=%d)", out_path, len(gdf_out) if hasattr(gdf_out, "__len__") else 0)


def repair_geometry(geom):
    """
    Attempt to repair a single geometry:
    - if valid -> return
    - else try buffer(0)
    - if still invalid or None -> return None
    """
    if geom is None:
        return None
    try:
        if geom.is_valid:
            return geom
        repaired = geom.buffer(0)
        if repaired.is_valid and not repaired.is_empty:
            return repaired
    except Exception:
        pass
    return None


def normalize_geometry_to_linestring(geom):
    """
    Convert geometry into a representative LineString if possible:
    - LineString -> return as-is
    - MultiLineString -> take longest LineString component
    - Polygon/MultiPolygon -> return exterior boundary (LineString)
    - GeometryCollection -> try to extract a LineString
    - Point or unsupported -> return None
    """
    if geom is None:
        return None
    gt = geom.geom_type
    try:
        if gt == "LineString":
            return geom
        if gt == "MultiLineString":
            parts = list(geom.geoms)
            if not parts:
                return None
            return max(parts, key=lambda g: g.length)
        if gt == "Polygon":
            return geom.boundary
        if gt == "MultiPolygon":
            parts = list(geom.geoms)
            if not parts:
                return None
            largest = max(parts, key=lambda g: g.area)
            return largest.boundary
        if gt == "GeometryCollection":
            for g in geom.geoms:
                if g.geom_type == "LineString":
                    return g
                if g.geom_type == "MultiLineString":
                    return max(list(g.geoms), key=lambda x: x.length) if list(g.geoms) else None
            return None
        # Points or other types -> not usable as road geometry
        return None
    except Exception:
        return None


def clip_raster(src_path: Path, boundary_gdf: gpd.GeoDataFrame, out_path: Path) -> Path:
    """
    Clip raster by a vector boundary.
    - Reprojects boundary geometry into raster CRS if necessary.
    - Writes clipped raster to out_path.
    """
    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        rast_crs = src.crs
        # prepare geometries in raster CRS
        # unify boundary (may be multi-feature)
        union_geom = unary_union(boundary_gdf.geometry.values)
        geom_mapping = mapping(union_geom)
        # transform geom mapping to raster CRS if boundary not in same CRS
        try:
            b_crs = boundary_gdf.crs
            if b_crs is None:
                # assume VECTOR_OUT_CRS
                b_crs = {'init': f'epsg:{VECTOR_OUT_CRS}'}
            if rast_crs is None:
                raise ValueError("Raster has no CRS.")
            if hasattr(b_crs, "to_string"):
                src_crs = b_crs.to_string()
            else:
                src_crs = boundary_gdf.crs.to_string()
            # perform transform if needed
            if src_crs != rast_crs.to_string():
                # rasterio.transform_geom expects dict strings
                geom_trans = transform_geom(src_crs, rast_crs.to_string(), geom_mapping)
                geoms = [geom_trans]
            else:
                geoms = [geom_mapping]
        except Exception:
            # fallback: try writing boundary as-is; let rasterio handle if same
            geoms = [geom_mapping]

        # perform clipping
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
    logger.info("Clipped raster %s -> %s", src_path, out_path)
    return out_path


# -------------------------
# Cleaning functions
# -------------------------
def clean_boundary(raw_name: str = "hue_boundary.geojson") -> gpd.GeoDataFrame:
    src = RAW_DIR / raw_name
    if not src.exists():
        raise FileNotFoundError(f"Boundary file not found at {src}")
    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf, epsg=VECTOR_OUT_CRS)
    # repair geometries
    gdf["geometry"] = gdf.geometry.apply(repair_geometry)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    # dissolve into single polygon (study area)
    try:
        unified = unary_union(gdf.geometry.values)
        gdf = gpd.GeoDataFrame([{"geometry": unified}], crs=f"EPSG:{VECTOR_OUT_CRS}")
    except Exception:
        # fallback: keep original
        pass
    out_path = CLEAN_DIR / "hue_boundary_clean.geojson"
    safe_write_geojson(gdf, out_path)
    return gdf


def clean_roads(boundary_gdf: gpd.GeoDataFrame, raw_name: str = "roads_hue.geojson", out_name: str = "roads_hue_clean.geojson") -> gpd.GeoDataFrame:
    src = RAW_DIR / raw_name
    out_path = CLEAN_DIR / out_name
    if not src.exists():
        logger.warning("Roads file not found at %s; writing empty template", src)
        cols = ["edge_id", "length_m", "speed_kmh", "geometry"]
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf, epsg=VECTOR_OUT_CRS)
    # repair geometries
    gdf["geometry"] = gdf.geometry.apply(repair_geometry)

    # spatial intersection with boundary: try overlay then fallback to intersects
    try:
        clipped = gpd.overlay(gdf, boundary_gdf, how="intersection")
    except Exception:
        clipped = gdf[gdf.geometry.intersects(boundary_gdf.unary_union)]

    if clipped.empty:
        logger.warning("No road features after clipping. Writing empty template.")
        cols = ["edge_id", "length_m", "speed_kmh", "geometry"]
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    # normalize geometry types to LineString where possible
    logger.info("Normalizing road geometries to LineString where possible...")
    clipped["geometry"] = clipped.geometry.apply(normalize_geometry_to_linestring)
    clipped = clipped[clipped.geometry.notna()]
    clipped = clipped[~clipped.geometry.is_empty]

    if clipped.empty:
        logger.warning("No valid LineString roads left after normalization.")
        cols = ["edge_id", "length_m", "speed_kmh", "geometry"]
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    # compute length in meters (project to metric CRS)
    try:
        metric = clipped.to_crs(epsg=METRIC_CRS)
        clipped["length_m"] = metric.geometry.length
    except Exception:
        clipped["length_m"] = clipped.geometry.length

    clipped = clipped.reset_index(drop=True)
    clipped["edge_id"] = clipped.index.astype(int)
    if "speed_kmh" not in clipped.columns:
        clipped["speed_kmh"] = DEFAULT_ROAD_SPEED

    safe_write_geojson(clipped, out_path)
    return clipped


def clean_point_layer(filename: str, out_name: str, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src = RAW_DIR / filename
    out_path = CLEAN_DIR / out_name
    cols = ["id", "lon", "lat", "type", "geometry"]

    if not src.exists():
        logger.warning("%s not found; writing empty template %s", filename, out_name)
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf, epsg=VECTOR_OUT_CRS)
    gdf["geometry"] = gdf.geometry.apply(repair_geometry)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    if gdf.empty:
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    # clip/intersect
    try:
        clipped = gpd.overlay(gdf, boundary_gdf, how="intersection")
    except Exception:
        clipped = gdf[gdf.geometry.intersects(boundary_gdf.unary_union)]

    if clipped.empty:
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    # ensure point-type: convert non-point to centroid
    def to_point(geom):
        if geom is None:
            return None
        if geom.geom_type == "Point":
            return geom
        return geom.centroid

    clipped["geometry"] = clipped.geometry.apply(to_point)
    clipped = clipped[clipped.geometry.notna()]

    # standardize schema
    clipped = clipped.reset_index(drop=True)
    clipped["id"] = clipped.index.astype(int)
    centroids = clipped.geometry.centroid
    clipped["lon"] = centroids.x
    clipped["lat"] = centroids.y
    if "type" not in clipped.columns:
        clipped["type"] = "unknown"

    safe_write_geojson(clipped, out_path)
    return clipped


def clean_polygon_layer(filename: str, out_name: str, boundary_gdf: gpd.GeoDataFrame, compute_area: bool = True) -> gpd.GeoDataFrame:
    src = RAW_DIR / filename
    out_path = CLEAN_DIR / out_name
    cols = ["id", "type", "lon", "lat", "area_m2", "geometry"]

    if not src.exists():
        logger.warning("%s not found; writing empty template %s", filename, out_name)
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf, epsg=VECTOR_OUT_CRS)
    gdf["geometry"] = gdf.geometry.apply(repair_geometry)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    if gdf.empty:
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    try:
        clipped = gpd.overlay(gdf, boundary_gdf, how="intersection")
    except Exception:
        clipped = gdf[gdf.geometry.intersects(boundary_gdf.unary_union)]

    if clipped.empty:
        empty = gpd.GeoDataFrame(columns=cols, geometry=None, crs=f"EPSG:{VECTOR_OUT_CRS}")
        safe_write_geojson(empty, out_path)
        return empty

    clipped = clipped.reset_index(drop=True)
    clipped["id"] = clipped.index.astype(int)
    if "type" not in clipped.columns:
        clipped["type"] = "unknown"

    centroids = clipped.geometry.centroid
    clipped["lon"] = centroids.x
    clipped["lat"] = centroids.y

    if compute_area:
        try:
            metric = clipped.to_crs(epsg=METRIC_CRS)
            clipped["area_m2"] = metric.geometry.area
        except Exception:
            clipped["area_m2"] = None

    safe_write_geojson(clipped, out_path)
    return clipped


def clean_raster(name: str, boundary_gdf: gpd.GeoDataFrame) -> Optional[Path]:
    """
    Clip raster (e.g., DEM, slope, population) to boundary and write *_clean.tif.
    Returns path to cleaned raster.
    """
    src = RAW_DIR / name
    if not src.exists():
        logger.warning("Raster %s not found at %s", name, src)
        return None
    out_path = CLEAN_DIR / name.replace(".tif", "_clean.tif")
    try:
        return clip_raster(src, boundary_gdf, out_path)
    except Exception as e:
        logger.exception("Failed to clip raster %s: %s", src, e)
        return None


def assert_points_within_boundary(points_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame, label: str) -> bool:
    """Ensure all points are within boundary. Raises ValueError if not."""
    if points_gdf is None or points_gdf.empty:
        return True
    mask = points_gdf.geometry.within(boundary_gdf.unary_union)
    if not mask.all():
        missing = (~mask).sum()
        raise ValueError(f"{missing} {label} points lie outside boundary.")
    return True


# -------------------------
# Main pipeline runner
# -------------------------
def run_cleaning_pipeline():
    logger.info("Starting data cleaning pipeline...")

    # 1. Boundary
    boundary = clean_boundary()

    # 2. Roads (special handling)
    roads = clean_roads(boundary)

    # 3. Point layers
    schools = clean_point_layer("schools.geojson", "schools_clean.geojson", boundary)
    hospitals = clean_point_layer("hospitals.geojson", "hospitals_clean.geojson", boundary)
    residential = clean_point_layer("residential.geojson", "residential_clean.geojson", boundary)
    command_centers = clean_point_layer("command_centers.geojson", "command_centers_clean.geojson", boundary)
    medical_centers = clean_point_layer("medical_centers.geojson", "medical_centers_clean.geojson", boundary)

    # 4. Polygon layers
    industrial = clean_polygon_layer("industrial.geojson", "industrial_clean.geojson", boundary)
    water = clean_polygon_layer("water_hue.geojson", "water_hue_clean.geojson", boundary)

    # 5. Raster layers (attempt clipping; missing files are warned)
    for rname in ["elev_hue.tif", "slope_hue.tif", "pop_hue.tif"]:
        try:
            out_r = clean_raster(rname, boundary)
            if out_r:
                logger.info("Raster cleaned: %s", out_r)
        except Exception as e:
            logger.exception("Raster cleaning failed for %s: %s", rname, e)

    logger.info("DATA CLEANING COMPLETED")
    logger.info("Cleaned data saved at: %s", CLEAN_DIR)


if __name__ == "__main__":
    run_cleaning_pipeline()
