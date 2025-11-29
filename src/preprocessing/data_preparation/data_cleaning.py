# src/preprocessing/data_preparation/data_cleaning.py
"""
Data Cleaning & Normalization Pipeline – Stage 1

Objectives:
- Standardize all vector and raster data in data/raw
- Check and unify the CRS (Coordinate Reference System)
- Fix geometry errors (invalid geometry)
- Standardize the schema (id, lon, lat, type…)
- Clip all rasters to the Hue boundary
- Validate data integrity (assert BTS is within the boundary)

Output will be stored in: data/cleaned/
"""

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
CLEAN_DIR = Path(__file__).resolve().parents[3] / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Utility Functions
def ensure_crs(gdf, epsg=4326):
    """Ensure geodataframe has a CRS. Set if missing, then reproject to epsg."""
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg)
    else:
        # if it's not the target, convert
        if gdf.crs.to_epsg() != epsg:
            gdf = gdf.to_crs(epsg=epsg)
    return gdf

def project_to_metric(gdf, metric_epsg=3857):
    """Return a copy of gdf reprojected to a metric CRS for distance/length calculations."""
    return gdf.to_crs(epsg=metric_epsg)

def fix_invalid_geometries(gdf):
    """
    Fix invalid geometries:
    - For invalid ones, attempt buffer(0)
    - Drop empty / null geometries afterwards
    """
    # Work on copy
    gdf = gdf.copy()
    def repair_geom(g):
        try:
            if g is None:
                return None
            if g.is_valid:
                return g
            # buffer(0) often fixes simple invalid geometries
            repaired = g.buffer(0)
            if repaired.is_valid:
                return repaired
            return repaired  # may still be invalid but non-null
        except Exception:
            return None

    gdf["geometry"] = gdf["geometry"].apply(repair_geom)
    # drop None or empty
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf

def clip_raster(src_path, boundary_gdf, out_path):
    """Clip raster theo boundary tỉnh Huế."""
    with rasterio.open(src_path) as src:
        geoms = [boundary_gdf.unary_union]
        out_image, out_transform = mask(src, geoms, crop=True)

        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)

    return out_path

def standardize_schema_points(gdf, id_field="id", type_field="type"):
    """
    Chuẩn hoá schema cho các lớp điểm (BTS, bệnh viện, trường học...).

    Notes:
    - If input geometries are not Points (e.g., Polygons/LineStrings), use centroid as representative point.
    - This avoids ValueError when accessing .x/.y.
    """
    gdf = gdf.reset_index(drop=True)

    # Ensure we have geometry column and non-empty
    if "geometry" not in gdf:
        raise ValueError("GeoDataFrame has no geometry column.")

    # Use centroid for non-point geometries to extract lon/lat reliably
    # centroid returns a Point for any geometry.
    # If geometry is already Point, centroid is that point itself.
    centroid_series = gdf.geometry.centroid

    gdf[id_field] = gdf.index.astype(int)
    # extract lon/lat from centroid (works for all geometry types)
    gdf["lon"] = centroid_series.x
    gdf["lat"] = centroid_series.y

    if type_field not in gdf:
        gdf[type_field] = "unknown"
    return gdf

# Cleaning tasks for each dataset
def clean_boundary():
    path = RAW_DIR / "hue_boundary.geojson"
    gdf = gpd.read_file(path)
    gdf = ensure_crs(gdf)
    gdf = fix_invalid_geometries(gdf)
    # force single-part polygons and valid
    gdf.to_file(CLEAN_DIR / "hue_boundary_clean.geojson", driver="GeoJSON")
    return gdf

def clean_roads(boundary):
    path = RAW_DIR / "roads_hue.geojson"
    gdf = gpd.read_file(path)
    gdf = ensure_crs(gdf)
    gdf = fix_invalid_geometries(gdf)

    # Intersect with boundary (safe if gdf empty)
    if not gdf.empty:
        try:
            gdf = gpd.overlay(gdf, boundary, how="intersection")
        except Exception:
            # fallback to spatial join clip
            gdf = gdf[gdf.geometry.intersects(boundary.unary_union)]

    if gdf.empty:
        # produce empty template with expected columns
        cols = ["edge_id", "length_m", "speed_kmh", "geometry"]
        empty = gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / "roads_hue_clean.geojson", driver="GeoJSON")
        return empty

    # compute length in meters using metric projection for accuracy
    gdf_metric = project_to_metric(gdf)
    gdf["length_m"] = gdf_metric.geometry.length  # meters

    gdf = gdf.reset_index(drop=True)
    gdf["edge_id"] = gdf.index
    if "speed_kmh" not in gdf:
        gdf["speed_kmh"] = 40

    # keep output in EPSG:4326 for GeoJSON
    gdf.to_file(CLEAN_DIR / "roads_hue_clean.geojson", driver="GeoJSON")
    return gdf

def clean_point_layer(filename, output_name, boundary):
    src = RAW_DIR / filename
    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf)
    gdf = fix_invalid_geometries(gdf)

    # intersect with boundary safely
    if gdf.empty:
        empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    try:
        clipped = gpd.overlay(gdf, boundary, how="intersection")
    except Exception:
        clipped = gdf[gdf.geometry.intersects(boundary.unary_union)]

    if clipped.empty:
        # export empty template with standardized schema
        empty = gpd.GeoDataFrame(columns=["id", "lon", "lat", "type", "geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    clipped = standardize_schema_points(clipped)
    clipped.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
    return clipped

def clean_raster(name, boundary):
    """Clip các raster DEM, slope, population."""
    src = RAW_DIR / name
    out = CLEAN_DIR / name.replace(".tif", "_clean.tif")
    clip_raster(src, boundary, out)
    return out

def assert_points_within_boundary(points, boundary, label):
    """Đảm bảo mọi điểm (BTS, facility) nằm trong biên giới."""
    if points.empty:
        return True
    mask = points.within(boundary.unary_union)
    if not mask.all():
        missing = (~mask).sum()
        raise ValueError(f"[ERROR] {missing} {label} points lie outside boundary.")
    return True

# Main Pipeline
def run_cleaning_pipeline():
    # 1) Boundary
    boundary = clean_boundary()

    # 2) Vector layers
    clean_roads(boundary)

    schools = clean_point_layer("schools.geojson", "schools_clean.geojson", boundary)
    hospitals = clean_point_layer("hospitals.geojson", "hospitals_clean.geojson", boundary)
    residential = clean_point_layer("residential.geojson", "residential_clean.geojson", boundary)
    command_centers = clean_point_layer("command_centers.geojson", "command_centers_clean.geojson", boundary)

    # 3) Raster data
    clean_raster("elev_hue.tif", boundary)
    clean_raster("slope_hue.tif", boundary)
    clean_raster("pop_hue.tif", boundary)

    print("DATA CLEANING COMPLETED")
    print("Cleaned data saved at:", CLEAN_DIR)

if __name__ == "__main__":
    run_cleaning_pipeline()
