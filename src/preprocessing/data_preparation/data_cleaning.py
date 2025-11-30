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

# Project-level data dirs (3 levels up from src/.../data_cleaning.py -> project root)
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
        try:
            if gdf.crs.to_epsg() != epsg:
                gdf = gdf.to_crs(epsg=epsg)
        except Exception:
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
            return repaired
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
    centroid_series = gdf.geometry.centroid

    gdf[id_field] = gdf.index.astype(int)
    # extract lon/lat from centroid (works for all geometry types)
    gdf["lon"] = centroid_series.x
    gdf["lat"] = centroid_series.y

    if type_field not in gdf:
        gdf[type_field] = "unknown"
    return gdf

def standardize_schema_polygons(gdf, id_field="id", type_field="type"):
    """
    Standardize polygon schema:
    - add id, type (if missing)
    - compute centroid lon/lat and area_m (using metric projection)
    """
    gdf = gdf.reset_index(drop=True)
    gdf[id_field] = gdf.index.astype(int)
    if type_field not in gdf:
        gdf[type_field] = "unknown"

    # compute centroid lon/lat in EPSG:4326 (safe because gdf is in 4326)
    centroids = gdf.geometry.centroid
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y

    # compute area in meters using metric projection
    try:
        g_metric = project_to_metric(gdf)
        gdf["area_m2"] = g_metric.geometry.area
    except Exception:
        gdf["area_m2"] = None

    return gdf

# Cleaning tasks for each dataset
def clean_boundary():
    path = RAW_DIR / "hue_boundary.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found at {path}")
    gdf = gpd.read_file(path)
    gdf = ensure_crs(gdf)
    gdf = fix_invalid_geometries(gdf)
    # force single-part polygons and valid
    gdf.to_file(CLEAN_DIR / "hue_boundary_clean.geojson", driver="GeoJSON")
    return gdf

def clean_roads(boundary):
    path = RAW_DIR / "roads_hue.geojson"
    if not path.exists():
        # create an empty template
        cols = ["edge_id", "length_m", "speed_kmh", "geometry"]
        empty = gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / "roads_hue_clean.geojson", driver="GeoJSON")
        return empty

    gdf = gpd.read_file(path)
    gdf = ensure_crs(gdf)
    gdf = fix_invalid_geometries(gdf)

    # Intersect with boundary (safe if gdf empty)
    if not gdf.empty:
        try:
            gdf = gpd.overlay(gdf, boundary, how="intersection")
        except Exception:
            # fallback to spatial filter
            gdf = gdf[gdf.geometry.intersects(boundary.unary_union)]

    if gdf.empty:
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
    if not src.exists():
        # export empty template
        empty = gpd.GeoDataFrame(columns=["id", "lon", "lat", "type", "geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf)
    gdf = fix_invalid_geometries(gdf)

    # intersect with boundary safely
    if gdf.empty:
        empty = gpd.GeoDataFrame(columns=["id", "lon", "lat", "type", "geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    try:
        clipped = gpd.overlay(gdf, boundary, how="intersection")
    except Exception:
        clipped = gdf[gdf.geometry.intersects(boundary.unary_union)]

    if clipped.empty:
        empty = gpd.GeoDataFrame(columns=["id", "lon", "lat", "type", "geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    clipped = standardize_schema_points(clipped)
    clipped.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
    return clipped

def clean_polygon_layer(filename, output_name, boundary, compute_area=True):
    """
    Clean polygon layers (e.g., industrial zones, water bodies).
    - ensures CRS, fixes geometries
    - intersects/crops to boundary
    - standardizes id/type, computes centroid lon/lat and area_m2
    """
    src = RAW_DIR / filename
    if not src.exists():
        empty = gpd.GeoDataFrame(columns=["id", "type", "lon", "lat", "area_m2", "geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    gdf = gpd.read_file(src)
    gdf = ensure_crs(gdf)
    gdf = fix_invalid_geometries(gdf)

    if gdf.empty:
        empty = gpd.GeoDataFrame(columns=["id", "type", "lon", "lat", "area_m2", "geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    try:
        clipped = gpd.overlay(gdf, boundary, how="intersection")
    except Exception:
        clipped = gdf[gdf.geometry.intersects(boundary.unary_union)]

    if clipped.empty:
        empty = gpd.GeoDataFrame(columns=["id", "type", "lon", "lat", "area_m2", "geometry"], geometry="geometry", crs=boundary.crs)
        empty.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
        return empty

    # standardize and compute area
    clipped = standardize_schema_polygons(clipped)
    clipped.to_file(CLEAN_DIR / output_name, driver="GeoJSON")
    return clipped

def clean_raster(name, boundary):
    """Clip các raster DEM, slope, population."""
    src = RAW_DIR / name
    if not src.exists():
        raise FileNotFoundError(f"Raster {name} not found at {src}")
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

    # existing point layers
    schools = clean_point_layer("schools.geojson", "schools_clean.geojson", boundary)
    hospitals = clean_point_layer("hospitals.geojson", "hospitals_clean.geojson", boundary)
    residential = clean_point_layer("residential.geojson", "residential_clean.geojson", boundary)
    command_centers = clean_point_layer("command_centers.geojson", "command_centers_clean.geojson", boundary)
    medical = clean_point_layer("medical_centers.geojson", "medical_centers_clean.geojson", boundary)
    industrial = clean_polygon_layer("industrial.geojson", "industrial_clean.geojson", boundary)
    water = clean_polygon_layer("water_hue.geojson", "water_hue_clean.geojson", boundary)

    # 3) Raster data
    # Only attempt to clean rasters if they exist
    for rname in ["elev_hue.tif", "slope_hue.tif", "pop_hue.tif"]:
        try:
            clean_raster(rname, boundary)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

    print("DATA CLEANING COMPLETED")
    print("Cleaned data saved at:", CLEAN_DIR)

if __name__ == "__main__":
    run_cleaning_pipeline()
