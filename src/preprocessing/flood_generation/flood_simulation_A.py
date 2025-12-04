"""
Optimized Flood Simulation A (Buffer-based, Cleaned Version)
------------------------------------------------------------
Fixes:
1. Remove flood outside boundary.
2. Water bodies always assigned permanent high-depth (e.g., 2.0 m).
3. Water bodies excluded from percentile allocation.
4. Clean raster + polygon output added.
"""

from pathlib import Path
import os
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio import features as rfeatures
from shapely.geometry import shape
from shapely.ops import unary_union
from scipy import ndimage

# ===========================================================
# CONFIG
# ===========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "flood"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

DEM_PATH = CLEAN_DIR / "elev_hue_clean.tif"
WATER_PATH = CLEAN_DIR / "water_hue_clean.geojson"
BOUNDARY_PATH = CLEAN_DIR / "hue_boundary_clean.geojson"

FLOOD_LEVELS = [0.0, 0.2, 0.5, 1.0, 2.0]
FLOOD_PERCENT = [0.2, 0.4, 0.2, 0.1, 0.1]

BUFFER_KM = 10.0
ELEVATION_MAX_M = 200.0
WATER_FIXED_DEPTH = 2.0  # always flood rivers/lakes at 2m

SIMPLIFY_TOLERANCE = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================
# UTILITY FUNCTIONS
# ===========================================================

def deg_to_m(lat_deg):
    lat_rad = math.radians(lat_deg)
    m_per_lon = 111320 * math.cos(lat_rad)
    m_per_lat = 110540
    return m_per_lon, m_per_lat


def pixel_resolution_m(transform, lat_center):
    px_lon_deg = abs(transform.a)
    px_lat_deg = abs(transform.e)
    m_lon, m_lat = deg_to_m(lat_center)
    return px_lon_deg * m_lon, px_lat_deg * m_lat


def load_dem_clipped(dem_src, boundary_gdf):
    """DEM is clipped by boundary → ensures we NEVER flood outside province."""
    geoms = [boundary_gdf.unary_union]
    clipped, transform = mask(dem_src, geoms, crop=True)
    return clipped[0], transform


def rasterize_water(shape, transform, water_gdf):
    """Rasterize water polygons → ensures exact water mask."""
    if water_gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    shapes_list = [(geom, 1) for geom in water_gdf.geometry]
    return rfeatures.rasterize(
        shapes_list,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )


def compute_distance_to_water(water_mask, dem_transform, dem_bounds):
    h, w = water_mask.shape
    minx, miny, maxx, maxy = dem_bounds
    lat_center = (miny + maxy) / 2.0

    px_x_m, px_y_m = pixel_resolution_m(dem_transform, lat_center)
    px_m = (px_x_m + px_y_m) / 2.0

    inv = ~(water_mask.astype(bool))
    dist_pixels = ndimage.distance_transform_edt(inv)
    return dist_pixels * px_m


def build_floodable_area(dem_array, dem_transform, dem_bounds, water_gdf):
    water_mask = rasterize_water(dem_array.shape, dem_transform, water_gdf)
    if water_mask.sum() == 0:
        return np.zeros_like(dem_array), water_mask

    dist_m = compute_distance_to_water(water_mask, dem_transform, dem_bounds)
    floodable = (dist_m <= BUFFER_KM * 1000).astype(np.uint8)

    # elevation filter
    if ELEVATION_MAX_M is not None:
        floodable = np.where(dem_array <= ELEVATION_MAX_M, floodable, 0)

    return floodable, water_mask


def allocate_depth(dem_array, floodable_mask, water_mask):
    """Percentile allocation WITH water bodies treated separately."""
    floodable_bool = floodable_mask.astype(bool)

    # Remove rivers from allocation pool
    allocation_mask = floodable_bool & (~water_mask.astype(bool))

    elev_vals = dem_array[allocation_mask]
    if elev_vals.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    idx = np.argsort(elev_vals)
    n = len(idx)

    counts = [int(n * p) for p in FLOOD_PERCENT]
    counts[-1] += n - sum(counts)

    result = np.zeros_like(dem_array, dtype=np.float32)
    flat = result[allocation_mask]

    start = 0
    for cnt, lvl in zip(counts, FLOOD_LEVELS):
        end = start + cnt
        flat[idx[start:end]] = lvl
        start = end

    result[allocation_mask] = flat

    # Assign fixed depth to water bodies
    result = np.where(water_mask == 1, WATER_FIXED_DEPTH, result)

    # Outside floodable → 0
    result = np.where(floodable_bool, result, 0)

    return result.astype(np.float32)


def clip_raster_to_boundary(raster_arr, transform, boundary_gdf):
    """Force remove any flood outside boundary (safety)."""
    boundary_union = boundary_gdf.unary_union
    boundary_mask = rfeatures.rasterize(
        [(boundary_union, 1)],
        out_shape=raster_arr.shape,
        transform=transform,
        dtype=np.uint8,
        fill=0
    )
    return np.where(boundary_mask == 1, raster_arr, 0)


def save_raster(path, array, ref_ds, transform):
    profile = ref_ds.meta.copy()
    profile.update({
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype.name,
        "transform": transform,
        "compress": "lzw",
        "crs": ref_ds.crs,
        "nodata": 0.0
    })

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def raster_to_polygon(raster_array, transform, crs):
    mask_arr = raster_array > 0
    shapes_gen = shapes(raster_array, mask=mask_arr, transform=transform)

    polys, vals = [], []
    for geom, val in shapes_gen:
        if val > 0:
            polys.append(shape(geom))
            vals.append(float(val))

    return gpd.GeoDataFrame({"flood_level": vals, "geometry": polys}, crs=crs)


def clean_polygon(poly_gdf, boundary_gdf):
    try:
        clean = gpd.overlay(poly_gdf, boundary_gdf, how="intersection")
    except Exception:
        clean = poly_gdf[poly_gdf.geometry.intersects(boundary_gdf.unary_union)]
    return clean


# ===========================================================
# MAIN
# ===========================================================

def main():
    print("\n=== Optimized Flood Simulation A ===")

    water_gdf = gpd.read_file(WATER_PATH)
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)

    with rasterio.open(DEM_PATH) as dem_src:

        dem_arr, transform = load_dem_clipped(dem_src, boundary_gdf)
        dem_bounds = dem_src.bounds

        floodable_mask, water_mask = build_floodable_area(
            dem_arr, transform, dem_bounds, water_gdf
        )

        combined_depth = allocate_depth(dem_arr, floodable_mask, water_mask)

        # Ensure clean flood inside boundary only
        combined_depth_clean = clip_raster_to_boundary(combined_depth, transform, boundary_gdf)

        # Save combined raster
        save_raster(str(OUTPUT_DIR / "flood_depth_combined_clean.tif"),
                    combined_depth_clean, dem_src, transform)

        # Convert to polygons
        poly = raster_to_polygon(combined_depth_clean, transform, dem_src.crs.to_string())
        poly_clean = clean_polygon(poly, boundary_gdf)
        poly_clean = poly_clean.dissolve(by="flood_level")
        poly_clean.to_file(str(OUTPUT_DIR / "flood_area_combined_clean.geojson"), driver="GeoJSON")

        print("\nFlood Simulation A optimized & cleaned successfully.")
        print("Output saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
