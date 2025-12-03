"""
flood_simulation_A.py

Buffer-based flood simulation + percentage allocation.

Behavior:
- Rasterize water bodies to DEM grid.
- Compute distance (meters) from each pixel to nearest water.
- Floodable zone = distance <= BUFFER_M AND (optional) elevation <= ELEVATION_MAX_M.
- Within floodable zone, assign flood levels by percentiles (sorted by elevation low->high).
- Outside floodable zone -> 0.0m (no flood).
- Outputs:
    - flood_mask_{level}m.tif, flood_depth_{level}m.tif for each level
    - flood_depth_combined.tif
    - flood_area_combined.geojson
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

# -----------------------
# CONFIG (tweak here)
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "flood"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

DEM_PATH = CLEAN_DIR / "elev_hue_clean.tif"
WATER_PATH = CLEAN_DIR / "water_hue_clean.geojson"
BOUNDARY_PATH = CLEAN_DIR / "hue_boundary_clean.geojson"

# Flood levels and percentages (including 0.0 = no flood)
FLOOD_LEVELS = [0.0, 0.2, 0.5, 1.0, 2.0]
FLOOD_PERCENT = [0.2, 0.3, 0.3, 0.1, 0.1]   # 20% no-flood, 30%, 30%, 10%, 10%

# Buffer limit (km) from water bodies to allow flooding. Set to 5 or 10 as you like.
BUFFER_KM = 10.0   # default 10 km; change to 5.0 for 5 km

# Optional: maximum elevation (meters) allowed to flood. Set None to disable.
ELEVATION_MAX_M = 200.0  # set e.g. 200m to exclude mountains; or None to disable

# Simplify tolerance for polygons (map units)
SIMPLIFY_TOLERANCE = 1.5

# -----------------------
# Utilities
# -----------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dem_clipped(dem_src, boundary_gdf):
    """Clip DEM to boundary and return (array2d, transform)."""
    if boundary_gdf is None or boundary_gdf.empty:
        arr = dem_src.read(1)
        return arr, dem_src.transform
    boundary_geom = [boundary_gdf.unary_union]
    dem_clip, transform = mask(dem_src, boundary_geom, crop=True)
    return dem_clip[0], transform


def rasterize_water_to_dem(dem_shape, dem_transform, water_gdf):
    """Rasterize water polygons onto DEM grid; return uint8 mask (1=water)."""
    if water_gdf is None or water_gdf.empty:
        return np.zeros(dem_shape, dtype=np.uint8)
    shapes_list = [(geom, 1) for geom in water_gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes_list:
        return np.zeros(dem_shape, dtype=np.uint8)
    water_mask = rfeatures.rasterize(
        shapes_list,
        out_shape=dem_shape,
        transform=dem_transform,
        fill=0,
        dtype=np.uint8
    )
    return water_mask


def pixel_size_meters_from_transform(transform, crs, dem_bounds=None):
    """
    Estimate pixel size in meters using transform and CRS.
    If CRS is geographic (degrees), approximate conversion using center latitude.
    Returns pixel_size_x_m, pixel_size_y_m
    """
    px_w = abs(transform.a)
    px_h = abs(transform.e)
    try:
        is_geographic = crs.is_geographic
    except Exception:
        # fallback: check if EPSG:4326 in string
        is_geographic = False if crs is None else ('4326' in str(crs))

    if not is_geographic:
        return px_w, px_h

    # if geographic degrees, convert degrees -> meters using central latitude
    if dem_bounds is not None:
        minx, miny, maxx, maxy = dem_bounds
        center_lat = (miny + maxy) / 2.0
    else:
        # default mid-lat Vietnam ~16 deg
        center_lat = 16.0

    # meters per degree latitude ≈ 111132.954 - 559.822 * cos(2φ) + 1.175 * cos(4φ)
    lat_rad = math.radians(center_lat)
    meters_per_degree = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    # meters per degree longitude ≈ (π/180) * R * cos(lat)
    meters_per_deg_lon = 111320.0 * math.cos(lat_rad)

    px_w_m = px_w * meters_per_deg_lon
    px_h_m = px_h * meters_per_degree
    return px_w_m, px_h_m


def compute_distance_to_water_meters(water_mask, dem_transform, dem_crs, dem_bounds=None):
    """
    Compute Euclidean distance in meters from each pixel to nearest water pixel.
    Uses scipy.ndimage.distance_transform_edt on inverse mask.
    """
    # water_mask: 1 where water, 0 elsewhere
    if water_mask.dtype != np.bool_:
        w = water_mask.astype(bool)
    else:
        w = water_mask

    # invert: True where not water
    inv = (~w)
    # distance in pixels to nearest water pixel
    dist_pixels = ndimage.distance_transform_edt(inv)
    # pixel size in meters
    px_w_m, px_h_m = pixel_size_meters_from_transform(dem_transform, dem_crs, dem_bounds)
    # approximate euclidean distance in meters (use average pixel size)
    px_mean = (abs(px_w_m) + abs(px_h_m)) / 2.0
    dist_m = dist_pixels * px_mean
    return dist_m


def build_floodable_mask_by_buffer(dem_array, dem_transform, dem_crs, water_gdf, buffer_km, elevation_max_m=None, dem_bounds=None):
    """
    Build floodable mask:
      - rasterize water
      - compute distance_m
      - floodable = (distance_m <= buffer_km*1000)
      - optionally AND (dem_array <= elevation_max_m)
    """
    water_mask = rasterize_water_to_dem(dem_array.shape, dem_transform, water_gdf)
    if water_mask.sum() == 0:
        # no water found -> no floodable unless we choose to allow all; we choose none
        print("Warning: no water polygons found; floodable mask will be empty.")
        floodable = np.zeros_like(dem_array, dtype=np.uint8)
        return floodable, water_mask

    dist_m = compute_distance_to_water_meters(water_mask, dem_transform, dem_crs, dem_bounds=dem_bounds)
    buffer_m = float(buffer_km) * 1000.0
    floodable = (dist_m <= buffer_m).astype(np.uint8)

    if elevation_max_m is not None:
        floodable = np.where((floodable == 1) & (dem_array <= float(elevation_max_m)), 1, 0).astype(np.uint8)

    return floodable, water_mask


def allocate_within_floodable(dem_array, floodable_mask, percentages, levels):
    """
    Allocate flood levels only within floodable_mask.
    Pixels outside floodable remain 0.0 (no flood).
    percentages and levels must have same length and include 0.0 level if desired.
    """
    # ensure arrays
    mask_bool = floodable_mask.astype(bool)
    valid_vals = dem_array[mask_bool]

    # if no floodable pixels, return zeros
    if valid_vals.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    # sort indices of valid pixels by elevation (low -> high)
    order = np.argsort(valid_vals)
    n = order.size

    # compute counts per bucket
    counts = [int(n * p) for p in percentages]
    remainder = n - sum(counts)
    if remainder > 0:
        counts[-1] += remainder

    combined = np.zeros_like(dem_array, dtype=np.float32)

    start = 0
    # We'll assign levels in order corresponding to ascending elevation groups
    for cnt, lvl in zip(counts, levels):
        if cnt <= 0:
            start += cnt
            continue
        end = start + cnt
        sel = order[start:end]  # indices into valid_vals
        # create a temp array for valid positions
        temp = combined[mask_bool].copy()
        temp[sel] = float(lvl)
        combined[mask_bool] = temp
        start = end

    return combined


def _compatible_nodata(np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        return 0.0
    if np.issubdtype(np_dtype, np.uint8):
        return None
    return None


def save_raster(path, array, ref_ds, transform):
    """Save raster with profile adapted to array dtype and transform/CRS from ref_ds."""
    profile = ref_ds.meta.copy()
    profile.update({
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype.name,
        "transform": transform,
        "crs": ref_ds.crs,
        "compress": "lzw"
    })
    if "nodata" in profile:
        profile.pop("nodata", None)
    nodata = _compatible_nodata(array.dtype)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def raster_to_polygon(raster_array, transform, crs):
    """Convert raster >0 to polygons with flood_level attribute."""
    mask_arr = raster_array > 0
    shapes_gen = shapes(raster_array, mask=mask_arr, transform=transform)
    polys = []
    levels = []
    for geom, val in shapes_gen:
        if val <= 0:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        polys.append(poly)
        levels.append(float(val))
    if not polys:
        return gpd.GeoDataFrame(columns=["flood_level", "geometry"], geometry="geometry", crs=crs)
    gdf = gpd.GeoDataFrame({"flood_level": levels, "geometry": polys}, crs=crs)
    return gdf


def optimize_polygon(gdf, tolerance=SIMPLIFY_TOLERANCE):
    """Dissolve by level, fix geometries, simplify."""
    if gdf.empty:
        return gdf
    dissolved = gdf.dissolve(by="flood_level").reset_index()
    cleaned = []
    for _, row in dissolved.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            try:
                geom = geom.buffer(0)
            except Exception:
                pass
        try:
            geom = geom.simplify(tolerance, preserve_topology=True)
        except Exception:
            pass
        cleaned.append({"flood_level": float(row.flood_level), "geometry": geom})
    if not cleaned:
        return gpd.GeoDataFrame(columns=["flood_level", "geometry"], geometry="geometry")
    return gpd.GeoDataFrame(cleaned, geometry="geometry", crs=gdf.crs)


# -----------------------
# Main
# -----------------------

def main():
    ensure_dir(OUTPUT_DIR)

    print("Flood simulation (buffer-based) starting...")
    # read inputs
    water_gdf = gpd.read_file(WATER_PATH) if WATER_PATH.exists() else gpd.GeoDataFrame()
    boundary_gdf = gpd.read_file(BOUNDARY_PATH) if BOUNDARY_PATH.exists() else gpd.GeoDataFrame()

    if not DEM_PATH.exists():
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")

    with rasterio.open(DEM_PATH) as dem_src:
        # clip DEM
        dem_arr, dem_transform = load_dem_clipped(dem_src, boundary_gdf)
        dem_arr = dem_arr.astype(float)

        # rasterize water onto DEM grid
        water_mask = rasterize_water_to_dem(dem_arr.shape, dem_transform, water_gdf)

        # compute floodable mask using buffer_km and optional elevation cap
        dem_bounds = dem_src.bounds
        floodable_mask, _ = None, None
        floodable_mask, _water = None, None  # placeholders

        floodable_mask, water_mask = build_floodable_mask_by_buffer(
            dem_arr, dem_transform, dem_src.crs, water_gdf, BUFFER_KM, elevation_max_m=ELEVATION_MAX_M, dem_bounds=(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
        )

        # allocate flood levels WITHIN floodable mask
        combined_depth = allocate_within_floodable(dem_arr, floodable_mask, FLOOD_PERCENT, FLOOD_LEVELS)

        # ensure pixels outside floodable are 0.0
        combined_depth = np.where(floodable_mask == 1, combined_depth, 0.0).astype(np.float32)

        # Save per-level masks/depths and combined
        for level in FLOOD_LEVELS:
            lvl = float(level)
            mask_lvl = (combined_depth == lvl).astype(np.uint8)
            depth_lvl = np.where(mask_lvl == 1, lvl, 0.0).astype(np.float32)

            mask_path = OUTPUT_DIR / f"flood_mask_{str(lvl).replace('.', '_')}m.tif"
            depth_path = OUTPUT_DIR / f"flood_depth_{str(lvl).replace('.', '_')}m.tif"

            save_raster(str(mask_path), mask_lvl, dem_src, dem_transform)
            save_raster(str(depth_path), depth_lvl, dem_src, dem_transform)

        # combined raster
        combined_path = OUTPUT_DIR / "flood_depth_combined.tif"
        save_raster(str(combined_path), combined_depth.astype(np.float32), dem_src, dem_transform)

        # convert to polygons (exclude 0.0)
        crs = dem_src.crs.to_string() if dem_src.crs is not None else None
        poly_gdf = raster_to_polygon(combined_depth, dem_transform, crs)
        poly_gdf = optimize_polygon(poly_gdf)
        out_geojson = OUTPUT_DIR / "flood_area_combined.geojson"
        if crs is not None and not poly_gdf.empty:
            poly_gdf.set_crs(crs, inplace=True)
        poly_gdf.to_file(str(out_geojson), driver="GeoJSON")

        # summary stats
        vals, counts = np.unique(combined_depth, return_counts=True)
        total = counts.sum()
        print("\nFlood allocation summary (value : percentage_of_floodable_area):")
        for v, c in zip(vals, counts):
            print(f"  {v} m : {c/total*100:.2f}% (count {c})")

        print("\nOutputs written to:", OUTPUT_DIR)
        print("Buffer-based flood simulation completed.")

if __name__ == "__main__":
    main()
