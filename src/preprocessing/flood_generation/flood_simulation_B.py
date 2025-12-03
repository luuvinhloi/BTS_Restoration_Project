"""
flood_simulation.py
Hydrologic flood-fill model + percentile-based flood depth allocation.

Steps:
1. Load DEM + water polygons + boundary.
2. Rasterize water polygons.
3. Perform hydrologic flood fill:
      - Water spreads only to lower or equal elevation cells (hydrologic connectivity).
4. The resulting floodable zone is REALISTIC (only valleys and plains).
5. Inside floodable zone, assign flood levels using percentage rule:
      - 20% no flood
      - 30% 0.2m
      - 30% 0.5m
      - 10% 1.0m
      - 10% 2.0m
6. Outside floodable → depth = 0.0m
7. Export:
      - flood_mask_*.tif
      - flood_depth_*.tif
      - flood_depth_combined.tif
      - flood_area_combined.geojson
"""

from pathlib import Path
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio import features as rfeatures
from shapely.geometry import shape
from shapely.ops import unary_union
from collections import deque

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "flood"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

DEM_PATH = CLEAN_DIR / "elev_hue_clean.tif"
WATER_PATH = CLEAN_DIR / "water_hue_clean.geojson"
BOUNDARY_PATH = CLEAN_DIR / "hue_boundary_clean.geojson"

# Flood levels + percentages
FLOOD_LEVELS = [0.0, 0.2, 0.5, 1.0, 2.0]
FLOOD_PERCENT = [0.2, 0.3, 0.3, 0.1, 0.1]


# ============================================================
# UTILITIES
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dem_clipped(dem_src, boundary_gdf):
    """Clip DEM by boundary polygon."""
    boundary_geom = [boundary_gdf.unary_union]
    dem_clip, transform = mask(dem_src, boundary_geom, crop=True)
    return dem_clip[0], transform


def rasterize_water(dem_shape, dem_transform, water_gdf):
    """Rasterize water polygons into DEM grid."""
    if water_gdf.empty:
        return np.zeros(dem_shape, dtype=np.uint8)

    shapes_list = [(geom, 1) for geom in water_gdf.geometry if geom is not None]
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


def hydrologic_flood_fill(dem_array, water_mask):
    """
    Hydrologic flood fill:
    Water starts from initial water_mask (1) and spreads ONLY to neighbors
    that have elevation <= current cell elevation.

    Returns: floodable_mask (1 = can be flooded)
    """

    h, w = dem_array.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    # seed queue with all water pixels
    water_pixels = np.argwhere(water_mask == 1)
    for (r, c) in water_pixels:
        visited[r, c] = 1
        q.append((r, c))

    # BFS hydrologic expansion
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while q:
        r, c = q.popleft()
        base_h = dem_array[r, c]

        for dr, dc in dirs:
            rr, cc = r+dr, c+dc
            if 0 <= rr < h and 0 <= cc < w:
                if visited[rr, cc] == 0:
                    # hydrologic rule: water can move *downhill or flat*
                    if dem_array[rr, cc] <= base_h:
                        visited[rr, cc] = 1
                        q.append((rr, cc))

    return visited  # 1=floodable, 0=not floodable


def allocate_flood_by_percentile(dem_array, floodable_mask, percentages, levels):
    """
    Allocate flood depths only within floodable pixels.
    Pixels sorted by elevation (low->high).
    """

    mask_bool = floodable_mask.astype(bool)
    valid_elev = dem_array[mask_bool]

    if valid_elev.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    order = np.argsort(valid_elev)
    n = order.size

    counts = [int(n * p) for p in percentages]
    remainder = n - sum(counts)
    if remainder > 0:
        counts[-1] += remainder

    combined = np.zeros_like(dem_array, dtype=np.float32)

    start = 0
    for count, lvl in zip(counts, levels):
        end = start + count
        sel = order[start:end]
        arr_part = combined[mask_bool].copy()
        arr_part[sel] = float(lvl)
        combined[mask_bool] = arr_part
        start = end

    return combined


def _compatible_nodata(np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        return 0.0
    return None


def save_raster(path, array, ref_ds, transform):
    """Write GeoTIFF safely."""
    profile = ref_ds.meta.copy()
    profile.update({
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "transform": transform,
        "crs": ref_ds.crs,
        "dtype": array.dtype.name,
        "compress": "lzw"
    })
    profile.pop("nodata", None)

    nodata = _compatible_nodata(array.dtype)
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array, 1)


def raster_to_polygon(raster_array, transform, crs):
    """Convert raster>0 to polygons."""
    mask = raster_array > 0
    shapes_gen = shapes(raster_array, mask=mask, transform=transform)
    polys, levels = [], []

    for geom, val in shapes_gen:
        if val <= 0:
            continue
        polys.append(shape(geom))
        levels.append(float(val))

    if not polys:
        return gpd.GeoDataFrame(columns=["flood_level", "geometry"], geometry="geometry", crs=crs)

    return gpd.GeoDataFrame({"flood_level": levels, "geometry": polys}, crs=crs)


def optimize_polygon(gdf):
    """Simplify + dissolve."""
    if gdf.empty:
        return gdf

    dissolved = gdf.dissolve(by="flood_level").reset_index()
    cleaned = []

    for _, row in dissolved.iterrows():
        geom = row.geometry
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = geom.simplify(1.5, preserve_topology=True)
        cleaned.append({"flood_level": row.flood_level, "geometry": geom})

    return gpd.GeoDataFrame(cleaned, geometry="geometry", crs=gdf.crs)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("Hydrologic flood-fill simulation started...")

    water_gdf = gpd.read_file(WATER_PATH)
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)

    with rasterio.open(DEM_PATH) as dem_src:
        dem_array, dem_transform = load_dem_clipped(dem_src, boundary_gdf)
        dem_array = dem_array.astype(float)

        # STEP 1 — Rasterize water bodies
        water_mask = rasterize_water(dem_array.shape, dem_transform, water_gdf)

        # STEP 2 — Hydrologic flood fill
        floodable_mask = hydrologic_flood_fill(dem_array, water_mask)

        # STEP 3 — Allocate flood levels only inside floodable
        combined_depth = allocate_flood_by_percentile(
            dem_array,
            floodable_mask,
            FLOOD_PERCENT,
            FLOOD_LEVELS
        )

        # Outside floodable = 0.0
        combined_depth = np.where(floodable_mask == 1, combined_depth, 0.0).astype(np.float32)

        # STEP 4 — Save per-level rasters
        for lvl in FLOOD_LEVELS:
            mask_lvl = (combined_depth == lvl).astype(np.uint8)
            depth_lvl = np.where(mask_lvl == 1, lvl, 0.0).astype(np.float32)

            save_raster(str(OUTPUT_DIR / f"flood_mask_{lvl}m_B.tif"), mask_lvl, dem_src, dem_transform)
            save_raster(str(OUTPUT_DIR / f"flood_depth_{lvl}m_B.tif"), depth_lvl, dem_src, dem_transform)

        # STEP 5 — Save combined
        save_raster(str(OUTPUT_DIR / "flood_depth_combined_B.tif"), combined_depth, dem_src, dem_transform)

        # STEP 6 — Convert to polygons
        crs = dem_src.crs.to_string()
        poly_gdf = raster_to_polygon(combined_depth, dem_transform, crs)
        poly_gdf = optimize_polygon(poly_gdf)
        poly_gdf.to_file(str(OUTPUT_DIR / "flood_area_combined_B.geojson"), driver="GeoJSON")

        print("\nHydrologic flood-fill simulation completed.")
        print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
