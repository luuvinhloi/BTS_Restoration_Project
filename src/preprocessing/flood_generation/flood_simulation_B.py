"""
flood_simulation_B.py
Hydrologic flood-fill model + percentile-based depth allocation.
Now includes CLEANED OUTPUTS:
- flood_depth_combined_B_clean.tif
- flood_area_combined_B_clean.geojson
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

FLOOD_LEVELS = [0.0, 0.2, 0.5, 1.0, 2.0]
FLOOD_PERCENT = [0.2, 0.4, 0.2, 0.1, 0.1]


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dem_clipped(dem_src, boundary_gdf):
    """Clip DEM by boundary polygon."""
    geom = [boundary_gdf.unary_union]
    arr, transform = mask(dem_src, geom, crop=True)
    return arr[0], transform


def rasterize_water(shape_hw, transform, water_gdf):
    """Rasterize water polygons -> DEM grid."""
    if water_gdf.empty:
        return np.zeros(shape_hw, dtype=np.uint8)

    shapes_list = [(geom, 1) for geom in water_gdf.geometry if geom is not None]

    return rfeatures.rasterize(
        shapes_list,
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )


def hydrologic_flood_fill(dem_array, water_mask):
    """
    BFS propagation: water spreads only to neighbors with elevation <= current elevation.
    """
    h, w = dem_array.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    # Seed queue
    for r, c in np.argwhere(water_mask == 1):
        visited[r, c] = 1
        q.append((r, c))

    # 4-way adjacency
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while q:
        r, c = q.popleft()
        base_h = dem_array[r, c]

        for dr, dc in dirs:
            rr, cc = r+dr, c+dc
            if 0 <= rr < h and 0 <= cc < w:
                if visited[rr, cc] == 0 and dem_array[rr, cc] <= base_h:
                    visited[rr, cc] = 1
                    q.append((rr, cc))

    return visited


def allocate_flood(dem_array, floodable_mask):
    """Assign flood depths via percentile rule inside floodable mask."""
    mask_bool = floodable_mask.astype(bool)
    elev = dem_array[mask_bool]

    if elev.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    idx_sorted = np.argsort(elev)
    n = len(idx_sorted)

    counts = [int(n * p) for p in FLOOD_PERCENT]
    counts[-1] += n - sum(counts)

    out = np.zeros_like(dem_array, dtype=np.float32)
    out_mask = out[mask_bool]

    start = 0
    for cnt, lvl in zip(counts, FLOOD_LEVELS):
        end = start + cnt
        sel = idx_sorted[start:end]
        out_mask[sel] = lvl
        start = end

    out[mask_bool] = out_mask
    return out.astype(np.float32)


def save_raster(path, array, ref, transform):
    """Write GeoTIFF safely."""
    profile = ref.meta.copy()
    profile.update({
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "transform": transform,
        "dtype": array.dtype.name,
        "compress": "lzw",
        "crs": ref.crs,
        "nodata": 0.0
    })

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def raster_to_polygon(raster_array, transform, crs):
    mask = raster_array > 0
    shapes_gen = shapes(raster_array, mask=mask, transform=transform)

    polys = []
    vals = []

    for geom, val in shapes_gen:
        if val > 0:
            polys.append(shape(geom))
            vals.append(float(val))

    if not polys:
        return gpd.GeoDataFrame(columns=["flood_level", "geometry"], crs=crs)

    return gpd.GeoDataFrame({"flood_level": vals, "geometry": polys}, crs=crs)


def optimize_polygon(gdf):
    if gdf.empty:
        return gdf

    dissolved = gdf.dissolve(by="flood_level")
    cleaned = []

    for lvl, row in dissolved.iterrows():
        geom = row.geometry
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = geom.simplify(1.5, preserve_topology=True)
        cleaned.append({"flood_level": float(lvl), "geometry": geom})

    return gpd.GeoDataFrame(cleaned, crs=gdf.crs)


# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean_outputs(boundary_gdf, transform, ref_ds):
    """
    Clean flood_depth_combined_B.* outputs:
    - Raster outside boundary = 0
    - Polygon clipped strictly to boundary
    """

    # --- Paths ---
    raster_path = OUTPUT_DIR / "flood_depth_combined_B.tif"
    raster_clean = OUTPUT_DIR / "flood_depth_combined_B_clean.tif"

    polygon_path = OUTPUT_DIR / "flood_area_combined_B.geojson"
    polygon_clean = OUTPUT_DIR / "flood_area_combined_B_clean.geojson"

    # ---------------------------
    # CLEAN RASTER
    # ---------------------------
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        meta = src.meta.copy()

    # Rasterize boundary mask
    boundary_union = boundary_gdf.unary_union
    boundary_mask = rfeatures.rasterize(
        [(boundary_union, 1)],
        out_shape=arr.shape,
        transform=transform,
        dtype=np.uint8,
        fill=0
    )

    arr_clean = np.where(boundary_mask == 1, arr, 0).astype(np.float32)

    with rasterio.open(raster_clean, "w", **meta) as dst:
        dst.write(arr_clean, 1)

    print("[CLEAN] Saved:", raster_clean)

    # ---------------------------
    # CLEAN POLYGON
    # ---------------------------
    if polygon_path.exists():
        poly = gpd.read_file(polygon_path)
        boundary = boundary_gdf

        try:
            poly_clean = gpd.overlay(poly, boundary, how="intersection")
        except Exception:
            poly_clean = poly[poly.geometry.intersects(boundary.unary_union)]

        poly_clean.to_file(polygon_clean, driver="GeoJSON")
        print("[CLEAN] Saved:", polygon_clean)

    else:
        print("[WARNING] polygon missing, skip clean.")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("\n=== Hydrologic Flood Simulation B START ===")

    water_gdf = gpd.read_file(WATER_PATH)
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)

    with rasterio.open(DEM_PATH) as dem_src:

        dem_array, dem_transform = load_dem_clipped(dem_src, boundary_gdf)
        dem_array = dem_array.astype(float)

        # 1) Rasterize water
        water_mask = rasterize_water(dem_array.shape, dem_transform, water_gdf)

        # 2) Hydrologic flood propagation
        floodable_mask = hydrologic_flood_fill(dem_array, water_mask)

        # 3) Allocate flood depths
        combined = allocate_flood(dem_array, floodable_mask)

        # 4) Outside floodable = 0
        combined = np.where(floodable_mask == 1, combined, 0).astype(np.float32)

        # 5) Save per-level + combined rasters
        for lvl in FLOOD_LEVELS:
            mask_lvl = (combined == lvl).astype(np.uint8)
            depth_lvl = np.where(mask_lvl == 1, lvl, 0).astype(np.float32)

            save_raster(OUTPUT_DIR / f"flood_mask_{lvl}m_B.tif", mask_lvl, dem_src, dem_transform)
            save_raster(OUTPUT_DIR / f"flood_depth_{lvl}m_B.tif", depth_lvl, dem_src, dem_transform)

        combined_path = OUTPUT_DIR / "flood_depth_combined_B.tif"
        save_raster(combined_path, combined, dem_src, dem_transform)

        # 6) Convert combined raster → polygons
        poly = raster_to_polygon(combined, dem_transform, dem_src.crs.to_string())
        poly = optimize_polygon(poly)
        poly.to_file(OUTPUT_DIR / "flood_area_combined_B.geojson", driver="GeoJSON")

        # 7) CLEAN OUTPUT FILES
        clean_outputs(boundary_gdf, dem_transform, dem_src)

    print("\n=== Hydrologic Flood Simulation B COMPLETE ===\n")


if __name__ == "__main__":
    main()
