# dem_processing.py
"""
DEM Processing for realistic flood modeling (A+B):
 - DEM offset correction
 - Depression filling (Wang-Liu)
 - Flow direction (D8)
 - Flow accumulation
"""
from pathlib import Path

import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio import Affine
from .utils import ensure_dir, write_raster
import logging
import os

logger = logging.getLogger(__name__)


def apply_dem_offset(dem_arr: np.ndarray, offset: float, nodata=None):
    """Subtract a uniform DEM bias (Huế DEM typically +1.0m higher than reality)."""
    corrected = dem_arr.astype("float32") - offset
    if nodata is not None:
        corrected[dem_arr == nodata] = nodata
    corrected[corrected < -5] = -5  # safety clamp
    return corrected


def fill_depressions_simple(dem: np.ndarray):
    """
    Very fast depression-filling algorithm (Wang & Liu approximation).
    Ensures flow connectivity for river & pluvial flooding.
    """
    filled = dem.copy()
    for _ in range(3):  # iterative fill
        nbr_min = np.minimum.reduce([
            np.roll(filled, 1, axis=0),
            np.roll(filled, -1, axis=0),
            np.roll(filled, 1, axis=1),
            np.roll(filled, -1, axis=1)
        ])
        filled = np.maximum(filled, nbr_min)
    return filled


def compute_flow_direction(dem: np.ndarray):
    """
    Compute simple D8 flow direction matrix.
    Returns flowdir (0-7 values).
    """
    h, w = dem.shape
    flowdir = np.zeros_like(dem, dtype=np.uint8)

    # kernel offsets: E, SE, S, SW, W, NW, N, NE
    dirs = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    for y in range(1, h-1):
        for x in range(1, w-1):
            cell = dem[y,x]
            neigh = [dem[y+dy, x+dx] for dy, dx in dirs]
            min_i = np.argmin(neigh)
            if neigh[min_i] < cell:
                flowdir[y,x] = min_i
            else:
                flowdir[y,x] = 255  # pit
    return flowdir


def compute_flow_accum(flowdir: np.ndarray):
    """
    Compute flow accumulation (very simplified).
    """
    h, w = flowdir.shape
    acc = np.ones_like(flowdir, dtype=np.float32)

    # 8-direction movement
    dirs = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    for _ in range(5):  # propagate 5 iterations
        for y in range(1, h-1):
            for x in range(1, w-1):
                d = flowdir[y,x]
                if d == 255:
                    continue
                dy, dx = dirs[d]
                acc[y+dy, x+dx] += acc[y,x]

    return acc


def process_dem_pipeline(dem_path: str, out_path: str, offset: float = 1.0):
    logger.info(f"[DEM] Loading: {dem_path}")

    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        dem = src.read(1).astype("float32")
        nodata = src.nodata

    logger.info(f"[DEM] Applying offset correction = {offset} m")
    dem_corr = apply_dem_offset(dem, offset, nodata)

    logger.info("[DEM] Filling depressions...")
    dem_filled = fill_depressions_simple(dem_corr)

    logger.info("[DEM] Computing flow direction...")
    flowdir = compute_flow_direction(dem_filled)

    logger.info("[DEM] Computing flow accumulation...")
    accum = compute_flow_accum(flowdir)

    # --- FIX: ensure parent directory exists ---
    out_dir = Path(out_path).parent
    ensure_dir(out_dir)

    # --- FIX: sanitize profile for writing ---
    profile.update({
        "dtype": "float32",
        "count": 1,
        "compress": "lzw"
    })
    profile.pop("tiled", None)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    # --- Write DEM ---
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem_filled, 1)

    return dem_filled, flowdir, accum, profile

