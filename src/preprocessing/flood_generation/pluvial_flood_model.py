# pluvial_flood_model.py
"""
Pluvial (rainfall) flooding:
 - Identify basins using DEM + accum
 - Fill with rainfall depth
 - Spread water based on flow direction
"""

import numpy as np
import logging
from .utils import write_raster

logger = logging.getLogger(__name__)


def run_pluvial_flood(dem_filled: np.ndarray, accumulation: np.ndarray,
                      rainfall_m: float, out_path: str, profile):
    """
    Pluvial flood approximation:
      depth_pluvial = rainfall_m * (accumulation_normalized)
    """
    logger.info(f"[PLUVIAL] Running rainfall flood, rainfall = {rainfall_m} m")

    acc_norm = accumulation / accumulation.max()
    depth = acc_norm * rainfall_m
    depth = depth.astype("float32")

    profile.update(dtype="float32", count=1)
    write_raster(out_path, depth, profile)

    return depth
