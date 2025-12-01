# river_flood_model.py
"""
Stage-based river flooding:
 - Given flood stage m (meters above reference), compute water surface elevation
 - Flood depth = max(0, stage - DEM_filled)
"""

import numpy as np
import logging
from .utils import write_raster, ensure_dir

logger = logging.getLogger(__name__)


def run_river_flood(dem_filled: np.ndarray, stage_m: float, out_path: str, profile):
    """
    Simple river flood model:
     flood_depth = max(0, stage_m - dem_filled)
    """
    logger.info(f"[RIVER] Running river flood with stage = {stage_m} m")

    depth = (stage_m - dem_filled).astype("float32")
    depth[depth < 0] = 0.0

    profile.update(dtype="float32", count=1)
    write_raster(out_path, depth, profile)

    return depth
