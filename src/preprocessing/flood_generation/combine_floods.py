# combine_floods.py
"""
Combine river flood + pluvial flood into final flood depth.
"""

import numpy as np
from .utils import write_raster
import logging

logger = logging.getLogger(__name__)


def combine_river_and_pluvial(river: np.ndarray, pluvial: np.ndarray,
                              out_path: str, profile):
    """
    Final depth = river_depth + pluvial_depth
    """
    combined = river + pluvial
    combined = combined.astype("float32")

    profile.update(dtype="float32", count=1)
    write_raster(out_path, combined, profile)

    logger.info("[COMBINE] Combined flood depth written.")

    return combined
