"""
flow_accum.py
Optional flow-accumulation based ponding adjustment.
This module tries to use pysheds first; if not available, it will fallback to a simple no-op.
"""
import logging
import numpy as np
import rasterio
from .utils import ensure_dir
from typing import Optional

logger = logging.getLogger(__name__)


def adjust_with_flow_accum(dem_path: str, depth_raster_in: str, depth_raster_out: str,
                           accumulation_threshold: int = 500, method: str = 'pysheds') -> str:
    """
    If pysheds is available, compute flow accumulation and boost depths in cells with high accumulation.
    Otherwise just copy input to output (no-op) and warn.

    The adjustment is heuristic: additional_depth = k * log(1 + accumulation)
    k tuned relative to dem resolution and accumulation_threshold.
    """
    try:
        if method == 'pysheds':
            from pysheds.grid import Grid
        else:
            logger.warning("Unknown flow accumulation method '%s'. Attempting pysheds.", method)
            from pysheds.grid import Grid
    except Exception:
        logger.warning("pysheds not available: skipping flow-accumulation adjustment.")
        # fallback: copy input raster to output
        import shutil
        shutil.copy(depth_raster_in, depth_raster_out)
        return depth_raster_out

    logger.info("Running flow-accumulation adjustment (pysheds)...")
    grid = Grid.from_raster(dem_path, data_name='dem')
    # fill depressions
    grid.fill_depressions('dem', out_name='dem_filled')
    grid.compute_flowdirs(data='dem_filled', out_name='fdir')
    grid.accumulation(data='fdir', out_name='acc')

    # read depth raster
    with rasterio.open(depth_raster_in) as src:
        depth = src.read(1).astype('float32')
        profile = src.profile

    acc = grid.acc
    # make sure shapes match; if not, we will reproject/resize not handled here — assume same
    if acc.shape != depth.shape:
        logger.warning("accumulation shape != depth shape; skipping adjustment")
        with rasterio.open(depth_raster_out, 'w', **profile) as dst:
            dst.write(depth, 1)
        return depth_raster_out

    # heuristic adjust
    k = 0.1  # meters scaling factor (tune)
    extra = k * np.log1p(acc)
    extra[acc < accumulation_threshold] = 0.0
    depth_adj = depth + extra.astype('float32')

    # keep nodata consistent
    nodata = profile.get('nodata', None)
    if nodata is not None:
        depth_adj[depth == nodata] = nodata

    ensure_dir(depth_raster_out.rsplit('/', 1)[0])
    profile.update(dtype='float32', count=1, compress='lzw')
    with rasterio.open(depth_raster_out, 'w', **profile) as dst:
        dst.write(depth_adj, 1)

    logger.info("Wrote adjusted depth raster to %s", depth_raster_out)
    return depth_raster_out
