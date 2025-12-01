"""
bathtub.py
Functions to compute simple level-fill (bathtub) flood depth raster.
"""
from typing import Optional, Dict, Any
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio import Affine
import logging
from .utils import ensure_dir, write_raster

logger = logging.getLogger(__name__)


def resample_dem_if_needed(src_path: str, target_res: Optional[float]) -> Dict[str, Any]:
    """
    If target_res is None, return the original raster path and its profile.
    If target_res provided (meters), resample DEM to square pixels of that size; return profile and array in memory.
    """
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        if target_res is None:
            return {'array': src.read(1).astype('float32'), 'transform': src.transform, 'crs': src.crs, 'profile': profile}
        # compute scale factor
        old_res_x = src.transform.a
        old_res_y = -src.transform.e
        scale = old_res_x / target_res
        if scale == 1:
            return {'array': src.read(1).astype('float32'), 'transform': src.transform, 'crs': src.crs, 'profile': profile}

        new_width = int(src.width * scale)
        new_height = int(src.height * scale)
        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.bilinear
        ).astype('float32')

        # new transform
        new_transform = Affine(target_res, src.transform.b, src.transform.c,
                               src.transform.d, -target_res, src.transform.f)
        new_profile = profile.copy()
        new_profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'dtype': 'float32'
        })
        return {'array': data, 'transform': new_transform, 'crs': src.crs, 'profile': new_profile}


def compute_bathtub(dem_path: str, m: float, out_raster: str, dem_resample: Optional[float] = None) -> str:
    """
    Compute flood depth raster: depth = max(0, m - elev)
    Writes float32 tif to out_raster and returns out_raster path.
    """
    logger.info("Computing bathtub for m=%s using DEM=%s", m, dem_path)
    res = resample_dem_if_needed(dem_path, dem_resample)
    dem = res['array']
    profile = res['profile']
    transform = res['transform']
    crs = res['crs']

    # compute depth
    depth = (m - dem).astype('float32')
    depth[depth < 0] = 0.0

    # set nodata if DEM had nodata
    nodata = profile.get('nodata', None)
    if nodata is not None:
        mask = dem == nodata
        depth[mask] = profile.get('nodata', nodata)

    profile.update(dtype='float32', count=1, compress='lzw', nodata=profile.get('nodata', None))
    ensure_dir(out_raster.rsplit('/', 1)[0] if '/' in out_raster else '.')
    with rasterio.open(out_raster, 'w', **profile) as dst:
        dst.write(depth, 1)
    logger.info("Wrote flood depth raster to %s", out_raster)
    return out_raster
