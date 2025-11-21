# Xử lý dữ liệu đầu vào
"""
Data cleaning and normalization.
- Ensures CRS consistency
- Saves processed copies
"""
import geopandas as gpd
import rasterio
from pathlib import Path
from src.utils.io_utils import read_geojson, save_json
import os

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def reproject_vector_to_epsg(gdf, epsg=4326):
    return gdf.to_crs(epsg)

def clip_raster_by_boundary(raster_path, boundary_gdf, out_path):
    import rasterio
    from rasterio.mask import mask
    with rasterio.open(raster_path) as src:
        geoms = [boundary_gdf.unary_union]
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
    return out_path

def main():
    # Example usage: read boundary and clip rasters
    boundary = read_geojson(os.path.join(DATA_DIR, "raw", "hue_boundary.geojson"))
    # Reproject boundary if needed
    if boundary.crs is None:
        boundary = boundary.set_crs(epsg=4326)
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Clip population raster
    pop_in = DATA_DIR / "raw" / "pop_hue.tif"
    pop_out = processed_dir / "pop_hue_clipped.tif"
    clip_raster_by_boundary(str(pop_in), boundary, str(pop_out))
    # Similarly for elev/slope
    print("Clipped rasters to processed folder:", processed_dir)
