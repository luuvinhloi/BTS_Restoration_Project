# utils.py
import os, json
import numpy as np
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio import Affine

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def write_raster(path, arr, profile):
    ensure_dir(os.path.dirname(path))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)

def save_geojson(gdf, path):
    ensure_dir(os.path.dirname(path))
    gdf.to_file(path, driver="GeoJSON")

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
