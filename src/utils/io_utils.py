# Đọc/ghi file dữ liệu
"""
I/O utilities: read/write common geospatial and tabular data.
"""
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path

def read_csv(path):
    return pd.read_csv(path)

def read_geojson(path, crs=None):
    gdf = gpd.read_file(path)
    if crs:
        gdf = gdf.to_crs(crs)
    return gdf

def write_geojson(gdf, path):
    gdf.to_file(path, driver='GeoJSON')

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_yaml(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
