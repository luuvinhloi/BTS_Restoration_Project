# FILE: src/optimization/GA_PSO/utils.py
import math
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def read_params(params_path: Path):
    if not Path(params_path).exists():
        logging.warning(f"params file {params_path} not found, returning empty dict")
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_roads_graph(graphml_path: Path):
    try:
        G = nx.read_graphml(str(graphml_path))
        return G
    except Exception as e:
        logging.warning(f"Failed to load roads graph {graphml_path}: {e}")
        return None


def point_flood_depth(flood_tif_path: Path, lat: float, lon: float):
    """Return flood depth (meters) at given lat/lon. None if raster missing or outside."""
    try:
        with rasterio.open(str(flood_tif_path)) as src:
            # raster in EPSG:4326? assume lat/lon; else user must ensure same CRS.
            row, col = src.index(lon, lat)
            val = src.read(1)[row, col]
            if val is None:
                return None
            try:
                return float(val)
            except:
                return None
    except Exception:
        return None


def nearest_nonflood_site(J_df: pd.DataFrame, idx: int, flood_tif_path: Path, max_search_km=10.0):
    """
    Given a J_df row index that is flooded, search nearest J site (within max_search_km)
    with flood depth < 0.5m. Returns row (series) or None.
    """
    src_row = J_df.iloc[idx]
    src_lat = float(src_row["latitude"])
    src_lon = float(src_row["longitude"])
    # transform to numpy arrays and compute haversine distances
    lats = J_df["latitude"].astype(float).values
    lons = J_df["longitude"].astype(float).values
    # compute haversine (km)
    def hav_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1 = math.radians(lat1); phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2.0)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dists = np.array([hav_km(src_lat, src_lon, la, lo) for la, lo in zip(lats, lons)])
    order = np.argsort(dists)
    for o in order[1:]:  # skip self
        if dists[o] > max_search_km:
            break
        depth = point_flood_depth(flood_tif_path, float(lats[o]), float(lons[o]))
        if depth is None or depth < 0.5:
            return J_df.iloc[o]
    return None


def haversine_m(lat1, lon1, lat2, lon2):
    """Return haversine distance in meters"""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default
