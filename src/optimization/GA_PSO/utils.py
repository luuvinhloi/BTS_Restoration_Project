# FILE: src/optimization/GA_PSO/utils.py
import math
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def read_params(params_path: Path):
    if not Path(params_path).exists():
        logging.warning(f"params file {params_path} not found, returning empty dict")
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
