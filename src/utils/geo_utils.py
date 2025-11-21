# Hàm tính khoảng cách, buffer, mask
"""
Geospatial helper functions.
"""
from shapely.geometry import Point
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio import Affine
import geopandas as gpd
import math

def lonlat_to_meters(lon, lat):
    # approximate conversion: using haversine where needed instead of projection
    return lon, lat

def haversine(lon1, lat1, lon2, lat2):
    # returns meters (great-circle)
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c

def raster_to_points(raster_path, value_threshold=0, max_points=None, weighted_sampling=False, seed=0):
    """
    Convert raster to list of points (centroids) where raster value > threshold.
    Returns list of dict: [{'x':lon, 'y':lat, 'val':value}, ...]
    """
    pts = []
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        mask = (arr > value_threshold) & (~np.isnan(arr))
        rows, cols = np.where(mask)
        values = arr[rows, cols].astype(float)
        total = len(rows)
        indices = list(range(total))
        import random
        random.seed(seed)
        if max_points is not None and total > max_points:
            if weighted_sampling:
                probs = values / values.sum()
                chosen = np.random.choice(indices, size=max_points, replace=False, p=probs)
            else:
                chosen = random.sample(indices, k=max_points)
        else:
            chosen = indices
        transform = src.transform
        for idx in chosen:
            r = rows[idx]; c = cols[idx]
            x, y = transform * (c + 0.5, r + 0.5)
            pts.append({'x': float(x), 'y': float(y), 'val': float(values[idx])})
    return pts

def sample_candidate_sites_from_population(pop_points, K):
    """
    Simple method: cluster/pop-weighted sampling to create candidate sites.
    We use KMeans to get cluster centers (requires scikit-learn).
    """
    from sklearn.cluster import KMeans
    coords = np.array([[p['x'], p['y']] for p in pop_points])
    k = min(K, len(coords))
    km = KMeans(n_clusters=k, random_state=0).fit(coords)
    centers = km.cluster_centers_
    cand = [{'x': float(cx), 'y': float(cy)} for cx, cy in centers]
    return cand

def point_in_polygon(point, polygon):
    return polygon.contains(Point(point[0], point[1]))

def compute_distance_matrix(points_I, points_J, metric='haversine'):
    import numpy as np
    N = len(points_I); M = len(points_J)
    mat = np.zeros((N, M), dtype=float)
    for i, p in enumerate(points_I):
        for j, q in enumerate(points_J):
            if metric == 'haversine':
                mat[i, j] = haversine(p['x'], p['y'], q['x'], q['y'])
            else:
                dx = p['x'] - q['x']; dy = p['y'] - q['y']
                mat[i, j] = math.hypot(dx, dy)
    return mat
