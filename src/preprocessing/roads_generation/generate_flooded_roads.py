#!/usr/bin/env python3
"""
generate_flooded_roads_with_graph.py   (FIXED VERSION)

Fixes applied:
 - Road length must be computed in METRIC CRS (EPSG:3857), not degrees
 - Roads kept in EPSG:4326 for geometry, but copied to metric CRS for length calc
 - Segment filtering now works correctly
"""

import sys
import math
from pathlib import Path
import warnings

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape, LineString, Point
from shapely.ops import unary_union
import networkx as nx

# ------------------------
# CONFIG
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

ROADS_FILE = CLEAN_DIR / "roads_hue_clean.geojson"
FLOOD_RASTER = DATA_DIR / "flood" / "flood_depth_combined_B_clean.tif"
BOUNDARY_FILE = CLEAN_DIR / "hue_boundary_clean.geojson"

OUT_GEOJSON = DATA_DIR / "road" / "roads_flooded.geojson"
OUT_GRAPHML = DATA_DIR / "road" / "roads_flooded.graphml"

DEPTH_THRESHOLD = 1.0      # >1.0 m => blocked (impassable)
MIN_SEG_LEN_M = 0.5        # minimum segment length to keep (meters)
METRIC_CRS = "EPSG:3857"   # used ONLY for length calculation

# ------------------------
# Utility functions
# ------------------------
def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

def reproject_to_raster(gdf: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        target_crs = src.crs
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf

def polygonize_threshold(raster_path: Path, threshold: float):
    """
    Polygonize raster pixels where value > threshold.
    Returns unary_union geometry (MultiPolygon) of flooded areas.
    """
    with rasterio.open(raster_path) as src:
        band1 = src.read(1, masked=True)
        transform = src.transform

        # valid pixels mask
        valid_mask = ~band1.mask if np.ma.is_masked(band1) else np.ones(band1.shape, bool)
        data = band1.data if np.ma.is_masked(band1) else band1

        # condition mask
        cond_mask = (data > threshold) & valid_mask
        if not cond_mask.any():
            return None, src.crs

        shapes_gen = features.shapes(cond_mask.astype("uint8"), transform=transform)
        polygons = [shape(geom) for geom, val in shapes_gen if val == 1]

        if not polygons:
            return None, src.crs

        return unary_union(polygons), src.crs

def sample_stats_along_line(line: LineString, raster_src, px_size: float):
    """
    Sample flood raster along line.
    """
    length = line.length
    if length == 0:
        return None, None

    n_samples = max(2, int(math.ceil(length / px_size)))
    pts = [line.interpolate(i / (n_samples - 1), normalized=True) for i in range(n_samples)]
    coords = [(p.x, p.y) for p in pts]

    values = []
    for val in raster_src.sample(coords):
        v = val[0]
        if v is None:
            continue
        try:
            v = float(v)
        except:
            continue
        if raster_src.nodata is not None and v == raster_src.nodata:
            continue
        if np.isnan(v):
            continue
        values.append(v)

    if not values:
        return None, None

    return float(np.mean(values)), float(np.max(values))

def lines_from_geometry_collection(geom):
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return [g for g in geom.geoms if g.geom_type == "LineString"]
    if geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if g.geom_type == "LineString"]
    return []

def round_coord(pt, digits=6):
    return (round(pt[0], digits), round(pt[1], digits))

# ------------------------
# Main
# ------------------------
def main():
    ensure_exists(ROADS_FILE)
    ensure_exists(FLOOD_RASTER)
    ensure_exists(BOUNDARY_FILE)

    print("Loading data...")
    roads = gpd.read_file(ROADS_FILE)
    boundary = gpd.read_file(BOUNDARY_FILE)

    # Load raster
    raster = rasterio.open(FLOOD_RASTER)
    raster_crs = raster.crs
    px_size = abs(raster.transform.a)
    print(f"Raster CRS: {raster_crs} | pixel size ≈ {px_size}")

    # Reproject roads/boundary to raster CRS (4326)
    roads = reproject_to_raster(roads, FLOOD_RASTER)
    boundary = reproject_to_raster(boundary, FLOOD_RASTER)

    # Clip roads to boundary
    bound_union = unary_union(boundary.geometry)
    roads = roads[roads.intersects(bound_union)].copy()
    roads["geometry"] = roads.geometry.intersection(bound_union)
    roads = roads[~roads.geometry.is_empty].reset_index(drop=True)

    # Create metric version for length calculation
    roads_metric = roads.to_crs(METRIC_CRS)

    print("Polygonizing flood raster...")
    blocked_union, _ = polygonize_threshold(FLOOD_RASTER, DEPTH_THRESHOLD)
    if blocked_union is None:
        print("Warning: No blocked flood area found (depth > threshold).")
    else:
        print("Flood polygon created.")

    out_features = []
    seg_id = 0

    print("Splitting & sampling roads...")
    for idx, row in roads.iterrows():
        geom_4326 = row.geometry
        geom_metric = roads_metric.geometry.iloc[idx]

        if geom_4326 is None or geom_4326.is_empty:
            continue

        if blocked_union is None:
            segs = [geom_4326]
            flags = [False]
        else:
            inside = geom_4326.intersection(blocked_union)
            outside = geom_4326.difference(blocked_union)

            inside_segs = lines_from_geometry_collection(inside)
            outside_segs = lines_from_geometry_collection(outside)

            segs = outside_segs + inside_segs
            flags = [False] * len(outside_segs) + [True] * len(inside_segs)

            if not segs:
                segs = [geom_4326]
                flags = [False]

        # Process each segment
        for sline, inside_flag in zip(segs, flags):

            # Compute length in meters via metric CRS
            seg_metric = gpd.GeoSeries([sline], crs=roads.crs).to_crs(METRIC_CRS).iloc[0]
            length_m = seg_metric.length

            if length_m < MIN_SEG_LEN_M:
                continue

            mean_d, max_d = sample_stats_along_line(sline, raster, px_size)
            mean_d = mean_d or 0.0
            max_d = max_d or 0.0

            is_passable = not (max_d > DEPTH_THRESHOLD)
            flood_class = "passable" if is_passable else "blocked"

            out_features.append({
                "road_id": int(idx),
                "segment_id": int(seg_id),
                "mean_depth": float(mean_d),
                "max_depth": float(max_d),
                "flood_class": flood_class,
                "is_passable": bool(is_passable),
                "length_m": float(length_m),
                "geometry": sline
            })
            seg_id += 1

    if not out_features:
        print("ERROR: No road segments generated. Check CRS or threshold.")
        return

    out_gdf = gpd.GeoDataFrame(out_features, crs=raster_crs)

    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    print("Saving:", OUT_GEOJSON)
    out_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")

    # ------------------------
    # Build NetworkX graph
    # ------------------------
    print("Building road graph...")
    G = nx.Graph()

    for _, r in out_gdf.iterrows():
        geom = r.geometry
        start = round_coord(geom.coords[0])
        end = round_coord(geom.coords[-1])

        if start not in G:
            G.add_node(start, x=start[0], y=start[1])
        if end not in G:
            G.add_node(end, x=end[0], y=end[1])

        weight = r["length_m"] if r["is_passable"] else 1e12

        G.add_edge(start, end,
                   road_id=r["road_id"],
                   segment_id=r["segment_id"],
                   mean_depth=r["mean_depth"],
                   max_depth=r["max_depth"],
                   length_m=r["length_m"],
                   is_passable=r["is_passable"],
                   flood_class=r["flood_class"],
                   weight=weight)

    print("Saving graph:", OUT_GRAPHML)
    nx.write_graphml(G, OUT_GRAPHML)

    print("Done.")
    print("Total segments:", len(out_gdf))
    print("Blocked segments:", sum(~out_gdf["is_passable"]))
    print("Total length (m):", out_gdf["length_m"].sum())


if __name__ == "__main__":
    main()
