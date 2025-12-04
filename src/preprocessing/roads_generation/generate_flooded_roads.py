#!/usr/bin/env python3
"""
generate_flooded_roads_with_graph.py

Inputs:
 - BTS_Restoration_Project/data/cleaned/roads_hue_clean.geojson
 - BTS_Restoration_Project/data/cleaned/flood_depth_combined_clean.tif
 - BTS_Restoration_Project/data/cleaned/hue_boundary_clean.geojson

Outputs:
 - BTS_Restoration_Project/data/cleaned/roads_flooded.geojson  (split segments with flood attributes)
 - BTS_Restoration_Project/data/cleaned/roads_flooded.graphml  (NetworkX graph)

Behavior:
 - polygonize raster where depth > DEPTH_THRESHOLD (blocked)
 - split lines by polygon union (segments inside polygon = flooded-blocked)
 - sample raster along each segment to compute mean_depth and max_depth
 - classify each segment: is_passable = max_depth <= DEPTH_THRESHOLD
 - build an undirected NetworkX graph with nodes at segment endpoints and edges representing segments
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
from shapely.ops import split as shapely_split
import networkx as nx

# ------------------------
# CONFIG
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ROADS_FILE = PROJECT_ROOT / "data" / "cleaned" / "roads_hue_clean.geojson"
FLOOD_RASTER = PROJECT_ROOT / "data" / "processed" / "flood" / "flood_depth_combined_clean.tif"
BOUNDARY_FILE = PROJECT_ROOT / "data" / "cleaned" / "hue_boundary_clean.geojson"

OUT_GEOJSON = PROJECT_ROOT / "data" / "processed" / "road" / "roads_flooded.geojson"
OUT_GRAPHML = PROJECT_ROOT / "data" / "processed" / "road" / "roads_flooded.graphml"

DEPTH_THRESHOLD = 1.0  # >1.0 m => blocked (vehicle cannot pass)
MIN_SEG_LEN_M = 0.5    # drop extremely short segments (meters)

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

def polygonize_threshold(raster_path: Path, threshold: float, nodata_value=None):
    """
    Polygonize raster pixels where value > threshold.
    Returns a unary_union geometry (possibly MultiPolygon) of flooded areas.
    """
    with rasterio.open(raster_path) as src:
        band1 = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs

        # Create boolean mask where pixel > threshold and not masked
        if np.ma.is_masked(band1):
            valid_mask = ~band1.mask
            data = band1.data
        else:
            valid_mask = np.ones(band1.shape, dtype=bool)
            data = band1

        cond_mask = (data > threshold) & valid_mask

        # If no pixels, return empty geometry
        if not cond_mask.any():
            return None, crs

        # polygonize using rasterio.features.shapes (we polygonize cond_mask as uint8)
        shapes_gen = features.shapes(cond_mask.astype('uint8'), transform=transform)
        geoms = []
        for geom, val in shapes_gen:
            if val == 1:
                geoms.append(shape(geom))

        if not geoms:
            return None, crs

        union = unary_union(geoms)
        return union, crs

def sample_stats_along_line(line: LineString, raster_src, px_size: float):
    """
    Sample raster values along a LineString at approx interval = px_size,
    return (mean, max) ignoring nodata.
    """
    length = line.length
    if length == 0:
        return None, None

    n_samples = max(2, int(math.ceil(length / px_size)))
    # create normalized points along line
    pts = [line.interpolate(float(i) / (n_samples - 1), normalized=True) for i in range(n_samples)]
    coords = [(p.x, p.y) for p in pts]

    values = []
    for val in raster_src.sample(coords):
        # val is array-like (bands)
        v = val[0]
        # handle masked arrays and nodata
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        # raster nodata check
        if raster_src.nodata is not None and v == raster_src.nodata:
            continue
        if np.isnan(v):
            continue
        values.append(v)

    if not values:
        return None, None
    return float(np.mean(values)), float(np.max(values))

def lines_from_geometry_collection(geom):
    """
    Normalize: return list of LineString parts from various geometry types.
    """
    if geom is None:
        return []
    geom_type = geom.geom_type
    if geom_type == "LineString":
        return [geom]
    elif geom_type == "MultiLineString":
        return [g for g in geom.geoms if isinstance(g, LineString) and (not g.is_empty)]
    elif geom_type == "GeometryCollection":
        return [g for g in geom.geoms if isinstance(g, LineString) and (not g.is_empty)]
    else:
        return []

def round_coord(pt_tuple, digits=6):
    return (round(pt_tuple[0], digits), round(pt_tuple[1], digits))

# ------------------------
# Main
# ------------------------
def main():
    # existence checks
    ensure_exists(ROADS_FILE)
    ensure_exists(FLOOD_RASTER)
    ensure_exists(BOUNDARY_FILE)

    print("Loading data...")
    roads = gpd.read_file(ROADS_FILE)
    boundary = gpd.read_file(BOUNDARY_FILE)

    # open raster
    raster = rasterio.open(FLOOD_RASTER)
    raster_crs = raster.crs
    px_size = abs(raster.transform.a)  # pixel width (assumes square ~)
    print(f"Raster CRS: {raster_crs}; pixel size ~{px_size}")

    # reproject layers to raster CRS if needed
    roads = reproject_to_raster(roads, FLOOD_RASTER)
    boundary = reproject_to_raster(boundary, FLOOD_RASTER)

    # warn if CRS is geographic (degrees)
    try:
        if not raster.crs.is_projected:
            warnings.warn("Raster CRS is not a projected CRS. Lengths will be in degrees, not meters. "
                          "Consider using a projected CRS for metric lengths.", UserWarning)
    except Exception:
        pass

    # Clip roads to study boundary
    bound_union = unary_union(boundary.geometry)
    roads = roads[roads.intersects(bound_union)].copy()
    # safe intersection to crop lines to boundary
    roads["geometry"] = roads.geometry.intersection(bound_union)
    roads = roads[~roads.geometry.is_empty]
    roads = roads.reset_index(drop=True)

    print("Polygonizing raster for blocked areas (depth >", DEPTH_THRESHOLD, "m)...")
    blocked_union, _ = polygonize_threshold(FLOOD_RASTER, DEPTH_THRESHOLD)
    if blocked_union is None:
        print("No blocked cells found (depth > threshold). All segments will be passable.")
    else:
        print("Blocked geometry prepared.")

    out_features = []
    seg_counter = 0

    print("Splitting roads by blocked area and sampling depths...")
    for orig_idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # If blocked union is None (no blocked pixels), the whole line is passable
        if blocked_union is None:
            segments = [geom]  # no splitting
            inside_flags = [False]  # all false (not blocked)
        else:
            # We want segments separated by blocked area boundaries.
            # Compute intersection (inside blocked) and difference (outside)
            inside = geom.intersection(blocked_union)
            outside = geom.difference(blocked_union)

            inside_segs = lines_from_geometry_collection(inside)
            outside_segs = lines_from_geometry_collection(outside)

            # combine with flags
            segments = []
            inside_flags = []
            for s in outside_segs:
                segments.append(s)
                inside_flags.append(False)
            for s in inside_segs:
                segments.append(s)
                inside_flags.append(True)

            # If both empty (maybe the line lies exactly on boundary or other degeneracy), fallback to whole geom
            if not segments:
                segments = [geom]
                inside_flags = [False]

        # For each segment, compute depth stats (sampling), classify, and keep if length enough
        for seg, inside_flag in zip(segments, inside_flags):
            if seg is None or seg.is_empty:
                continue
            # skip segments that are points
            if seg.geom_type not in ("LineString", "MultiLineString"):
                # try to get lines
                seg_lines = lines_from_geometry_collection(seg)
            else:
                seg_lines = [seg] if seg.geom_type == "LineString" else lines_from_geometry_collection(seg)

            for sline in seg_lines:
                length = sline.length
                # If projected CRS, length is in CRS units (likely meters). If geographic, it's degrees.
                if length < MIN_SEG_LEN_M:
                    continue

                # sample stats
                mean_depth, max_depth = sample_stats_along_line(sline, raster, px_size)
                # handle None
                if mean_depth is None:
                    mean_depth = 0.0
                if max_depth is None:
                    max_depth = 0.0

                # classify based on max_depth (strict > threshold -> blocked)
                is_passable = not (max_depth > DEPTH_THRESHOLD)
                flood_class = "passable" if is_passable else "blocked"

                feat = {
                    "road_id": int(orig_idx),
                    "segment_id": int(seg_counter),
                    "mean_depth": float(mean_depth),
                    "max_depth": float(max_depth),
                    "flood_class": flood_class,
                    "is_passable": bool(is_passable),
                    "length_m": float(length),
                    "geometry": sline
                }
                out_features.append(feat)
                seg_counter += 1

    # build GeoDataFrame
    if not out_features:
        print("No segments generated. Exiting.")
        return

    out_gdf = gpd.GeoDataFrame(out_features, crs=raster_crs)
    # optionally simplify geometry slightly for storage (disabled by default)
    # out_gdf['geometry'] = out_gdf.geometry.simplify(0.5)

    print(f"Saving split segments to {OUT_GEOJSON} ...")
    out_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    print("Saved:", OUT_GEOJSON)

    # ------------------------
    # Build NetworkX graph
    # ------------------------
    print("Building NetworkX graph from segments...")

    G = nx.Graph()
    # We'll add nodes as coordinate tuples rounded to 6 decimal places to avoid floating duplicates
    for _, r in out_gdf.iterrows():
        geom: LineString = r.geometry
        # get endpoints
        start = (geom.coords[0][0], geom.coords[0][1])
        end = (geom.coords[-1][0], geom.coords[-1][1])
        start_r = round_coord(start)
        end_r = round_coord(end)

        # add nodes with coordinate attribute
        if start_r not in G:
            G.add_node(start_r, x=start_r[0], y=start_r[1])
        if end_r not in G:
            G.add_node(end_r, x=end_r[0], y=end_r[1])

        # edge attributes
        attr = {
            "road_id": int(r["road_id"]),
            "segment_id": int(r["segment_id"]),
            "mean_depth": float(r["mean_depth"]),
            "max_depth": float(r["max_depth"]),
            "is_passable": bool(r["is_passable"]),
            "flood_class": r["flood_class"],
            "length_m": float(r["length_m"])
        }

        # set weight: if blocked -> set very large weight (or you may choose to skip adding edge)
        if not attr["is_passable"]:
            # Option 1: add edge but set weight large
            attr["weight"] = float("inf")  # some algorithms cannot handle inf, you may set a large number instead
            # Option 2 (commented): skip adding blocked edges to force graph without those connections
            # continue
        else:
            # weight = length (cost)
            attr["weight"] = float(attr["length_m"])

        # For undirected graph, if multiple edges between same nodes, keep smallest weight
        if G.has_edge(start_r, end_r):
            # if existing weight is larger, replace
            existing = G[start_r][end_r]
            existing_w = existing.get("weight", float("inf"))
            if attr["weight"] < existing_w:
                G[start_r][end_r].update(attr)
        else:
            G.add_edge(start_r, end_r, **attr)

    # Save graph to GraphML (GraphML supports only string/numeric attributes; `inf` may be problematic)
    # Replace inf weights by a large number for GraphML compatibility
    LARGE_WEIGHT = 1e12
    for u, v, ed in G.edges(data=True):
        if ed.get("weight") is None:
            ed["weight"] = float(ed.get("length_m", 0.0))
        if ed["weight"] == float("inf") or (isinstance(ed["weight"], float) and np.isinf(ed["weight"])):
            ed["weight"] = LARGE_WEIGHT

    print(f"Writing graph to {OUT_GRAPHML} ...")
    nx.write_graphml(G, OUT_GRAPHML)
    print("Saved graph:", OUT_GRAPHML)

    print("Done. Summary:")
    total_segments = len(out_gdf)
    blocked_segments = sum(~out_gdf["is_passable"])
    total_length = out_gdf["length_m"].sum()
    print(f" - total segments: {total_segments}")
    print(f" - blocked segments: {int(blocked_segments)}")
    print(f" - total network length (CRS units): {total_length:.2f}")

if __name__ == "__main__":
    main()
