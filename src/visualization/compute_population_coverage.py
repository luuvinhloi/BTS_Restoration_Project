# src/visualization/compute_population_coverage.py
"""
Compute population in outage (failed BTS), population covered by active BTS, and population restored by COWs (GA_PSO or MILP).

Behavior:
    - Use raster zonal stats on unioned polygons (unique population, no double counting).
    - For outage (population lost due to failed BTS) compute:
      outage_geom = (union_failed_buffers - union_active_buffers) INTERSECT province_boundary
    - For active coverage, use union_active_buffers INTERSECT province_boundary.
    - For COW coverage, union COW buffers INTERSECT province_boundary.
    - Export per-site CSVs (individual buffers statistics) and union summaries.
"""

from pathlib import Path
import json
import logging
import pandas as pd
import rasterio
from rasterio.warp import transform_geom
from rasterstats import zonal_stats
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union
from pyproj import Transformer, CRS
import numpy as np
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED = DATA_DIR / "processed"
RAW = DATA_DIR / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "results"
SUMMARY_DIR = OUT_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

POP_RASTER = PROCESSED / "pop_hue_clipped.tif"
FAILED_BTS = PROCESSED / "failed_bts.csv"
ACTIVE_BTS = PROCESSED / "active_bts.csv"
I_POINTS = PROCESSED / "I_points.csv"
J_SITES = PROCESSED / "J_sites.csv"
COWS = RAW / "cow_dataset.csv"
BOUNDARY_GEOJSON = RAW / "hue_boundary.geojson"

GA_PSO_ASSIGN = OUT_DIR / "ga_pso_assignments.csv"
MILP_ASSIGN = OUT_DIR / "milp_assignments.csv"

# Utility / low-level APIs
def read_raster_crs_and_nodata(raster_path: Path):
    with rasterio.open(raster_path) as src:
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        nodata = src.nodata
    return crs, nodata

def compute_total_population_from_raster(raster_path: Path):
    """Sum all valid raster pixels (ignore nodata)."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)
        # arr is masked array; sum ignoring masked
        total = float(arr.filled(0).sum())
    logging.info(f"Total population from raster: {total:,.0f}")
    return total

def buffer_point_meters(lon: float, lat: float, radius_m: float, target_crs="EPSG:4326"):
    """
    Create circular buffer around lon,lat (input in lon/lat) with radius in meters.
    Implementation: project to WebMercator (EPSG:3857), buffer in meters, transform back to target_crs.
    Returns GeoJSON-like geometry (dict).
    """
    merc_crs = CRS.from_epsg(3857)
    src_crs = CRS.from_epsg(4326)
    t_to_merc = Transformer.from_crs(src_crs, merc_crs, always_xy=True)
    t_from_merc = Transformer.from_crs(merc_crs, CRS.from_user_input(target_crs), always_xy=True)

    x_m, y_m = t_to_merc.transform(lon, lat)
    circle_m = Point(x_m, y_m).buffer(radius_m)
    # transform coords back to target_crs via shapely mapping & rasterio.transform_geom
    geom_merc = mapping(circle_m)
    geom_target = transform_geom(merc_crs.to_string(), target_crs, geom_merc, precision=6)
    return geom_target

def compute_zonal_sum_geom(geom, raster_path: Path, nodata=None):
    """
    Compute zonal sum of raster over geom (geom must be GeoJSON-like mapping or shapely mapping).
    Returns float (0 if none).
    """
    if geom is None:
        return 0.0
    try:
        # Use all_touched=True so pixels that are touched count (safer for small/edge polygons).
        stats = zonal_stats([geom], str(raster_path), stats="sum", nodata=nodata, all_touched=True)
        s = stats[0].get("sum", 0.0)
        return float(s if s else 0.0)
    except Exception as e:
        logging.error(f"zonal_stats failed on geom: {e}")
        return 0.0

def union_geoms_list(geoms):
    """
    Input: list of GeoJSON-like geometries or shapely geometries.
    Return: shapely geometry union or None.
    """
    shapes = []
    for g in geoms:
        if g is None:
            continue
        if isinstance(g, dict):
            shapes.append(shape(g))
        else:
            shapes.append(g)
    if not shapes:
        return None
    u = unary_union(shapes)
    return u if not u.is_empty else None

def intersect_with_boundary(shapely_geom, boundary_path: Path, target_crs="EPSG:4326"):
    """
    Intersect shapely_geom with province boundary (read from boundary_path).
    Result is shapely geometry or None.
    """
    if shapely_geom is None:
        return None
    try:
        b = gpd.read_file(boundary_path)
        # ensure b geometry in target_crs
        if b.crs:
            b = b.to_crs(target_crs)
        boundary = b.geometry.unary_union
        inter = shapely_geom.intersection(boundary)
        if inter.is_empty:
            return None
        return inter
    except Exception as e:
        logging.warning(f"Could not intersect with boundary: {e} — returning original geometry clipped to nothing if fails.")
        # fallback: return original (but we prefer intersect)
        return shapely_geom

# Per-site stats helpers
def per_site_buffers_and_stats(df_sites, raster_crs, raster_path: Path, nodata=None, boundary_geom=None):
    """
    For each row in df_sites (expects lon/lat and coverage_radius_m), build buffer geom,
    optionally intersect with boundary_geom, compute zonal sum (per-site).
    Returns list of dicts (records) and list of shapely geoms (buffers original intersected).
    """
    records = []
    geoms = []
    for _, row in df_sites.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        radius = float(row.get("coverage_radius_m", 3000))
        geom = buffer_point_meters(lon, lat, radius, target_crs=raster_crs)
        if geom is None:
            continue
        shap = shape(geom)
        if boundary_geom is not None:
            shap = shap.intersection(boundary_geom)
            if shap.is_empty:
                pop = 0.0
                geoms.append(None)
                records.append({
                    "site_id": row.get("site_id"),
                    "latitude": lat,
                    "longitude": lon,
                    "coverage_radius_m": radius,
                    "pop_in_buffer": pop
                })
                continue
        pop = compute_zonal_sum_geom(mapping(shap), raster_path, nodata=nodata)
        geoms.append(shap)
        records.append({
            "site_id": row.get("site_id"),
            "latitude": lat,
            "longitude": lon,
            "coverage_radius_m": radius,
            "pop_in_buffer": pop
        })
    return records, geoms

# High-level computations
def compute_active_failed_unions_and_stats():
    """
    Compute:
      - union_active (shapely)
      - union_failed (shapely)
      - per-site CSVs for diagnostics
    """
    raster_crs, nodata = read_raster_crs_and_nodata(POP_RASTER)
    # read boundary and ensure CRS
    boundary = gpd.read_file(BOUNDARY_GEOJSON)
    target_crs = raster_crs
    if boundary.crs:
        boundary = boundary.to_crs(target_crs)
    boundary_geom = boundary.geometry.unary_union

    # read site tables
    active_df = pd.read_csv(ACTIVE_BTS)
    failed_df = pd.read_csv(FAILED_BTS)

    # compute per-site buffers and per-site pop (clipped to boundary)
    active_records, active_geoms = per_site_buffers_and_stats(active_df, target_crs, POP_RASTER, nodata=nodata, boundary_geom=boundary_geom)
    failed_records, failed_geoms = per_site_buffers_and_stats(failed_df, target_crs, POP_RASTER, nodata=nodata, boundary_geom=boundary_geom)

    # union them
    active_union = union_geoms_list(active_geoms)
    failed_union = union_geoms_list(failed_geoms)

    # ensure both unions are intersected with boundary (safety)
    if active_union is not None:
        active_union = active_union.intersection(boundary_geom)
        if active_union.is_empty:
            active_union = None
    if failed_union is not None:
        failed_union = failed_union.intersection(boundary_geom)
        if failed_union.is_empty:
            failed_union = None

    # compute union sums (unique populations)
    active_union_pop = compute_zonal_sum_geom(mapping(active_union) if active_union is not None else None, POP_RASTER, nodata=nodata)
    failed_union_pop = compute_zonal_sum_geom(mapping(failed_union) if failed_union is not None else None, POP_RASTER, nodata=nodata)

    # compute outage = failed_union - active_union (then clipped to boundary already)
    outage_geom = None
    if failed_union is not None:
        if active_union is not None:
            outage_geom = failed_union.difference(active_union)
        else:
            outage_geom = failed_union
        # clip to boundary (again defensive)
        outage_geom = outage_geom.intersection(boundary_geom) if outage_geom is not None else None
        if outage_geom is not None and outage_geom.is_empty:
            outage_geom = None

    outage_pop = compute_zonal_sum_geom(mapping(outage_geom) if outage_geom is not None else None, POP_RASTER, nodata=nodata)

    # save per-site CSVs
    pd.DataFrame(active_records).to_csv(SUMMARY_DIR / "active_bts_per_site_pop.csv", index=False)
    pd.DataFrame(failed_records).to_csv(SUMMARY_DIR / "failed_bts_per_site_pop.csv", index=False)

    # also save union geometries to GeoJSON features for visualization
    features = []
    if active_union is not None:
        features.append({
            "type": "Feature",
            "properties": {"layer": "active_union"},
            "geometry": mapping(active_union)
        })
    if failed_union is not None:
        features.append({
            "type": "Feature",
            "properties": {"layer": "failed_union"},
            "geometry": mapping(failed_union)
        })
    if outage_geom is not None:
        features.append({
            "type": "Feature",
            "properties": {"layer": "outage_union"},
            "geometry": mapping(outage_geom)
        })
    geo = {"type": "FeatureCollection", "features": features}
    with open(SUMMARY_DIR / "bts_union_layers.geojson", "w", encoding="utf-8") as f:
        json.dump(geo, f, ensure_ascii=False, indent=2)

    return {
        "active_union": active_union,
        "failed_union": failed_union,
        "outage_geom": outage_geom,
        "active_union_pop": active_union_pop,
        "failed_union_pop": failed_union_pop,
        "outage_pop": outage_pop
    }

def compute_cow_coverage_and_stats(method="GA_PSO", outage_geom=None):
    """
    Compute union of COW coverage and per-site COW pop stats (clipped to boundary).
    If outage_geom is provided (shapely geometry), compute also population of COW coverage
    *within the outage area* (i.e., restored population).
    Returns both total cow_union_pop and cow_union_pop_within_outage.
    """
    raster_crs, nodata = read_raster_crs_and_nodata(POP_RASTER)
    boundary = gpd.read_file(BOUNDARY_GEOJSON)
    target_crs = raster_crs
    if boundary.crs:
        boundary = boundary.to_crs(target_crs)
    boundary_geom = boundary.geometry.unary_union

    j_sites = pd.read_csv(J_SITES)
    cows = pd.read_csv(COWS)
    assign_path = GA_PSO_ASSIGN if method.upper() == "GA_PSO" else MILP_ASSIGN
    if not assign_path.exists():
        logging.warning(f"{method} assignment file missing: {assign_path}")
        # return empty
        return {"cow_union": None, "cow_union_pop": 0.0, "cow_union_pop_in_outage": 0.0, "cow_records": [], "cow_geoms": []}

    assign = pd.read_csv(assign_path)

    # Build mapping site_id -> list of cow radii (if multiple cows assigned to same site)
    cow_radius_map = {}
    for _, r in assign.iterrows():
        sid = r.get("assigned_site_id")
        if pd.isna(sid):
            continue
        sid = str(sid)
        cow_id = r.get("cow_id")
        cov_r = None
        # if assignment row contains coverage_radius_m directly, use it
        if "coverage_radius_m" in r and not pd.isna(r["coverage_radius_m"]):
            cov_r = float(r["coverage_radius_m"])
        else:
            # fallback: lookup in cows table by cow_id
            if pd.notna(cow_id):
                crow = cows[cows["cow_id"] == cow_id]
                if not crow.empty and "coverage_radius_m" in crow.columns:
                    cov_r = float(crow.iloc[0]["coverage_radius_m"])
        if cov_r is None:
            cov_r = 3000.0
        cow_radius_map.setdefault(sid, []).append(cov_r)

    # chosen site ids as strings
    chosen_sites = assign["assigned_site_id"].dropna().astype(str).unique().tolist()
    cow_records = []
    cow_geoms = []
    for sid in chosen_sites:
        jrow = j_sites[j_sites["site_id"] == sid]
        if jrow.empty:
            jrow = j_sites[j_sites["i_ref"] == sid]
        if jrow.empty:
            logging.debug(f"COW assigned site {sid} not found in J_sites.")
            continue
        lon = float(jrow.iloc[0]["longitude"])
        lat = float(jrow.iloc[0]["latitude"])
        # choose conservative radius: max of cow radii assigned to that site
        radii = cow_radius_map.get(sid, [])
        radius = max(radii) if radii else float(jrow.iloc[0].get("coverage_radius_m", 3000))
        # build buffer and clip to boundary
        geom = buffer_point_meters(lon, lat, radius, target_crs)
        shap = shape(geom).intersection(boundary_geom) if boundary_geom is not None else shape(geom)
        if shap.is_empty:
            pop_total = 0.0
            pop_in_outage = 0.0
            cow_geoms.append(None)
        else:
            pop_total = compute_zonal_sum_geom(mapping(shap), POP_RASTER, nodata=nodata)
            if outage_geom is not None:
                inter = shap.intersection(outage_geom)
                if inter is None or inter.is_empty:
                    pop_in_outage = 0.0
                else:
                    pop_in_outage = compute_zonal_sum_geom(mapping(inter), POP_RASTER, nodata=nodata)
            else:
                pop_in_outage = 0.0
            cow_geoms.append(shap)
        cow_records.append({
            "site_id": sid,
            "latitude": lat,
            "longitude": lon,
            "coverage_radius_m": radius,
            "pop_in_buffer_total": pop_total,
            "pop_in_buffer_in_outage": pop_in_outage
        })

    cow_union = union_geoms_list(cow_geoms)
    if cow_union is not None:
        if cow_union.geom_type == "MultiPolygon":
            cow_union = unary_union(cow_union)
        cow_union = cow_union.buffer(0)  #  fix polygon invalidity

    # total pop in cow union (all area)
    cow_union_pop = compute_zonal_sum_geom(mapping(cow_union) if cow_union is not None else None,
                                           POP_RASTER, nodata=nodata)

    # pop covered by cows but restricted to outage area (this is the actual 'restored' pop)
    cow_union_pop_in_outage = 0.0
    if outage_geom is not None and cow_union is not None:
        inter_union = cow_union.intersection(outage_geom)
        if inter_union is not None and not inter_union.is_empty:
            cow_union_pop_in_outage = compute_zonal_sum_geom(mapping(inter_union), POP_RASTER, nodata=nodata)

    # Save per-site (with two pop cols)
    pd.DataFrame(cow_records).to_csv(SUMMARY_DIR / f"cow_per_site_pop_{method.lower()}.csv", index=False)

    # Save union geometry
    features = []
    if cow_union is not None:
        features.append({"type": "Feature", "properties": {"layer": "cow_union"}, "geometry": mapping(cow_union)})
    with open(SUMMARY_DIR / f"cow_union_{method.lower()}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

    return {
        "cow_union": cow_union,
        "cow_union_pop": cow_union_pop,
        "cow_union_pop_in_outage": cow_union_pop_in_outage,
        "cow_records": cow_records,
        "cow_geoms": cow_geoms
    }

# Main orchestration
def main_compute_all(method="GA_PSO"):
    """
    Run full pipeline and produce:
      - outputs/results/coverage_report_<method>.json
      - per-site CSVs
      - union GeoJSONs
    """
    # 1) total pop
    total_pop = compute_total_population_from_raster(POP_RASTER)

    # 2) active / failed bases
    bts_results = compute_active_failed_unions_and_stats()
    active_union_pop = bts_results.get("active_union_pop", 0.0)
    failed_union_pop = bts_results.get("failed_union_pop", 0.0)
    outage_pop = bts_results.get("outage_pop", 0.0)
    outage_geom = bts_results.get("outage_geom", None)

    # 3) cows - pass outage_geom so we measure only pop restored within outage
    cow_results = compute_cow_coverage_and_stats(method=method, outage_geom=outage_geom)
    # use population restored within outage (not total cow union pop) as restored_by_cows
    cow_union_pop_in_outage = cow_results.get("cow_union_pop_in_outage", 0.0)
    cow_union_pop_total = cow_results.get("cow_union_pop", 0.0)

    # 4) Additional checks & derived metrics
    # population not covered by any active BTS (current outage by total-active)
    population_not_covered_now = max(0.0, total_pop - active_union_pop)

    # Save combined visualization geojson (active, failed, outage, cow)
    features = []
    # active union
    if bts_results.get("active_union") is not None:
        features.append({"type": "Feature", "properties": {"layer": "active_union"}, "geometry": mapping(bts_results["active_union"])})
    if bts_results.get("failed_union") is not None:
        features.append({"type": "Feature", "properties": {"layer": "failed_union"}, "geometry": mapping(bts_results["failed_union"])})
    if outage_geom is not None:
        features.append({"type": "Feature", "properties": {"layer": "outage_union"}, "geometry": mapping(outage_geom)})
    if cow_results.get("cow_union") is not None:
        features.append({"type": "Feature", "properties": {"layer": "cow_union"}, "geometry": mapping(cow_results["cow_union"])})

    with open(SUMMARY_DIR / f"coverage_layers_{method.lower()}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

    # Compose final summary (include both outage definitions)
    # outage_pop : population inside failed buffers but not inside active buffers (this is "lost due to failure")
    # population_not_covered_now : total - active_union_pop (this is population without active BTS coverage now; includes places that never had BTS)
    summary = {
        "method": method,
        "total_population": float(total_pop),
        "population_covered_by_active_bts": float(active_union_pop),
        "population_outage_due_failed_bts": float(outage_pop),
        "population_not_covered_now": float(population_not_covered_now),
        "failed_union_population": float(failed_union_pop),
        # Use cow_union_pop_in_outage (restored within outage) as the restored value
        "population_restored_by_cows": float(cow_union_pop_in_outage),
        "coverage_restored_percent_of_outage": round((cow_union_pop_in_outage / outage_pop * 100) if outage_pop > 0 else 0.0, 2),
        "coverage_active_percent": round(active_union_pop / total_pop * 100, 2),
        # coverage after restoration should use active + restored_in_outage (avoid double counting)
        "coverage_after_restoration_percent": round((active_union_pop + cow_union_pop_in_outage) / total_pop * 100, 2),
        # include also the total cow union pop (for diagnostics)
        "cow_union_pop_total": float(cow_union_pop_total)
    }

    out_json = SUMMARY_DIR / f"coverage_report_{method.lower()}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote coverage summary: {out_json}")

    return summary

if __name__ == "__main__":
    res = main_compute_all(method="GA_PSO")
    print("Done. Summary:", res)
