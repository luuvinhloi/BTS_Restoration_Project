#!/usr/bin/env python3
"""
compute_population_coverage.py

Extended compute pipeline that:
- keeps original zonal-statistics / union logic
- reads MILP/GUROBI assignment outputs:
    - assignments_cow_GUROBI.csv
    - assignments_power_GUROBI.csv
- computes:
    - total population
    - lost coverage (population in failed/power_outage BTS buffers not covered by active BTS)
    - population restored by COWs (unique, non-overlapping)
    - population restored by power assignments (unique, non-overlapping)
    - combined restored population (COW U POWER) inside outage
    - percentages before/after restoration
    - max deployment time
    - total deployment cost (COW cost + travel + power cost + travel)
- writes per-site CSVs, geojson outputs and a final summary JSON.

Author: Generated for user (Lợi Lưu) — 2025
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
import geopandas as gpd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Project paths (adjust if your repo structure differs)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
OUT_DIR = PROJECT_ROOT / "outputs"
SUMMARY_DIR = OUT_DIR / "summary" / "milp"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Input datasets (existing)
POP_RASTER = CLEANED_DIR / "pop_hue_clean.tif"
FAILED_BTS = PROCESSED_DIR / "damage_bts" / "failed_bts.csv"
ACTIVE_BTS = PROCESSED_DIR / "damage_bts" / "active_bts.csv"
I_POINTS = PROCESSED_DIR / "position_I_J" / "I_points.csv"
J_SITES = PROCESSED_DIR / "position_I_J" / "J_sites.csv"
COWS = PROCESSED_DIR / "cow" / "cow_dataset.csv"
BOUNDARY_GEOJSON = CLEANED_DIR / "hue_boundary_clean.geojson"

# Assignment outputs to read (from MILP/Gurobi)
ASSIGN_COW_GUROBI = OUT_DIR / "milp_runs" / "milp_gurobi" / "assignments_cow_GUROBI.csv"
ASSIGN_POWER_GUROBI = OUT_DIR / "milp_runs" / "milp_gurobi"  / "assignments_power_GUROBI.csv"

# Keep compatibility with previous variable names
GA_PSO_ASSIGN = OUT_DIR / "ga_pso_assignments.csv"
MILP_ASSIGN = OUT_DIR / "milp_assignments.csv"

# ---------------------------------------------------------------------------
# Utility / low-level APIs (kept and reused from original implementation)
# ---------------------------------------------------------------------------
def read_raster_crs_and_nodata(raster_path: Path):
    with rasterio.open(raster_path) as src:
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        nodata = src.nodata
    return crs, nodata

def compute_total_population_from_raster(raster_path: Path):
    """Sum all valid raster pixels (ignore nodata)."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)
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
        if b.crs:
            b = b.to_crs(target_crs)
        boundary = b.geometry.unary_union
        inter = shapely_geom.intersection(boundary)
        if inter.is_empty:
            return None
        return inter
    except Exception as e:
        logging.warning(f"Could not intersect with boundary: {e} — returning original geometry.")
        return shapely_geom

# ---------------------------------------------------------------------------
# Reused per-site helper
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Existing logic for active/failed unions
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# New functions to compute COW and POWER coverage from Gurobi assignment outputs
# ---------------------------------------------------------------------------
def compute_cow_coverage_from_assignments(assign_path: Path, j_sites_path: Path, method_tag="gurobi", outage_geom=None):
    """
    Read assignments_cow_GUROBI.csv and compute:
      - per-site cow pop (total and in outage)
      - cow_union shapely
      - cow union pop and cow union pop inside outage
      - per-site records
    """
    raster_crs, nodata = read_raster_crs_and_nodata(POP_RASTER)
    boundary = gpd.read_file(BOUNDARY_GEOJSON)
    target_crs = raster_crs
    if boundary.crs:
        boundary = boundary.to_crs(target_crs)
    boundary_geom = boundary.geometry.unary_union

    if not assign_path.exists():
        logging.warning(f"COW assignment file not found: {assign_path}")
        return {"cow_union": None, "cow_union_pop": 0.0, "cow_union_pop_in_outage": 0.0, "cow_records": [], "cow_geoms": [], "max_deploy_time":0.0, "total_cost":0.0, "cow_count":0}

    assign = pd.read_csv(assign_path)
    j_sites = pd.read_csv(j_sites_path)

    cow_records = []
    cow_geoms = []
    max_deploy = 0.0
    total_cost = 0.0

    # iterate assignment rows; many rows correspond to one cow per row
    for _, r in assign.iterrows():
        try:
            cow_id = r.get("cow_id", None)
            sid = r.get("site_id") or r.get("assigned_site_id") or r.get("site") or r.get("site_id_assigned")
            lat = r.get("lat") if "lat" in r else r.get("latitude") if "latitude" in r else r.get("lat")
            lon = r.get("lon") if "lon" in r else r.get("longitude") if "longitude" in r else r.get("lon")
            # fallback: try to locate site in j_sites by site id
            if pd.isna(lat) or pd.isna(lon):
                if pd.notna(sid):
                    js = j_sites[(j_sites["site_id"].astype(str) == str(sid)) | (j_sites.get("i_ref", "").astype(str) == str(sid))]
                    if not js.empty:
                        lat = float(js.iloc[0]["latitude"])
                        lon = float(js.iloc[0]["longitude"])
            if pd.isna(lat) or pd.isna(lon):
                logging.debug(f"Skipping cow row with missing coords: {r}")
                continue
            lat = float(lat)
            lon = float(lon)
            radius = float(r.get("coverage_radius_m", r.get("coverage_radius", 3000.0)))
            # cost fields
            cost_vnd = float(r.get("cost_vnd", r.get("cost", 0.0)))
            travel_cost_vnd = float(r.get("travel_cost_vnd", r.get("travel_cost", 0.0)))
            total_cost += cost_vnd + travel_cost_vnd
            # deployment times
            travel_time = float(r.get("travel_time_hr", 0.0))
            setup_time = float(r.get("setup_time_h", r.get("setup_time_hr", 0.0)))
            deployment_time = float(r.get("deployment_time_hr", travel_time + setup_time))
            if deployment_time > max_deploy:
                max_deploy = deployment_time

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
                "cow_id": cow_id,
                "site_id": sid,
                "latitude": lat,
                "longitude": lon,
                "coverage_radius_m": radius,
                "pop_in_buffer_total": pop_total,
                "pop_in_buffer_in_outage": pop_in_outage,
                "cost_vnd": cost_vnd,
                "travel_cost_vnd": travel_cost_vnd,
                "deployment_time_hr": deployment_time
            })
        except Exception as e:
            logging.error(f"Error processing cow assignment row: {e}")
            continue

    cow_union = union_geoms_list(cow_geoms)
    if cow_union is not None:
        cow_union = cow_union.buffer(0)

    cow_union_pop = compute_zonal_sum_geom(mapping(cow_union) if cow_union is not None else None, POP_RASTER, nodata=nodata)
    cow_union_pop_in_outage = 0.0
    if outage_geom is not None and cow_union is not None:
        inter_union = cow_union.intersection(outage_geom)
        if inter_union is not None and not inter_union.is_empty:
            cow_union_pop_in_outage = compute_zonal_sum_geom(mapping(inter_union), POP_RASTER, nodata=nodata)

    pd.DataFrame(cow_records).to_csv(SUMMARY_DIR / f"cow_per_site_pop_{method_tag}.csv", index=False)
    features = []
    if cow_union is not None:
        features.append({"type":"Feature","properties":{"layer":"cow_union"},"geometry":mapping(cow_union)})
    with open(SUMMARY_DIR / f"cow_union_{method_tag}.geojson","w",encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":features}, f, ensure_ascii=False, indent=2)

    cow_count = len(pd.Series([r.get("cow_id") for _,r in assign.iterrows() if pd.notna(r.get("cow_id"))]).unique())

    return {"cow_union": cow_union,
            "cow_union_pop": cow_union_pop,
            "cow_union_pop_in_outage": cow_union_pop_in_outage,
            "cow_records": cow_records,
            "cow_geoms": cow_geoms,
            "max_deploy_time": max_deploy,
            "total_cost": total_cost,
            "cow_count": cow_count}

def compute_power_coverage_from_assignments(assign_path: Path, failed_bts_path: Path, method_tag="gurobi", outage_geom=None):
    """
    Read assignments_power_GUROBI.csv and compute:
      - per-power assignment pop restored (using failed_bts coverage_radius_m)
      - power_union shapely
      - union pops and union pop inside outage
    """
    raster_crs, nodata = read_raster_crs_and_nodata(POP_RASTER)
    boundary = gpd.read_file(BOUNDARY_GEOJSON)
    target_crs = raster_crs
    if boundary.crs:
        boundary = boundary.to_crs(target_crs)
    boundary_geom = boundary.geometry.unary_union

    if not assign_path.exists():
        logging.warning(f"Power assignment file not found: {assign_path}")
        return {"power_union": None, "power_union_pop": 0.0, "power_union_pop_in_outage": 0.0, "power_records": [], "power_geoms": [], "total_cost":0.0, "power_count":0}

    assign = pd.read_csv(assign_path)
    # load failed_bts to map bts_id -> coverage_radius_m and coordinates if needed
    failed_df = pd.read_csv(failed_bts_path) if failed_bts_path.exists() else pd.DataFrame()

    power_records = []
    power_geoms = []
    total_cost = 0.0

    for _, r in assign.iterrows():
        try:
            power_id = r.get("power_id")
            bts_id = r.get("bts_id")
            lat = r.get("lat") if "lat" in r else r.get("latitude") if "latitude" in r else r.get("lat")
            lon = r.get("lon") if "lon" in r else r.get("longitude") if "longitude" in r else r.get("lon")
            # If lat/lon missing, try to get from failed_bts by bts_id
            if pd.isna(lat) or pd.isna(lon):
                if pd.notna(bts_id) and not failed_df.empty:
                    bf = failed_df[failed_df["site_id"].astype(str) == str(bts_id)]
                    if bf.empty and "bts_id" in failed_df.columns:
                        bf = failed_df[failed_df["bts_id"].astype(str) == str(bts_id)]
                    if not bf.empty:
                        lat = float(bf.iloc[0]["latitude"])
                        lon = float(bf.iloc[0]["longitude"])
            if pd.isna(lat) or pd.isna(lon):
                logging.debug(f"Skipping power row with missing coords: {r}")
                continue
            lat = float(lat)
            lon = float(lon)
            # get radius from failed_bts (preferred)
            radius = None
            if pd.notna(bts_id) and not failed_df.empty:
                bf = failed_df[(failed_df.get("site_id", failed_df.get("bts_id")) .astype(str) == str(bts_id))] if "site_id" in failed_df.columns or "bts_id" in failed_df.columns else failed_df
                if not bf.empty and "coverage_radius_m" in bf.columns:
                    radius = float(bf.iloc[0]["coverage_radius_m"])
            if radius is None:
                radius = float(r.get("coverage_radius_m", r.get("coverage_radius", 3000.0)))
            # cost fields
            cost_vnd_24h = float(r.get("cost_vnd_24h", r.get("cost_vnd", 0.0)))
            travel_cost_vnd = float(r.get("travel_cost_vnd", r.get("travel_cost", 0.0)))
            total_cost += float(cost_vnd_24h) + float(travel_cost_vnd)

            geom = buffer_point_meters(lon, lat, radius, target_crs)
            shap = shape(geom).intersection(boundary_geom) if boundary_geom is not None else shape(geom)
            if shap.is_empty:
                pop_total = 0.0
                pop_in_outage = 0.0
                power_geoms.append(None)
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
                power_geoms.append(shap)
            power_records.append({
                "power_id": power_id,
                "bts_id": bts_id,
                "latitude": lat,
                "longitude": lon,
                "coverage_radius_m": radius,
                "pop_in_buffer_total": pop_total,
                "pop_in_buffer_in_outage": pop_in_outage,
                "cost_vnd_24h": cost_vnd_24h,
                "travel_cost_vnd": travel_cost_vnd
            })
        except Exception as e:
            logging.error(f"Error processing power assignment row: {e}")
            continue

    power_union = union_geoms_list(power_geoms)
    if power_union is not None:
        power_union = power_union.buffer(0)

    power_union_pop = compute_zonal_sum_geom(mapping(power_union) if power_union is not None else None, POP_RASTER, nodata=nodata)
    power_union_pop_in_outage = 0.0
    if outage_geom is not None and power_union is not None:
        inter_union = power_union.intersection(outage_geom)
        if inter_union is not None and not inter_union.is_empty:
            power_union_pop_in_outage = compute_zonal_sum_geom(mapping(inter_union), POP_RASTER, nodata=nodata)

    pd.DataFrame(power_records).to_csv(SUMMARY_DIR / f"power_per_site_pop_{method_tag}.csv", index=False)
    features = []
    if power_union is not None:
        features.append({"type":"Feature","properties":{"layer":"power_union"},"geometry":mapping(power_union)})
    with open(SUMMARY_DIR / f"power_union_{method_tag}.geojson","w",encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":features}, f, ensure_ascii=False, indent=2)

    power_count = len(pd.Series([r.get("power_id") for _,r in assign.iterrows() if pd.notna(r.get("power_id"))]).unique())

    return {"power_union": power_union,
            "power_union_pop": power_union_pop,
            "power_union_pop_in_outage": power_union_pop_in_outage,
            "power_records": power_records,
            "power_geoms": power_geoms,
            "total_cost": total_cost,
            "power_count": power_count}

# ---------------------------------------------------------------------------
# Main orchestration (keeps same overall structure, adds new metrics)
# ---------------------------------------------------------------------------
def main_compute_all(method="MILP_GUROBI"):
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
    active_union = bts_results.get("active_union", None)

    # lost coverage percent BEFORE restoration
    lost_coverage_percent = round(outage_pop / total_pop * 100, 2) if total_pop > 0 else 0.0

    # 3) cows - from Gurobi assignments
    cow_results = compute_cow_coverage_from_assignments(ASSIGN_COW_GUROBI, J_SITES, method_tag="gurobi", outage_geom=outage_geom)
    cow_union_pop_in_outage = cow_results.get("cow_union_pop_in_outage", 0.0)
    cow_union_pop_total = cow_results.get("cow_union_pop", 0.0)
    cow_union = cow_results.get("cow_union")
    cow_max_deploy = cow_results.get("max_deploy_time", 0.0)
    cow_total_cost = cow_results.get("total_cost", 0.0)
    cow_count = cow_results.get("cow_count", 0)

    # 4) power - from Gurobi assignments
    power_results = compute_power_coverage_from_assignments(ASSIGN_POWER_GUROBI, FAILED_BTS, method_tag="gurobi", outage_geom=outage_geom)
    power_union_pop_in_outage = power_results.get("power_union_pop_in_outage", 0.0)
    power_union_pop_total = power_results.get("power_union_pop", 0.0)
    power_union = power_results.get("power_union")
    power_total_cost = power_results.get("total_cost", 0.0)
    power_count = power_results.get("power_count", 0)

    # 5) Combined restored (unique, no double-count)
    combined_geoms = []
    if cow_union is not None:
        combined_geoms.append(cow_union)
    if power_union is not None:
        combined_geoms.append(power_union)
    combined_union = union_geoms_list(combined_geoms)
    if combined_union is not None:
        combined_union = combined_union.buffer(0)
    restored_total_in_outage = 0.0
    if outage_geom is not None and combined_union is not None:
        inter = combined_union.intersection(outage_geom)
        if inter is not None and not inter.is_empty:
            restored_total_in_outage = compute_zonal_sum_geom(mapping(inter), POP_RASTER, nodata=read_raster_crs_and_nodata(POP_RASTER)[1])

    # coverage after restoration percent
    coverage_after_restoration_percent = round((active_union_pop + restored_total_in_outage) / total_pop * 100, 2) if total_pop > 0 else 0.0
    coverage_restored_percent_of_outage = round((restored_total_in_outage / outage_pop * 100) if outage_pop > 0 else 0.0, 2)

    # max deployment time overall
    # For safety, include cow_max_deploy (power assignments usually have travel_time in file; here we compute from cow file)
    max_deploy_time = cow_max_deploy
    # attempt to see deployment_time_hr in power assignments if present -> not computed earlier
    try:
        if ASSIGN_POWER_GUROBI.exists():
            pa = pd.read_csv(ASSIGN_POWER_GUROBI)
            if "deployment_time_hr" in pa.columns:
                max_dep_power = pa["deployment_time_hr"].max(skipna=True)
                if not np.isnan(max_dep_power) and max_dep_power > max_deploy_time:
                    max_deploy_time = float(max_dep_power)
            else:
                # fallback to travel_time_hr if present
                if "travel_time_hr" in pa.columns:
                    max_travel_plus = float((pa["travel_time_hr"].fillna(0)).max())
                    if max_travel_plus > max_deploy_time:
                        max_deploy_time = max_travel_plus
    except Exception:
        pass

    # total cost
    total_deployment_cost = float(cow_total_cost) + float(power_total_cost)

    # Save combined visualization geojson (active, failed, outage, cow, power)
    features = []
    if active_union is not None:
        features.append({"type":"Feature","properties":{"layer":"active_union"},"geometry":mapping(active_union)})
    if bts_results.get("failed_union") is not None:
        features.append({"type":"Feature","properties":{"layer":"failed_union"},"geometry":mapping(bts_results.get("failed_union"))})
    if outage_geom is not None:
        features.append({"type":"Feature","properties":{"layer":"outage_union"},"geometry":mapping(outage_geom)})
    if cow_union is not None:
        features.append({"type":"Feature","properties":{"layer":"cow_union"},"geometry":mapping(cow_union)})
    if power_union is not None:
        features.append({"type":"Feature","properties":{"layer":"power_union"},"geometry":mapping(power_union)})
    if combined_union is not None:
        features.append({"type":"Feature","properties":{"layer":"combined_restored_union"},"geometry":mapping(combined_union)})

    with open(SUMMARY_DIR / f"coverage_layers_{method.lower()}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":features}, f, ensure_ascii=False, indent=2)

    # Compose final summary
    summary = {
        "method": method,
        "total_population": float(total_pop),
        "population_covered_by_active_bts": float(active_union_pop),
        "population_outage_due_failed_bts": float(outage_pop),
        "lost_coverage_percent": float(lost_coverage_percent),
        "failed_union_population": float(failed_union_pop),
        "population_restored_by_cows": float(cow_union_pop_in_outage),
        "population_restored_by_power": float(power_union_pop_in_outage),
        "population_restored_total": float(restored_total_in_outage),
        "coverage_restored_percent_of_outage": float(coverage_restored_percent_of_outage),
        "coverage_after_restoration_percent": float(coverage_after_restoration_percent),
        "cow_union_pop_total": float(cow_union_pop_total),
        "power_union_pop_total": float(power_union_pop_total),
        "max_deploy_time_hr": float(max_deploy_time),
        "total_deployment_cost_vnd": float(total_deployment_cost),
        "cow_count_used": int(cow_count),
        "power_units_used": int(power_count)
    }

    out_json = SUMMARY_DIR / f"coverage_report_{method.lower()}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote coverage summary: {out_json}")

    return summary

if __name__ == "__main__":
    res = main_compute_all(method="MILP_GUROBI")
    print("Done. Summary:", res)
