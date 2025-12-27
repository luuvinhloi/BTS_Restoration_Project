#!/usr/bin/env python3
"""
compute_population_coverage_hybrid.py

Compute population coverage summary using MILP_GA-PSO assignment outputs.
Inputs (MILP_GA-PSO):
 - outputs/results_hybrid/solution_cow_assignments.csv
 - outputs/results_hybrid/solution_power_assignments.csv

Output:
 - outputs/summary/milp_ga_pso_B/*.csv, *.geojson, coverage_report_hybrid.json

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

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------
# Paths (adjust if needed)
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
OUT_DIR = PROJECT_ROOT / "outputs"

# GA-PSO assignment results (expected)
ASSIGN_COW_HYBRID = OUT_DIR / "results_hybrid" / "solution_cow_assignments.csv"
ASSIGN_POWER_HYBRID = OUT_DIR / "results_hybrid" / "solution_power_assignments.csv"

# Inputs reused from MILP pipeline
POP_RASTER = CLEANED_DIR / "pop_hue_clean.tif"
FAILED_BTS = PROCESSED_DIR / "damage_bts" / "failed_bts.csv"
ACTIVE_BTS = PROCESSED_DIR / "damage_bts" / "active_bts.csv"
J_SITES = PROCESSED_DIR / "position_I_J" / "J_sites.csv"
COWS = PROCESSED_DIR / "cow" / "cow_dataset.csv"
BOUNDARY_GEOJSON = CLEANED_DIR / "hue_boundary_clean.geojson"

# Output folder for GA-PSO summary
SUMMARY_DIR = OUT_DIR / "summary" / "milp_ga_pso"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def read_raster_crs_and_nodata(raster_path: Path):
    with rasterio.open(raster_path) as src:
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        nodata = src.nodata
    return crs, nodata


def compute_total_population_from_raster(raster_path: Path):
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)
        total = float(arr.filled(0).sum())
    logging.info(f"Total population from raster: {total:,.0f}")
    return total


def buffer_point_meters(lon: float, lat: float, radius_m: float, target_crs="EPSG:4326"):
    """Buffer a lon/lat point by radius (meters)."""
    merc = CRS.from_epsg(3857)
    wgs = CRS.from_epsg(4326)
    t_to_merc = Transformer.from_crs(wgs, merc, always_xy=True)
    x_m, y_m = t_to_merc.transform(lon, lat)
    circle = Point(x_m, y_m).buffer(radius_m)
    geom = mapping(circle)
    return transform_geom("EPSG:3857", target_crs, geom, precision=6)


def compute_zonal_sum_geom(geom, raster_path: Path, nodata=None):
    """Return zonal sum (population) for a geometry on raster_path."""
    if geom is None:
        return 0.0
    try:
        stats = zonal_stats([geom], str(raster_path), stats="sum", nodata=nodata, all_touched=True)
        s = stats[0].get("sum", 0.0)
        return float(s if s else 0.0)
    except Exception as e:
        logging.error("zonal_stats failed: %s", e)
        return 0.0


def union_geoms_list(geoms):
    arr = []
    for g in geoms:
        if g is None:
            continue
        if isinstance(g, dict):
            arr.append(shape(g))
        else:
            arr.append(g)
    if not arr:
        return None
    u = unary_union(arr)
    return u if not u.is_empty else None


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# -------------------------
# Normalization helpers
# -------------------------
def normalize_j_site_id(s):
    """Convert 'J_00093' or '93' to uniform string '93'."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s.upper().startswith("J_"):
        s = s[2:]
    return s.lstrip("0") or "0"


def normalize_bts_id(s):
    """Convert 'BTS_03053' or '03053' to '3053' (string)."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s.upper().startswith("BTS_"):
        s = s[4:]
    return s.lstrip("0") or "0"


# -------------------------
# Per-site buffer routine (reused)
# -------------------------
def per_site_buffers_and_stats(df_sites, raster_crs, raster_path, nodata, boundary_geom):
    records = []
    geoms = []
    for _, row in df_sites.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        radius = float(row.get("coverage_radius_m", 3000.0))
        geom = buffer_point_meters(lon, lat, radius, raster_crs)
        shp = shape(geom)
        if boundary_geom is not None:
            shp = shp.intersection(boundary_geom)
            if shp.is_empty:
                records.append({
                    "site_id": row.get("site_id"),
                    "latitude": lat,
                    "longitude": lon,
                    "coverage_radius_m": radius,
                    "pop_in_buffer": 0.0
                })
                geoms.append(None)
                continue
        pop = compute_zonal_sum_geom(mapping(shp), raster_path, nodata)
        records.append({
            "site_id": row.get("site_id"),
            "latitude": lat,
            "longitude": lon,
            "coverage_radius_m": radius,
            "pop_in_buffer": pop
        })
        geoms.append(shp)
    return records, geoms


# -------------------------
# Active / Failed Unions (same logic)
# -------------------------
def compute_active_failed_unions_and_stats():
    raster_crs, nodata = read_raster_crs_and_nodata(POP_RASTER)
    boundary = gpd.read_file(BOUNDARY_GEOJSON)
    if boundary.crs:
        boundary = boundary.to_crs(raster_crs)
    boundary_geom = boundary.geometry.unary_union

    active_df = pd.read_csv(ACTIVE_BTS)
    failed_df = pd.read_csv(FAILED_BTS)

    active_records, active_geoms = per_site_buffers_and_stats(active_df, raster_crs, POP_RASTER, nodata, boundary_geom)
    failed_records, failed_geoms = per_site_buffers_and_stats(failed_df, raster_crs, POP_RASTER, nodata, boundary_geom)

    active_union = union_geoms_list(active_geoms)
    failed_union = union_geoms_list(failed_geoms)

    if active_union is not None:
        active_union = active_union.intersection(boundary_geom)
        if active_union.is_empty:
            active_union = None
    if failed_union is not None:
        failed_union = failed_union.intersection(boundary_geom)
        if failed_union.is_empty:
            failed_union = None

    active_union_pop = compute_zonal_sum_geom(mapping(active_union) if active_union else None, POP_RASTER, nodata)
    failed_union_pop = compute_zonal_sum_geom(mapping(failed_union) if failed_union else None, POP_RASTER, nodata)

    outage_geom = None
    if failed_union is not None:
        if active_union is not None:
            outage_geom = failed_union.difference(active_union)
        else:
            outage_geom = failed_union
        outage_geom = outage_geom.intersection(boundary_geom)
        if outage_geom.is_empty:
            outage_geom = None

    outage_pop = compute_zonal_sum_geom(mapping(outage_geom) if outage_geom else None, POP_RASTER, nodata)

    pd.DataFrame(active_records).to_csv(SUMMARY_DIR / "active_bts_per_site_pop.csv", index=False)
    pd.DataFrame(failed_records).to_csv(SUMMARY_DIR / "failed_bts_per_site_pop.csv", index=False)

    feats = []
    if active_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "active_union"}, "geometry": mapping(active_union)})
    if failed_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "failed_union"}, "geometry": mapping(failed_union)})
    if outage_geom is not None:
        feats.append({"type": "Feature", "properties": {"layer": "outage_union"}, "geometry": mapping(outage_geom)})

    with open(SUMMARY_DIR / "bts_union_layers.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

    return {
        "active_union": active_union,
        "failed_union": failed_union,
        "outage_geom": outage_geom,
        "active_union_pop": active_union_pop,
        "failed_union_pop": failed_union_pop,
        "outage_pop": outage_pop
    }


# -------------------------
# Compute COW coverage from hybrid assignments
# -------------------------
def compute_cow_coverage_from_hybrid(assign_path: Path, j_sites_path: Path, cows_table_path: Path,
                                    method_tag="milp_ga_pso_B", outage_geom=None):
    raster_crs, nodata = read_raster_crs_and_nodata(POP_RASTER)
    boundary = gpd.read_file(BOUNDARY_GEOJSON)
    if boundary.crs:
        boundary = boundary.to_crs(raster_crs)
    boundary_geom = boundary.geometry.unary_union

    if not assign_path.exists():
        logging.warning("COW assignment file not found: %s", assign_path)
        return {
            "cow_union": None,
            "cow_union_pop": 0.0,
            "cow_union_pop_in_outage": 0.0,
            "cow_records": [],
            "cow_geoms": [],
            "max_deploy_time": 0.0,
            "total_cost": 0.0,
            "cow_count": 0
        }

    assign = pd.read_csv(assign_path, dtype=str)
    # keep numeric columns too (distance, travel_time, travel_cost) - try to read numeric if present
    assign_num = pd.read_csv(assign_path) if any(c in ("distance_km", "travel_time_hr", "travel_cost_vnd") for c in pd.read_csv(assign_path, nrows=0).columns) else None

    j_sites = pd.read_csv(j_sites_path, dtype=str)
    # ensure lat/lon numeric in j_sites
    if "latitude" in j_sites.columns and "longitude" in j_sites.columns:
        j_sites["latitude"] = j_sites["latitude"].astype(float)
        j_sites["longitude"] = j_sites["longitude"].astype(float)
    else:
        raise RuntimeError("J_sites.csv must contain 'latitude' and 'longitude' columns.")

    cows_table = pd.read_csv(cows_table_path) if cows_table_path.exists() else pd.DataFrame()

    cow_records = []
    cow_geoms = []
    max_deploy = 0.0
    total_cost = 0.0

    # We'll iterate the rows; support both string and numeric parsing
    for idx, row in assign.iterrows():
        try:
            cow_id = row.get("cow_id") or row.get("id") or row.get("COW_ID")
            raw_site = row.get("site_id") or row.get("site") or row.get("assigned_site") or row.get("J_site")

            if pd.isna(raw_site) or raw_site == "":
                # skip unassigned
                continue

            site_id = normalize_j_site_id(raw_site)

            # locate J site coordinates
            # J_sites might have site_id like '93' or 'J_00093' - try both matches
            js = j_sites[(j_sites["site_id"].astype(str) == str(site_id)) | (j_sites["site_id"].astype(str) == str(raw_site))]
            if js.empty:
                logging.debug("COW site not found in J_sites: %s (row %s)", raw_site, idx)
                continue

            lat = float(js.iloc[0]["latitude"])
            lon = float(js.iloc[0]["longitude"])

            # radius: from cows table if available, else fallback to 3000
            radius = 3000.0
            if not cows_table.empty and cow_id is not None:
                try:
                    ct = cows_table[cows_table["cow_id"].astype(str) == str(cow_id)]
                    if not ct.empty and "coverage_radius_m" in ct.columns:
                        radius = float(ct.iloc[0]["coverage_radius_m"])
                except Exception:
                    pass

            # # deployment/travel/cost fields (try numeric file)
            # travel_cost = 0.0
            # travel_time = 0.0
            # if "travel_cost_vnd" in row.index:
            #     travel_cost = safe_float(row.get("travel_cost_vnd"), 0.0)
            # else:
            #     # try numeric file if loaded
            #     if assign_num is not None and "travel_cost_vnd" in assign_num.columns:
            #         travel_cost = safe_float(assign_num.loc[idx, "travel_cost_vnd"], 0.0)
            #
            # if "travel_time_hr" in row.index:
            #     travel_time = safe_float(row.get("travel_time_hr"), 0.0)
            # else:
            #     if assign_num is not None and "travel_time_hr" in assign_num.columns:
            #         travel_time = safe_float(assign_num.loc[idx, "travel_time_hr"], 0.0)
            #
            # # base cost (per-COW) from cows_table if available
            # base_cost = 0.0
            # if not cows_table.empty and cow_id is not None:
            #     try:
            #         ct = cows_table[cows_table["cow_id"].astype(str) == str(cow_id)]
            #         if not ct.empty and "cost_vnd" in ct.columns:
            #             base_cost = safe_float(ct.iloc[0]["cost_vnd"], 0.0)
            #     except Exception:
            #         pass
            #
            # total_cost += (travel_cost + base_cost)
            # max_deploy = max(max_deploy, travel_time)

            # === COST & TIME: READ DIRECTLY FROM HYBRID OUTPUT ===
            deployment_time = safe_float(row.get("total_time_hr"), 0.0)
            total_cost_vnd = safe_float(row.get("total_cost_vnd"), 0.0)

            total_cost += total_cost_vnd
            max_deploy = max(max_deploy, deployment_time)

            geom = buffer_point_meters(lon, lat, radius, raster_crs)
            shap = shape(geom).intersection(boundary_geom)

            if shap.is_empty:
                pop_total = 0.0
                pop_in_outage = 0.0
                cow_geoms.append(None)
            else:
                pop_total = compute_zonal_sum_geom(mapping(shap), POP_RASTER, nodata)
                if outage_geom is not None:
                    inter = shap.intersection(outage_geom)
                    if inter is None or inter.is_empty:
                        pop_in_outage = 0.0
                    else:
                        pop_in_outage = compute_zonal_sum_geom(mapping(inter), POP_RASTER, nodata)
                else:
                    pop_in_outage = 0.0
                cow_geoms.append(shap)

            cow_records.append({
                "cow_id": cow_id,
                "site_id": site_id,
                "latitude": lat,
                "longitude": lon,
                "coverage_radius_m": radius,
                "pop_in_buffer_total": pop_total,
                "pop_in_buffer_in_outage": pop_in_outage,
                "total_cost_vnd": total_cost_vnd,
                "deployment_time_hr": deployment_time
            })

        except Exception as e:
            logging.error("Error processing COW row %s: %s", idx, e)
            continue

    cow_union = union_geoms_list(cow_geoms)
    if cow_union is not None:
        cow_union = cow_union.buffer(0)

    cow_union_pop = compute_zonal_sum_geom(mapping(cow_union) if cow_union else None, POP_RASTER, nodata)
    cow_union_pop_in_outage = 0.0
    if outage_geom is not None and cow_union is not None:
        inter = cow_union.intersection(outage_geom)
        if inter is not None and not inter.is_empty:
            cow_union_pop_in_outage = compute_zonal_sum_geom(mapping(inter), POP_RASTER, nodata)

    pd.DataFrame(cow_records).to_csv(SUMMARY_DIR / f"cow_per_site_pop_{method_tag}.csv", index=False)

    feats = []
    if cow_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "cow_union"}, "geometry": mapping(cow_union)})
    with open(SUMMARY_DIR / f"cow_union_{method_tag}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

    cow_count = len(pd.Series([r.get("cow_id") for r in cow_records if r.get("cow_id") is not None]).unique())

    return {
        "cow_union": cow_union,
        "cow_union_pop": cow_union_pop,
        "cow_union_pop_in_outage": cow_union_pop_in_outage,
        "cow_records": cow_records,
        "cow_geoms": cow_geoms,
        "max_deploy_time": max_deploy,
        "total_cost": total_cost,
        "cow_count": cow_count
    }


# -------------------------
# Compute POWER coverage from HYBRID assignments
# -------------------------
def compute_power_coverage_from_hybrid(assign_path: Path, failed_bts_path: Path,
                                      method_tag="milp_ga_pso_B", outage_geom=None):
    raster_crs, nodata = read_raster_crs_and_nodata(POP_RASTER)
    boundary = gpd.read_file(BOUNDARY_GEOJSON)
    if boundary.crs:
        boundary = boundary.to_crs(raster_crs)
    boundary_geom = boundary.geometry.unary_union

    if not assign_path.exists():
        logging.warning("POWER assignment file not found: %s", assign_path)
        return {
            "power_union": None,
            "power_union_pop": 0.0,
            "power_union_pop_in_outage": 0.0,
            "power_records": [],
            "power_geoms": [],
            "total_cost": 0.0,
            "power_count": 0
        }

    assign = pd.read_csv(assign_path, dtype=str)
    # attempt to read numeric columns separately if present
    try:
        assign_num = pd.read_csv(assign_path)
    except Exception:
        assign_num = None

    failed_df = pd.read_csv(failed_bts_path, dtype=str)
    # ensure numeric lat/lon + radius in failed_df
    if "latitude" in failed_df.columns and "longitude" in failed_df.columns:
        failed_df["latitude"] = failed_df["latitude"].astype(float)
        failed_df["longitude"] = failed_df["longitude"].astype(float)
    # coverage_radius may or may not be present; handle safe_float

    power_records = []
    power_geoms = []
    total_cost = 0.0

    for idx, row in assign.iterrows():
        try:
            raw_bts = row.get("bts_id") or row.get("BTS_ID") or row.get("site_id")
            power_id = row.get("power_id") or row.get("power")

            if pd.isna(raw_bts):
                continue
            bts_id = normalize_bts_id(raw_bts)

            # find BTS in failed_df by site_id or bts_id (try multiple columns)
            bf = failed_df[(failed_df.get("site_id", failed_df.get("site_id", pd.Series())).astype(str) == str(bts_id))] \
                 if "site_id" in failed_df.columns else pd.DataFrame()

            if (bf.empty) and ("bts_id" in failed_df.columns):
                bf = failed_df[failed_df["bts_id"].astype(str) == str(bts_id)]

            if bf.empty:
                # try matching raw forms (like 'BTS_03053')
                bf = failed_df[failed_df.apply(lambda r: str(r.astype(str)).find(str(raw_bts)) >= 0, axis=1)]
                if bf.empty:
                    logging.debug("POWER bts not found in failed_bts: %s (row %s)", raw_bts, idx)
                    continue

            lat = float(bf.iloc[0]["latitude"])
            lon = float(bf.iloc[0]["longitude"])
            radius = safe_float(bf.iloc[0].get("coverage_radius_m"), 3000.0)

            # cost: GA-PSO uses 'total_cost_vnd' or 'total_cost' commonly
            # cost = 0.0
            # if "total_cost_vnd" in row.index:
            #     cost = safe_float(row.get("total_cost_vnd"), 0.0)
            # elif assign_num is not None and "total_cost_vnd" in assign_num.columns:
            #     cost = safe_float(assign_num.loc[idx, "total_cost_vnd"], 0.0)
            # elif "total_cost" in row.index:
            #     cost = safe_float(row.get("total_cost"), 0.0)
            #
            # total_cost += cost

            # === COST & TIME: READ DIRECTLY FROM HYBRID OUTPUT ===
            deployment_time = safe_float(row.get("total_time_hr"), 0.0)
            total_cost_vnd = safe_float(row.get("total_cost_vnd"), 0.0)

            total_cost += total_cost_vnd

            geom = buffer_point_meters(lon, lat, radius, raster_crs)
            shp = shape(geom).intersection(boundary_geom)

            if shp.is_empty:
                pop_total = 0.0
                pop_in_outage = 0.0
                power_geoms.append(None)
            else:
                pop_total = compute_zonal_sum_geom(mapping(shp), POP_RASTER, nodata)
                if outage_geom is not None:
                    inter = shp.intersection(outage_geom)
                    if inter is None or inter.is_empty:
                        pop_in_outage = 0.0
                    else:
                        pop_in_outage = compute_zonal_sum_geom(mapping(inter), POP_RASTER, nodata)
                else:
                    pop_in_outage = 0.0
                power_geoms.append(shp)

            power_records.append({
                "power_id": power_id,
                "bts_id": bts_id,
                "latitude": lat,
                "longitude": lon,
                "coverage_radius_m": radius,
                "pop_in_buffer_total": pop_total,
                "pop_in_buffer_in_outage": pop_in_outage,
                "total_cost_vnd": total_cost_vnd,
                "deployment_time_hr": deployment_time
            })

        except Exception as e:
            logging.error("Error processing POWER row %s: %s", idx, e)
            continue

    power_union = union_geoms_list(power_geoms)
    if power_union is not None:
        power_union = power_union.buffer(0)

    power_union_pop = compute_zonal_sum_geom(mapping(power_union) if power_union else None, POP_RASTER, nodata)
    power_union_pop_in_outage = 0.0
    if outage_geom is not None and power_union is not None:
        inter = power_union.intersection(outage_geom)
        if inter is not None and not inter.is_empty:
            power_union_pop_in_outage = compute_zonal_sum_geom(mapping(inter), POP_RASTER, nodata)

    pd.DataFrame(power_records).to_csv(SUMMARY_DIR / f"power_per_site_pop_{method_tag}.csv", index=False)

    feats = []
    if power_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "power_union"}, "geometry": mapping(power_union)})
    with open(SUMMARY_DIR / f"power_union_{method_tag}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

    power_count = len(pd.Series([r.get("power_id") for r in power_records if r.get("power_id") is not None]).unique())

    return {
        "power_union": power_union,
        "power_union_pop": power_union_pop,
        "power_union_pop_in_outage": power_union_pop_in_outage,
        "power_records": power_records,
        "power_geoms": power_geoms,
        "total_cost": total_cost,
        "power_count": power_count
    }


# -------------------------
# Main orchestration
# -------------------------
def main_compute_all(method="MILP_GA_PSO"):
    method_key = str(method).lower()
    method_summary_dir = OUT_DIR / "summary_new" / method_key
    method_summary_dir.mkdir(parents=True, exist_ok=True)

    global SUMMARY_DIR
    SUMMARY_DIR = method_summary_dir

    total_pop = compute_total_population_from_raster(POP_RASTER)

    bts_results = compute_active_failed_unions_and_stats()
    active_union_pop = bts_results.get("active_union_pop", 0.0)
    failed_union_pop = bts_results.get("failed_union_pop", 0.0)
    outage_pop = bts_results.get("outage_pop", 0.0)
    outage_geom = bts_results.get("outage_geom", None)
    active_union = bts_results.get("active_union", None)

    lost_coverage_percent = round(outage_pop / total_pop * 100, 2) if total_pop > 0 else 0.0

    # compute GA-PSO results
    cow_results = compute_cow_coverage_from_hybrid(ASSIGN_COW_HYBRID, J_SITES, COWS, method_tag="milp_ga_pso_B", outage_geom=outage_geom)
    power_results = compute_power_coverage_from_hybrid(ASSIGN_POWER_HYBRID, FAILED_BTS, method_tag="milp_ga_pso_B", outage_geom=outage_geom)

    cow_union_pop_in_outage = cow_results.get("cow_union_pop_in_outage", 0.0)
    cow_union_pop_total = cow_results.get("cow_union_pop", 0.0)
    cow_union = cow_results.get("cow_union")
    cow_max_deploy = cow_results.get("max_deploy_time", 0.0)
    cow_total_cost = cow_results.get("total_cost", 0.0)
    cow_count = cow_results.get("cow_count", 0)

    power_union_pop_in_outage = power_results.get("power_union_pop_in_outage", 0.0)
    power_union_pop_total = power_results.get("power_union_pop", 0.0)
    power_union = power_results.get("power_union")
    power_total_cost = power_results.get("total_cost", 0.0)
    power_count = power_results.get("power_count", 0)

    # Combined union
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

    coverage_after_restoration_percent = round((active_union_pop + restored_total_in_outage) / total_pop * 100, 2) if total_pop > 0 else 0.0
    coverage_restored_percent_of_outage = round((restored_total_in_outage / outage_pop * 100) if outage_pop > 0 else 0.0, 2)

    # max deployment time (HYBRID): makespan from solution files
    max_deploy_time = cow_max_deploy

    if ASSIGN_POWER_HYBRID.exists():
        pa = pd.read_csv(ASSIGN_POWER_HYBRID)
        if "total_time_hr" in pa.columns:
            max_power_time = pa["total_time_hr"].astype(float).max(skipna=True)
            if not np.isnan(max_power_time):
                max_deploy_time = max(max_deploy_time, float(max_power_time))

    total_deployment_cost = float(cow_total_cost) + float(power_total_cost)

    # Save combined visualization geojson
    feats = []
    if active_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "active_union"}, "geometry": mapping(active_union)})
    if bts_results.get("failed_union") is not None:
        feats.append({"type": "Feature", "properties": {"layer": "failed_union"}, "geometry": mapping(bts_results.get("failed_union"))})
    if outage_geom is not None:
        feats.append({"type": "Feature", "properties": {"layer": "outage_union"}, "geometry": mapping(outage_geom)})
    if cow_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "cow_union"}, "geometry": mapping(cow_union)})
    if power_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "power_union"}, "geometry": mapping(power_union)})
    if combined_union is not None:
        feats.append({"type": "Feature", "properties": {"layer": "combined_restored_union"}, "geometry": mapping(combined_union)})

    with open(SUMMARY_DIR / f"coverage_layers_{method_key}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)

    # summary
    summary = {
        "method": method,
        "total_population": float(total_pop),
        "population_covered_by_active_bts": float(active_union_pop),
        "population_outage_due_failed_bts": float(outage_pop),
        "lost_coverage_percent": float(lost_coverage_percent),
        "coverage_after_restoration_percent": float(coverage_after_restoration_percent),
        "coverage_restored_percent_of_outage": float(coverage_restored_percent_of_outage),
        "failed_union_population": float(failed_union_pop),
        "population_restored_by_cows": float(cow_union_pop_in_outage),
        "population_restored_by_power": float(power_union_pop_in_outage),
        "population_restored_total": float(restored_total_in_outage),
        "cow_union_pop_total": float(cow_union_pop_total),
        "power_union_pop_total": float(power_union_pop_total),
        "max_deploy_time_hr": float(max_deploy_time),
        "total_deployment_cost_vnd": float(total_deployment_cost),
        "cow_count_used": int(cow_count),
        "power_units_used": int(power_count)
    }

    out_json = SUMMARY_DIR / f"coverage_report_{method_key}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Wrote coverage summary: %s", out_json)
    return summary


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute population coverage summary for GA-PSO")
    parser.add_argument("--method", type=str, default="MILP_GA_PSO", help="Method key (MILP_GA_PSO)")
    args = parser.parse_args()

    res = main_compute_all(method=args.method)
    print("Done. Summary:", json.dumps(res, indent=2, ensure_ascii=False))
