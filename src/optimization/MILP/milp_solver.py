"""
milp_solver.py
MILP lexicographic solver for BTS restoration (COW deployment + backup power allocation).

Place under:
BTS_Restoration_Project/src/optimization/MILP/milp_solver.py

Usage: call main_solve(...) at bottom or import run_lexicographic_for_solver(...)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pulp
import networkx as nx
import rasterio
from shapely.geometry import Point
from math import radians, sin, cos, asin, sqrt
import json
import time

# Utilities
def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def read_csv_safe(path, **kwargs):
    return pd.read_csv(path, dtype=str, **kwargs).fillna("")

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# Data loaders
def load_all_data(processed_dir: str):
    p = Path(processed_dir)
    # files (as described in prompt)
    j_sites = pd.read_csv(p / "position_I_J" / "J_sites.csv")
    cows = pd.read_csv(p / "cow" / "cow_dataset.csv")
    backup_power = pd.read_csv(p / "backup_power" / "backup_power.csv")
    failed_bts = pd.read_csv(p / "damage_bts" / "failed_bts.csv")
    cow_travel = pd.read_csv(p / "travel_cost" / "cow_to_J_sites.csv")
    power_travel = pd.read_csv(p / "travel_cost" / "backup_to_failed_bts.csv")
    roads_graphml = p / "road" / "roads_flooded.graphml"
    flood_raster = p / "flood" / "flood_depth_combined_clean.tif"

    # numeric coercions
    for df, cols in [
        (j_sites, ["latitude", "longitude", "pop", "priority_weight"]),
        (cows, ["lat", "lon", "coverage_radius_m", "power_kw", "speed_kmh", "endurance_hr", "cost_vnd"]),
        (backup_power, ["lat", "lon", "runtime_h", "cost_vnd_24h", "resource_amount"]),
        (failed_bts, ["latitude", "longitude", "coverage_radius_m", "power_W", "pop_covered"])
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # travel costs numeric
    for df, cols in [(cow_travel, ["distance_km", "travel_time_hr", "travel_cost_vnd"]),
                     (power_travel, ["distance_km", "total_time_hr", "total_cost_vnd"])]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # load graph (if exists)
    G = None
    if roads_graphml.exists():
        try:
            G = nx.read_graphml(str(roads_graphml))
        except Exception as e:
            print(f"[WARN] Failed to read graphml {roads_graphml}: {e}")
            G = None

    # load raster dataset handle
    flood_ds = None
    if flood_raster.exists():
        try:
            flood_ds = rasterio.open(str(flood_raster))
        except Exception as e:
            print(f"[WARN] Failed to open flood raster {flood_raster}: {e}")
            flood_ds = None

    return {
        "j_sites": j_sites,
        "cows": cows,
        "backup_power": backup_power,
        "failed_bts": failed_bts,
        "cow_travel": cow_travel,
        "power_travel": power_travel,
        "roads_graph": G,
        "flood_ds": flood_ds
    }

# Preprocessing: feasible assignments + cover matrices
def build_cover_and_travel_maps(data, params):
    """
    Build:
      - cow_cover[(demand_j, deploy_j, cow_id)] -> 0/1  (we use demand==site set here)
      - cow_travel_map[(cow_id, site_id)] -> travel dict (uses cow_to_J_sites.csv)
      - power_travel_map[(power_id, bts_id)] -> travel dict (uses backup_to_failed_bts.csv)
      - feasible flags: exclude J that are flooded > threshold (0.5 m) or in_water==True
    """
    j_sites = data["j_sites"]
    cows = data["cows"]
    backup_power = data["backup_power"]
    failed_bts = data["failed_bts"]
    cow_travel = data["cow_travel"]
    power_travel = data["power_travel"]
    G = data["roads_graph"]
    flood_ds = data["flood_ds"]

    flood_threshold = float(params.get("flood_deploy_threshold_m", 0.5))
    find_neighbour_search_m = float(params.get("neighbour_search_m", 500.0))

    # Prepare dictionaries
    cow_travel_map = {}
    for _, row in cow_travel.iterrows():
        cow_travel_map[(str(row["cow_id"]), str(row["site_id"]))] = {
            "distance_km": to_float(row.get("distance_km", 0.0)),
            "travel_time_hr": to_float(row.get("travel_time_hr", 0.0)),
            "travel_cost_vnd": to_float(row.get("travel_cost_vnd", 0.0))
        }

    power_travel_map = {}
    for _, row in power_travel.iterrows():
        power_travel_map[(str(row["power_id"]), str(row["bts_id"]))] = {
            "distance_km": to_float(row.get("distance_km", 0.0)),
            "total_time_hr": to_float(row.get("total_time_hr", 0.0)),
            "travel_cost_vnd": to_float(row.get("travel_cost_vnd", 0.0)),
            "note": row.get("note", "")
        }

    # feasible J: check flood raster and 'in_water' column
    j_sites = j_sites.copy()
    j_sites["feasible_deploy"] = True
    if flood_ds is not None:
        # for each J compute flood depth
        coords = [(row["longitude"], row["latitude"]) for _, row in j_sites.iterrows()]
        # rasterio sample expects (x,y) lon/lat if raster in same crs; assume rasters and coords match
        try:
            samp = list(flood_ds.sample(coords))
            depths = [s[0] if len(s) > 0 else 0.0 for s in samp]
        except Exception:
            depths = [0.0]*len(coords)
        j_sites["flood_depth_m"] = depths
        # mark infeasible if depth > threshold or in_water True
        if "in_water" in j_sites.columns:
            j_sites["in_water_bool"] = j_sites["in_water"].astype(bool)
        else:
            j_sites["in_water_bool"] = False
        j_sites.loc[(j_sites["flood_depth_m"] > flood_threshold) | (j_sites["in_water_bool"]), "feasible_deploy"] = False
    else:
        # no raster: only use in_water flag
        if "in_water" in j_sites.columns:
            j_sites["in_water_bool"] = j_sites["in_water"].astype(bool)
            j_sites.loc[j_sites["in_water_bool"], "feasible_deploy"] = False
        j_sites["flood_depth_m"] = 0.0

    # For J infeasible, try to find nearest feasible J within radius find_neighbour_search_m
    # build mapping j -> alternative_j if available
    j_alternative = {}
    for idx, row in j_sites.iterrows():
        if row["feasible_deploy"]:
            j_alternative[row["site_id"]] = row["site_id"]
            continue
        # search nearest feasible point
        lat0 = row["latitude"]
        lon0 = row["longitude"]
        found = None
        # brute-force search among feasible sites
        for idx2, row2 in j_sites[j_sites["feasible_deploy"]].iterrows():
            dist = haversine_km(lat0, lon0, row2["latitude"], row2["longitude"]) * 1000.0
            if dist <= find_neighbour_search_m:
                found = row2["site_id"]
                break
        if found is None:
            j_alternative[row["site_id"]] = None
        else:
            j_alternative[row["site_id"]] = found

    # build cover indicator: cow deployed at site j covers demand i if distance(site_i, site_j) <= cow.coverage_radius_m
    cover_indicator = {}
    # demand set = j_sites (I_points), deploy sites = j_sites (J_sites)
    # We will consider demands indexed by site_id
    demands = list(j_sites["site_id"].astype(str).values)
    deploy_sites = list(j_sites["site_id"].astype(str).values)
    cow_ids = list(cows["cow_id"].astype(str).values)

    # prepare lat/lon lookup
    site_lookup = {str(r["site_id"]): (to_float(r["latitude"]), to_float(r["longitude"])) for _, r in j_sites.iterrows()}
    cow_lookup = {str(r["cow_id"]): {"coverage_radius_m": to_float(r["coverage_radius_m"]), "base_id": r.get("base_id", "")} for _, r in cows.iterrows()}

    for i in demands:
        lat_i, lon_i = site_lookup[i]
        for j in deploy_sites:
            lat_j, lon_j = site_lookup[j]
            d_km = haversine_km(lat_i, lon_i, lat_j, lon_j)
            d_m = d_km * 1000.0
            for c in cow_ids:
                cov_r = cow_lookup[c]["coverage_radius_m"]
                cover_indicator[(i, j, c)] = 1 if d_m <= cov_r else 0

    # build compatibility matrix C(b_i, g_k): check power capacity >= bts.power_W
    power_ids = list(backup_power["power_id"].astype(str).values)
    bts_ids = list(failed_bts["site_id"].astype(str).values)
    power_lookup = {}
    for _, r in backup_power.iterrows():
        power_lookup[str(r["power_id"])] = {
            "type": r.get("type", "").upper(),
            "base_id": r.get("base_id", ""),
            "lat": to_float(r.get("lat", 0.0)),
            "lon": to_float(r.get("lon", 0.0)),
            "runtime_h": to_float(r.get("runtime_h", 0.0)),
            "cost_vnd_24h": to_float(r.get("cost_vnd_24h", 0.0)),
            # resource_amount left in lookup but will be ignored by MILP (PHƯƠNG ÁN 1)
            "resource_amount": to_float(r.get("resource_amount", 1.0))
        }

    bts_lookup = {}
    for _, r in failed_bts.iterrows():
        bts_lookup[str(r["site_id"])] = {
            "power_W": to_float(r.get("power_W", 0.0)),
            "latitude": to_float(r.get("latitude", 0.0)),
            "longitude": to_float(r.get("longitude", 0.0)),
            "status": r.get("status", "")
        }

    compatibility = {}
    for b in bts_ids:
        for g in power_ids:
            # assume GENSET produces e.g. 5-10kW; but in backup_power we don't have power rating column
            # We'll use simple rule: if power resource model contains '10KW' or resource_amount suggests capacity large, else use conservative:
            # But safer approach: check that power has runtime and assume it's sufficient; however user requested: "must have power >= consumption"
            # If backup_power lacks explicit power_kW, we will assume GENSET type meets most BTS loads if model contains '10KW' else reject.
            row = backup_power[backup_power["power_id"] == g]
            meets = False
            if "power_W" in bts_lookup[b]:
                b_power = bts_lookup[b]["power_W"]
            else:
                b_power = 0.0
            # try parse model
            try:
                model = str(backup_power.loc[backup_power["power_id"] == g, "model"].iloc[0])
            except Exception:
                model = ""
            model_upper = model.upper()
            # naive parse for KW in model string
            found_kw = None
            import re
            m = re.search(r"(\d{1,3})\s*KW", model_upper)
            if m:
                found_kw = float(m.group(1))*1000.0
            # allow if found_kw >= b_power OR if resource type BATTERY and runtime sufficient to meet energy need (we don't have E_req so skip)
            if found_kw is not None:
                meets = (found_kw >= b_power)
            else:
                # fallback: if type GENSET assume 10000 W if model mentions 10 or resource_amount large
                ptype = str(backup_power.loc[backup_power["power_id"] == g, "type"].iloc[0]).upper()
                if "GEN" in ptype or "GENSET" in ptype:
                    meets = True  # assume genset adequate; user data mentions 5-10kW gensets
                else:
                    # battery: assume 10kWh typical -> if BTS power small accept
                    meets = (b_power <= 10000)
            compatibility[(b, g)] = 1 if meets else 0

    # Done
    preproc = {
        "cover_indicator": cover_indicator,
        "cow_travel_map": cow_travel_map,
        "power_travel_map": power_travel_map,
        "j_sites": j_sites,
        "j_alternative": j_alternative,
        "compatibility": compatibility,
        "power_lookup": power_lookup,
        "bts_lookup": bts_lookup
    }
    return preproc

# MILP builder
def build_milp_problem(preproc, data, params):
    """
    Build PuLP MILP problem containing variables + common constraints for lexicographic steps.
    Returns (prob_base, var_dict)
    """
    j_sites = preproc["j_sites"]
    cows = data["cows"]
    backup_power = data["backup_power"]
    failed_bts = data["failed_bts"]

    # index sets
    cow_ids = list(cows["cow_id"].astype(str).values)
    j_ids = list(j_sites["site_id"].astype(str).values)   # deploy sites and demand sites
    bts_ids = list(failed_bts["site_id"].astype(str).values)
    power_ids = list(backup_power["power_id"].astype(str).values)

    budget_max = float(params.get("budget_max", 1e9))
    setup_time_h = float(params.get("default_setup_time_h", 0.5))
    M_max = int(params.get("M_max", len(cow_ids)))

    # problem
    prob = pulp.LpProblem("MILP_BTS_Restoration", pulp.LpMinimize)

    # decision variables
    # COW: x[cow, j] = 1 if cow deployed at J
    x = pulp.LpVariable.dicts("x", (cow_ids, j_ids), cat="Binary")

    # y[demand_i, j] = 1 if demand (I point) i served by deployment at J
    # demand set is j_ids (I_points ~ J_sites's i_ref could be different, but for simplicity we use j_ids as demand set)
    y = pulp.LpVariable.dicts("y", (j_ids, j_ids), cat="Binary")

    # z[g_id, bts_id] = 1 if power g assigned to bts
    z = pulp.LpVariable.dicts("z", (power_ids, bts_ids), cat="Binary")

    # u[bts] = 1 if BTS restored (has power assigned)
    u = pulp.LpVariable.dicts("u", bts_ids, cat="Binary")

    # w_bts_serve[bts, demand] = 1 if demand served by bts
    w_bts = pulp.LpVariable.dicts("w_bts", (bts_ids, j_ids), cat="Binary")

    # w_cow_serve[cow, demand_deploy_j] -> we already have y[demand, j] combining all cows via x; to follow doc we'll keep y as above and
    # also allow relation: y[demand,j] <= sum_k cover * x[k,j], we'll build that.
    # T_max
    T_max = pulp.LpVariable("T_max", lowBound=0, cat="Continuous")

    # Helper mappings
    cover = preproc["cover_indicator"]
    cow_travel_map = preproc["cow_travel_map"]
    power_travel_map = preproc["power_travel_map"]
    compatibility = preproc["compatibility"]
    j_alternative = preproc["j_alternative"]
    power_lookup = preproc["power_lookup"]
    bts_lookup = preproc["bts_lookup"]

    # COMMON CONSTRAINTS

    # 1) Each cow at most 1 deployment
    for k in cow_ids:
        prob += pulp.lpSum([x[k][j] for j in j_ids]) <= 1, f"one_deploy_per_cow_{k}"

    # 2) Each deploy site (J) at most 1 cow
    for j in j_ids:
        prob += pulp.lpSum([x[k][j] for k in cow_ids]) <= 1, f"one_cow_per_site_{j}"

    # 3) y (demand served by J) at most 1 (non-overlap)
    for i in j_ids:
        prob += pulp.lpSum([y[i][j] for j in j_ids]) <= 1, f"non_overlap_demand_{i}"

    # 4) y only if there exists deployed cow at j that can cover demand i:
    for i in j_ids:
        for j in j_ids:
            coverable_cows = [k for k in cow_ids if cover.get((i, j, k), 0) == 1]
            if len(coverable_cows) == 0:
                # force y=0
                prob += y[i][j] == 0, f"no_cover_possible_{i}_{j}"
            else:
                prob += y[i][j] <= pulp.lpSum([x[k][j] for k in coverable_cows]), f"y_implies_x_cover_{i}_{j}"

    # 5) z and u relation: u[b] <= sum_k z[g,b]*C(b,g)  (if any compatible power assigned then can be restored)
    for bi in bts_ids:
        prob += u[bi] <= pulp.lpSum([z[g][bi] * compatibility.get((bi, g), 0) for g in power_ids]), f"u_implies_z_{bi}"

    # 6) energy capacity: if z[g,b] then that g must meet energy constraints - we rely on precomputed travel and power specs;
    #    We don't have E_req per BTS; but we can ensure assigned power compatibility only via compatibility matrix.
    #    Also each BTS must receive at most one power unit:
    for bi in bts_ids:
        prob += pulp.lpSum([z[g][bi] for g in power_ids]) <= 1, f"one_power_per_bts_{bi}"

    # 7) Each power unit (power_id) is treated as a single physical device (PHƯƠNG ÁN 1):
    #    enforce that each power_id can be assigned to at most one BTS.
    #    (This intentionally **ignores** the 'resource_amount' column in CSV: each power_id is one device.)
    for g in power_ids:
        prob += pulp.lpSum([z[g][bi] for bi in bts_ids]) <= 1, f"one_bts_per_power_{g}"

    # 8) w_bts <= u and distance constraints: a demand can be served by bts only if bts restored and within its coverage radius
    # build coverage matrix from bts to demands
    # prepare bts location lookup and radius
    bts_loc = {}
    for bid, info in bts_lookup.items():
        bts_loc[bid] = (info["latitude"], info["longitude"], info.get("coverage_radius_m", 0.0))

    # but failed_bts dataframe had coverage_radius_m column - prefer that
    bts_df_map = {str(r["site_id"]): r for _, r in failed_bts.iterrows()}
    for bi in bts_ids:
        for j in j_ids:
            # compute distance between bts and demand j
            lat_b = to_float(bts_df_map[bi].get("latitude", 0.0))
            lon_b = to_float(bts_df_map[bi].get("longitude", 0.0))
            # fallback: get j position
            try:
                lat_j = to_float(preproc["j_sites"].set_index("site_id").loc[j]["latitude"])
                lon_j = to_float(preproc["j_sites"].set_index("site_id").loc[j]["longitude"])
            except Exception:
                lat_j = None
                lon_j = None
            # use coverage radius from bts_df_map
            try:
                r_cov = to_float(bts_df_map[bi].get("coverage_radius_m", 0.0))
            except Exception:
                r_cov = 0.0
            if lat_j is None or lat_b is None:
                # cannot compute -> forbid
                prob += w_bts[bi][j] == 0, f"bts_cover_unknown_{bi}_{j}"
            else:
                d_m = haversine_km(lat_b, lon_b, lat_j, lon_j) * 1000.0
                if d_m <= r_cov:
                    # can be served only if u_bts = 1
                    prob += w_bts[bi][j] <= u[bi], f"w_bts_implies_u_{bi}_{j}"
                else:
                    prob += w_bts[bi][j] == 0, f"bts_out_of_range_{bi}_{j}"

    # 9) each demand at most 1 served by bts + cow combined
    for j in j_ids:
        prob += pulp.lpSum([w_bts[bi][j] for bi in bts_ids]) + pulp.lpSum([y[j][jj] for jj in j_ids]) <= 1, f"single_service_{j}"

    # 10) Each cow-deploy used implies y for some demands; already enforced above.

    # 11) T_max constraints for cow deployments:
    for k in cow_ids:
        for j in j_ids:
            travel_time = to_float(preproc["cow_travel_map"].get((k, j), {}).get("travel_time_hr", 0.0))
            prob += T_max >= (travel_time + setup_time_h) * x[k][j], f"Tmax_def_cow_{k}_{j}"

    # 12) For power: we might define a T_max_power but per model only global T_max is used for COW; we'll also consider power travel times when compute final objectives
    # But we still may push T_max >= power travel time if assigned (to reflect overall deployment time)
    for g in power_ids:
        for bi in bts_ids:
            travel_t = to_float(preproc["power_travel_map"].get((g, bi), {}).get("total_time_hr", 0.0))
            prob += T_max >= travel_t * z[g][bi], f"Tmax_def_power_{g}_{bi}"

    for j in j_ids:
        if preproc["j_alternative"].get(j) is None:
            # không có điểm thay thế → cấm deploy
            prob += pulp.lpSum(x[k][j] for k in cow_ids) == 0, f"forbid_flooded_J_{j}"

    # 13) Budget constraint placeholder (objective uses costs). We'll add final cost expr externally.
    # Save var dict
    var_dict = {
        "x": x,
        "y": y,
        "z": z,
        "u": u,
        "w_bts": w_bts,
        "T_max": T_max,
        "cow_ids": cow_ids,
        "j_ids": j_ids,
        "bts_ids": bts_ids,
        "power_ids": power_ids
    }
    return prob, var_dict

def build_coverage_expression(y, w_bts, j_ids, bts_ids, pop_map):
    """
    Total covered population by COW + restored BTS
    """
    cov_cow = pulp.lpSum(
        pop_map[j] * pulp.lpSum(y[j][jj] for jj in j_ids)
        for j in j_ids
    )
    cov_bts = pulp.lpSum(
        pop_map[j] * pulp.lpSum(w_bts[bi][j] for bi in bts_ids)
        for j in j_ids
    )
    return cov_cow + cov_bts

# Lexicographic solve orchestration
def run_lexicographic_for_solver(data, preproc, params, solver_name="CBC", out_dir="./outputs/milp_runs"):
    """
    Runs lexicographic MILP as:
      Step1: maximize covered population (both bts restored coverage and cow coverage)
      Step2: minimize T_max subject to coverage_opt
      Step3: minimize total cost subject to coverage and T_max fixed.

    Uses travel matrices precomputed for costs/times.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_base, var_dict = build_milp_problem(preproc, data, params)

    # build cost & pop weights
    j_sites = preproc["j_sites"]
    cows = data["cows"]
    backup_power = data["backup_power"]
    failed_bts = data["failed_bts"]
    cow_travel_map = preproc["cow_travel_map"]
    power_travel_map = preproc["power_travel_map"]
    power_lookup = preproc["power_lookup"]
    budget_max = float(params.get("budget_max", 1e9))

    # demand pop mapping (use j_sites pop as population to be covered)
    pop_map = {str(r["site_id"]): to_float(r.get("pop", 0.0)) for _, r in j_sites.iterrows()}
    total_pop_on_I = sum(pop_map.values())

    priority_map = {
        str(r["site_id"]): to_float(r.get("priority_weight", 0.0))
        for _, r in j_sites.iterrows()
    }

    # convenience
    x = var_dict["x"]
    y = var_dict["y"]
    z = var_dict["z"]
    u = var_dict["u"]
    w_bts = var_dict["w_bts"]
    T_max = var_dict["T_max"]
    cow_ids = var_dict["cow_ids"]
    j_ids = var_dict["j_ids"]
    bts_ids = var_dict["bts_ids"]
    power_ids = var_dict["power_ids"]

    # solver objects
    time_limit = int(params.get("milp", {}).get("solver", {}).get("time_limit", 600))
    if solver_name.upper() == "GUROBI":
        solver = pulp.GUROBI_CMD(
            timeLimit=time_limit,
            msg=True
        )
    else:
        solver = pulp.PULP_CBC_CMD(
            msg=True,
            timeLimit=time_limit
        )

    def build_cow_priority_expression(x, cow_ids, j_ids, priority_map):
        return pulp.lpSum(
            priority_map[j] * pulp.lpSum(x[k][j] for k in cow_ids)
            for j in j_ids
        )

    # Step 1: maximize covered population
    prob1, var_dict1 = build_milp_problem(preproc, data, params)
    x = var_dict1["x"];
    y = var_dict1["y"]
    z = var_dict1["z"];
    w_bts = var_dict1["w_bts"]
    T_max = var_dict1["T_max"]
    prob1.sense = pulp.LpMaximize

    # covered population expression:
    # Population covered by cows: sum_j pop_j * (sum_i y[i,j] where i=j as demand)
    # Population covered by BTS: sum_demands pop_j * (sum_b w_bts[b,j])
    # cov_expr = pulp.lpSum([pop_map.get(j, 0.0) * pulp.lpSum([y[j][jj] for jj in j_ids]) for j in j_ids]) \
    #            + pulp.lpSum([pop_map.get(j, 0.0) * pulp.lpSum([w_bts[bi][j] for bi in bts_ids]) for j in j_ids])
    #
    # prob1.setObjective(cov_expr)

    cov_expr_1 = build_coverage_expression(y, w_bts, j_ids, bts_ids, pop_map)
    cow_priority_expr = build_cow_priority_expression(
        x, cow_ids, j_ids, priority_map
    )

    EPS_PRIORITY = 1e-4
    # EPS_PRIORITY is chosen sufficiently small so that
    # coverage objective always dominates cow priority.

    prob1.setObjective(
        cov_expr_1 + EPS_PRIORITY * cow_priority_expr
    )

    print("Solving Step 1: maximize covered population...")
    t0 = time.time()
    prob1.solve(solver)
    t1 = time.time()
    status1 = pulp.LpStatus[prob1.status]
    covered_pop = pulp.value(cov_expr_1)
    print(f" Step1 status={status1}, covered_pop={covered_pop:.2f}, time={t1-t0:.2f}s")

    if status1 not in ("Optimal", "Integer Feasible", "Feasible"):
        print("Step1 failed - aborting.")
        return {"solver": solver_name, "status_step1": status1}

    optimal_covered_pop = covered_pop

    # Step 2: minimize T_max subject to covered_pop >= optimal_covered_pop
    # prob2 = prob_base.copy()
    # prob2 += cov_expr >= optimal_covered_pop, "fix_covered_pop"
    prob2, var_dict2 = build_milp_problem(preproc, data, params)
    x = var_dict2["x"];
    y = var_dict2["y"]
    z = var_dict2["z"];
    w_bts = var_dict2["w_bts"]
    T_max = var_dict2["T_max"]
    cov_expr_2 = build_coverage_expression(y, w_bts, j_ids, bts_ids, pop_map)
    EPS = 1e-6
    prob2 += cov_expr_2 >= optimal_covered_pop - EPS, "fix_covered_pop"
    prob2.setObjective(T_max)
    prob2.sense = pulp.LpMinimize

    print("Solving Step 2: minimize T_max with coverage fixed...")
    t0 = time.time()
    prob2.solve(solver)
    t1 = time.time()
    status2 = pulp.LpStatus[prob2.status]
    T_max_val = pulp.value(T_max)
    print(f" Step2 status={status2}, T_max={T_max_val}, time={t1-t0:.2f}s")

    if status2 not in ("Optimal", "Integer Feasible", "Feasible"):
        print("Step2 failed - aborting.")
        return {"solver": solver_name, "status_step1": status1, "status_step2": status2}

    optimal_T_max = T_max_val

    # Step 3: minimize total cost subject to coverage and T_max fixed
    # prob3 = prob_base.copy()
    # prob3 += cov_expr >= optimal_covered_pop, "fix_covered_pop"
    prob3, var_dict3 = build_milp_problem(preproc, data, params)
    x = var_dict3["x"];
    y = var_dict3["y"]
    z = var_dict3["z"];
    w_bts = var_dict3["w_bts"]
    T_max = var_dict3["T_max"]
    cov_expr_3 = build_coverage_expression(y, w_bts, j_ids, bts_ids, pop_map)
    EPS = 1e-6
    prob3 += cov_expr_3 >= optimal_covered_pop - EPS, "fix_covered_pop"
    prob3 += T_max <= optimal_T_max + 1e-9, "fix_T_max"

    # total cost: sum cow fixed cost + cow travel cost + power travel cost + power op cost (cost_vnd_24h)
    total_cost_terms = []
    # cows fixed cost from cows_df
    for _, crow in cows.iterrows():
        k = str(crow["cow_id"])
        cow_cost = to_float(crow.get("cost_vnd", 0.0))
        for j in j_ids:
            travel_cost = to_float(cow_travel_map.get((k, j), {}).get("travel_cost_vnd", 0.0))
            total_cost_terms.append((cow_cost + travel_cost) * x[k][j])

    # power assignment costs
    for g in power_ids:
        power_operating_cost = to_float(
            power_lookup.get(g, {}).get("cost_vnd_24h", 0.0)
        )
        for bi in bts_ids:
            power_travel_cost = to_float(
                power_travel_map.get((g, bi), {}).get("travel_cost_vnd", 0.0)
            )

            # Tổng chi phí triển khai nguồn điện dự phòng
            power_total_cost = power_travel_cost + power_operating_cost

            total_cost_terms.append(power_total_cost * z[g][bi])

    # Budget constraint
    budget_cost_terms = []

    # COW costs
    for _, crow in cows.iterrows():
        k = str(crow["cow_id"])
        cow_cost = to_float(crow.get("cost_vnd", 0.0))
        for j in j_ids:
            travel_cost = to_float(
                cow_travel_map.get((k, j), {}).get("travel_cost_vnd", 0.0)
            )
            budget_cost_terms.append((cow_cost + travel_cost) * x[k][j])

    # Power costs
    for g in power_ids:
        power_operating_cost = to_float(
            power_lookup.get(g, {}).get("cost_vnd_24h", 0.0)
        )
        for bi in bts_ids:
            power_travel_cost = to_float(
                power_travel_map.get((g, bi), {}).get("travel_cost_vnd", 0.0)
            )
            budget_cost_terms.append(
                (power_operating_cost + power_travel_cost) * z[g][bi]
            )

    # Add budget constraint
    prob3 += pulp.lpSum(budget_cost_terms) <= budget_max, "Budget_Constraint"

    prob3.setObjective(pulp.lpSum(total_cost_terms))
    prob3.sense = pulp.LpMinimize

    print("Solving Step 3: minimize total cost with coverage and T_max fixed...")
    t0 = time.time()
    prob3.solve(solver)
    t1 = time.time()
    status3 = pulp.LpStatus[prob3.status]
    print(f" Step3 status={status3}, time={t1-t0:.2f}s")

    # Extract solution from prob3
    assignments_cow = []
    for k in cow_ids:
        for j in j_ids:
            val = None
            try:
                val = pulp.value(prob3.variablesDict()[f"x_{k}_{j}"])
            except KeyError:
                try:
                    val = pulp.value(prob3.variablesDict()[f"x_{k}__{j}"])
                except Exception:
                    val = None
            if val is not None and round(val) == 1:
                assignments_cow.append((k, j))

    assignments_power = []
    for g in power_ids:
        for bi in bts_ids:
            val = None
            try:
                val = pulp.value(prob3.variablesDict()[f"z_{g}_{bi}"])
            except KeyError:
                try:
                    val = pulp.value(prob3.variablesDict()[f"z_{g}__{bi}"])
                except Exception:
                    val = None
            if val is not None and round(val) == 1:
                assignments_power.append((g, bi))

    # served demands mapping
    demand_served = {}
    total_pop_served = 0.0
    for i in j_ids:
        served = False
        # check cow coverage y
        for j in j_ids:
            try:
                val = pulp.value(prob3.variablesDict()[f"y_{i}_{j}"])
            except KeyError:
                try:
                    val = pulp.value(prob3.variablesDict()[f"y_{i}__{j}"])
                except Exception:
                    val = None
            if val is not None and round(val) == 1:
                demand_served[i] = ("COW", j)
                total_pop_served += pop_map.get(i, 0.0)
                served = True
                break
        if served: continue
        # check bts coverage
        for bi in bts_ids:
            try:
                val = pulp.value(prob3.variablesDict()[f"w_bts_{bi}_{i}"])
            except KeyError:
                try:
                    val = pulp.value(prob3.variablesDict()[f"w_bts_{bi}__{i}"])
                except Exception:
                    val = None
            if val is not None and round(val) == 1:
                demand_served[i] = ("BTS", bi)
                total_pop_served += pop_map.get(i, 0.0)
                served = True
                break
        if not served:
            demand_served[i] = None

    # compute cost totals
    total_travel_cost = 0.0
    total_fixed_cost = 0.0
    # cows
    for (k,j) in assignments_cow:
        cow_cost = to_float(cows[cows["cow_id"] == k].iloc[0].get("cost_vnd", 0.0)) if not cows[cows["cow_id"] == k].empty else 0.0
        travel_cost = to_float(cow_travel_map.get((k,j), {}).get("travel_cost_vnd", 0.0))
        total_fixed_cost += cow_cost
        total_travel_cost += travel_cost
    # powers
    for (g, bi) in assignments_power:
        power_operating_cost = to_float(
            power_lookup.get(g, {}).get("cost_vnd_24h", 0.0)
        )
        power_travel_cost = to_float(
            power_travel_map.get((g, bi), {}).get("travel_cost_vnd", 0.0)
        )

        total_fixed_cost += power_operating_cost
        total_travel_cost += power_travel_cost

    total_cost_all = total_fixed_cost + total_travel_cost

    T_max_final = pulp.value(T_max)

    # Save outputs
    res = {
        "solver": solver_name,
        "status": (status1, status2, status3),
        "optimal_covered_pop": optimal_covered_pop,
        "optimal_T_max": optimal_T_max,
        "final_total_cost_vnd": total_cost_all,
        "final_total_travel_cost_vnd": total_travel_cost,
        "final_total_fixed_cost_vnd": total_fixed_cost,
        "assignments_cow": assignments_cow,
        "assignments_power": assignments_power,
        "demand_served": demand_served,
        "T_max_hr": T_max_final,
        "total_pop_served": total_pop_served,
        "time_elapsed_s": t1 - t0
    }

    # write CSVs similar to existing code
    out_dir_run = out_dir / f"milp_{solver_name.lower()}"
    out_dir_run.mkdir(parents=True, exist_ok=True)
    # assignments cow
    df_cow_assign = pd.DataFrame(assignments_cow, columns=["cow_id", "site_id"])

    # merge cow static attributes
    df_cow_assign = df_cow_assign.merge(
        cows[[
            "cow_id", "base_id", "base_name", "type",
            "lat", "lon",
            "coverage_radius_m", "power_kw",
            "speed_kmh", "endurance_hr",
            "cost_vnd"
        ]],
        on="cow_id",
        how="left"
    )

    # assigned_region = base_name
    df_cow_assign["assigned_region"] = df_cow_assign["base_name"]

    # travel time & cost
    df_cow_assign["travel_time_hr"] = df_cow_assign.apply(
        lambda r: float(
            cow_travel_map.get((r["cow_id"], r["site_id"]), {}).get("travel_time_hr", 0.0)
        ),
        axis=1
    )
    # Làm tròn travel_time_hr đến 10 chữ số thập phân
    df_cow_assign["travel_time_hr"] = df_cow_assign["travel_time_hr"].round(10)

    df_cow_assign["travel_cost_vnd"] = df_cow_assign.apply(
        lambda r: float(
            cow_travel_map.get((r["cow_id"], r["site_id"]), {}).get("travel_cost_vnd", 0.0)
        ),
        axis=1
    )
    # Làm tròn travel_cost_vnd đến 8 chữ số thập phân
    df_cow_assign["travel_cost_vnd"] = df_cow_assign["travel_cost_vnd"].round(8)

    # setup time
    df_cow_assign["setup_time_h"] = float(params.get("default_setup_time_h", 0.5))

    # total time
    df_cow_assign["total_time_hr"] = (
            df_cow_assign["travel_time_hr"] + df_cow_assign["setup_time_h"]
    )
    # Làm tròn total_time_hr đến 10 chữ số thập phân
    df_cow_assign["total_time_hr"] = df_cow_assign["total_time_hr"].round(10)

    # total_cost_vnd = travel_cost_vnd + cost_vnd
    df_cow_assign["total_cost_vnd"] = (
            df_cow_assign["cost_vnd"].astype(float)
            + df_cow_assign["travel_cost_vnd"]
    )
    # Làm tròn total_cost_vnd đến 8 chữ số thập phân
    df_cow_assign["total_cost_vnd"] = df_cow_assign["total_cost_vnd"].round(8)

    df_cow_assign = df_cow_assign[[
        "cow_id", "site_id",
        "base_id", "base_name", "type",
        "lat", "lon",
        "coverage_radius_m", "power_kw",
        "speed_kmh", "endurance_hr",
        "cost_vnd",
        "assigned_region",
        "total_cost_vnd",
        "travel_time_hr", "setup_time_h", "total_time_hr"
    ]]

    df_cow_assign.to_csv(
        out_dir_run / f"assignments_cow_{solver_name}.csv",
        index=False
    )

    # power assignments
    df_power_assign = pd.DataFrame(assignments_power, columns=["power_id", "bts_id"])

    df_power_assign = df_power_assign.merge(
        backup_power[[
            "power_id", "base_id", "base_name",
            "lat", "lon",
            "type", "model",
            "runtime_h", "cost_vnd_24h", "resource_amount"
        ]],
        on="power_id",
        how="left"
    )

    # total_time_hr
    df_power_assign["total_time_hr"] = df_power_assign.apply(
        lambda r: float(
            power_travel_map.get((r["power_id"], r["bts_id"]), {}).get("total_time_hr", 0.0)
        ),
        axis=1
    )
    # Làm tròn total_time_hr đến 10 chữ số thập phân
    df_power_assign["total_time_hr"] = df_power_assign["total_time_hr"].round(10)

    df_power_assign["travel_cost_vnd"] = df_power_assign.apply(
        lambda r: float(
            power_travel_map.get(
                (r["power_id"], r["bts_id"]), {}
            ).get("travel_cost_vnd", 0.0)
        ),
        axis=1
    )

    # total_cost_vnd = travel_cost_vnd + cost_vnd_24h
    df_power_assign["total_cost_vnd"] = (
            df_power_assign["travel_cost_vnd"]
            + df_power_assign["cost_vnd_24h"].astype(float)
    )
    # Làm tròn total_cost_vnd đến 8 chữ số thập phân
    df_power_assign["total_cost_vnd"] = df_power_assign["total_cost_vnd"].round(8)

    df_power_assign = df_power_assign[[
        "base_id", "power_id", "bts_id",
        "lat", "lon",
        "base_name",
        "type", "model",
        "runtime_h", "cost_vnd_24h", "resource_amount",
        "total_cost_vnd", "total_time_hr"
    ]]

    df_power_assign.to_csv(
        out_dir_run / f"assignments_power_{solver_name}.csv",
        index=False
    )

    # summary json
    summary = {
        "solver": solver_name,
        "status": (status1, status2, status3),
        "optimal_covered_pop": optimal_covered_pop,
        "optimal_T_max": optimal_T_max,
        "final_total_cost_vnd": total_cost_all,
        "total_pop_served": total_pop_served,
        "num_cow_used": len(assignments_cow),
        "num_power_used": len(assignments_power),
        "time_elapsed_s": t1 - t0
    }
    with open(out_dir_run / f"summary_{solver_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_dir_run}")
    return res

# Main runner
def main_solve(config_params: dict, processed_data_dir: str, outputs_dir: str = None):
    """
    config_params: dict with params (budget_max, default_setup_time_h, etc.)
    processed_data_dir: path to data/processed
    outputs_dir: where to write outputs
    """
    if outputs_dir is None:
        outputs_dir = Path.cwd() / "outputs" / "milp_runs"
    else:
        outputs_dir = Path(outputs_dir)

    data = load_all_data(processed_data_dir)
    preproc = build_cover_and_travel_maps(data, config_params)

    # Run with available solvers: try GUROBI, fallback to CBC
    solvers = []
    if pulp.GUROBI_CMD().available():
        solvers.append("GUROBI")
    else:
        print("GUROBI not available → fallback to CBC")

    solvers.append("CBC")

    results = []
    for s in solvers:
        print(f"\n=== Running MILP with solver {s} ===")
        res = run_lexicographic_for_solver(data, preproc, config_params, solver_name=s, out_dir=outputs_dir)
        results.append((s, res))

    # choose canonical
    chosen = None
    for s, r in results:
        if r.get("status") and r["status"][0] in ("Optimal", "Integer Feasible", "Feasible"):
            chosen = (s, r)
            break
    if chosen:
        print(f"Chosen solver result: {chosen[0]}")
    else:
        print("No feasible solver result produced.")

    return results

# if __name__ usage example
if __name__ == "__main__":
    # minimal params
    params = {
        "budget_max": 1e9,
        "default_setup_time_h": 0.5,
        "flood_deploy_threshold_m": 0.5,
        "neighbour_search_m": 500.0,
        "milp": {"solver": {"time_limit": 600}}
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[4]  # adjust depending on layout when executing from script
    processed_dir = PROJECT_ROOT / "data" / "processed"
    outdir = PROJECT_ROOT / "outputs" / "milp_runs"
    main_solve(params, str(processed_dir), str(outdir))
