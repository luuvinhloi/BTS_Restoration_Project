"""
milp_solver.py
MILP lexicographic solver for BTS restoration
(KEEP 100% OUTPUT STRUCTURE OF OLD VERSION)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import rasterio
from math import radians, sin, cos, asin, sqrt
import json
import time

# Solver backends
import pulp
import pyomo.environ as pyo


# =====================================================
# Utilities
# =====================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# =====================================================
# Data loaders (UNCHANGED OUTPUT)
# =====================================================
def load_all_data(processed_dir: str):
    p = Path(processed_dir)

    j_sites = pd.read_csv(p / "position_I_J" / "J_sites.csv")
    cows = pd.read_csv(p / "cow" / "cow_dataset.csv")
    backup_power = pd.read_csv(p / "backup_power" / "backup_power.csv")
    failed_bts = pd.read_csv(p / "damage_bts" / "failed_bts.csv")

    cow_travel = pd.read_csv(p / "travel_cost" / "cow_to_J_sites.csv")
    power_travel = pd.read_csv(p / "travel_cost" / "backup_to_failed_bts.csv")

    roads_graphml = p / "road" / "roads_flooded.graphml"
    flood_raster = p / "flood" / "flood_depth_combined_B_clean.tif"

    # ---------- numeric coercions ----------
    for df, cols in [
        (j_sites, ["latitude", "longitude", "pop", "priority_weight"]),
        (cows, ["lat", "lon", "coverage_radius_m", "power_kw",
                "speed_kmh", "endurance_hr", "cost_vnd"]),
        (backup_power, ["lat", "lon", "runtime_h",
                        "cost_vnd_24h", "resource_amount"]),
        (failed_bts, ["latitude", "longitude",
                      "coverage_radius_m", "power_W", "pop_covered"])
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for df, cols in [
        (cow_travel, ["distance_km", "travel_time_hr", "travel_cost_vnd"]),
        (power_travel, ["distance_km", "total_time_hr", "total_cost_vnd"])
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ---------- road graph ----------
    G = None
    if roads_graphml.exists():
        try:
            G = nx.read_graphml(str(roads_graphml))
        except Exception as e:
            print(f"[WARN] Cannot read graphml: {e}")

    # ---------- flood raster ----------
    flood_ds = None
    if flood_raster.exists():
        try:
            flood_ds = rasterio.open(str(flood_raster))
        except Exception as e:
            print(f"[WARN] Cannot open flood raster: {e}")

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


# =====================================================
# Preprocessing (GIỮ LOGIC CŨ + thêm j_feasible)
# =====================================================
def build_cover_and_travel_maps(data, params):
    """
    Build:
      - cover_indicator[(i, j, k)]
      - cow_travel_map[(cow_id, site_id)]
      - power_travel_map[(power_id, bts_id)]
      - j_alternative
      - j_feasible (FORCE 100% COVERAGE)
      - compatibility[(bts_id, power_id)]
    """

    j_sites = data["j_sites"].copy()
    cows = data["cows"]
    backup_power = data["backup_power"]
    failed_bts = data["failed_bts"]
    cow_travel = data["cow_travel"]
    power_travel = data["power_travel"]
    flood_ds = data["flood_ds"]

    flood_threshold = float(params.get("flood_deploy_threshold_m", 0.5))
    neighbour_search_m = float(params.get("neighbour_search_m", 500.0))

    # ---------- travel maps ----------
    cow_travel_map = {
        (str(r["cow_id"]), str(r["site_id"])): {
            "distance_km": to_float(r.get("distance_km")),
            "travel_time_hr": to_float(r.get("travel_time_hr")),
            "travel_cost_vnd": to_float(r.get("travel_cost_vnd"))
        }
        for _, r in cow_travel.iterrows()
    }

    power_travel_map = {
        (str(r["power_id"]), str(r["bts_id"])): {
            "distance_km": to_float(r.get("distance_km")),
            "total_time_hr": to_float(r.get("total_time_hr")),
            "total_cost_vnd": to_float(r.get("total_cost_vnd")),
            "note": r.get("note", "")
        }
        for _, r in power_travel.iterrows()
    }

    # ---------- flood feasibility ----------
    j_sites["feasible_deploy"] = True
    if flood_ds is not None:
        coords = [(r["longitude"], r["latitude"]) for _, r in j_sites.iterrows()]
        try:
            depths = [v[0] for v in flood_ds.sample(coords)]
        except Exception:
            depths = [0.0] * len(coords)
        j_sites["flood_depth_m"] = depths
        j_sites.loc[j_sites["flood_depth_m"] > flood_threshold,
                    "feasible_deploy"] = False
    else:
        j_sites["flood_depth_m"] = 0.0

    if "in_water" in j_sites.columns:
        j_sites.loc[j_sites["in_water"].astype(bool),
                    "feasible_deploy"] = False

    # ---------- alternative J ----------
    j_alternative = {}
    for _, r in j_sites.iterrows():
        sid = str(r["site_id"])
        if r["feasible_deploy"]:
            j_alternative[sid] = sid
            continue

        lat0, lon0 = r["latitude"], r["longitude"]
        found = None
        for _, r2 in j_sites[j_sites["feasible_deploy"]].iterrows():
            d = haversine_km(lat0, lon0,
                             r2["latitude"], r2["longitude"]) * 1000
            if d <= neighbour_search_m:
                found = str(r2["site_id"])
                break
        j_alternative[sid] = found

    # ---------- j_feasible (FORCE 100% COVERAGE SET) ----------
    j_feasible = []
    for _, r in j_sites.iterrows():
        sid = str(r["site_id"])
        if r["feasible_deploy"] or j_alternative.get(sid) is not None:
            j_feasible.append(sid)

    # ---------- coverage indicator ----------
    demands = j_sites["site_id"].astype(str).tolist()
    cow_ids = cows["cow_id"].astype(str).tolist()

    site_xy = {
        str(r["site_id"]): (r["latitude"], r["longitude"])
        for _, r in j_sites.iterrows()
    }

    cow_radius = {
        str(r["cow_id"]): to_float(r["coverage_radius_m"])
        for _, r in cows.iterrows()
    }

    cover_indicator = {}
    for i in demands:
        lat_i, lon_i = site_xy[i]
        for j in demands:
            lat_j, lon_j = site_xy[j]
            d_m = haversine_km(lat_i, lon_i, lat_j, lon_j) * 1000
            for k in cow_ids:
                cover_indicator[(i, j, k)] = int(d_m <= cow_radius[k])

    # ---------- power compatibility ----------
    power_ids = backup_power["power_id"].astype(str).tolist()
    bts_ids = failed_bts["site_id"].astype(str).tolist()

    power_lookup = {
        str(r["power_id"]): {
            "type": str(r.get("type", "")).upper(),
            "cost_vnd_24h": to_float(r.get("cost_vnd_24h"))
        }
        for _, r in backup_power.iterrows()
    }

    bts_lookup = {
        str(r["site_id"]): {
            "power_W": to_float(r.get("power_W")),
            "latitude": to_float(r.get("latitude")),
            "longitude": to_float(r.get("longitude")),
            "status": r.get("status", "")
        }
        for _, r in failed_bts.iterrows()
    }

    compatibility = {(b, g): 1 for b in bts_ids for g in power_ids}

    return {
        "cover_indicator": cover_indicator,
        "cow_travel_map": cow_travel_map,
        "power_travel_map": power_travel_map,
        "j_sites": j_sites,
        "j_alternative": j_alternative,
        "j_feasible": j_feasible,
        "compatibility": compatibility,
        "power_lookup": power_lookup,
        "bts_lookup": bts_lookup
    }

# =====================================================
# MILP builder — Pyomo (GUROBI)
# =====================================================
def build_milp_problem_pyomo(preproc, data, params):
    """
    Build Pyomo MILP model (GUROBI backend).
    """

    model = pyo.ConcreteModel("MILP_BTS_Restoration")

    j_sites = preproc["j_sites"]
    cows = data["cows"]
    backup_power = data["backup_power"]
    failed_bts = data["failed_bts"]

    cow_ids = cows["cow_id"].astype(str).tolist()
    j_ids = j_sites["site_id"].astype(str).tolist()
    bts_ids = failed_bts["site_id"].astype(str).tolist()
    power_ids = backup_power["power_id"].astype(str).tolist()

    setup_time_h = float(params.get("default_setup_time_h", 0.5))

    # ---------------- Sets ----------------
    model.K = pyo.Set(initialize=cow_ids)
    model.J = pyo.Set(initialize=j_ids)
    model.B = pyo.Set(initialize=bts_ids)
    model.G = pyo.Set(initialize=power_ids)

    # ---------------- Variables ----------------
    model.x = pyo.Var(model.K, model.J, domain=pyo.Binary)
    model.y = pyo.Var(model.J, model.J, domain=pyo.Binary)
    model.z = pyo.Var(model.G, model.B, domain=pyo.Binary)
    model.u = pyo.Var(model.B, domain=pyo.Binary)
    model.w_bts = pyo.Var(model.B, model.J, domain=pyo.Binary)
    model.T_max = pyo.Var(domain=pyo.NonNegativeReals)

    cover = preproc["cover_indicator"]
    cow_travel = preproc["cow_travel_map"]
    power_travel = preproc["power_travel_map"]
    compatibility = preproc["compatibility"]
    j_alternative = preproc["j_alternative"]

    # ---------------- Constraints ----------------
    model.OneDeployPerCow = pyo.Constraint(
        model.K,
        rule=lambda m, k: sum(m.x[k, j] for j in m.J) <= 1
    )

    model.OneCowPerSite = pyo.Constraint(
        model.J,
        rule=lambda m, j: sum(m.x[k, j] for k in m.K) <= 1
    )

    model.DemandCoveredOnce = pyo.Constraint(
        model.J,
        rule=lambda m, i: sum(m.y[i, j] for j in m.J) <= 1
    )

    def y_implies_x(m, i, j):
        feasible = [k for k in m.K if cover.get((i, j, k), 0) == 1]
        if not feasible:
            return m.y[i, j] == 0
        return m.y[i, j] <= sum(m.x[k, j] for k in feasible)

    model.YImpliesX = pyo.Constraint(model.J, model.J, rule=y_implies_x)

    model.OnePowerPerBTS = pyo.Constraint(
        model.B,
        rule=lambda m, b: sum(m.z[g, b] for g in m.G) <= 1
    )

    model.OneBTSPerPower = pyo.Constraint(
        model.G,
        rule=lambda m, g: sum(m.z[g, b] for b in m.B) <= 1
    )

    model.UImpliesZ = pyo.Constraint(
        model.B,
        rule=lambda m, b: m.u[b] <= sum(
            m.z[g, b] * compatibility.get((b, g), 0)
            for g in m.G
        )
    )

    model.SingleService = pyo.Constraint(
        model.J,
        rule=lambda m, j:
            sum(m.w_bts[b, j] for b in m.B)
            + sum(m.y[j, jj] for jj in m.J)
            <= 1
    )

    model.WImpliesU = pyo.Constraint(
        model.B, model.J,
        rule=lambda m, b, j: m.w_bts[b, j] <= m.u[b]
    )

    def forbid_flooded(m, j):
        if j_alternative.get(j) is None:
            return sum(m.x[k, j] for k in m.K) == 0
        return pyo.Constraint.Skip

    model.ForbidFloodedJ = pyo.Constraint(model.J, rule=forbid_flooded)

    model.TmaxCow = pyo.Constraint(
        model.K, model.J,
        rule=lambda m, k, j:
            m.T_max >= (cow_travel.get((k, j), {}).get("travel_time_hr", 0.0)
                        + setup_time_h) * m.x[k, j]
    )

    model.TmaxPower = pyo.Constraint(
        model.G, model.B,
        rule=lambda m, g, b:
            m.T_max >= power_travel.get((g, b), {}).get("total_time_hr", 0.0)
            * m.z[g, b]
    )

    return model

# =====================================================
# MILP builder — PuLP (CBC)
# =====================================================
def build_milp_problem_pulp(preproc, data, params):
    """
    Build PuLP MILP problem (CBC backend).
    GIỮ CẤU TRÚC BIẾN CŨ + ép 100% coverage cho j_feasible
    Returns: prob, var_dict
    """

    j_sites = preproc["j_sites"]
    j_feasible = preproc.get("j_feasible", [])

    cows = data["cows"]
    backup_power = data["backup_power"]
    failed_bts = data["failed_bts"]

    cow_ids = cows["cow_id"].astype(str).tolist()
    j_ids = j_sites["site_id"].astype(str).tolist()
    bts_ids = failed_bts["site_id"].astype(str).tolist()
    power_ids = backup_power["power_id"].astype(str).tolist()

    setup_time_h = float(params.get("default_setup_time_h", 0.5))

    # ---------------- Problem ----------------
    prob = pulp.LpProblem("MILP_BTS_Restoration", pulp.LpMinimize)

    # ---------------- Variables ----------------
    x = pulp.LpVariable.dicts("x", (cow_ids, j_ids), cat="Binary")          # COW -> J
    y = pulp.LpVariable.dicts("y", (j_ids, j_ids), cat="Binary")            # demand i covered by COW at j
    z = pulp.LpVariable.dicts("z", (power_ids, bts_ids), cat="Binary")      # Power -> BTS
    u = pulp.LpVariable.dicts("u", bts_ids, cat="Binary")                   # BTS powered
    w_bts = pulp.LpVariable.dicts("w_bts", (bts_ids, j_ids), cat="Binary")  # BTS covers J

    # Coverage indicator (PHỤC VỤ STEP 1)
    covered = pulp.LpVariable.dicts("covered", j_ids, cat="Binary")

    # Max deployment time
    T_max = pulp.LpVariable("T_max", lowBound=0)

    cover = preproc["cover_indicator"]
    cow_travel = preproc["cow_travel_map"]
    power_travel = preproc["power_travel_map"]
    compatibility = preproc["compatibility"]
    j_alternative = preproc["j_alternative"]

    # =================================================
    # Constraints
    # =================================================

    # ---------- Each COW at most one site ----------
    for k in cow_ids:
        prob += pulp.lpSum(x[k][j] for j in j_ids) <= 1

    # ---------- Each site at most one COW ----------
    for j in j_ids:
        prob += pulp.lpSum(x[k][j] for k in cow_ids) <= 1

    # ---------- Each demand covered at most once by COW ----------
    for i in j_ids:
        prob += pulp.lpSum(y[i][j] for j in j_ids) <= 1

    # ---------- y implies feasible x ----------
    for i in j_ids:
        for j in j_ids:
            feasible = [k for k in cow_ids if cover.get((i, j, k), 0) == 1]
            if not feasible:
                prob += y[i][j] == 0
            else:
                prob += y[i][j] <= pulp.lpSum(x[k][j] for k in feasible)

    # ---------- Power assignment ----------
    for b in bts_ids:
        prob += pulp.lpSum(z[g][b] for g in power_ids) <= 1

    for g in power_ids:
        prob += pulp.lpSum(z[g][b] for b in bts_ids) <= 1

    for b in bts_ids:
        prob += u[b] <= pulp.lpSum(
            z[g][b] * compatibility.get((b, g), 0)
            for g in power_ids
        )

    # ---------- Demand served by BTS or COW (exclusive) ----------
    for j in j_ids:
        prob += (
            pulp.lpSum(w_bts[b][j] for b in bts_ids)
            + pulp.lpSum(y[j][jj] for jj in j_ids)
            <= 1
        )

    for b in bts_ids:
        for j in j_ids:
            prob += w_bts[b][j] <= u[b]

    # ---------- Coverage definition ----------
    for j in j_ids:
        prob += covered[j] <= (
            pulp.lpSum(y[j][jj] for jj in j_ids)
            + pulp.lpSum(w_bts[b][j] for b in bts_ids)
        )

    # ---------- FORCE 100% COVERAGE FOR FEASIBLE J ----------
    for j in j_feasible:
        prob += covered[j] == 1

    # ---------- Forbid flooded J without alternative ----------
    for j in j_ids:
        if j_alternative.get(j) is None:
            prob += pulp.lpSum(x[k][j] for k in cow_ids) == 0

    # ---------- T_max constraints ----------
    for k in cow_ids:
        for j in j_ids:
            prob += T_max >= (
                cow_travel.get((k, j), {}).get("travel_time_hr", 0.0)
                + setup_time_h
            ) * x[k][j]

    for g in power_ids:
        for b in bts_ids:
            prob += T_max >= (
                power_travel.get((g, b), {}).get("total_time_hr", 0.0)
            ) * z[g][b]

    var_dict = {
        "x": x,
        "y": y,
        "z": z,
        "u": u,
        "w_bts": w_bts,
        "covered": covered,
        "T_max": T_max,
        "cow_ids": cow_ids,
        "j_ids": j_ids,
        "bts_ids": bts_ids,
        "power_ids": power_ids
    }

    return prob, var_dict

# =====================================================
# Coverage expression — PuLP
# =====================================================
def build_coverage_expression_pulp(y, w_bts, j_ids, bts_ids, pop_map):
    """
    Total covered population by COW + restored BTS
    """
    cov_cow = pulp.lpSum(
        pop_map[j] * pulp.lpSum(y[j][jj] for jj in j_ids)
        for j in j_ids
    )
    cov_bts = pulp.lpSum(
        pop_map[j] * pulp.lpSum(w_bts[b][j] for b in bts_ids)
        for j in j_ids
    )
    return cov_cow + cov_bts


# =====================================================
# Lexicographic solve — PuLP (CBC)
# =====================================================
def solve_lexicographic_pulp(
    data,
    preproc,
    params,
    solver_name="CBC",
    out_dir="./outputs/milp_runs"
):
    """
    Lexicographic MILP with PuLP + CBC
    OUTPUT FILES IDENTICAL TO OLD VERSION
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- aliases ----------
    j_sites = preproc["j_sites"]
    cows = data["cows"]
    backup_power = data["backup_power"]
    failed_bts = data["failed_bts"]

    cow_travel_map = preproc["cow_travel_map"]
    power_travel_map = preproc["power_travel_map"]
    power_lookup = preproc["power_lookup"]

    # ---------- population & priority ----------
    pop_map = {str(r["site_id"]): to_float(r.get("pop", 0.0))
               for _, r in j_sites.iterrows()}
    priority_map = {str(r["site_id"]): to_float(r.get("priority_weight", 0.0))
                    for _, r in j_sites.iterrows()}

    # ---------- solver ----------
    time_limit = int(params.get("milp", {}).get("solver", {}).get("time_limit", 600))
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit)

    # =================================================
    # STEP 1 — Max coverage (+ priority tie-break)
    # =================================================
    prob1, v1 = build_milp_problem_pulp(preproc, data, params)
    cov1 = build_coverage_expression_pulp(
        v1["y"], v1["w_bts"], v1["j_ids"], v1["bts_ids"], pop_map
    )

    EPS_PRIORITY = 1e-4
    prob1.sense = pulp.LpMaximize
    prob1.setObjective(
        cov1
        + EPS_PRIORITY * pulp.lpSum(
            priority_map[j] * pulp.lpSum(v1["x"][k][j] for k in v1["cow_ids"])
            for j in v1["j_ids"]
        )
    )

    print("[CBC] Step 1: maximize coverage")
    t0 = time.time()
    prob1.solve(solver)
    t1 = time.time()

    status1 = pulp.LpStatus[prob1.status]
    optimal_covered_pop = pulp.value(cov1)

    if status1 not in ("Optimal", "Integer Feasible", "Feasible"):
        return {"solver": solver_name, "status_step1": status1}

    # =================================================
    # STEP 2 — Minimize T_max (fix coverage)
    # =================================================
    prob2, v2 = build_milp_problem_pulp(preproc, data, params)
    cov2 = build_coverage_expression_pulp(
        v2["y"], v2["w_bts"], v2["j_ids"], v2["bts_ids"], pop_map
    )

    prob2 += cov2 >= optimal_covered_pop - 1e-6, "FixCoverage"
    prob2.sense = pulp.LpMinimize
    prob2.setObjective(v2["T_max"])

    print("[CBC] Step 2: minimize T_max")
    prob2.solve(solver)

    status2 = pulp.LpStatus[prob2.status]
    optimal_T_max = pulp.value(v2["T_max"])

    if status2 not in ("Optimal", "Integer Feasible", "Feasible"):
        return {
            "solver": solver_name,
            "status_step1": status1,
            "status_step2": status2
        }

    # =================================================
    # STEP 3 — Minimize total cost (fix coverage & T_max)
    # =================================================
    prob3, v3 = build_milp_problem_pulp(preproc, data, params)
    cov3 = build_coverage_expression_pulp(
        v3["y"], v3["w_bts"], v3["j_ids"], v3["bts_ids"], pop_map
    )

    prob3 += cov3 >= optimal_covered_pop - 1e-6, "FixCoverage"
    prob3 += v3["T_max"] <= optimal_T_max + 1e-9, "FixTmax"

    total_cost_terms = []

    # --- COW costs ---
    for _, r in cows.iterrows():
        k = str(r["cow_id"])
        c_fix = to_float(r.get("cost_vnd", 0.0))
        for j in v3["j_ids"]:
            c_tr = to_float(cow_travel_map.get((k, j), {}).get("travel_cost_vnd", 0.0))
            total_cost_terms.append((c_fix + c_tr) * v3["x"][k][j])

    # --- Power costs ---
    for g in v3["power_ids"]:
        c_fix = to_float(power_lookup.get(g, {}).get("cost_vnd_24h", 0.0))
        for b in v3["bts_ids"]:
            c_tr = to_float(power_travel_map.get((g, b), {}).get("total_cost_vnd", 0.0))
            total_cost_terms.append((c_fix + c_tr) * v3["z"][g][b])

    # --- Budget ---
    budget_max = float(params.get("budget_max", 1e9))
    prob3 += pulp.lpSum(total_cost_terms) <= budget_max, "Budget"

    prob3.sense = pulp.LpMinimize
    prob3.setObjective(pulp.lpSum(total_cost_terms))

    print("[CBC] Step 3: minimize total cost")
    prob3.solve(solver)

    status3 = pulp.LpStatus[prob3.status]

    # =================================================
    # Extract solution (IDENTICAL TO OLD CODE)
    # =================================================
    assignments_cow = [
        (k, j)
        for k in v3["cow_ids"]
        for j in v3["j_ids"]
        if round(pulp.value(v3["x"][k][j]) or 0) == 1
    ]

    assignments_power = [
        (g, b)
        for g in v3["power_ids"]
        for b in v3["bts_ids"]
        if round(pulp.value(v3["z"][g][b]) or 0) == 1
    ]

    # --- demand served ---
    demand_served = {}
    total_pop_served = 0.0
    for i in v3["j_ids"]:
        served = False
        for j in v3["j_ids"]:
            if round(pulp.value(v3["y"][i][j]) or 0) == 1:
                demand_served[i] = ("COW", j)
                total_pop_served += pop_map.get(i, 0.0)
                served = True
                break
        if served:
            continue
        for b in v3["bts_ids"]:
            if round(pulp.value(v3["w_bts"][b][i]) or 0) == 1:
                demand_served[i] = ("BTS", b)
                total_pop_served += pop_map.get(i, 0.0)
                served = True
                break
        if not served:
            demand_served[i] = None

    # --- cost totals ---
    total_fixed_cost = 0.0
    total_travel_cost = 0.0

    for (k, j) in assignments_cow:
        total_fixed_cost += to_float(
            cows[cows["cow_id"] == k].iloc[0].get("cost_vnd", 0.0)
        )
        total_travel_cost += to_float(
            cow_travel_map.get((k, j), {}).get("travel_cost_vnd", 0.0)
        )

    for (g, b) in assignments_power:
        total_fixed_cost += to_float(
            power_lookup.get(g, {}).get("cost_vnd_24h", 0.0)
        )
        total_travel_cost += to_float(
            power_travel_map.get((g, b), {}).get("total_cost_vnd", 0.0)
        )

    total_cost_all = total_fixed_cost + total_travel_cost
    T_max_final = pulp.value(v3["T_max"])

    # =================================================
    # Save OUTPUT FILES (CSV / JSON) — IDENTICAL
    # =================================================
    out_run = out_dir / f"milp_{solver_name.lower()}"
    out_run.mkdir(parents=True, exist_ok=True)

    # --- COW CSV ---
    df_cow = pd.DataFrame(assignments_cow, columns=["cow_id", "site_id"])
    df_cow = df_cow.merge(cows, on="cow_id", how="left")
    df_cow["travel_cost_vnd"] = df_cow.apply(
        lambda r: to_float(
            cow_travel_map.get((r["cow_id"], r["site_id"]), {}).get("travel_cost_vnd", 0.0)
        ),
        axis=1
    )
    df_cow["travel_time_hr"] = df_cow.apply(
        lambda r: to_float(
            cow_travel_map.get((r["cow_id"], r["site_id"]), {}).get("travel_time_hr", 0.0)
        ),
        axis=1
    )
    df_cow["setup_time_h"] = float(params.get("default_setup_time_h", 0.5))
    df_cow["deployment_time_hr"] = df_cow["travel_time_hr"] + df_cow["setup_time_h"]
    df_cow.to_csv(out_run / f"assignments_cow_{solver_name}.csv", index=False)

    # --- Power CSV ---
    df_power = pd.DataFrame(assignments_power, columns=["power_id", "bts_id"])
    df_power = df_power.merge(backup_power, on="power_id", how="left")
    df_power["travel_cost_vnd"] = df_power.apply(
        lambda r: to_float(
            power_travel_map.get((r["power_id"], r["bts_id"]), {}).get("total_cost_vnd", 0.0)
        ),
        axis=1
    )
    df_power["operating_cost_vnd_24h"] = df_power["cost_vnd_24h"]
    df_power["total_deployment_cost_vnd"] = (
        df_power["travel_cost_vnd"] + df_power["operating_cost_vnd_24h"]
    )
    df_power["travel_time_hr"] = df_power.apply(
        lambda r: to_float(
            power_travel_map.get((r["power_id"], r["bts_id"]), {}).get("total_time_hr", 0.0)
        ),
        axis=1
    )
    df_power.to_csv(out_run / f"assignments_power_{solver_name}.csv", index=False)

    # --- Summary JSON ---
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

    with open(out_run / f"summary_{solver_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
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

# =====================================================
# Lexicographic solve — Pyomo (GUROBI)
# =====================================================
def solve_lexicographic_pyomo(
    data,
    preproc,
    params,
    solver_name="GUROBI",
    out_dir="./outputs/milp_runs"
):
    """
    Lexicographic MILP using Pyomo + GUROBI
    OUTPUT STRUCTURE IDENTICAL TO PuLP VERSION
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    j_sites = preproc["j_sites"]
    pop_map = {str(r["site_id"]): to_float(r.get("pop", 0.0))
               for _, r in j_sites.iterrows()}
    priority_map = {str(r["site_id"]): to_float(r.get("priority_weight", 0.0))
                    for _, r in j_sites.iterrows()}

    cow_travel = preproc["cow_travel_map"]
    power_travel = preproc["power_travel_map"]
    power_lookup = preproc["power_lookup"]

    time_limit = int(params.get("milp", {}).get("solver", {}).get("time_limit", 600))
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = time_limit

    # =================================================
    # STEP 1 — Max coverage (+ priority)
    # =================================================
    model1 = build_milp_problem_pyomo(preproc, data, params)

    cov_expr1 = (
        sum(pop_map[j] * sum(model1.y[j, jj] for jj in model1.J) for j in model1.J)
        + sum(pop_map[j] * sum(model1.w[b, j] for b in model1.B) for j in model1.J)
    )
    priority_expr = sum(
        priority_map[j] * sum(model1.x[k, j] for k in model1.K)
        for j in model1.J
    )

    EPS_PRIORITY = 1e-4
    model1.Obj = pyo.Objective(
        expr=cov_expr1 + EPS_PRIORITY * priority_expr,
        sense=pyo.maximize
    )

    print("[GUROBI] Step 1: maximize coverage")
    t0 = time.time()
    solver.solve(model1, tee=True)
    t1 = time.time()

    optimal_covered_pop = pyo.value(cov_expr1)
    status1 = "Optimal"

    # =================================================
    # STEP 2 — Minimize T_max (fix coverage)
    # =================================================
    model2 = build_milp_problem_pyomo(preproc, data, params)

    cov_expr2 = (
        sum(pop_map[j] * sum(model2.y[j, jj] for jj in model2.J) for j in model2.J)
        + sum(pop_map[j] * sum(model2.w[b, j] for b in model2.B) for j in model2.J)
    )

    model2.CovFix = pyo.Constraint(expr=cov_expr2 >= optimal_covered_pop - 1e-6)
    model2.Obj = pyo.Objective(expr=model2.T_max, sense=pyo.minimize)

    print("[GUROBI] Step 2: minimize T_max")
    solver.solve(model2, tee=True)

    optimal_T_max = pyo.value(model2.T_max)
    status2 = "Optimal"

    # =================================================
    # STEP 3 — Minimize total cost (fix coverage & T_max)
    # =================================================
    model3 = build_milp_problem_pyomo(preproc, data, params)

    cov_expr3 = (
        sum(pop_map[j] * sum(model3.y[j, jj] for jj in model3.J) for j in model3.J)
        + sum(pop_map[j] * sum(model3.w[b, j] for b in model3.B) for j in model3.J)
    )

    model3.CovFix = pyo.Constraint(expr=cov_expr3 >= optimal_covered_pop - 1e-6)
    model3.TFix = pyo.Constraint(expr=model3.T_max <= optimal_T_max + 1e-9)

    cost_expr = 0

    for _, r in data["cows"].iterrows():
        k = str(r["cow_id"])
        c_fix = to_float(r.get("cost_vnd", 0.0))
        for j in model3.J:
            c_tr = to_float(cow_travel.get((k, j), {}).get("travel_cost_vnd", 0.0))
            cost_expr += (c_fix + c_tr) * model3.x[k, j]

    for g in model3.G:
        c_fix = to_float(power_lookup.get(g, {}).get("cost_vnd_24h", 0.0))
        for b in model3.B:
            c_tr = to_float(power_travel.get((g, b), {}).get("total_cost_vnd", 0.0))
            cost_expr += (c_fix + c_tr) * model3.z[g, b]

    budget_max = float(params.get("budget_max", 1e9))
    model3.Budget = pyo.Constraint(expr=cost_expr <= budget_max)
    model3.Obj = pyo.Objective(expr=cost_expr, sense=pyo.minimize)

    print("[GUROBI] Step 3: minimize total cost")
    solver.solve(model3, tee=True)

    status3 = "Optimal"

    # =================================================
    # Extract solution (Pyomo → identical structure)
    # =================================================
    def val(v):
        try:
            return int(round(pyo.value(v)))
        except Exception:
            return 0

    assignments_cow = [(k, j) for k in model3.K for j in model3.J if val(model3.x[k, j]) == 1]
    assignments_power = [(g, b) for g in model3.G for b in model3.B if val(model3.z[g, b]) == 1]

    demand_served = {}
    total_pop_served = 0.0
    for j in model3.J:
        served = False
        for jj in model3.J:
            if val(model3.y[j, jj]) == 1:
                demand_served[j] = ("COW", jj)
                total_pop_served += pop_map.get(j, 0.0)
                served = True
                break
        if served:
            continue
        for b in model3.B:
            if val(model3.w[b, j]) == 1:
                demand_served[j] = ("BTS", b)
                total_pop_served += pop_map.get(j, 0.0)
                served = True
                break
        if not served:
            demand_served[j] = None

    total_fixed_cost = 0.0
    total_travel_cost = 0.0

    for (k, j) in assignments_cow:
        total_fixed_cost += to_float(
            data["cows"][data["cows"]["cow_id"] == k].iloc[0].get("cost_vnd", 0.0)
        )
        total_travel_cost += to_float(
            cow_travel.get((k, j), {}).get("travel_cost_vnd", 0.0)
        )

    for (g, b) in assignments_power:
        total_fixed_cost += to_float(power_lookup.get(g, {}).get("cost_vnd_24h", 0.0))
        total_travel_cost += to_float(power_travel.get((g, b), {}).get("total_cost_vnd", 0.0))

    total_cost_all = total_fixed_cost + total_travel_cost
    T_max_final = pyo.value(model3.T_max)

    # =================================================
    # Save outputs — IDENTICAL TO CBC
    # =================================================
    out_run = Path(out_dir) / f"milp_{solver_name.lower()}"
    out_run.mkdir(parents=True, exist_ok=True)

    # --- COW CSV ---
    df_cow = pd.DataFrame(assignments_cow, columns=["cow_id", "site_id"])
    df_cow = df_cow.merge(data["cows"], on="cow_id", how="left")
    df_cow["travel_cost_vnd"] = df_cow.apply(
        lambda r: to_float(cow_travel.get((r["cow_id"], r["site_id"]), {}).get("travel_cost_vnd", 0.0)),
        axis=1
    )
    df_cow["travel_time_hr"] = df_cow.apply(
        lambda r: to_float(cow_travel.get((r["cow_id"], r["site_id"]), {}).get("travel_time_hr", 0.0)),
        axis=1
    )
    df_cow["setup_time_h"] = float(params.get("default_setup_time_h", 0.5))
    df_cow["deployment_time_hr"] = df_cow["travel_time_hr"] + df_cow["setup_time_h"]
    df_cow.to_csv(out_run / f"assignments_cow_{solver_name}.csv", index=False)

    # --- Power CSV ---
    df_power = pd.DataFrame(assignments_power, columns=["power_id", "bts_id"])
    df_power = df_power.merge(data["backup_power"], on="power_id", how="left")
    df_power["travel_cost_vnd"] = df_power.apply(
        lambda r: to_float(power_travel.get((r["power_id"], r["bts_id"]), {}).get("total_cost_vnd", 0.0)),
        axis=1
    )
    df_power["operating_cost_vnd_24h"] = df_power["cost_vnd_24h"]
    df_power["total_deployment_cost_vnd"] = df_power["travel_cost_vnd"] + df_power["operating_cost_vnd_24h"]
    df_power["travel_time_hr"] = df_power.apply(
        lambda r: to_float(power_travel.get((r["power_id"], r["bts_id"]), {}).get("total_time_hr", 0.0)),
        axis=1
    )
    df_power.to_csv(out_run / f"assignments_power_{solver_name}.csv", index=False)

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

    with open(out_run / f"summary_{solver_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
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


# =====================================================
# Solver dispatcher (UNCHANGED API)
# =====================================================
def run_milp_backend(data, preproc, params, solver_backend="AUTO", out_dir="./outputs/milp_runs"):
    solver_backend = solver_backend.upper()

    if solver_backend == "CBC":
        return solve_lexicographic_pulp(data, preproc, params, solver_name="CBC", out_dir=out_dir)

    if solver_backend == "GUROBI":
        return solve_lexicographic_pyomo(data, preproc, params, solver_name="GUROBI", out_dir=out_dir)

    # AUTO
    try:
        if pyo.SolverFactory("gurobi").available():
            return solve_lexicographic_pyomo(data, preproc, params, solver_name="GUROBI", out_dir=out_dir)
    except Exception:
        pass

    return solve_lexicographic_pulp(data, preproc, params, solver_name="CBC", out_dir=out_dir)


# =====================================================
# Main runner (UNCHANGED)
# =====================================================
def main_solve_cov(config_params: dict, processed_data_dir: str, outputs_dir: str = None, solver_backend="AUTO"):
    if outputs_dir is None:
        outputs_dir = Path.cwd() / "outputs" / "milp_runs"
    else:
        outputs_dir = Path(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("[MAIN] Loading data...")
    data = load_all_data(processed_data_dir)

    print("[MAIN] Preprocessing...")
    preproc = build_cover_and_travel_maps(data, config_params)

    print(f"[MAIN] Running MILP backend = {solver_backend}")
    return run_milp_backend(
        data=data,
        preproc=preproc,
        params=config_params,
        solver_backend=solver_backend,
        out_dir=outputs_dir
    )
