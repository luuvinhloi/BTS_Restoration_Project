# src/optimization/milp_pulp.py
"""
MILP builder using PuLP for COW deployment lexicographic optimization.

Provides:
- build_base_problem(...) : constructs a fresh PuLP problem with variables and constraints common to all 3 lexicographic steps.
- helper functions to extract solution and metrics.

Assumptions / notes:
- Demand points are provided by J_sites.csv (site_id, latitude, longitude, pop, ...).
- Deployment candidates are the same set of J_sites (deploying at site j).
- travel_cost_matrix contains travel_time_hr and travel_cost_vnd per cow_id/site_id pair.
- Coverage decision: a cow k deployed at site j covers demand i if distance(i,j) * 1000 <= cow.coverage_radius_m.
- Non-overlap: sum_j y[i,j] <= 1 (each demand served by at most one site).
"""

from math import radians, sin, cos, asin, sqrt
import pulp
from collections import defaultdict


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def build_base_problem(demand_list, site_list, cows, travel_matrix,
                       cover_indicator, params,
                       solver_name="CBC"):
    """
    Build base PuLP problem (variables + common constraints) for lexicographic steps.

    Inputs:
      - demand_list: list of demand dicts each with keys: site_id, latitude, longitude, pop, ...
      - site_list: list of candidate site dicts (same structure)
      - cows: list of cow dicts each with cow_id, base_id, coverage_radius_m, endurance_hr, cost_vnd, speed_kmh, ...
      - travel_matrix: dict keyed (cow_id, site_id) -> dict(distance_km, travel_time_hr, travel_cost_vnd)
      - cover_indicator: dict keyed (i_site_id, deploy_site_id, cow_id) -> 0/1 whether cow at deploy_site covers demand i
      - params: dict from params.yaml (contains budget_max, M_max, default_setup_time_h, etc.)
      - solver_name: "GUROBI" or "CBC" (affects solver selection when solve called)
    Returns:
      - prob: pulp.LpProblem object
      - variables dicts: x[(cow_id, site_id)], y[(demand_id, site_id)], z[demand_id], T_max variable
      - helper data structures for constraints
    """

    # Create problem (objective to be set per lexicographic step)
    prob = pulp.LpProblem("COW_Lexicographic_Base", pulp.LpMinimize)

    # Index sets
    demand_ids = [d["site_id"] for d in demand_list]
    site_ids = [s["site_id"] for s in site_list]
    cow_ids = [c["cow_id"] for c in cows]

    # Parameters
    budget_max = float(params.get("budget_max", 5e8))
    M_max = int(params.get("M_max", len(cows)))
    setup_time_h = float(params.get("default_setup_time_h", 0.5))

    # Decision variables
    # x_kj : 1 if cow k is deployed at site j
    x = pulp.LpVariable.dicts("x", (cow_ids, site_ids), cat="Binary")

    # y_ij : 1 if demand i is served by deployment at site j (note: non-overlap constraint ensures each demand at most 1)
    y = pulp.LpVariable.dicts("y", (demand_ids, site_ids), cat="Binary")

    # z_i : 1 if demand i is covered by any deployment
    z = pulp.LpVariable.dicts("z", (demand_ids,), cat="Binary")

    # T_max : maximum deployment completion time (hours)
    T_max = pulp.LpVariable("T_max", lowBound=0, cat="Continuous")

    # Common constraints

    # 1) Each cow at most 1 deployment (some cows may remain unused)
    for k in cow_ids:
        prob += pulp.lpSum([x[k][j] for j in site_ids]) <= 1, f"one_deploy_per_cow_{k}"

    # 2) Each site may have at most 1 cow deployed
    for j in site_ids:
        prob += pulp.lpSum([x[k][j] for k in cow_ids]) <= 1, f"one_cow_per_site_{j}"

    # 3) Non-overlap: each demand at most one serving site
    for i in demand_ids:
        prob += pulp.lpSum([y[i][j] for j in site_ids]) <= 1, f"non_overlap_demand_{i}"

    # 4) y_ij only if there exists a deployed cow at j that can cover i:
    #    y[i,j] <= sum_k cover_ikj * x[k,j]
    for i in demand_ids:
        for j in site_ids:
            # compute set of cows that can cover demand i when deployed at j
            coverable_cows = [k for k in cow_ids if cover_indicator.get((i, j, k), 0) == 1]
            if len(coverable_cows) == 0:
                # If no cow can cover i from j, force y[i,j] = 0
                prob += y[i][j] == 0, f"no_cover_possible_{i}_{j}"
            else:
                prob += y[i][j] <= pulp.lpSum([x[k][j] for k in coverable_cows]), f"y_implies_x_cover_{i}_{j}"

    # 5) z_i <= sum_j y_ij  (if any y assigned then z can be 1)
    for i in demand_ids:
        prob += z[i] <= pulp.lpSum([y[i][j] for j in site_ids]), f"z_def_{i}"

    # 6) Budget constraint (total deployment cost + travel cost <= budget)
    # We'll compute cost expression from travel_matrix and cow base cost.
    total_cost_expr = []
    for k in cow_ids:
        cow_cost = float(next(c for c in cows if c["cow_id"] == k).get("cost_vnd", 0.0))
        for j in site_ids:
            travel_cost_vnd = float(travel_matrix.get((k, j), {}).get("travel_cost_vnd", 0.0))
            total_cost_expr.append(cow_cost * x[k][j] + travel_cost_vnd * x[k][j])
    prob += pulp.lpSum(total_cost_expr) <= budget_max, "budget_constraint"

    # 7) M_max constraint: total number of deployed cows <= M_max
    prob += pulp.lpSum([x[k][j] for k in cow_ids for j in site_ids]) <= M_max, "M_max_constraint"

    # 8) Endurance constraint: disallow assignments where travel_time > endurance of cow
    for k in cow_ids:
        cow_endurance = float(next(c for c in cows if c["cow_id"] == k).get("endurance_hr", 0.0))
        for j in site_ids:
            travel_time_hr = float(travel_matrix.get((k, j), {}).get("travel_time_hr", 0.0))
            if travel_time_hr > cow_endurance:
                # cannot assign cow k to site j
                prob += x[k][j] == 0, f"endurance_violation_{k}_{j}"

    # 9) T_max constraints:
    #    For each possible assignment (k,j): T_max >= (travel_time_hr + setup_time_h) * x[k][j]
    for k in cow_ids:
        for j in site_ids:
            travel_time_hr = float(travel_matrix.get((k, j), {}).get("travel_time_hr", 0.0))
            # (travel_time + setup_time) * x_kj <= T_max  -> T_max >= ...
            prob += T_max >= (travel_time_hr + setup_time_h) * x[k][j], f"Tmax_def_{k}_{j}"

    # Return problem and variable references
    var_dict = {
        "x": x,
        "y": y,
        "z": z,
        "T_max": T_max,
        "demand_ids": demand_ids,
        "site_ids": site_ids,
        "cow_ids": cow_ids,
        "budget_max": budget_max,
        "setup_time_h": setup_time_h
    }
    return prob, var_dict


def extract_solution(var_dict, cows, site_list, demand_list, travel_matrix):
    """
    Extract and summarize solution from variables after solve.
    Returns a dict with assignment, coverage, costs, times.
    """
    x = var_dict["x"]
    y = var_dict["y"]
    z = var_dict["z"]
    T_max_var = var_dict["T_max"]
    demand_ids = var_dict["demand_ids"]
    site_ids = var_dict["site_ids"]
    cow_ids = var_dict["cow_ids"]

    # Assignments: list of (cow_id, site_id) with x=1
    assignments = []
    for k in cow_ids:
        for j in site_ids:
            val = pulp.value(x[k][j])
            if val is not None and round(val) == 1:
                assignments.append((k, j))

    # Served demands: mapping demand_id -> serving site (or None)
    demand_served = {}
    total_pop_served = 0.0
    for i in demand_ids:
        served = False
        for j in site_ids:
            val = pulp.value(y[i][j])
            if val is not None and round(val) == 1:
                demand_served[i] = j
                served = True
                break
        if not served:
            demand_served[i] = None
        pop_i = float(next(d for d in demand_list if d["site_id"] == i).get("pop", 0.0))
        if served:
            total_pop_served += pop_i

    # Total cost and time
    total_travel_cost = 0.0
    total_broadcast_cost = 0.0
    for (k, j) in assignments:
        travel_cost_vnd = float(travel_matrix.get((k, j), {}).get("travel_cost_vnd", 0.0))
        cow_cost = float(next(c for c in cows if c["cow_id"] == k).get("cost_vnd", 0.0))
        total_travel_cost += travel_cost_vnd
        total_broadcast_cost += cow_cost

    total_cost = total_travel_cost + total_broadcast_cost

    # T_max value
    T_max_value = pulp.value(T_max_var)

    summary = {
        "assignments": assignments,
        "demand_served": demand_served,
        "total_pop_served": total_pop_served,
        "total_cost_vnd": total_cost,
        "total_travel_cost_vnd": total_travel_cost,
        "total_broadcast_cost_vnd": total_broadcast_cost,
        "T_max_hr": T_max_value
    }
    return summary
