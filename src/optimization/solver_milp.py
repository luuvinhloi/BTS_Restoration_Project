# src/optimization/solve_milp.py
"""
Orchestrator for MILP lexicographic optimization using PuLP.

- Reads inputs:
    data/processed/J_sites.csv
    data/raw/cow_dataset.csv
    data/processed/travel_cost_matrix_A.csv
    config params.yaml

- Builds cover indicators, travel matrix dictionaries.
- Runs lexicographic (3-step) optimization with two solvers (Gurobi and CBC).
- Saves results to outputs/ directory and prints summary.

Requirements:
- pulp
- pandas
- pyyaml
"""

import os
from pathlib import Path
import pandas as pd
import yaml
import time
import json
import pulp
from src.optimization.milp_pulp import (
    haversine_km, build_base_problem, extract_solution
)

# --- Utility functions ---------------------------------------------------


def read_params(params_path):
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(base_dir):
    """
    Expect files:
      - data/raw/cow_dataset.csv
      - data/processed/J_sites.csv
      - data/processed/travel_cost_matrix_A.csv
    """
    base_dir = Path(base_dir)
    cow_csv = base_dir.parent / "raw" / "cow_dataset.csv"
    j_sites_csv = base_dir / "J_sites.csv"
    travel_csv = base_dir / "travel_cost_matrix_A.csv"

    cows_df = pd.read_csv(cow_csv, dtype=str).fillna("")
    # coerce numeric where needed
    numeric_cols = ["coverage_radius_m", "cost_vnd", "speed_kmh", "endurance_hr", "lat", "lon"]
    for col in numeric_cols:
        if col in cows_df.columns:
            cows_df[col] = pd.to_numeric(cows_df[col], errors="coerce").fillna(0.0)

    sites_df = pd.read_csv(j_sites_csv, dtype=str).fillna("")
    # coerce numeric pop/lat/lon
    if "pop" in sites_df.columns:
        sites_df["pop"] = pd.to_numeric(sites_df["pop"], errors="coerce").fillna(0.0)
    for col in ["latitude", "longitude"]:
        if col in sites_df.columns:
            sites_df[col] = pd.to_numeric(sites_df[col], errors="coerce").fillna(0.0)

    travel_df = pd.read_csv(travel_csv, dtype=str).fillna("")
    # numeric conversions
    for col in ["distance_km", "travel_time_hr", "travel_cost_vnd"]:
        if col in travel_df.columns:
            travel_df[col] = pd.to_numeric(travel_df[col], errors="coerce").fillna(0.0)

    return cows_df, sites_df, travel_df


def build_travel_matrix(travel_df):
    """
    Build dict keyed by (cow_id, site_id) -> {distance_km, travel_time_hr, travel_cost_vnd}
    """
    travel = {}
    for _, row in travel_df.iterrows():
        k = str(row["cow_id"])
        j = str(row["site_id"])
        travel[(k, j)] = {
            "distance_km": float(row.get("distance_km", 0.0)),
            "travel_time_hr": float(row.get("travel_time_hr", 0.0)),
            "travel_cost_vnd": float(row.get("travel_cost_vnd", 0.0))
        }
    return travel


def build_lists(cows_df, sites_df):
    # cows list of dicts
    cows = []
    for _, r in cows_df.iterrows():
        cows.append({
            "cow_id": str(r["cow_id"]),
            "base_id": r.get("base_id", ""),
            "base_name": r.get("base_name", ""),
            "type": r.get("type", ""),
            "lat": float(r.get("lat", 0.0)),
            "lon": float(r.get("lon", 0.0)),
            "coverage_radius_m": float(r.get("coverage_radius_m", 0.0)),
            "power_kw": float(r.get("power_kw", 0.0)) if "power_kw" in r else 0.0,
            "speed_kmh": float(r.get("speed_kmh", 0.0)),
            "endurance_hr": float(r.get("endurance_hr", 0.0)),
            "cost_vnd": float(r.get("cost_vnd", 0.0)),
            "assigned_region": r.get("assigned_region", "")
        })

    # site / demand list
    sites = []
    for _, r in sites_df.iterrows():
        sites.append({
            "site_id": str(r["site_id"]),
            "i_ref": r.get("i_ref", ""),
            "latitude": float(r.get("latitude", 0.0)),
            "longitude": float(r.get("longitude", 0.0)),
            "pop": float(r.get("pop", 0.0)),
            "priority_category": r.get("priority_category", ""),
            "priority_weight": float(r.get("priority_weight", 0.0)) if "priority_weight" in r else 0.0
        })

    return cows, sites


def build_cover_indicator(cows, sites):
    """
    Build cover_indicator dict keyed (demand_id, deploy_site_id, cow_id) -> 1/0
    We assume deploy_site is same set as sites. A cow deployed at site j covers demand i if
    distance(site_i, site_j) * 1000 <= cow.coverage_radius_m
    """
    cover = {}
    for i in sites:
        for j in sites:
            d_km = haversine_km(i["latitude"], i["longitude"], j["latitude"], j["longitude"])
            for c in cows:
                cover[(i["site_id"], j["site_id"], c["cow_id"])] = 1 if (d_km * 1000.0) <= c["coverage_radius_m"] else 0
    return cover


def _write_assignments_csv(assignments, cows_df, sites_df, out_path):
    """
    assignments: list of tuples (cow_id, site_id)
    Write CSV with columns: cow_id,type,base_id,assigned_site_index,assigned_site_id,coverage_radius_m
    """
    rows = []
    for (cow_id, site_id) in assignments:
        cow_row = cows_df[cows_df["cow_id"] == cow_id]
        if cow_row.empty:
            cow_type = ""
            base_id = ""
            coverage_radius_m = ""
        else:
            cow_type = cow_row.iloc[0].get("type", "")
            base_id = cow_row.iloc[0].get("base_id", "")
            coverage_radius_m = cow_row.iloc[0].get("coverage_radius_m", "")
        # find site index in sites_df by site_id or by i_ref
        idx_match = sites_df[sites_df["site_id"] == site_id]
        if idx_match.empty:
            idx_match = sites_df[sites_df["i_ref"] == site_id]
        assigned_site_index = ""
        if not idx_match.empty:
            assigned_site_index = int(idx_match.index[0]) + 1
        rows.append({
            "cow_id": cow_id,
            "type": cow_type,
            "base_id": base_id,
            "assigned_site_index": assigned_site_index,
            "assigned_site_id": site_id,
            "coverage_radius_m": coverage_radius_m
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def _build_summary_json(result, cows_df, sites_df, travel_matrix, out_json_path):
    """
    Build milp_solution_summary.json content from 'result' object returned by solver run.
    result must contain:
      - assignments: list of (cow_id, site_id)
      - demand_served: dict demand->site
      - total_pop_served (optional)
      - final_total_travel_cost_vnd (optional)
      - final_total_broadcast_cost_vnd (optional)
      - T_max_hr or optimal_T_max
    """
    assignments = result.get("assignments", [])
    demand_served = result.get("demand_served", {})
    total_pop_served = float(result.get("total_pop_served", 0.0))
    # total pop on I_points (use sites_df pop)
    total_pop_on_I_points = float(sites_df["pop"].sum()) if "pop" in sites_df.columns else 0.0

    # covered pop: sum pop for served demand ids (demand_served mapping)
    covered_pop = 0.0
    chosen_site_set = set()
    for i, served_j in (demand_served.items() if isinstance(demand_served, dict) else []):
        if served_j is not None and served_j != "":
            try:
                pop_i = float(sites_df[sites_df["site_id"] == i]["pop"].iloc[0])
            except Exception:
                pop_i = 0.0
            covered_pop += pop_i

    # fallback if result["total_pop_served"] has sensible value
    if total_pop_served > 0 and covered_pop == 0:
        covered_pop = total_pop_served

    # chosen site ids from assignments
    chosen_sites = []
    for (cow_id, site_id) in assignments:
        if site_id not in chosen_sites:
            chosen_sites.append(site_id)
        chosen_site_set.add(site_id)

    # compute costs from travel_matrix and cows_df if available
    total_broadcast_cost = 0.0
    total_travel_cost = 0.0
    for (cow_id, site_id) in assignments:
        # broadcast cost
        crow = cows_df[cows_df["cow_id"] == cow_id]
        if not crow.empty and "cost_vnd" in crow.columns:
            total_broadcast_cost += float(crow.iloc[0]["cost_vnd"])
        # travel cost from travel_matrix
        t = travel_matrix.get((cow_id, site_id), {})
        total_travel_cost += float(t.get("travel_cost_vnd", 0.0))

    total_cost_all = total_broadcast_cost + total_travel_cost

    # T_max
    T_max = result.get("T_max_hr", result.get("optimal_T_max", None))

    # coverage percent on I_points (using numeric total_pop_on_I_points)
    coverage_pct = round((covered_pop / total_pop_on_I_points * 100) if total_pop_on_I_points > 0 else 0.0, 2)

    summary = {
        "num_COW_used": len(assignments),
        "chosen_site_ids": chosen_sites,
        "coverage_pct_on_I_points": coverage_pct,
        "covered_pop_on_I_points": covered_pop,
        "total_pop_on_I_points": total_pop_on_I_points,
        "total_fixed_cost_vnd": total_broadcast_cost,
        "total_travel_cost_vnd": total_travel_cost,
        "total_cost_all_vnd": total_cost_all,
        "total_time_hours": float(T_max) if T_max is not None else None,
        # note: raster-based metrics will be computed by compute_population_coverage and merged later
    }

    # write json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def run_lexicographic_for_solver(solver_name, cows, sites, travel_matrix, cover_indicator, params, out_dir):
    """
    Execute lexicographic 3-step optimization using PuLP with the specified solver.
    solver_name: "GUROBI" or "CBC"
    Returns: result dict (same structure as before)
    """
    print(f"\n--- Running lexicographic optimization with solver: {solver_name} ---")
    start_all = time.time()

    # Build base problem (variables + common constraints). We will copy for each step to avoid stale objectives.
    base_prob, var_dict = build_base_problem(
        demand_list=sites,
        site_list=sites,
        cows=cows,
        travel_matrix=travel_matrix,
        cover_indicator=cover_indicator,
        params=params,
        solver_name=solver_name
    )

    demand_ids = var_dict["demand_ids"]
    site_ids = var_dict["site_ids"]
    cow_ids = var_dict["cow_ids"]
    x_vars = var_dict["x"]
    y_vars = var_dict["y"]
    z_vars = var_dict["z"]
    T_max_var = var_dict["T_max"]
    setup_time_h = var_dict["setup_time_h"]

    # Helper: create solver object
    if solver_name.upper() == "GUROBI":
        solver = pulp.GUROBI_CMD(timeLimit=int(params.get("milp", {}).get("solver", {}).get("time_limit", 600)), msg=True)
    else:
        # Use CBC
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=int(params.get("milp", {}).get("solver", {}).get("time_limit", 600)))

    # --- Step 1: Maximize covered population ---
    prob1 = base_prob.copy()
    prob1.sense = pulp.LpMaximize
    objective_expr_1 = pulp.lpSum([float(next(s for s in sites if s["site_id"] == i)["pop"]) * z_vars[i] for i in demand_ids])
    prob1.setObjective(objective_expr_1)

    print("Solving Step 1 (maximize covered population)...")
    t0 = time.time()
    prob1.solve(solver)
    t1 = time.time()
    status1 = pulp.LpStatus[prob1.status]
    covered_pop = pulp.value(objective_expr_1)
    print(f" Step1 status: {status1}, covered_pop={covered_pop:.2f}, time={t1-t0:.2f}s")

    if status1 not in ("Optimal", "Integer Feasible", "Feasible"):
        print("Step1 did not find feasible solution. Aborting this solver run.")
        return {"solver": solver_name, "status_step1": status1}

    optimal_covered_pop = covered_pop

    # --- Step 2: Minimize T_max subject to covered_pop >= optimal_covered_pop ---
    prob2 = base_prob.copy()
    prob2 += pulp.lpSum([float(next(s for s in sites if s["site_id"] == i)["pop"]) * var_dict["z"][i] for i in demand_ids]) >= optimal_covered_pop, "fix_covered_pop"
    prob2.setObjective(var_dict["T_max"])
    prob2.sense = pulp.LpMinimize

    print("Solving Step 2 (minimize T_max subject to coverage)...")
    t0 = time.time()
    prob2.solve(solver)
    t1 = time.time()
    status2 = pulp.LpStatus[prob2.status]
    T_max_val = pulp.value(var_dict["T_max"])
    print(f" Step2 status: {status2}, T_max={T_max_val}, time={t1-t0:.2f}s")

    if status2 not in ("Optimal", "Integer Feasible", "Feasible"):
        print("Step2 did not find feasible solution. Aborting this solver run.")
        return {"solver": solver_name, "status_step1": status1, "status_step2": status2}

    optimal_T_max = T_max_val

    # --- Step 3: Minimize total cost subject to coverage and T_max fixed ---
    prob3 = base_prob.copy()
    prob3 += pulp.lpSum([float(next(s for s in sites if s["site_id"] == i)["pop"]) * var_dict["z"][i] for i in demand_ids]) >= optimal_covered_pop, "fix_covered_pop"
    prob3 += var_dict["T_max"] <= optimal_T_max + 1e-6, "fix_T_max"

    total_cost_expr = []
    for k in cow_ids:
        cow_cost = float(next(c for c in cows if c["cow_id"] == k).get("cost_vnd", 0.0))
        for j in site_ids:
            travel_cost_vnd = float(travel_matrix.get((k, j), {}).get("travel_cost_vnd", 0.0))
            total_cost_expr.append((cow_cost + travel_cost_vnd) * x_vars[k][j])

    prob3.setObjective(pulp.lpSum(total_cost_expr))
    prob3.sense = pulp.LpMinimize

    print("Solving Step 3 (minimize total cost subject to coverage and T_max)...")
    t0 = time.time()
    prob3.solve(solver)
    t1 = time.time()
    status3 = pulp.LpStatus[prob3.status]
    print(f" Step3 status: {status3}, time={t1-t0:.2f}s")

    # Extract assignments from prob3
    assignments = []
    for k in cow_ids:
        for j in site_ids:
            try:
                val = pulp.value(prob3.variablesDict()[f"x_{k}_{j}"])
            except KeyError:
                try:
                    val = pulp.value(prob3.variablesDict()[f"x_{k}__{j}"])
                except Exception:
                    val = None
            if val is not None and round(val) == 1:
                assignments.append((k, j))

    # Served demands
    demand_served = {}
    total_pop_served = 0.0
    for i in demand_ids:
        demand_served[i] = None
        for j in site_ids:
            try:
                val = pulp.value(prob3.variablesDict()[f"y_{i}_{j}"])
            except KeyError:
                try:
                    val = pulp.value(prob3.variablesDict()[f"y_{i}__{j}"])
                except Exception:
                    val = None
            if val is not None and round(val) == 1:
                demand_served[i] = j
                try:
                    pop_i = float(next(s for s in sites if s["site_id"] == i)["pop"])
                except Exception:
                    pop_i = 0.0
                total_pop_served += pop_i
                break

    # Compute costs
    total_travel_cost = 0.0
    total_broadcast_cost = 0.0
    for (k, j) in assignments:
        travel_cost_vnd = float(travel_matrix.get((k, j), {}).get("travel_cost_vnd", 0.0))
        cow_cost = float(next(c for c in cows if c["cow_id"] == k).get("cost_vnd", 0.0))
        total_travel_cost += travel_cost_vnd
        total_broadcast_cost += cow_cost
    total_cost = total_travel_cost + total_broadcast_cost

    # T_max value
    T_max_val_final = None
    try:
        T_max_val_final = pulp.value(prob3.variablesDict()["T_max"])
    except Exception:
        if "T_max" in prob3.variablesDict():
            T_max_val_final = pulp.value(prob3.variablesDict()["T_max"])
        else:
            T_max_val_final = None

    elapsed = time.time() - start_all

    result = {
        "solver": solver_name,
        "status": (status1, status2, status3),
        "optimal_covered_pop": optimal_covered_pop,
        "optimal_T_max": optimal_T_max,
        "final_total_cost_vnd": total_cost,
        "final_total_travel_cost_vnd": total_travel_cost,
        "final_total_broadcast_cost_vnd": total_broadcast_cost,
        "assignments": assignments,
        "demand_served": demand_served,
        "T_max_hr": T_max_val_final,
        "time_elapsed_s": elapsed,
        "total_pop_served": total_pop_served
    }

    # Save outputs in run folder (keeps previous behaviour)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # assignments simple CSV
    assign_df_simple = pd.DataFrame(result["assignments"], columns=["cow_id", "site_id"])
    assign_df_simple.to_csv(out_path / f"assignments_{solver_name}.csv", index=False)
    # save summary yaml (as before)
    try:
        import yaml as _yaml
        with open(out_path / f"summary_{solver_name}.yaml", "w", encoding="utf-8") as f:
            _yaml.safe_dump(result, f, allow_unicode=True)
    except Exception:
        pass

    print(f"Step results saved to {out_path}")
    return result


def main_solve(config_path, processed_data_dir, outputs_dir=None):
    """
    Main entrypoint.
    - config_path: path to params.yaml
    - processed_data_dir: directory containing cow_dataset.csv, J_sites.csv, travel_cost_matrix_A.csv
    - outputs_dir: directory to put outputs (default: ./outputs/milp_runs)
    """
    if outputs_dir is None:
        outputs_dir = Path.cwd() / "outputs" / "milp_runs"
    else:
        outputs_dir = Path(outputs_dir)

    params = read_params(config_path)
    cows_df, sites_df, travel_df = load_data(processed_data_dir)
    travel_matrix = build_travel_matrix(travel_df)
    cows, sites = build_lists(cows_df, sites_df)
    cover_indicator = build_cover_indicator(cows, sites)

    # Run with Gurobi and CBC (if both available)
    solvers = []
    try:
        pu = pulp.GUROBI_CMD
        solvers.append("GUROBI")
    except Exception:
        print("GUROBI not available in this environment (PuLP). Skipping GUROBI run.")
    solvers.append("CBC")

    results = []
    for s in solvers:
        out_dir_run = Path(outputs_dir) / f"milp_{s.lower()}"
        res = run_lexicographic_for_solver(s, cows, sites, travel_matrix, cover_indicator, params, out_dir_run)
        results.append((s, res))

    # Summarize comparison
    print("\n=== Comparison summary ===")
    for s, r in results:
        print(f"Solver: {r.get('solver')}, covered_pop={r.get('optimal_covered_pop')}, T_max={r.get('optimal_T_max')}, total_cost={r.get('final_total_cost_vnd')}, time={r.get('time_elapsed_s'):.2f}s")

    # --- EXPORT results that downstream steps expect (outputs/results/) ---
    # Prepare outputs/results dir
    project_outputs_results = Path.cwd() / "outputs" / "results"
    project_outputs_results.mkdir(parents=True, exist_ok=True)

    # For each solver: write milp_assignments_{solver}.csv (rich) in outputs/results
    saved_result_map = {}
    for solver_name, res in results:
        solver_lower = solver_name.lower()
        assignments = res.get("assignments", [])
        out_assign_path = project_outputs_results / f"milp_assignments_{solver_lower}.csv"
        # write enriched csv using cows_df and sites_df
        df_assign = _write_assignments_csv(assignments, cows_df, sites_df, out_assign_path)
        saved_result_map[solver_lower] = {
            "assign_df": df_assign,
            "result": res,
            "assign_path": out_assign_path
        }
        print(f"Wrote solver-specific assignments to {out_assign_path}")

    # Choose which solver to expose as the canonical MILP result used by compute_population_coverage.py and simulation_scenario.py
    # Strategy: prefer CBC if available, else GUROBI
    chosen_solver_key = None
    if "cbc" in saved_result_map:
        chosen_solver_key = "cbc"
    elif "gurobi" in saved_result_map:
        chosen_solver_key = "gurobi"

    if chosen_solver_key is not None:
        chosen = saved_result_map[chosen_solver_key]
        # also write a canonical milp_assignments.csv (simple name)
        canonical_assign_path = project_outputs_results / "milp_assignments.csv"
        # copy DataFrame to canonical file
        chosen["assign_df"].to_csv(canonical_assign_path, index=False)
        print(f"Wrote canonical assignments to {canonical_assign_path}")

        # build and write milp_solution_summary.json (used by simulation_scenario.py)
        canonical_summary_path = project_outputs_results / "milp_solution_summary.json"
        summary_obj = _build_summary_json(chosen["result"], cows_df, sites_df, travel_matrix, canonical_summary_path)
        print(f"Wrote canonical solution summary to {canonical_summary_path}")
    else:
        print("No solver produced assignments; skipping creation of canonical outputs for compute_population_coverage/simulation.")

    return [r for (_, r) in results]


if __name__ == "__main__":
    # For direct run (useful for testing)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    config_path = PROJECT_ROOT / "config" / "params.yaml"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    outputs_dir = PROJECT_ROOT / "outputs" / "milp_runs"
    main_solve(str(config_path), str(processed_dir), outputs_dir)
