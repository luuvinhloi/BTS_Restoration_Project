# src/optimization/MILP/solve_milp_full.py
"""
Orchestrator to run the full MILP lexicographic optimization.

Saves:
 - outputs/results/milp_assignments_{solver}.csv
 - outputs/results/milp_solution_summary.json
"""

from pathlib import Path
import pandas as pd
import yaml
import time
import json
import pulp

from src.optimization.MILP.milp_pulp import build_base_problem, haversine_km

# -------------------------
# Data loaders / matrix builders
# -------------------------
def read_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv_safe(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(p)


def load_all(processed_dir):
    """Load I, J, failed BTS, cow dataset, backup power, travel matrices."""
    pd_dir = Path(processed_dir)
    I_csv = pd_dir / "position_I_J" / "I_points.csv"
    J_csv = pd_dir / "position_I_J" / "J_sites.csv"
    failed_bts_csv = pd_dir / "damage_bts" / "failed_bts.csv"
    cow_csv = pd_dir / "cow" / "cow_dataset.csv"
    backup_csv = pd_dir / "backup_power" / "backup_power.csv"
    cow_travel_csv = pd_dir / "travel_cost" / "cow_to_J_sites.csv"
    backup_travel_csv = pd_dir / "travel_cost" / "backup_to_failed_bts.csv"

    I_df = load_csv_safe(I_csv)
    J_df = load_csv_safe(J_csv)
    bts_df = load_csv_safe(failed_bts_csv)
    cows_df = load_csv_safe(cow_csv)
    backup_df = load_csv_safe(backup_csv)
    cow_travel_df = load_csv_safe(cow_travel_csv)
    backup_travel_df = load_csv_safe(backup_travel_csv)

    # Normalize numeric columns
    for col in ["latitude", "longitude", "pop"]:
        if col in I_df.columns:
            I_df[col] = pd.to_numeric(I_df[col], errors="coerce").fillna(0.0)
    for col in ["latitude", "longitude"]:
        if col in J_df.columns:
            J_df[col] = pd.to_numeric(J_df[col], errors="coerce").fillna(0.0)
    if "power_W" in bts_df.columns:
        bts_df["power_W"] = pd.to_numeric(bts_df["power_W"], errors="coerce").fillna(0.0)
    # cows numeric
    for col in ["coverage_radius_m", "cost_vnd", "speed_kmh", "endurance_hr", "lat", "lon", "power_kw"]:
        if col in cows_df.columns:
            cows_df[col] = pd.to_numeric(cows_df[col], errors="coerce").fillna(0.0)
    # backup numeric
    for col in ["resource_amount", "cost_vnd_24h", "power_kw"]:
        if col in backup_df.columns:
            backup_df[col] = pd.to_numeric(backup_df[col], errors="coerce").fillna(0.0)
    # travel numeric
    for col in ["distance_km", "travel_time_hr", "travel_cost_vnd", "total_time_hr", "total_cost_vnd"]:
        if col in cow_travel_df.columns:
            cow_travel_df[col] = pd.to_numeric(cow_travel_df[col], errors="coerce").fillna(0.0)
        if col in backup_travel_df.columns:
            backup_travel_df[col] = pd.to_numeric(backup_travel_df[col], errors="coerce").fillna(0.0)

    return I_df, J_df, bts_df, cows_df, backup_df, cow_travel_df, backup_travel_df


def build_travel_dict(df, key_from, key_to, time_col=None, cost_col=None, dist_col=None):
    """
    Build dict keyed (from_id, to_id) -> {travel_time_hr, travel_cost_vnd, distance_km}
    key_from/key_to specify column names in df.
    """
    d = {}
    for _, r in df.iterrows():
        a = str(r[key_from])
        b = str(r[key_to])
        rec = {}
        if time_col in r:
            rec["travel_time_hr"] = float(r.get(time_col, 0.0))
        elif "travel_time_hr" in r:
            rec["travel_time_hr"] = float(r.get("travel_time_hr", 0.0))
        else:
            rec["travel_time_hr"] = float(r.get("total_time_hr", 0.0))
        rec["travel_cost_vnd"] = float(r.get(cost_col, r.get("travel_cost_vnd", r.get("total_cost_vnd", 0.0))))
        rec["distance_km"] = float(r.get(dist_col, r.get("distance_km", 0.0)))
        d[(a, b)] = rec
    return d


def df_to_list_of_dicts(df, id_col, expected_cols):
    out = []
    for _, r in df.iterrows():
        d = {"site_id": str(r[id_col])}
        for c in expected_cols:
            if c in r:
                d[c] = r[c]
        out.append(d)
    return out


# -------------------------
# Solver orchestration
# -------------------------
def run_lexicographic(params_path, processed_dir, outputs_dir=None):
    params = read_params(params_path)
    I_df, J_df, bts_df, cows_df, backup_df, cow_travel_df, backup_travel_df = load_all(processed_dir)

    # convert to lists/dicts expected by model builder
    I_points = df_to_list_of_dicts(I_df, id_col="site_id", expected_cols=["latitude", "longitude", "pop"])
    J_sites = df_to_list_of_dicts(J_df, id_col="site_id", expected_cols=["latitude", "longitude"])
    BTS_failed = df_to_list_of_dicts(bts_df, id_col="site_id", expected_cols=["latitude", "longitude", "power_W", "coverage_radius_m"])
    # cows list: ensure fields names match expected keys
    cows = []
    for _, r in cows_df.iterrows():
        cows.append({
            "cow_id": str(r.get("cow_id")),
            "base_id": r.get("base_id", ""),
            "coverage_radius_m": float(r.get("coverage_radius_m", 0.0)),
            "endurance_hr": float(r.get("endurance_hr", 0.0)),
            "cost_vnd": float(r.get("cost_vnd", 0.0)),
            "speed_kmh": float(r.get("speed_kmh", 0.0)),
            "lat": float(r.get("lat", 0.0)),
            "lon": float(r.get("lon", 0.0))
        })
    backup_powers = []
    for _, r in backup_df.iterrows():
        backup_powers.append({
            "power_id": str(r.get("power_id")),
            "base_id": r.get("base_id", ""),
            "type": r.get("type", ""),
            # resource_amount used as kW if present; otherwise 0
            "resource_amount": float(r.get("resource_amount", r.get("power_kw", 0.0))),
            "cost_vnd_24h": float(r.get("cost_vnd_24h", r.get("cost_vnd", 0.0))),
            "lat": float(r.get("lat", 0.0)),
            "lon": float(r.get("lon", 0.0))
        })

    # travel dictionaries
    cow_travel = build_travel_dict(cow_travel_df, "cow_id", "site_id", time_col="travel_time_hr", cost_col="travel_cost_vnd", dist_col="distance_km")
    backup_travel = build_travel_dict(backup_travel_df, "power_id", "bts_id", time_col="total_time_hr", cost_col="total_cost_vnd", dist_col="distance_km")

    # Build base problem
    base_prob, var = build_base_problem(I_points, J_sites, BTS_failed, cows, backup_powers, cow_travel, backup_travel, params, solver_name="CBC")

    # select solvers list
    solvers = []
    try:
        _ = pulp.GUROBI_CMD
        solvers.append("GUROBI")
    except Exception:
        print("GUROBI not available; will use CBC.")
    solvers.append("CBC")

    results = []
    for solver_name in solvers:
        print(f"\n=== Running lexicographic with solver {solver_name} ===")
        prob = base_prob.copy()

        # variable shortcuts
        x = var["x"]; w_cow = var["w_cow"]; w_bts = var["w_bts"]; z = var["z"]; u = var["u"]; y = var["y"]; T_max = var["T_max"]
        I_ids = var["I_ids"]; J_ids = var["J_ids"]; B_ids = var["B_ids"]; C_ids = var["C_ids"]; G_ids = var["G_ids"]
        maps = var["maps"]

        # Configure pulp solver instance
        if solver_name == "GUROBI":
            solver = pulp.GUROBI_CMD(timeLimit=int(params.get("milp", {}).get("solver", {}).get("time_limit", 600)), msg=True)
        else:
            solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=int(params.get("milp", {}).get("solver", {}).get("time_limit", 600)))

        # Step 1: maximize covered population
        prob1 = prob.copy()
        prob1.sense = pulp.LpMaximize
        obj1 = pulp.lpSum([float(maps["I"][i]["pop"]) * y[i] for i in I_ids])
        prob1.setObjective(obj1)
        t0 = time.time()
        prob1.solve(solver)
        t1 = time.time()
        status1 = pulp.LpStatus[prob1.status]
        covered_pop = pulp.value(obj1) if pulp.value(obj1) is not None else 0.0
        print(f" Step1 status={status1}, covered_pop={covered_pop:.2f}, time={t1-t0:.2f}s")
        if status1 not in ("Optimal", "Integer Feasible", "Feasible"):
            results.append({"solver": solver_name, "status_step1": status1})
            continue
        opt_covered_pop = covered_pop

        # Step 2: minimize T_max with coverage fixed
        prob2 = prob.copy()
        prob2 += pulp.lpSum([float(maps["I"][i]["pop"]) * y[i] for i in I_ids]) >= opt_covered_pop, "fix_covered_pop"
        prob2.setObjective(T_max)
        prob2.sense = pulp.LpMinimize
        t0 = time.time()
        prob2.solve(solver)
        t1 = time.time()
        status2 = pulp.LpStatus[prob2.status]
        T_max_val = pulp.value(T_max)
        print(f" Step2 status={status2}, T_max={T_max_val}, time={t1-t0:.2f}s")
        if status2 not in ("Optimal", "Integer Feasible", "Feasible"):
            results.append({"solver": solver_name, "status_step1": status1, "status_step2": status2})
            continue
        opt_T_max = T_max_val

        # Step 3: minimize total cost with coverage & T_max fixed
        prob3 = prob.copy()
        prob3 += pulp.lpSum([float(maps["I"][i]["pop"]) * y[i] for i in I_ids]) >= opt_covered_pop, "fix_covered_pop"
        prob3 += T_max <= opt_T_max + 1e-6, "fix_T_max"

        # total cost expression
        total_cost = []
        # cows
        for c in C_ids:
            cow_cost = float(next((cc for cc in cows if cc["cow_id"] == c), {}).get("cost_vnd", 0.0))
            for j in J_ids:
                travel_cost = float(cow_travel.get((c, j), {}).get("travel_cost_vnd", 0.0))
                total_cost.append((cow_cost + travel_cost) * x[c][j])
        # backup
        for g in G_ids:
            g_cost = float(next((gg for gg in backup_powers if gg["power_id"] == g), {}).get("cost_vnd_24h", 0.0))
            for b in B_ids:
                travel_cost = float(backup_travel.get((g, b), {}).get("travel_cost_vnd", 0.0))
                total_cost.append((g_cost + travel_cost) * z[g][b])

        prob3.setObjective(pulp.lpSum(total_cost))
        prob3.sense = pulp.LpMinimize
        t0 = time.time()
        prob3.solve(solver)
        t1 = time.time()
        status3 = pulp.LpStatus[prob3.status]
        print(f" Step3 status={status3}, time={t1-t0:.2f}s")

        # Extract assignments and metrics
        assignments_cow = []
        for c in C_ids:
            for j in J_ids:
                val = pulp.value(prob3.variablesDict().get(f"x_{c}_{j}", None))
                if val is None:
                    # variable name might include double underscore if original id contained special chars
                    val = pulp.value(prob3.variablesDict().get(f"x_{c}__{j}", 0))
                if val is not None and round(val) == 1:
                    assignments_cow.append((c, j))

        assignments_power = []
        for g in G_ids:
            for b in B_ids:
                val = pulp.value(prob3.variablesDict().get(f"z_{g}_{b}", None))
                if val is None:
                    val = pulp.value(prob3.variablesDict().get(f"z_{g}__{b}", 0))
                if val is not None and round(val) == 1:
                    assignments_power.append((g, b))

        # covered population final
        total_pop_served = 0.0
        demand_served = {}
        for i in I_ids:
            yi = pulp.value(prob3.variablesDict().get(f"y_{i}", None))
            if yi is None:
                yi = pulp.value(prob3.variablesDict().get(f"y_{i}", 0))
            if yi is not None and round(yi) == 1:
                total_pop_served += float(maps["I"][i]["pop"])
                demand_served[i] = 1
            else:
                demand_served[i] = 0

        # cost calc
        total_travel_cost = 0.0
        total_fixed_cost = 0.0
        # cows
        for (c, j) in assignments_cow:
            crow = next((cc for cc in cows if cc["cow_id"] == c), {})
            total_fixed_cost += float(crow.get("cost_vnd", 0.0))
            t_cost = float(cow_travel.get((c, j), {}).get("travel_cost_vnd", 0.0))
            total_travel_cost += t_cost
        # backup
        for (g, b) in assignments_power:
            grow = next((gg for gg in backup_powers if gg["power_id"] == g), {})
            total_fixed_cost += float(grow.get("cost_vnd_24h", 0.0))
            t_cost = float(backup_travel.get((g, b), {}).get("travel_cost_vnd", 0.0))
            total_travel_cost += t_cost

        total_cost_all = total_fixed_cost + total_travel_cost

        elapsed = time.time() - t0

        res = {
            "solver": solver_name,
            "status": (status1, status2, status3),
            "optimal_covered_pop": opt_covered_pop,
            "optimal_T_max": opt_T_max,
            "final_total_cost_vnd": total_cost_all,
            "assignments_cow": assignments_cow,
            "assignments_power": assignments_power,
            "demand_served": demand_served,
            "time_elapsed_s": elapsed,
            "total_pop_served": total_pop_served
        }

        # Save outputs
        out_dir = Path(outputs_dir or Path.cwd() / "outputs" / "milp_runs") / f"milp_{solver_name.lower()}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write assignment CSVs
        df_cow = pd.DataFrame(res["assignments_cow"], columns=["cow_id", "site_id"])
        df_power = pd.DataFrame(res["assignments_power"], columns=["power_id", "bts_id"])
        df_cow.to_csv(out_dir / f"assignments_cow_{solver_name}.csv", index=False)
        df_power.to_csv(out_dir / f"assignments_power_{solver_name}.csv", index=False)

        # write summary json (simple)
        summary_path = out_dir / f"summary_{solver_name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        print(f"Solver {solver_name} finished. Results in {out_dir}")
        results.append(res)

    # Write canonical results to outputs/results
    results_out = Path.cwd() / "outputs" / "results"
    results_out.mkdir(parents=True, exist_ok=True)
    if len(results) > 0:
        # choose last solver's result as canonical (or choose CBC preferentially)
        chosen = None
        for r in results:
            if r["solver"].lower() == "cbc":
                chosen = r
                break
        if chosen is None:
            chosen = results[0]
        # save as canonical milp_assignments.csv
        df_cow = pd.DataFrame(chosen["assignments_cow"], columns=["cow_id", "site_id"])
        df_power = pd.DataFrame(chosen["assignments_power"], columns=["power_id", "bts_id"])
        # Combine into single file with type column
        rows = []
        for _, row in df_cow.iterrows():
            rows.append({"type": "cow", "id_from": row["cow_id"], "id_to": row["site_id"]})
        for _, row in df_power.iterrows():
            rows.append({"type": "power", "id_from": row["power_id"], "id_to": row["bts_id"]})
        pd.DataFrame(rows).to_csv(results_out / "milp_assignments.csv", index=False)

        # summary
        with open(results_out / "milp_solution_summary.json", "w", encoding="utf-8") as f:
            json.dump(chosen, f, indent=2, ensure_ascii=False)

        print(f"Wrote canonical results to {results_out}")

    return results


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    config_path = PROJECT_ROOT / "config" / "params.yaml"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    out_dir = PROJECT_ROOT / "outputs" / "milp_runs"
    run_lexicographic(str(config_path), str(processed_dir), str(out_dir))
