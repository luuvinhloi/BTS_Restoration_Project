# FILE: src/optimization/GA_PSO/ga_pso_hybrid_new.py
"""
Main GA-PSO Hybrid runner (uses the 4 helper modules).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import json
import time

from .utils import read_params
from .cover_travel import load_inputs, build_cover_indicator_array
from .eval_repair import evaluate_solution, repair_solution, compute_coverage_from_solution
from .operators import initialize_population, mutation_discrete, uniform_crossover_arrays

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def ga_pso_hybrid_main(processed_data_dir: str, outputs_dir: str, config: dict):
    t0 = time.time()
    processed = Path(processed_data_dir)
    raw = processed.parent / "raw"
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    params_path = Path(config.get("params_path", Path.cwd() / "config" / "params.yaml"))
    params = read_params(params_path)

    # parameters (priority: config dict then params.yaml)
    pop_size = int(config.get("pop_size", params.get("ga_pso", {}).get("pop_size", 80)))
    max_iter = int(config.get("max_iter", params.get("ga_pso", {}).get("max_iter", 200)))
    mutation_rate = float(config.get("mutation_rate", params.get("ga_pso", {}).get("mutation_rate", 0.08)))
    ga_period = int(config.get("ga_period", params.get("ga_pso", {}).get("ga_period", 10)))
    elitism = float(config.get("elitism", params.get("ga_pso", {}).get("elitism", 0.1)))
    budget_max = float(config.get("budget_max", params.get("budget_max", 5e8)))
    budget_hard = bool(config.get("budget_hard", True))
    w_time = float(config.get("w_time", params.get("ga_pso", {}).get("weights", {}).get("w_time", 0.2)))
    w_cost = float(config.get("w_cost", params.get("ga_pso", {}).get("weights", {}).get("w_cost", 0.4)))
    seed = int(config.get("seed", params.get("seed", 42)))
    default_setup_time_h = float(params.get("default_setup_time_h", config.get("default_setup_time_h", 0.5)))
    np.random.seed(seed)

    logging.info(f"GA-PSO config pop={pop_size}, iter={max_iter}, budget={budget_max}")

    # Load data
    I_df, J_df, cow_df, travel_dict = load_inputs(processed, raw, params)
    n_cows = len(cow_df)
    n_I = len(I_df)
    n_J = len(J_df)
    logging.info(f"Loaded I({n_I}), J({n_J}), cows({n_cows})")

    # Build cover arr
    logging.info("Building cover 3D indicator (cow, I, J)...")
    cover_arr = build_cover_indicator_array(I_df, J_df, cow_df)
    logging.info("Cover array built.")

    # depot capacities
    depot_capacity_map = {}
    for base_id, g in cow_df.groupby("base_id"):
        depot_capacity_map[str(base_id)] = int(len(g))

    # time normalization baseline
    all_travel_times = [v.get("travel_time_hr", 0.0) for v in travel_dict.values()] if travel_dict else [0.0]
    time_max_norm = max(max(all_travel_times) + default_setup_time_h, 1.0)

    # initialize population
    population = initialize_population(pop_size, n_cows, n_J, cover_arr, cow_df, J_df)
    pop_array = np.array(population)

    pbest = [pop_array[i].copy() for i in range(pop_size)]
    pbest_fit = [1e9] * pop_size
    best_global = None
    best_global_fit = 1e9
    fitness_history = []

    logging.info("Start optimization loop")
    for it in range(max_iter):
        iter_best_fit = 1e9
        iter_best_idx = -1
        # evaluate + repair + update pbest
        for i in range(pop_size):
            Xi = pop_array[i].copy()
            Xi = repair_solution(Xi, cover_arr, cow_df, J_df, travel_dict, depot_capacity_map, I_df, budget_max, default_setup_time_h, budget_hard=budget_hard, enforce_site_unique=True)
            f, cov, cost, makespan, used, covered_mask = evaluate_solution(Xi, cover_arr, cow_df, I_df, J_df, travel_dict, default_setup_time_h, budget_max, time_max_norm, w_time, w_cost)
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = Xi.copy()
            if f < iter_best_fit:
                iter_best_fit = f
                iter_best_idx = i
            pop_array[i] = Xi
        # update global best
        if iter_best_idx >= 0:
            candidate = pop_array[iter_best_idx].copy()
            if iter_best_fit < best_global_fit:
                best_global_fit = iter_best_fit
                best_global = candidate.copy()
        # generate new pop via DPSO-like operators
        new_pop = []
        for i in range(pop_size):
            Xi = pop_array[i].copy()
            X_mut = mutation_discrete(Xi, n_J, mutation_rate)
            X_p = uniform_crossover_arrays(X_mut, pbest[i])
            if best_global is not None:
                X_g = uniform_crossover_arrays(X_p, best_global)
            else:
                X_g = X_p
            X_rep = repair_solution(X_g, cover_arr, cow_df, J_df, travel_dict, depot_capacity_map, I_df, budget_max, default_setup_time_h, budget_hard=budget_hard, enforce_site_unique=True)
            new_pop.append(X_rep)
        new_pop = np.array(new_pop)
        # elitism
        elite_n = max(1, int(elitism * pop_size))
        elite_indices = np.argsort(pbest_fit)[:elite_n]
        for k, idx in enumerate(elite_indices):
            new_pop[k] = pbest[idx].copy()
        pop_array = new_pop
        # periodic GA
        if (it % ga_period) == 0 and pop_size >= 2:
            parents_idx = np.random.choice(pop_size, size=2, replace=False)
            child = uniform_crossover_arrays(pop_array[parents_idx[0]], pop_array[parents_idx[1]])
            child = mutation_discrete(child, n_J, mutation_rate)
            child = repair_solution(child, cover_arr, cow_df, J_df, travel_dict, depot_capacity_map, I_df, budget_max, default_setup_time_h, budget_hard=budget_hard, enforce_site_unique=True)
            worst_idx = int(np.argmax(pbest_fit))
            pop_array[worst_idx] = child

        fitness_history.append(best_global_fit)
        if it % 10 == 0:
            logging.info(f"Iter {it}/{max_iter} best_fit {best_global_fit:.6f}")

    runtime = time.time() - t0
    logging.info(f"Optimization finished. best_fit={best_global_fit:.6f} runtime={runtime:.1f}s")

    # prepare outputs (milp-like)
    if best_global is None:
        best_global = np.zeros(n_cows, dtype=int)

    np.save(outputs / "ga_pso_best_solution.npy", best_global)
    np.save(outputs / "ga_pso_fitness_history.npy", np.array(fitness_history))

    # build assignments dataframe similar to MILP writer
    rows = []
    for p_idx, site in enumerate(best_global):
        site_idx = int(site)
        assigned_site_id = None
        travel_cost_vnd = 0.0
        travel_time_hr = 0.0
        if site_idx > 0 and 1 <= site_idx <= n_J:
            site_id = str(J_df.iloc[site_idx - 1]["site_id"])
            assigned_site_id = site_id
            cow_id = str(cow_df.iloc[p_idx]["cow_id"])
            tinfo = travel_dict.get((cow_id, site_id), {})
            travel_cost_vnd = float(tinfo.get("travel_cost_vnd", 0.0))
            travel_time_hr = float(tinfo.get("travel_time_hr", 0.0))
            assigned_site_index = int(site_idx)
        else:
            assigned_site_index = None

        cow_row = cow_df.iloc[p_idx]
        fixed_cost = float(cow_row.get("cost_vnd", 0.0))
        total_cost_vnd = fixed_cost + travel_cost_vnd
        setup_time_h = default_setup_time_h
        deployment_time_hr = travel_time_hr + setup_time_h

        rows.append({
            "cow_id": cow_row.get("cow_id"),
            "type": cow_row.get("type"),
            "base_id": cow_row.get("base_id"),
            "assigned_site_index": assigned_site_index,
            "assigned_site_id": assigned_site_id,
            "coverage_radius_m": cow_row.get("coverage_radius_m"),
            "cost_vnd": fixed_cost,
            "travel_cost_vnd": travel_cost_vnd,
            "total_cost_vnd": total_cost_vnd,
            "travel_time_hr": travel_time_hr,
            "setup_time_h": setup_time_h,
            "deployment_time_hr": deployment_time_hr
        })

    assign_df = pd.DataFrame(rows)
    assign_path = outputs / "ga_pso_assignments.csv"
    assign_df.to_csv(assign_path, index=False)

    # compute coverage precisely
    covered_mask = np.zeros(n_I, dtype=bool)
    for p in range(n_cows):
        site = int(best_global[p])
        if site == 0:
            continue
        j_idx = site - 1
        covered_mask = covered_mask | cover_arr[p, :, j_idx]
    total_pop_on_I = float(I_df["pop"].sum()) if "pop" in I_df.columns else 0.0
    covered_pop = float((I_df["pop"].values * covered_mask).sum()) if total_pop_on_I > 0 else 0.0
    coverage_pct = round((covered_pop / total_pop_on_I * 100) if total_pop_on_I > 0 else 0.0, 2)

    total_travel_cost = float(assign_df["travel_cost_vnd"].sum())
    total_broadcast_cost = float(assign_df["cost_vnd"].sum())
    total_cost_all = total_travel_cost + total_broadcast_cost
    max_time_hours = float(assign_df["deployment_time_hr"].max()) if not assign_df.empty else 0.0
    num_COW_used = int((assign_df["assigned_site_id"].notnull()).sum())
    chosen_site_ids = list(assign_df["assigned_site_id"].dropna().unique())

    summary = {
        "num_COW_used": num_COW_used,
        "chosen_site_ids": chosen_site_ids,
        "coverage_pct_on_I_points": coverage_pct,
        "covered_pop_on_I_points": covered_pop,
        "total_pop_on_I_points": total_pop_on_I,
        "total_fixed_cost_vnd": total_broadcast_cost,
        "total_travel_cost_vnd": total_travel_cost,
        "total_cost_all_vnd": total_cost_all,
        "max_time_hours": max_time_hours,
        "runtime_s": runtime
    }
    summary_path = outputs / "ga_pso_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # copy canonical outputs to outputs/results
    try:
        central = Path.cwd() / "outputs" / "results"
        central.mkdir(parents=True, exist_ok=True)
        assign_df.to_csv(central / "ga_pso_assignments.csv", index=False)
        with open(central / "ga_pso_solution_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        np.save(central / "ga_pso_best_solution.npy", best_global)
        np.save(central / "ga_pso_fitness_history.npy", np.array(fitness_history))
    except Exception as e:
        logging.warning(f"Failed to copy canonical outputs: {e}")

    logging.info(f"Outputs written to {outputs} and outputs/results")
    return summary

if __name__ == "__main__":
    # quick local run
    import argparse
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--outputs", default=str(PROJECT_ROOT / "outputs" / "ga_pso_run_01"))
    args = parser.parse_args()
    cfg = {"params_path": str(PROJECT_ROOT / "config" / "params.yaml")}
    s = ga_pso_hybrid_main(args.processed, args.outputs, cfg)
    print(s)
