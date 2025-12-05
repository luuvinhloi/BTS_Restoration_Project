# FILE: src/optimization/GA_PSO/ga_pso_hybrid_new.py
"""
GA-PSO hybrid runner (COW + Backup Power).
Entrypoint for method 2 as requested.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import json
import time

from .utils import read_params, load_roads_graph, point_flood_depth, nearest_nonflood_site
from .cover_travel import load_inputs, build_cover_indicator_array, build_travel_dicts
from .eval_repair import (evaluate_solution, repair_solution, compute_coverage_from_solution,
                          build_power_index_maps)
from .operators import (initialize_population, mutation_discrete, uniform_crossover_arrays,
                        decode_solution_to_assignments)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ga_pso_hybrid_main(processed_data_dir: str, outputs_dir: str, config: dict):
    t0 = time.time()
    processed = Path(processed_data_dir)
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    params_path = Path(config.get("params_path", Path.cwd() / "config" / "params.yaml"))
    params = read_params(params_path)

    # parameters
    pop_size = int(config.get("pop_size", params.get("ga_pso", {}).get("pop_size", 100)))
    max_iter = int(config.get("max_iter", params.get("ga_pso", {}).get("max_iter", 200)))
    mutation_rate = float(config.get("mutation_rate", params.get("ga_pso", {}).get("mutation_rate", 0.08)))
    ga_period = int(config.get("ga_period", params.get("ga_pso", {}).get("ga_period", 10)))
    elitism = float(config.get("elitism", params.get("ga_pso", {}).get("elitism", 0.1)))
    budget_max = float(config.get("budget_max", params.get("budget_max", 1e19)))
    budget_hard = bool(config.get("budget_hard", True))
    w_time = float(config.get("w_time", params.get("ga_pso", {}).get("weights", {}).get("w_time", 0.2)))
    w_cost = float(config.get("w_cost", params.get("ga_pso", {}).get("weights", {}).get("w_cost", 0.4)))
    coverage_weight = float(params.get("ga_pso", {}).get("coverage_weight", 0.5))
    seed = int(config.get("seed", params.get("seed", 42)))
    default_setup_time_h = float(params.get("default_setup_time_h", config.get("default_setup_time_h", 0.5)))
    np.random.seed(seed)

    logging.info(f"GA-PSO config pop={pop_size}, iter={max_iter}, budget={budget_max}")

    # Load inputs (I, J, cows, backup_powers, failed_bts, travel files, roads, flood raster)
    I_df, J_df, cow_df, power_df, failed_bts_df = load_inputs(processed, params)
    # travel dicts
    cow_travel_dict, power_travel_dict = build_travel_dicts(processed)
    # read roads graph (for passability checks) and flood raster for J checks
    roads_graph = load_roads_graph(processed / "road" / "roads_flooded.graphml")
    flood_path = processed / "flood" / "flood_depth_combined_B_clean.tif"

    # ensure J sites not in high flood: if flooded >=0.5m -> find nearest replacement
    J_df = J_df.copy().reset_index(drop=True)
    for idx, row in J_df.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        depth = point_flood_depth(flood_path, lat, lon)
        if depth is not None and depth >= 0.5:
            # find replacement J (nearest) that is not flooded
            repl = nearest_nonflood_site(J_df, idx, flood_path, max_search_km=5.0)
            if repl is not None:
                logging.info(f"J site {row['site_id']} flooded {depth:.2f}m -> redirect to {repl['site_id']}")
                J_df.at[idx, "site_id"] = repl["site_id"]
                J_df.at[idx, "latitude"] = repl["latitude"]
                J_df.at[idx, "longitude"] = repl["longitude"]
                J_df.at[idx, "in_water"] = repl.get("in_water", False)

    # Build coverage array: shape (n_cows, n_I, n_J_candidates)
    cover_arr = build_cover_indicator_array(I_df, J_df, cow_df)
    n_cows = len(cow_df)
    n_J = len(J_df)
    # Build power index maps for gene encoding/decoding
    power_index_map, index_power_map = build_power_index_maps(power_df)
    n_powers = len(power_df)
    # list of BTS that require power assignment: only status == power_outage
    bts_power_list = list(failed_bts_df[failed_bts_df["status"] == "power_outage"]["site_id"].values)
    n_bts_power = len(bts_power_list)

    logging.info(f"Loaded I({len(I_df)}), J({n_J}), cows({n_cows}), powers({n_powers}), bts_power_targets({n_bts_power})")

    # time normalization baseline
    all_travel_times = []
    all_travel_times += [v.get("travel_time_hr", 0.0) for v in cow_travel_dict.values()]
    all_travel_times += [v.get("total_time_hr", 0.0) for v in power_travel_dict.values()]
    time_max_norm = max(max(all_travel_times) + default_setup_time_h if all_travel_times else default_setup_time_h, 1.0)

    # initialize population: each individual is concatenation [cow_assignments (len n_cows), power_assignments (len n_bts_power)]
    population = initialize_population(pop_size, n_cows, n_J, n_bts_power, power_index_map, cover_arr, cow_df, J_df, bts_power_list)
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
            Xi = repair_solution(Xi,
                                 cover_arr,
                                 cow_df,
                                 J_df,
                                 cow_travel_dict,
                                 power_df,
                                 power_travel_dict,
                                 bts_power_list,
                                 power_index_map,
                                 index_power_map,
                                 failed_bts_df,
                                 flood_path,
                                 roads_graph,
                                 budget_max,
                                 default_setup_time_h,
                                 budget_hard=budget_hard,
                                 enforce_site_unique=True)
            f, cov, cost, makespan, used, covered_mask = evaluate_solution(Xi,
                                                                            cover_arr,
                                                                            cow_df,
                                                                            I_df,
                                                                            J_df,
                                                                            cow_travel_dict,
                                                                            power_df,
                                                                            power_travel_dict,
                                                                            bts_power_list,
                                                                            power_index_map,
                                                                            default_setup_time_h,
                                                                            budget_max,
                                                                            time_max_norm,
                                                                            w_time,
                                                                            w_cost,
                                                                            coverage_weight)
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
            X_mut = mutation_discrete(Xi, n_J, n_powers, mutation_rate)
            X_p = uniform_crossover_arrays(X_mut, pbest[i])
            if best_global is not None:
                X_g = uniform_crossover_arrays(X_p, best_global)
            else:
                X_g = X_p
            X_rep = repair_solution(X_g,
                                    cover_arr,
                                    cow_df,
                                    J_df,
                                    cow_travel_dict,
                                    power_df,
                                    power_travel_dict,
                                    bts_power_list,
                                    power_index_map,
                                    index_power_map,
                                    failed_bts_df,
                                    flood_path,
                                    roads_graph,
                                    budget_max,
                                    default_setup_time_h,
                                    budget_hard=budget_hard,
                                    enforce_site_unique=True)
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
            child = mutation_discrete(child, n_J, n_powers, mutation_rate)
            child = repair_solution(child,
                                    cover_arr,
                                    cow_df,
                                    J_df,
                                    cow_travel_dict,
                                    power_df,
                                    power_travel_dict,
                                    bts_power_list,
                                    power_index_map,
                                    index_power_map,
                                    failed_bts_df,
                                    flood_path,
                                    roads_graph,
                                    budget_max,
                                    default_setup_time_h,
                                    budget_hard=budget_hard,
                                    enforce_site_unique=True)
            worst_idx = int(np.argmax(pbest_fit))
            pop_array[worst_idx] = child

        fitness_history.append(best_global_fit)
        if it % 10 == 0:
            logging.info(f"Iter {it}/{max_iter} best_fit {best_global_fit:.6f}")

    runtime = time.time() - t0
    logging.info(f"Optimization finished. best_fit={best_global_fit:.6f} runtime={runtime:.1f}s")

    # prepare outputs
    if best_global is None:
        best_global = np.zeros(n_cows + n_bts_power, dtype=int)

    np.save(outputs / "ga_pso_best_solution.npy", best_global)
    np.save(outputs / "ga_pso_fitness_history.npy", np.array(fitness_history))

    # decode to assignments
    assign_rows = decode_solution_to_assignments(best_global,
                                                cow_df,
                                                J_df,
                                                power_df,
                                                bts_power_list,
                                                power_index_map,
                                                cow_travel_dict,
                                                power_travel_dict,
                                                default_setup_time_h)
    assign_df = pd.DataFrame(assign_rows)
    assign_path = outputs / "ga_pso_assignments.csv"
    assign_df.to_csv(assign_path, index=False)

    # coverage summary
    covered_mask = compute_coverage_from_solution(best_global[:n_cows], cover_arr, I_df)[2]
    total_pop_on_I = float(I_df["pop"].sum()) if "pop" in I_df.columns else 0.0
    covered_pop = float((I_df["pop"].values * covered_mask).sum()) if total_pop_on_I > 0 else 0.0
    coverage_pct = round((covered_pop / total_pop_on_I * 100) if total_pop_on_I > 0 else 0.0, 2)

    total_travel_cost = float(assign_df["travel_cost_vnd"].sum())
    total_broadcast_cost = float(assign_df[assign_df["type"] == "COW"]["cost_vnd"].sum())
    total_power_cost = float(assign_df[assign_df["type"] == "POWER"]["travel_cost_vnd"].sum() + assign_df[assign_df["type"] == "POWER"]["cost_vnd"].sum())
    total_cost_all = total_travel_cost + total_broadcast_cost + total_power_cost
    max_time_hours = float(assign_df["deployment_time_hr"].max()) if not assign_df.empty else 0.0
    num_COW_used = int((assign_df[assign_df["type"] == "COW"]["assigned_site_id"].notnull()).sum())
    num_power_used = int((assign_df[assign_df["type"] == "POWER"]["assigned_target_bts"].notnull()).sum())

    summary = {
        "num_COW_used": num_COW_used,
        "num_POWER_used": num_power_used,
        "coverage_pct_on_I_points": coverage_pct,
        "covered_pop_on_I_points": covered_pop,
        "total_pop_on_I_points": total_pop_on_I,
        "total_fixed_cost_vnd": total_broadcast_cost,
        "total_travel_cost_vnd": total_travel_cost,
        "total_power_cost_vnd": total_power_cost,
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
