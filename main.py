# main.py
import argparse
from pathlib import Path
import time

# Utils
from src.utils.io_utils import read_yaml

# Stage 1: Data Clean
from src.preprocessing.data_preparation.data_cleaning import main as run_cleaning_pipeline

# Stage 2: Generate BTS network, Generate COW dataset and Generate Backup Power dataset
from src.preprocessing.bts_generation.generate_bts_network import main as generate_bts_network
from src.preprocessing.cow_generation.cow_dataset_generator import generate_cow_dataset
from src.preprocessing.backup_power.backup_power_generation import generate_backup_power_dataset

# Stage 3: Generating Flood and Generating Flood road
from src.preprocessing.flood_generation.flood_simulation import main as run_flood_simulation
from src.preprocessing.roads_generation.generate_flooded_roads import main as generate_flooded_roads

# Stage 4: Damage scenario generation
from src.preprocessing.damage_bts.generate_damage_scenario import generate_bts_damage_dataset

# Stage 5: Generate data sets I, J and Calculate travel time and costs
from src.preprocessing.I_J_generation.feature_extraction_final import main as feature_extraction_final

# Calculate travel time and costs
from src.preprocessing.travel_cost.compute_travel_costs import (
    compute_cow_travel_matrix,
    compute_backup_travel_matrix
)

# Stage 6: Optimization
# MILP Solver
from src.optimization.MILP.milp_solver import main_solve as milp_main
# GA-PSO Solver
from src.optimization.GA_PSO.ga_pso_solver import run_from_config as ga_pso_main
# Hybrid MILP + GA-PSO Solver
from src.optimization.MILP_GA_PSO.hybrid_milp_ga_pso import run_hybrid

# Stage 7: Population coverage computation
from src.compute_pop_cover.compute_population_coverage_milp import (
    main_compute_all as compute_coverage_milp
)

from src.compute_pop_cover.compute_population_coverage_gapso import (
    main_compute_all as compute_coverage_gapso
)

from src.compute_pop_cover.compute_population_coverage_hybrid import (
    main_compute_all as compute_coverage_hybrid
)

# Stage 8: Simulation of COW and Backup Power Deployment
from src.simulation.spatial.spatial_flood_map import run as run_spatial_flood_map
from src.simulation.spatial.spatial_bts_status import run as run_spatial_bts_status
from src.simulation.spatial.spatial_deployment_map import run as run_spatial_deployment_map
from src.simulation.spatial.spatial_routes_map import run as run_spatial_routes_map

from src.simulation.coverage.coverage_population_map import run as run_coverage_population_map
from src.simulation.coverage.coverage_population_stats import run as run_coverage_population_stats

from src.simulation.comparison.method_comparison_plots import run as run_method_comparison

# Measure total execution time
start = time.perf_counter()

PROJECT_ROOT = Path(__file__).resolve().parent

def run_pipeline(config_path):
    cfg = read_yaml(config_path)
    method = cfg.get("method", "MILP").upper()

    print("Stage 1: Data Cleaning...")
    # run_cleaning_pipeline()

    print("Stage 2: Generating BTS network, Generating COW dataset and Generating Backup Power dataset...")
    print("Generating COW dataset...")
    # generate_bts_network()
    print("Generating COW dataset...")
    # generate_cow_dataset(str(PROJECT_ROOT / "data" / "processed"))
    print("Generating Backup Power dataset")
    # generate_backup_power_dataset(
    #     outage_csv_path=str(PROJECT_ROOT / "data" / "processed" / "damage_bts" / "failed_bts.csv"),
    #     output_csv_path=str(PROJECT_ROOT / "data" / "processed" / "backup_power" / "backup_power.csv")
    # )

    print("Stage 3: Generating Flood and Generating Flood road...")
    # run_flood_simulation()

    print("Generating Flood roads...")
    # generate_flooded_roads()

    print("Stage 4: Generating damage scenario...")
    # generate_bts_damage_dataset( # Kịch bản B hư hại 70%, mất nguồn 20%, hoạt động 10%
    #     bts_csv_path=str(PROJECT_ROOT / "data/processed/bts_network/bts_ga.csv"),
    #     flood_tif_path=str(PROJECT_ROOT / "data/processed/flood/flood_depth_combined_clean.tif"),
    #     output_dir=str(PROJECT_ROOT / "data/processed/damage_bts"),
    #     active_rate=0.10,
    #     power_outage_rate=0.20,
    #     failed_rate=0.70,
    #     seed=cfg.get("seed", 42)
    # )

    print("Stage 5: Generate data sets I, J and Calculate travel time and costs")
    feature_extraction_final(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))

    print("Computing travel time & cost matrix...")
    print("Computing travel time & cost for COW to J_sites...")
    # compute_cow_travel_matrix(
    #     cow_csv=str(PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"),
    #     site_csv=str(PROJECT_ROOT / "data/processed/position_I_J/J_sites_B.csv"),
    #     graphml_path=str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
    #     output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites_B.csv"),
    #     graph_pickle_cache=str(PROJECT_ROOT / "cache/flood_graph.pkl"),
    # )

    print("Computing travel time & cost for Backup Power → Failed BTS...")
    # compute_backup_travel_matrix(
    #     backup_csv=str(PROJECT_ROOT / "data/processed/backup_power/backup_power.csv"),
    #     outage_bts_csv=str(PROJECT_ROOT / "data/processed/damage_bts/failed_bts_B.csv"),
    #     graphml_path=str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
    #     output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts_B.csv"),
    #     graph_pickle_cache=str(PROJECT_ROOT / "cache/flood_graph.pkl"),
    #     optimize_for="time"
    # )

    print("Stage 6: Optimization")
    processed_dir = str(PROJECT_ROOT / "data" / "processed")
    outputs_dir = str(PROJECT_ROOT / "outputs" / "milp_runs_new")

    if method == "MILP":
        print("Solving with MILP (Full model - lexicographic)...")
        results = milp_main(cfg, processed_dir, outputs_dir)

        print("MILP finished. Results object returned.")

    elif method == "GA_PSO":
        print("Running GA-PSO Hybrid (Method 2)...")
        runs = int(cfg.get("ga_pso", {}).get("runs", 1))
        for i in range(runs):
            run_out_dir = str(PROJECT_ROOT / "outputs" / f"ga_pso_run_{i + 1:02d}")
            print(f"  • Run {i + 1}/{runs} → output: {run_out_dir}")
            # call module mới
            summary = ga_pso_main(cfg)
            print(f"Completed Run {i + 1}: best_fitness={summary.fitness}")

    elif method == "MILP_GA_PSO":
        print("Running Hybrid MILP + GA-PSO (Phương pháp 3)...")
        max_iter = int(cfg.get("hybrid", {}).get("max_iter", 300))
        top_k = int(cfg.get("hybrid", {}).get("top_k", 5))
        result = run_hybrid(max_iter=max_iter, top_k=top_k)
        print("\n=== Hybrid optimization finished ===")
        print("Best fitness:", result["refined_best_f"])
        print("Hybrid output saved to hybrid_result_summary.json")

    else:
        raise ValueError(f"Unknown method '{method}'. Must be MILP, GA_PSO, or HYBRID.")

    print("Stage 7: Computing population coverage...")
    # try:
    #     if method == "MILP":
    #         print("Computing population coverage for MILP...")
    #         coverage_summary = compute_coverage_milp(method="MILP_GUROBI")
    #
    #     elif method == "GA_PSO":
    #         print("Computing population coverage for GA-PSO...")
    #         coverage_summary = compute_coverage_gapso(method="GA_PSO")
    #
    #     elif method == "MILP_GA_PSO":
    #         print("Computing population coverage for Hybrid MILP + GA-PSO...")
    #         coverage_summary = compute_coverage_hybrid(method="MILP_GA_PSO")
    #
    #     else:
    #         raise ValueError(f"Unsupported method for coverage: {method}")
    #
    #     print("Coverage computation finished successfully.")
    #     print("Coverage summary:", coverage_summary)
    #
    # except Exception as e:
    #     print("Population coverage computation failed:")
    #     print(e)

    #  Simulation
    print("Stage 8: Simulation...")
    # try:
    #     print("8.1: Spatial flood map...")
    #     run_spatial_flood_map()
    #
    #     print("8.2: Spatial BTS status map...")
    #     run_spatial_bts_status()
    #
    #     print("8.3: Spatial deployment & coverage maps...")
    #     run_spatial_deployment_map()
    #
    #     print("8.4: Spatial deployment routes...")
    #     run_spatial_routes_map()
    #
    #     print("8.5: Population coverage maps...")
    #     run_coverage_population_map()
    #
    #     print("8.6: Population coverage statistics...")
    #     run_coverage_population_stats()
    #
    #     print("8.7: Method comparison analysis...")
    #     run_method_comparison()
    #
    #     print("Stage 8 completed successfully.")
    #
    # except Exception as e:
    #     print("[ERROR] Simulation stage failed:")
    #     print(e)

    print("Pipeline finished successfully.")

    # Measure total execution time
    end = time.perf_counter()
    print("Runtime:", end - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "params.yaml"))
    args = parser.parse_args()
    run_pipeline(args.config)