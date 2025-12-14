# main.py
import argparse
from pathlib import Path

# Utils
from src.utils.io_utils import read_yaml

# Stage 1: Data Clean
from src.preprocessing.data_preparation.data_cleaning import main as run_cleaning_pipeline

# Stage 2: Generate BTS network, Generate COW dataset and Generate Backup Power dataset
from src.preprocessing.bts_generation.generate_bts_network import main as generate_bts_network
from src.preprocessing.cow_generation.cow_dataset_generator import generate_cow_dataset
from src.preprocessing.backup_power.backup_power_generation import generate_backup_power_dataset

# Stage 3: Generating Flood and Generating Flood road
from src.preprocessing.flood_generation.flood_simulation_B import main as run_flood_simulation_B
from src.preprocessing.roads_generation.generate_flooded_roads import main as generate_flooded_roads

# Stage 4: Damage scenario generation
from src.preprocessing.damage_bts.generate_damage_scenario import generate_bts_damage_dataset

# Stage 5: Generate data sets I, J and Calculate travel time and costs
from src.preprocessing.I_J_generation.feature_extraction_final import main as feature_extraction_final

# Stage 6: Optimization
from src.optimization.MILP.milp_solver import main_solve as milp_main
from src.optimization.MILP.milp_solver_new import main_solve
from src.optimization.MILP.milp_solver_cov import main_solve_cov

from src.optimization.GA_PSO.ga_pso_solver import run_from_config as ga_pso_main
from src.optimization.GA_PSO.ga_pso_solver_cov import run_from_config as ga_pso_main_cov

from src.optimization.MILP_GA_PSO.hybrid_milp_ga_pso import run_hybrid
from src.optimization.MILP_GA_PSO.hybrid_milp_ga_pso_cov import run_hybrid_cov

# Stage 7: Visualization
from src.visualization.simulation_scenario import run_simulation_scenario
from src.visualization.compute_population_coverage import main_compute_all
from src.visualization.flood_visualization_A import run_flood_map_visualization_A
from src.visualization.flood_visualization_B import run_flood_map_visualization_B
from src.visualization.visualization_all import run_visualization_combined

PROJECT_ROOT = Path(__file__).resolve().parent

def run_pipeline(config_path):
    cfg = read_yaml(config_path)
    method = cfg.get("method", "MILP").upper()

    print("Stage 1: Data Cleaning...")
    # run_cleaning_pipeline()

    print("Stage 2: Generating BTS network, Generating COW dataset and Generating Backup Power dataset...")
    # print("Generating COW dataset...")
    # generate_bts_network()
    # print("Generating COW dataset...")
    # generate_cow_dataset(str(PROJECT_ROOT / "data" / "processed"))
    # print("Generating Backup Power dataset")
    # generate_backup_power_dataset(
    #     outage_csv_path=str(PROJECT_ROOT / "data" / "processed" / "damage_bts" / "failed_bts.csv"),
    #     output_csv_path=str(PROJECT_ROOT / "data" / "processed" / "backup_power" / "backup_power.csv")
    # )

    print("Stage 3: Generating Flood and Generating Flood road...")
    # run_flood_simulation_A()
    # run_flood_simulation_B()

    print("Generating Flood roads...")
    # generate_flooded_roads()

    print("Flood Map Visualization...")
    # run_flood_map_visualization_A()
    # run_flood_map_visualization_B()

    print("Stage 4: Generating damage scenario...")
    # generate_bts_damage_dataset(
    #     bts_csv_path=str(PROJECT_ROOT / "data/processed/bts_network/bts_ga.csv"),
    #     flood_tif_path=str(PROJECT_ROOT / "data/processed/flood/flood_depth_combined_B_clean.tif"),
    #     output_dir=str(PROJECT_ROOT / "data/processed/damage_bts"),
    #     active_rate=0.20,
    #     power_outage_rate=0.15,
    #     failed_rate=0.65,
    #     seed=cfg.get("seed", 42)
    # )

    print("Stage 5: Generate data sets I, J and Calculate travel time and costs")
    # feature_extraction_final(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))

    print("Computing travel time & cost matrix...")
    # print("Computing travel time & cost for COW → J_sites...")
    # compute_cow_travel_matrix(
    #     cow_csv=str(PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"),
    #     site_csv=str(PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"),
    #     graphml_path=str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
    #     output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites.csv"),
    #     graph_pickle_cache=str(PROJECT_ROOT / "cache/flood_graph.pkl"),
    # )
    #
    # print("Computing travel time & cost for Backup Power → Failed BTS...")
    # compute_backup_travel_matrix(
    #     backup_csv=str(PROJECT_ROOT / "data/processed/backup_power/backup_power.csv"),
    #     outage_bts_csv=str(PROJECT_ROOT / "data/processed/damage_bts/failed_bts.csv"),
    #     graphml_path=str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
    #     output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts.csv"),
    #     graph_pickle_cache=str(PROJECT_ROOT / "cache/flood_graph.pkl"),
    #     optimize_for="time"
    # )

    print("Stage 6: Optimization")
    processed_dir = str(PROJECT_ROOT / "data" / "processed")
    outputs_dir = str(PROJECT_ROOT / "outputs" / "milp_runs")

    if method == "MILP":
        print("6) Solving with MILP (Full model - lexicographic)...")
        results = milp_main(cfg, processed_dir, outputs_dir)

        results = main_solve( # Run solver GUROBI bằng Pymo
            config_params=cfg,
            processed_data_dir=processed_dir,
            outputs_dir=outputs_dir,
            solver_backend="AUTO"  # hoặc "CBC" hoặc "GUROBI"
        )

        # results = main_solve_cov(  # Run solver GUROBI bằng Pymo
        #     config_params=cfg,
        #     processed_data_dir=processed_dir,
        #     outputs_dir=outputs_dir,
        #     solver_backend="AUTO"  # hoặc "CBC" hoặc "GUROBI"
        # )

        print("MILP finished. Results object returned.")

    # elif method == "GA_PSO":
    #     print("Running GA-PSO Hybrid (Method 2)...")
    #     runs = int(cfg.get("ga_pso", {}).get("runs", 1))
    #     for i in range(runs):
    #         run_out_dir = str(PROJECT_ROOT / "outputs" / f"ga_pso_run_{i + 1:02d}")
    #         print(f"  • Run {i + 1}/{runs} → output: {run_out_dir}")
    #         # call module mới
    #         summary = ga_pso_main(cfg)
    #         print(f"Completed Run {i + 1}: best_fitness={summary.fitness}")

    elif method == "GA_PSO":
        print("Running GA-PSO Hybrid (Method 2)...")
        runs = int(cfg.get("ga_pso", {}).get("runs", 1))

        for i in range(runs):
            print(f"  • Run {i + 1}/{runs} → output: outputs/results_ga_pso")

            summary = ga_pso_main(cfg)

            print(f"Completed Run {i + 1}: best_fitness={summary.fitness}")

    elif method == "MILP_GA_PSO":
        print("Running Hybrid MILP + GA-PSO (Phương pháp 3)...")
        max_iter = int(cfg.get("hybrid", {}).get("max_iter", 300))
        top_k = int(cfg.get("hybrid", {}).get("top_k", 5))
        result = run_hybrid(max_iter=max_iter, top_k=top_k)

        result = run_hybrid_cov(max_iter=max_iter, top_k=top_k)
        print("\n=== Hybrid optimization finished ===")
        print("Best fitness:", result["refined_best_f"])
        print("Hybrid output saved to hybrid_result_summary.json")

    else:
        raise ValueError(f"Unknown method '{method}'. Must be MILP, GA_PSO, or HYBRID.")

    #  Visualization
    print("Stage 7: Visualization")
    # if method == "MILP":
    #     if cfg["milp"]["simulation"]["enable"]:
    #         run_simulation_scenario("MILP")
    #
    # elif method == "GA_PSO":
    #     if cfg["ga_pso"]["simulation"]["enable"]:
    #         run_simulation_scenario("GA_PSO")

    # Compute population coverage report
    # run_map_visualization_all()

    print("Stage 8: Computing population coverage (outage / COW coverage)...")
    # try:
    #     summary_cov = main_compute_all(method=method)
    #     print("Coverage summary:", summary_cov)
    # except Exception as e:
    #     print("Coverage computation failed:", e)

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "params.yaml"))
    args = parser.parse_args()
    run_pipeline(args.config)