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
from src.preprocessing.flood_generation.flood_simulation_A import main as run_flood_simulation_A
from src.preprocessing.flood_generation.flood_simulation_B import main as run_flood_simulation_B
from src.preprocessing.roads_generation.generate_flooded_roads import main as generate_flooded_roads

# Stage 4: Damage scenario generation
from src.preprocessing.damage_bts.generate_damage_scenario import generate_bts_damage_dataset

# Stage 5: Generate data sets I, J and Calculate travel time and costs
from src.preprocessing.I_J_generation.feature_extraction_final import main as feature_extraction_final
from src.preprocessing.I_J_generation.feature_extraction import main as feature_extraction
from src.preprocessing.I_J_generation.feature_extraction_optimize_A import main as feature_extraction_optimize_A
from src.preprocessing.I_J_generation.feature_extraction_optimize_B import main as feature_extraction_optimize_B
from src.preprocessing.I_J_generation.feature_extraction_A import main as feature_extraction_A
# Calculate travel time and costs
from src.preprocessing.travel_cost.compute_travel_costs_A import (
    compute_cow_travel_matrix,
    compute_backup_travel_matrix
)

# Stage 6: Optimization
# from src.optimization import solver_milp
from src.optimization.MILP.solver_milp import main_solve as milp_lexi_solve
from src.optimization.GA_PSO.ga_pso_hybrid_new import ga_pso_hybrid_main
# from src.optimization.GA_PSO.ga_pso_hybrid import ga_pso_hybrid_main

# Stage 7: Visualization
from src.visualization.simulation_scenario import run_simulation_scenario
from src.visualization.compute_population_coverage import main_compute_all
from src.visualization.flood_visualization_A import run_flood_map_visualization_A
from src.visualization.flood_visualization_B import run_flood_map_visualization_B
from src.visualization.visualization_all import run_map_visualization_all

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
    # feature_extraction(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))
    # feature_extraction_A(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))
    # feature_extraction_optimize_A(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))
    # feature_extraction_optimize_B(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))

    print("Computing travel time & cost matrix...")
    # compute_travel_matrix(
    #     cow_csv=str(PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"),
    #     site_csv=str(PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"),
    #     roads_path=str(PROJECT_ROOT / "data/cleaned/roads_hue_clean.geojson"),
    #     output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/travel_cost_matrix_A.csv")
    # )

    print("Computing travel time & cost for COW → J_sites...")
    compute_cow_travel_matrix(
        cow_csv=str(PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"),
        site_csv=str(PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"),
        graphml_path=str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
        output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites.csv"),
        graph_pickle_cache=str(PROJECT_ROOT / "cache/flood_graph.pkl"),
    )

    print("Computing travel time & cost for Backup Power → Failed BTS...")
    compute_backup_travel_matrix(
        backup_csv=str(PROJECT_ROOT / "data/processed/backup_power/backup_power.csv"),
        outage_bts_csv=str(PROJECT_ROOT / "data/processed/damage_bts/failed_bts.csv"),
        graphml_path=str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
        output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts.csv"),
        graph_pickle_cache=str(PROJECT_ROOT / "cache/flood_graph.pkl"),
        optimize_for="time"
    )

    print("Stage 6: Optimization")
    # if method == "MILP":
    #     # print("Solving with MILP (Phương pháp 1)...")
    #     # out = solver_milp.solve_milp(config_path, str(PROJECT_ROOT / "data" / "processed"))
    #     print("6) Solving with MILP (Phương pháp 1 - PuLP Lexicographic)...")
    #     out = milp_lexi_solve(
    #         config_path=config_path,
    #         processed_data_dir=str(PROJECT_ROOT / "data" / "processed"),
    #         outputs_dir=str(PROJECT_ROOT / "outputs" / "milp_runs")
    #     )
    #
    # elif method == "GA_PSO":
    #     # print("Solving with GA–PSO (Phương pháp 2)...")
    #     # runs = int(cfg.get("ga_pso", {}).get("runs", 30))
    #     # for i in range(runs):
    #     #     print(f"    Run {i+1}/{runs}")
    #     #     summary = ga_pso_hybrid_main(
    #     #         str(PROJECT_ROOT / "data" / "processed"),
    #     #         str(PROJECT_ROOT / "outputs" / f"ga_pso_run_{i+1:02d}"),
    #     #         cfg["ga_pso"]
    #     #     )
    #     print("Solving with GA–PSO (Phương pháp 2)...")
    #     runs = int(cfg.get("ga_pso", {}).get("runs", 1))
    #     for i in range(runs):
    #         print(f"    Run {i + 1}/{runs}")
    #         # outputs per-run folder
    #         out_dir = str(PROJECT_ROOT / "outputs" / f"ga_pso_run_{i + 1:02d}")
    #         summary = ga_pso_hybrid_main(
    #             str(PROJECT_ROOT / "data" / "processed"),
    #             out_dir,
    #             cfg.get("ga_pso", {})
    #         )
    #
    # else:
    #     raise ValueError(f"Unknown method '{method}'. Must be 'MILP' or 'GA_PSO'.")

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
    run_map_visualization_all()

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