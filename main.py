# main.py
import argparse
from pathlib import Path

# Utils
from src.utils.io_utils import read_yaml

# Stage 1 - Data Cleaning
from src.preprocessing.data_preparation.data_cleaning import run_cleaning_pipeline

# Stage 2 - Flood simulation
from src.preprocessing.flood_generation.flood_simulation_A import main as run_flood_simulation_A
from src.preprocessing.flood_generation.flood_simulation_B import main as run_flood_simulation_B

# Stage 2-4 - Preprocessing
from src.preprocessing.bts_generation.generate_bts_network import main as generate_bts_network
from src.preprocessing.cow_generation.cow_dataset_generator import generate_cow_dataset
from src.preprocessing.damage_bts.generate_damage_scenario import main as generate_damage_scenario
from src.preprocessing.I_J_generation.feature_extraction_final import main as feature_extraction_final
from src.preprocessing.I_J_generation.feature_extraction import main as feature_extraction
from src.preprocessing.I_J_generation.feature_extraction_optimize_A import main as feature_extraction_optimize_A
from src.preprocessing.I_J_generation.feature_extraction_optimize_B import main as feature_extraction_optimize_B
from src.preprocessing.I_J_generation.feature_extraction_A import main as feature_extraction_A
from src.preprocessing.travel_cost.compute_travel_costs_A import compute_travel_matrix

# Optimization
# from src.optimization import solver_milp
from src.optimization.MILP.solver_milp import main_solve as milp_lexi_solve
from src.optimization.GA_PSO.ga_pso_hybrid_new import ga_pso_hybrid_main
# from src.optimization.GA_PSO.ga_pso_hybrid import ga_pso_hybrid_main

# Visualization
from src.visualization.simulation_scenario import run_simulation_scenario
from src.visualization.compute_population_coverage import main_compute_all
from src.visualization.flood_visualization import main as run_flood_visualization

PROJECT_ROOT = Path(__file__).resolve().parent

def run_pipeline(config_path):
    cfg = read_yaml(config_path)
    method = cfg.get("method", "MILP").upper()

    print("Stage 1: DATA CLEANING")
    # run_cleaning_pipeline()

    print("Stage 2: FLOOD SIMULATION")
    # run_flood_simulation_A()
    # run_flood_simulation_B()

    print("2) Generating BTS network and Generating COW dataset...")
    # print("Generating COW dataset...")
    # generate_bts_network()
    # print("Generating COW dataset...")
    # generate_cow_dataset(str(PROJECT_ROOT / "data" / "processed"))

    print("3) Generating damage scenario...")
    # generate_damage_scenario(
    #     str(PROJECT_ROOT / "data" / "processed" / "bts_network" / "bts_ga.csv"),
    #     str(PROJECT_ROOT / "data" / "processed" / "damage_bts"),
    #     cfg['damage_rate'],
    #     cfg.get('seed', 42)
    # )

    print("4) Feature extraction...")
    # feature_extraction_final(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))
    # feature_extraction(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))
    # feature_extraction_A(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))
    # feature_extraction_optimize_A(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))
    # feature_extraction_optimize_B(cfg, str(PROJECT_ROOT / "data" / "processed" / "position_I_J"))

    print("5) Computing travel time & cost matrix...")
    # compute_travel_matrix(
    #     cow_csv=str(PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"),
    #     site_csv=str(PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"),
    #     roads_path=str(PROJECT_ROOT / "data/cleaned/roads_hue_clean.geojson"),
    #     output_csv=str(PROJECT_ROOT / "data/processed/travel_cost/travel_cost_matrix_A.csv")
    # )

    print("6) Solving optimization problem...")
    # if method == "MILP":
    #     # print("6) Solving with MILP (Phương pháp 1)...")
    #     # out = solver_milp.solve_milp(config_path, str(PROJECT_ROOT / "data" / "processed"))
    #     print("6) Solving with MILP (Phương pháp 1 - PuLP Lexicographic)...")
    #     out = milp_lexi_solve(
    #         config_path=config_path,
    #         processed_data_dir=str(PROJECT_ROOT / "data" / "processed"),
    #         outputs_dir=str(PROJECT_ROOT / "outputs" / "milp_runs")
    #     )
    #
    # elif method == "GA_PSO":
    #     # print("6) Solving with GA–PSO (Phương pháp 2)...")
    #     # runs = int(cfg.get("ga_pso", {}).get("runs", 30))
    #     # for i in range(runs):
    #     #     print(f"    Run {i+1}/{runs}")
    #     #     summary = ga_pso_hybrid_main(
    #     #         str(PROJECT_ROOT / "data" / "processed"),
    #     #         str(PROJECT_ROOT / "outputs" / f"ga_pso_run_{i+1:02d}"),
    #     #         cfg["ga_pso"]
    #     #     )
    #     print("6) Solving with GA–PSO (Phương pháp 2)...")
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

    # Compute population coverage report
    print("7) Computing population coverage (outage / COW coverage)...")
    # try:
    #     summary_cov = main_compute_all(method=method)
    #     print("Coverage summary:", summary_cov)
    # except Exception as e:
    #     print("Coverage computation failed:", e)

    #  Visualization
    print("8) Running simulation scenario...")
    # if method == "MILP":
    #     if cfg["milp"]["simulation"]["enable"]:
    #         run_simulation_scenario("MILP")
    #
    # elif method == "GA_PSO":
    #     if cfg["ga_pso"]["simulation"]["enable"]:
    #         run_simulation_scenario("GA_PSO")

    print("Stage 9: FLOOD MAP VISUALIZATION")
    run_flood_visualization()

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "params.yaml"))
    args = parser.parse_args()
    run_pipeline(args.config)