# main.py
import argparse
from pathlib import Path

# Utils
from src.utils.io_utils import read_yaml

# Preprocessing modules
from src.preprocessing.data_preparation.data_cleaning import main as data_cleaning
from src.preprocessing.bts_generation.generate_bts_network import main as generate_bts_network
from src.preprocessing.cow_generation.cow_dataset_generator import generate_cow_dataset
from src.preprocessing.damage_simulation.generate_damage_scenario import main as generate_damage_scenario
from src.preprocessing.feature_engineering.feature_extraction import main as feature_extraction
from src.preprocessing.feature_engineering.feature_extraction_optimize_A import main as feature_extraction_optimize_A
from src.preprocessing.feature_engineering.feature_extraction_optimize_B import main as feature_extraction_optimize_B
from src.preprocessing.feature_engineering.feature_extraction_A import main as feature_extraction_A
from src.preprocessing.travel_cost.compute_travel_costs import compute_travel_matrix

# Optimization
# from src.optimization import solver_milp
from src.optimization.MILP.solver_milp import main_solve as milp_lexi_solve
from src.optimization.GA_PSO.ga_pso_hybrid_new import ga_pso_hybrid_main
# from src.optimization.GA_PSO.ga_pso_hybrid import ga_pso_hybrid_main

# Visualization
from src.visualization.simulation_scenario import run_simulation_scenario
from src.visualization.compute_population_coverage import main_compute_all

PROJECT_ROOT = Path(__file__).resolve().parent

def run_pipeline(config_path):
    cfg = read_yaml(config_path)
    method = cfg.get("method", "MILP").upper()

    print("1) Preprocessing...")
    # data_cleaning.main()

    print("2) Generating BTS network and Generating COW dataset...")
    print("Generating COW dataset...")
    generate_bts_network()
    print("Generating COW dataset...")
    generate_cow_dataset(str(PROJECT_ROOT / "data" / "raw"))

    print("3) Generating damage scenario...")
    # generate_damage_scenario.main(
    #     str(PROJECT_ROOT / "data" / "raw" / "bts_ga.csv"),
    #     str(PROJECT_ROOT / "data" / "processed"),
    #     cfg['damage_rate'],
    #     cfg.get('seed', 42)
    # )

    print("4) Feature extraction...")
    # feature_extraction.main(cfg, str(PROJECT_ROOT / "data" / "processed"))
    # feature_extraction_A.main(cfg, str(PROJECT_ROOT / "data" / "processed"))
    # feature_extraction_optimize_A.main(cfg, str(PROJECT_ROOT / "data" / "processed"))
    # feature_extraction_optimize_B.main(cfg, str(PROJECT_ROOT / "data" / "processed"))

    print("5) Computing travel time & cost matrix...")
    # compute_travel_matrix(
    #     cow_csv=str(PROJECT_ROOT / "data/raw/cow_dataset.csv"),
    #     site_csv=str(PROJECT_ROOT / "data/processed/J_sites.csv"),
    #     roads_path=str(PROJECT_ROOT / "data/raw/roads_hue.geojson"),
    #     output_csv=str(PROJECT_ROOT / "data/processed/travel_cost_matrix_A.csv")
    # )

    # if method == "MILP":
    #     # print("5) Solving with MILP (Phương pháp 1)...")
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
    #
    # # Compute population coverage report
    # print("7) Computing population coverage (outage / COW coverage)...")
    # try:
    #     summary_cov = main_compute_all(method=method)
    #     print("Coverage summary:", summary_cov)
    # except Exception as e:
    #     print("Coverage computation failed:", e)
    #
    # #  Visualization
    # print("8) Running simulation scenario...")
    # if method == "MILP":
    #     if cfg["milp"]["simulation"]["enable"]:
    #         run_simulation_scenario("MILP")
    #
    # elif method == "GA_PSO":
    #     if cfg["ga_pso"]["simulation"]["enable"]:
    #         run_simulation_scenario("GA_PSO")

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "params.yaml"))
    args = parser.parse_args()
    run_pipeline(args.config)