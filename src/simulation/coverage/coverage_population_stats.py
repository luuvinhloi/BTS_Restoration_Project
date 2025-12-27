#!/usr/bin/env python3
"""
coverage_population_stats.py

COVERAGE POPULATION STATISTICS

Simulate population coverage:
- Before disaster
- After disaster
- After restoration

Comparison across methods:
- MILP
- GA-PSO
- MILP-GA-PSO

Output:
- Bar chart (PNG, thesis-ready)

Author: Lợi Lưu
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# PATH CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORTS = {
    "MILP": PROJECT_ROOT / "outputs/summary/milp/coverage_report_milp_gurobi.json",
    "GA_PSO": PROJECT_ROOT / "outputs/summary/ga_pso/coverage_report_ga_pso.json",
    "MILP_GA_PSO": PROJECT_ROOT / "outputs/summary/milp_ga_pso_B/coverage_report_milp_ga_pso.json",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs/simulation/coverage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BAR_OUTPUT = OUTPUT_DIR / "population_coverage_comparison_graph.png"

# LOAD & PREPARE DATA
def load_population_stats():
    records = []

    for method, path in REPORTS.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_pop = data["total_population"]
        after_disaster = data["population_covered_by_active_bts"]
        after_restoration = (
            data["population_covered_by_active_bts"]
            + data["population_restored_total"]
        )

        records.append({
            "method": method,
            "Before disaster": total_pop,
            "After disaster": after_disaster,
            "After restoration": after_restoration,
        })

    return pd.DataFrame(records)

# BAR CHART
def plot_population_bar_chart(df):
    methods = df["method"]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(
        x - width,
        df["Before disaster"],
        width,
        label="Before disaster",
        color="#bdc3c7"
    )

    bars2 = ax.bar(
        x,
        df["After disaster"],
        width,
        label="After disaster",
        color="#e74c3c"
    )

    bars3 = ax.bar(
        x + width,
        df["After restoration"],
        width,
        label="After restoration",
        color="#2ecc71"
    )

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax.set_ylabel("Population")
    ax.set_xlabel("Method")
    ax.set_title("Population Coverage Comparison Across Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(BAR_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Bar chart saved at: {BAR_OUTPUT}")

# MAIN
def run():
    print("\nCoverage Impact Simulation – Population Maps")
    df = load_population_stats()
    plot_population_bar_chart(df)
    print("Completed: Simulation of Coverage Impact Simulation – Population Maps\n")

if __name__ == "__main__":
    run()
