#!/usr/bin/env python3
"""
method_comparison_plots.py

METHOD COMPARISON (CORE ANALYSIS)

Figures:
Coverage & Resource Comparison
Cost & Time Comparison

Author: Lợi Lưu
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# PATH CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_PATHS = {
    "MILP": PROJECT_ROOT / "outputs/summary/milp/coverage_report_milp_gurobi.json",
    "GA-PSO": PROJECT_ROOT / "outputs/summary/ga_pso/coverage_report_ga_pso.json",
    "Hybrid": PROJECT_ROOT / "outputs/summary/milp_ga_pso_B/coverage_report_milp_ga_pso.json",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs/simulation/comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LOAD REPORTS
def load_reports():
    reports = {}
    for name, path in REPORT_PATHS.items():
        with open(path, "r", encoding="utf-8") as f:
            reports[name] = json.load(f)
    return reports

# COVERAGE & RESOURCE COMPARISON
def plot_coverage_comparison(reports):
    methods = list(reports.keys())
    x = np.arange(len(methods))
    width = 0.25

    coverage_pct = [reports[m]["coverage_after_restoration_percent"] for m in methods]
    cow_used = [reports[m]["cow_count_used"] for m in methods]
    bts_restored = [reports[m]["power_units_used"] for m in methods]

    fig, ax = plt.subplots(figsize=(13, 7))

    bars1 = ax.bar(x - width, coverage_pct, width,
                   color="#3498db", label="Coverage after restoration (%)")
    bars2 = ax.bar(x, cow_used, width,
                   color="#f1c40f", label="COW used")
    bars3 = ax.bar(x + width, bts_restored, width,
                   color="#e74c3c", label="BTS restored (power units)")

    ax.set_title("Figure: Coverage & Resource Comparison", fontsize=14)
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

    # Annotate values on bars
    def annotate(bars, fmt="{:.0f}"):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=10
            )

    annotate(bars1, "{:.1f}")
    annotate(bars2, "{:.0f}")
    annotate(bars3, "{:.0f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coverage_comparison.png", dpi=300)
    plt.close()

# COST & TIME COMPARISON
def plot_cost_time_comparison(reports):
    methods = list(reports.keys())
    x = np.arange(len(methods))
    width = 0.35

    cost_million = [reports[m]["total_deployment_cost_vnd"] / 1e6 for m in methods]
    time_hr = [reports[m]["max_deploy_time_hr"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars_cost = ax1.bar(
        x - width / 2, cost_million, width,
        color="#9b59b6"
    )
    ax1.set_ylabel("Total deployment cost (million VND)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    bars_time = ax2.bar(
        x + width / 2, time_hr, width,
        color="#34495e"
    )
    ax2.set_ylabel("Max deployment time (hours)")

    ax1.set_title("Figure: Cost & Time Comparison", fontsize=14)

    # Legend (top-right)
    legend_elements = [
        Patch(facecolor="#9b59b6", label="Total deployment cost (million VND)"),
        Patch(facecolor="#34495e", label="Max deployment time (hours)")
    ]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=True)

    # Annotate values
    for b in bars_cost:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    for b in bars_time:
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cost_time_comparison.png", dpi=300)
    plt.close()

# MAIN
def run():
    print("\nMethod Comparison Simulation")
    reports = load_reports()

    plot_coverage_comparison(reports)
    plot_cost_time_comparison(reports)

    print(f"Results saved at: {OUTPUT_DIR}")
    print("Completed: Simulation of Method Comparison")

if __name__ == "__main__":
    run()
