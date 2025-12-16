#!/usr/bin/env python3
"""
coverage_population_map.py

COVERAGE IMPACT (POPULATION)

Figures:
Population loss after disaster
Population coverage before disaster
Population restored by COW & power (per method)
Population restored FULL (Active + Power + COW)

Author: Lợi Lưu
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# PATH CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]

POP_RASTER = PROJECT_ROOT / "data/cleaned/pop_hue_clean.tif"
BOUNDARY_PATH = PROJECT_ROOT / "data/cleaned/hue_boundary_clean.geojson"

BTS_ALL_PATH = PROJECT_ROOT / "data/processed/bts_network/bts_ga.csv"
ACTIVE_BTS_PATH = PROJECT_ROOT / "data/processed/damage_bts/active_bts.csv"
FAILED_BTS_PATH = PROJECT_ROOT / "data/processed/damage_bts/failed_bts.csv"

J_SITES_PATH = PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"
COW_DATASET_PATH = PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs/simulation/coverage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    "MILP": {
        "cow": PROJECT_ROOT / "outputs/milp_runs/milp_gurobi/assignments_cow_GUROBI.csv",
        "power": PROJECT_ROOT / "outputs/milp_runs/milp_gurobi/assignments_power_GUROBI.csv"
    },
    "GA_PSO": {
        "cow": PROJECT_ROOT / "outputs/results_ga_pso/solution_cow_assignments.csv",
        "power": PROJECT_ROOT / "outputs/results_ga_pso/solution_power_assignments.csv"
    },
    "MILP_GA_PSO": {
        "cow": PROJECT_ROOT / "outputs/results_hybrid/solution_cow_assignments.csv",
        "power": PROJECT_ROOT / "outputs/results_hybrid/solution_power_assignments.csv"
    }
}

# LOADERS
def load_boundary():
    return gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")

def load_population():
    return rasterio.open(POP_RASTER)

def load_bts(path):
    df = pd.read_csv(path)

    if "coverage_radius_m" not in df.columns:
        ref = pd.read_csv(BTS_ALL_PATH)[["site_id", "coverage_radius_m"]]
        df = df.merge(ref, on="site_id", how="left")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

def load_cow_assignments(path):
    df = pd.read_csv(path)

    if {"lat", "lon"}.issubset(df.columns):
        df["latitude"] = df["lat"]
        df["longitude"] = df["lon"]
    else:
        j_sites = pd.read_csv(J_SITES_PATH)
        df = df.merge(
            j_sites[["site_id", "latitude", "longitude"]],
            on="site_id",
            how="left"
        )

    if "coverage_radius_m" not in df.columns:
        cow_ref = pd.read_csv(COW_DATASET_PATH)
        df = df.merge(
            cow_ref[["cow_id", "coverage_radius_m"]],
            on="cow_id",
            how="left"
        )

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

# DRAWING UTIL
def plot_buffers(ax, gdf, color, alpha=0.25, zorder=2):
    gdf_m = gdf.to_crs("EPSG:3857")
    for geom, r in zip(gdf_m.geometry, gdf_m.coverage_radius_m):
        circle = geom.buffer(r)
        gpd.GeoSeries([circle], crs="EPSG:3857") \
           .to_crs("EPSG:4326") \
           .plot(ax=ax, color=color, alpha=alpha, linewidth=0, zorder=zorder)

# FIG 4.5 – POPULATION LOSS
def plot_population_loss():
    boundary = load_boundary()
    failed_all = load_bts(FAILED_BTS_PATH)

    failed = failed_all[failed_all.status == "failed"]
    power_out = failed_all[failed_all.status == "power_outage"]

    with load_population() as src:
        pop = src.read(1)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(pop, cmap="OrRd", extent=extent, alpha=0.7)

    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)

    plot_buffers(ax, failed_all, color="#4b6cff", alpha=0.25)

    failed.plot(ax=ax, color="red", markersize=16, zorder=5)
    power_out.plot(ax=ax, color="orange", markersize=16, zorder=5)

    ax.set_title("Figure: Population Loss due to BTS Outage", fontsize=14)
    ax.set_axis_off()

    ax.legend(handles=[
        Patch(color="#4b6cff", alpha=0.25, label="Lost coverage"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="orange", markersize=8,
               label="Power outage BTS"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="red", markersize=8,
               label="Failed BTS"),
    ], loc="upper right")

    plt.savefig(OUTPUT_DIR / "population_loss.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# FIG 4.5b – FULL COVERAGE BEFORE DISASTER
def plot_population_full_before():
    boundary = load_boundary()
    bts_all = load_bts(BTS_ALL_PATH)

    with load_population() as src:
        pop = src.read(1)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(pop, cmap="OrRd", extent=extent, alpha=0.75)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)

    plot_buffers(ax, bts_all, "#2ecc71", 0.35)

    ax.set_title("Figure: Population Coverage Before Disaster", fontsize=14)
    ax.set_axis_off()

    plt.savefig(OUTPUT_DIR / "population_full_before_disaster.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# FIG 4.6 – RESTORED (COW + POWER)
def plot_population_restored(method, cfg):
    boundary = load_boundary()
    cow = load_cow_assignments(cfg["cow"])
    power_assign = pd.read_csv(cfg["power"])

    bts_ref = load_bts(BTS_ALL_PATH)
    power_bts = bts_ref[bts_ref.site_id.isin(power_assign.bts_id)]

    with load_population() as src:
        pop = src.read(1)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(pop, cmap="OrRd", extent=extent, alpha=0.7)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)

    plot_buffers(ax, cow, "#2ecc71", 0.3)
    plot_buffers(ax, power_bts, "#3498db", 0.3)

    ax.set_title(f"Figure: Population Restored ({method})", fontsize=14)
    ax.set_axis_off()

    ax.legend(handles=[
        Patch(color="#2ecc71", alpha=0.3, label="COW coverage"),
        Patch(color="#3498db", alpha=0.3, label="Power-restored BTS"),
    ])

    plt.savefig(OUTPUT_DIR / f"population_restored_{method}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# FIG 4.6b – FULL RESTORED
def plot_population_restored_full(method, cfg):
    boundary = load_boundary()
    active = load_bts(ACTIVE_BTS_PATH)
    cow = load_cow_assignments(cfg["cow"])
    power_assign = pd.read_csv(cfg["power"])

    bts_ref = load_bts(BTS_ALL_PATH)
    power_bts = bts_ref[bts_ref.site_id.isin(power_assign.bts_id)]

    with load_population() as src:
        pop = src.read(1)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(pop, cmap="OrRd", extent=extent, alpha=0.7)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)

    plot_buffers(ax, active, "#3ed15b", 0.25)
    plot_buffers(ax, cow, "#2ecc71", 0.3)
    plot_buffers(ax, power_bts, "#3498db", 0.3)

    ax.set_title(f"Figure: Full Restored Coverage ({method})", fontsize=14)
    ax.set_axis_off()

    ax.legend(handles=[
        Patch(color="#3ed15b", alpha=0.25, label="Active BTS"),
        Patch(color="#2ecc71", alpha=0.3, label="COW"),
        Patch(color="#3498db", alpha=0.3, label="Power-restored BTS"),
    ])

    plt.savefig(
        OUTPUT_DIR / f"population_restored_full_{method}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()

# MAIN
def run():
    print("\nCoverage Impact Simulation – Population Maps")

    plot_population_loss()
    plot_population_full_before()

    for m, cfg in METHODS.items():
        plot_population_restored(m, cfg)
        plot_population_restored_full(m, cfg)

    print("Completed: Simulation of Coverage Impact Simulation – Population Maps\n")

if __name__ == "__main__":
    run()
