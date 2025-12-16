#!/usr/bin/env python3
"""
spatial_deployment_map.py

SPATIAL DEPLOYMENT & COVERAGE RESTORATION MAP

Spatial Deployment of COW and Backup Power Units
and Resulting Coverage Restoration

Outputs (per method):
- MILP
- GA_PSO
- MILP_GA_PSO

Each method exports:
- PNG (static, thesis-ready)
- HTML (interactive, layer-controlled)

Author: Lợi Lưu
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import folium

# PATH CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOUNDARY_PATH = PROJECT_ROOT / "data/cleaned/hue_boundary_clean.geojson"
FLOOD_RASTER  = PROJECT_ROOT / "data/processed/flood/flood_depth_combined_clean.tif"

ACTIVE_BTS_PATH  = PROJECT_ROOT / "data/processed/damage_bts/active_bts.csv"
FAILED_BTS_PATH  = PROJECT_ROOT / "data/processed/damage_bts/failed_bts.csv"
J_SITES_PATH     = PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"
COW_DATASET_PATH = PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/simulation/spatial"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# METHOD CONFIG
METHODS = {
    "MILP": {
        "cow": PROJECT_ROOT / "outputs/milp_runs/milp_gurobi/assignments_cow_GUROBI.csv",
        "power": PROJECT_ROOT / "outputs/milp_runs/milp_gurobi/assignments_power_GUROBI.csv",
        "out": OUTPUT_ROOT / "milp"
    },
    "GA_PSO": {
        "cow": PROJECT_ROOT / "outputs/results_ga_pso/solution_cow_assignments.csv",
        "power": PROJECT_ROOT / "outputs/results_ga_pso/solution_power_assignments.csv",
        "out": OUTPUT_ROOT / "ga_pso"
    },
    "MILP_GA_PSO": {
        "cow": PROJECT_ROOT / "outputs/results_hybrid/solution_cow_assignments.csv",
        "power": PROJECT_ROOT / "outputs/results_hybrid/solution_power_assignments.csv",
        "out": OUTPUT_ROOT / "hybrid"
    }
}

# FLOOD STYLE (CONSISTENT WITH spatial_flood_map.py)
FLOOD_VALUES = [0.0, 0.2, 0.5, 1.0, 2.0]
FLOOD_COLORS = [
    (1, 1, 1, 0.0),
    (198/255, 219/255, 239/255, 1.0),
    (158/255, 202/255, 225/255, 1.0),
    (107/255, 174/255, 214/255, 1.0),
    (33/255, 113/255, 181/255, 1.0),
]
FLOOD_BOUNDS = [-0.01, 0.1, 0.35, 0.75, 1.5, 2.5]

# STYLE CONFIG
COLOR_FAILED  = "red"
COLOR_POWERED = "#6aa0f6"
COLOR_COW     = "#2ecc71"
COLOR_J       = "purple"
HTML_J_COLOR = "#4981f3"
HTML_BTS_ACTIVE_COLOR = "#2ecc71"

# LOADERS
def load_boundary():
    return gpd.read_file(BOUNDARY_PATH)

def load_flood():
    with rasterio.open(FLOOD_RASTER) as src:
        arr = src.read(1)
        bounds = src.bounds
    return arr, bounds

def load_bts():
    df = pd.concat([
        pd.read_csv(ACTIVE_BTS_PATH),
        pd.read_csv(FAILED_BTS_PATH)
    ], ignore_index=True)

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

def load_j_sites():
    df = pd.read_csv(J_SITES_PATH)
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

def load_cow_assignments(path):
    """
    Robust loader for ALL COW assignment outputs:
    - MILP
    - GA_PSO
    - MILP_GA_PSO

    Rules:
    1) Resolve coordinates (lat/lon OR site_id -> J_sites)
    2) coverage_radius_m:
       - use directly if exists
       - else attach from cow_dataset via cow_id
    """

    df = pd.read_csv(path)

    # STEP 1: Resolve coordinates
    if {"lat", "lon"}.issubset(df.columns):
        df["latitude"] = df["lat"]
        df["longitude"] = df["lon"]
    else:
        if "site_id" not in df.columns:
            raise ValueError(
                f"[ERROR] {path.name} has no lat/lon and no site_id column."
            )

        j_sites = pd.read_csv(J_SITES_PATH)

        df = df.merge(
            j_sites[["site_id", "latitude", "longitude"]],
            on="site_id",
            how="left"
        )

        if df[["latitude", "longitude"]].isna().any().any():
            raise ValueError(
                f"[ERROR] Some site_id in {path.name} not found in J_sites.csv."
            )

    # STEP 2: Resolve coverage_radius_m
    if "coverage_radius_m" not in df.columns:
        # MUST attach from cow_dataset
        if "cow_id" not in df.columns:
            raise ValueError(
                f"[ERROR] {path.name} has no cow_id column."
            )

        cow_ref = pd.read_csv(COW_DATASET_PATH)

        df = df.merge(
            cow_ref[["cow_id", "coverage_radius_m"]],
            on="cow_id",
            how="left"
        )

        if df["coverage_radius_m"].isna().any():
            missing = df[df["coverage_radius_m"].isna()]["cow_id"].unique()
            raise ValueError(
                f"[ERROR] cow_id not found in cow_dataset.csv: {missing}"
            )

    # STEP 3: Final GeoDataFrame
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )


# CORE DRAW FUNCTION (USED FOR ALL METHODS)
def run_deployment_map(method_name, cow_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    PNG_OUTPUT  = output_dir / "deployment_map.png"
    HTML_OUTPUT = output_dir / "deployment_map.html"

    boundary = load_boundary()
    flood, bounds = load_flood()
    bts = load_bts()
    j_sites = load_j_sites()
    cows = load_cow_assignments(cow_path)

    # ================= PNG =================
    fig, ax = plt.subplots(figsize=(13, 11))

    cmap = ListedColormap(FLOOD_COLORS)
    norm = BoundaryNorm(FLOOD_BOUNDS, cmap.N)

    ax.imshow(
        flood, cmap=cmap, norm=norm,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
    )

    boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)

    failed = bts[bts["status"] != "active"]
    failed.plot(ax=ax, color=COLOR_FAILED, markersize=14)

    powered = bts[bts["status"] == "active"]
    for _, r in powered.iterrows():
        ax.add_patch(
            plt.Circle(
                (r.longitude, r.latitude),
                r.coverage_radius_m / 111000,
                color=COLOR_POWERED,
                alpha=0.08
            )
        )
    powered.plot(ax=ax, color=COLOR_POWERED, markersize=16)

    for _, r in cows.iterrows():
        ax.add_patch(
            plt.Circle(
                (r.geometry.x, r.geometry.y),
                r.coverage_radius_m / 111000,
                color=COLOR_COW,
                alpha=0.08
            )
        )
    cows.plot(ax=ax, color=COLOR_COW, marker="^", markersize=60)

    j_sites.plot(ax=ax, color=COLOR_J, marker="*", markersize=70)

    ax.set_title(
        f"Figure: Spatial Deployment and Coverage Restoration ({method_name})",
        fontsize=14
    )
    ax.set_axis_off()

    legend = [
        Patch(color=COLOR_FAILED, label="Failed / Power Outage BTS"),
        Patch(color=COLOR_POWERED, label="Restored BTS Coverage"),
        Patch(color=COLOR_COW, label="COW Coverage"),
        Line2D([0],[0], marker="*", color="w",
               markerfacecolor=COLOR_J, markersize=10, label="J Sites")
    ]
    ax.legend(handles=legend, loc="lower left")

    plt.tight_layout()
    plt.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    # ================= HTML =================
    idx = np.zeros(flood.shape, dtype=np.uint8)
    for i, v in enumerate(FLOOD_VALUES):
        idx[flood == v] = i

    overlay_png = output_dir / "flood_overlay_tmp.png"
    plt.imsave(overlay_png, idx, cmap=ListedColormap(FLOOD_COLORS))

    m = folium.Map(
        location=[(bounds.top+bounds.bottom)/2, (bounds.left+bounds.right)/2],
        zoom_start=11,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    fg_flood = folium.FeatureGroup(name="Flood Depth")
    folium.raster_layers.ImageOverlay(
        image=str(overlay_png),
        bounds=[[bounds.bottom, bounds.left],[bounds.top, bounds.right]],
        opacity=0.6
    ).add_to(fg_flood)
    fg_flood.add_to(m)

    folium.GeoJson(
        boundary,
        name="Boundary",
        style_function=lambda x: {"color":"yellow","weight":2,"fillOpacity":0}
    ).add_to(m)

    fg_failed = folium.FeatureGroup(name="Failed / Power Outage BTS")
    fg_powered = folium.FeatureGroup(name="Powered BTS Coverage")
    fg_cow = folium.FeatureGroup(name="COW Deployment & Coverage")
    fg_j = folium.FeatureGroup(name="J Sites")

    for _, r in failed.iterrows():
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=5,
            color=COLOR_FAILED,
            fill=True,
            fill_opacity=0.9
        ).add_to(fg_failed)

    for _, r in powered.iterrows():
        # 1. Fixed-size BTS marker (CENTER)
        folium.Marker(
            location=[r.latitude, r.longitude],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    width:12px;
                    height:12px;
                    background:{HTML_BTS_ACTIVE_COLOR};
                    border-radius:50%;
                    border:1px solid white;
                    box-shadow:0 0 3px rgba(0,0,0,0.6);
                "></div>
                """
            ),
            popup=f"""
            <b>BTS ID:</b> {r.get('site_id', 'N/A')}<br>
            <b>Status:</b> Restored<br>
            <b>Coverage radius:</b> {r.coverage_radius_m} m
            """
        ).add_to(fg_powered)

        # 2. Coverage buffer
        folium.Circle(
            location=[r.latitude, r.longitude],
            radius=float(r.coverage_radius_m),
            color=COLOR_POWERED,
            weight=1,
            opacity=0.6,
            fill=True,
            fill_color=COLOR_POWERED,
            fill_opacity=0.15
        ).add_to(fg_powered)

    for _, r in cows.iterrows():
        lat = r.geometry.y
        lon = r.geometry.x

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="green", icon="signal")
        ).add_to(fg_cow)

        folium.Circle(
            location=[lat, lon],
            radius=r.coverage_radius_m,
            color=COLOR_COW,
            fill=True,
            fill_opacity=0.15
        ).add_to(fg_cow)

    for _, r in j_sites.iterrows():
        folium.Marker(
            location=[r.latitude, r.longitude],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size:12px;
                    color:{HTML_J_COLOR};
                    text-shadow:0 0 2px white;
                ">★</div>
                """
            )
        ).add_to(fg_j)

    for fg in [fg_failed, fg_powered, fg_cow, fg_j]:
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(HTML_OUTPUT)

    print(f"[DONE] {method_name} at: PNG & HTML exported")

# MAIN
def run():
    print("\nSPATIAL DEPLOYMENT MAP (MULTI-METHOD)")
    for name, cfg in METHODS.items():
        run_deployment_map(name, cfg["cow"], cfg["out"])
    print("Completed: Simulation of Spatial Deployment and Coverage Restoration\n")

if __name__ == "__main__":
    run()
