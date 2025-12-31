"""
spatial_bts_status.py

BTS STATUS SPATIAL VISUALIZATION
Spatial Distribution of BTS Status After Natural Disaster

Mô phỏng:
- Flood depth
- Boundary khu vực nghiên cứu
- BTS theo trạng thái:
    + Active
    + Power outage
    + Failed
- PNG: flood raster + BTS
- HTML: Google Satellite + flood raster + BTS

Author: Lợi Lưu
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import folium

# PATH CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOUNDARY_PATH = PROJECT_ROOT / "data/cleaned/hue_boundary_clean.geojson"
FLOOD_RASTER  = PROJECT_ROOT / "data/processed/flood/flood_depth_combined_clean.tif"

ACTIVE_BTS_PATH = PROJECT_ROOT / "data/processed/damage_bts/active_bts_B.csv"
FAILED_BTS_PATH = PROJECT_ROOT / "data/processed/damage_bts/failed_bts_B.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs/simulation_B/spatial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUTPUT  = OUTPUT_DIR / "bts_status.png"
HTML_OUTPUT = OUTPUT_DIR / "bts_status.html"

# FLOOD STYLE (MUST MATCH spatial_flood_map.py)
FLOOD_VALUES = [0.0, 0.2, 0.5, 1.0, 2.0]

FLOOD_COLORS = [
    (1, 1, 1, 0.0),          # no flood
    (198/255, 219/255, 239/255, 1.0),  # 0.2 m
    (158/255, 202/255, 225/255, 1.0),  # 0.5 m
    (107/255, 174/255, 214/255, 1.0),  # 1.0 m
    (33/255, 113/255, 181/255, 1.0),   # 2.0 m
]

FLOOD_BOUNDS = [-0.01, 0.1, 0.35, 0.75, 1.5, 2.5]

# BTS STYLE
STATUS_STYLE = {
    "active":        {"color": "green",  "label": "Active BTS"},
    "power_outage":  {"color": "orange", "label": "Power Outage BTS"},
    "failed":        {"color": "red",    "label": "Failed BTS"},
}

PNG_MARKER_SIZE = 18
HTML_MARKER_RADIUS = 6   # fixed size, no scaling

# LOAD DATA
def load_boundary():
    return gpd.read_file(BOUNDARY_PATH)

def load_flood():
    with rasterio.open(FLOOD_RASTER) as src:
        arr = src.read(1)
        bounds = src.bounds
    return arr, bounds

def load_bts():
    active = pd.read_csv(ACTIVE_BTS_PATH)
    failed = pd.read_csv(FAILED_BTS_PATH)
    df = pd.concat([active, failed], ignore_index=True)

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

# EXPORT PNG (FLOOD RASTER + BTS)
def export_png():
    boundary = load_boundary()
    bts = load_bts()
    flood_arr, bounds = load_flood()

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = ListedColormap(FLOOD_COLORS)
    norm = BoundaryNorm(FLOOD_BOUNDS, cmap.N)

    ax.imshow(
        flood_arr,
        cmap=cmap,
        norm=norm,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
    )

    boundary.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.5
    )

    for status, cfg in STATUS_STYLE.items():
        subset = bts[bts["status"] == status]
        if subset.empty:
            continue

        subset.plot(
            ax=ax,
            color=cfg["color"],
            markersize=PNG_MARKER_SIZE,
            alpha=0.85,
            label=cfg["label"]
        )

    ax.set_title(
        "Spatial Distribution of BTS Status After Disaster",
        fontsize=14,
        pad=12
    )
    ax.set_axis_off()

    legend_items = (
        [Patch(color=FLOOD_COLORS[i], label=f"{FLOOD_VALUES[i]} m")
         for i in range(1, len(FLOOD_VALUES))] +
        [Line2D([0], [0], marker='o', color='w',
                markerfacecolor=cfg["color"], markersize=8,
                label=cfg["label"])
         for cfg in STATUS_STYLE.values()]
    )

    ax.legend(handles=legend_items, loc="lower left", frameon=True)

    # Coverage buffer for ACTIVE BTS
    active_bts = bts[bts["status"] == "active"]

    for _, row in active_bts.iterrows():
        radius_m = row.get("coverage_radius_m", None)
        if pd.notna(radius_m) and radius_m > 0:
            circle = plt.Circle(
                (row.longitude, row.latitude),
                radius_m / 111000,  # meter → degree (xấp xỉ)
                color="#6aa0f6",
                fill=True,
                alpha=0.08,
                linewidth=0
            )
            ax.add_patch(circle)

    plt.tight_layout()
    plt.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] PNG saved at: {PNG_OUTPUT}")

# EXPORT HTML (SATELLITE + FLOOD + FIXED BTS)
def export_html():
    flood_arr, bounds = load_flood()
    boundary = load_boundary().to_crs("EPSG:4326")
    bts = load_bts()

    idx = np.zeros(flood_arr.shape, dtype=np.uint8)
    for i, val in enumerate(FLOOD_VALUES):
        idx[flood_arr == val] = i

    overlay_png = OUTPUT_DIR / "flood_overlay_bts_tmp.png"
    plt.imsave(overlay_png, idx, cmap=ListedColormap(FLOOD_COLORS))

    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    folium.raster_layers.ImageOverlay(
        image=str(overlay_png),
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.65,
        name="Flood Depth"
    ).add_to(m)

    folium.GeoJson(
        boundary,
        name="Boundary",
        style_function=lambda x: {
            "color": "yellow",
            "weight": 2,
            "fillOpacity": 0
        }
    ).add_to(m)

    for status, cfg in STATUS_STYLE.items():
        fg = folium.FeatureGroup(name=cfg["label"], show=True)
        subset = bts[bts["status"] == status]

        for _, row in subset.iterrows():
            popup_html = f"""
            <b>BTS ID:</b> {row.get('site_id', 'N/A')}<br>
            <b>Status:</b> {status}<br>
            <b>Type:</b> {row.get('bts_type', 'N/A')}<br>
            <b>Power:</b> {row.get('power_W', 'N/A')} W<br>
            <b>Coverage radius:</b> {row.get('coverage_radius_m', 'N/A')} m
            """

            #BTS marker
            folium.Marker(
                location=[row.latitude, row.longitude],
                icon=folium.DivIcon(
                    html=f"""
                            <div style="
                                width:12px;
                                height:12px;
                                background:{cfg['color']};
                                border-radius:50%;
                                border:0.5px solid white;
                                box-shadow:0 0 3px rgba(0,0,0,0.6);
                            "></div>
                            """
                ),
                popup=popup_html
            ).add_to(fg)

            # Coverage buffer (ONLY ACTIVE BTS)
            if status == "active":
                radius = row.get("coverage_radius_m", None)
                if pd.notna(radius) and radius > 0:
                    folium.Circle(
                        location=[row.latitude, row.longitude],
                        radius=float(radius),
                        color="#6aa0f6",
                        weight=1,
                        opacity=0.6,
                        fill=True,
                        fill_color="#6aa0f6",
                        fill_opacity=0.15
                    ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(HTML_OUTPUT)

    print(f"[INFO] HTML saved at: {HTML_OUTPUT}")

# MAIN
def run():
    print("\nBTS Status Map Simulation")
    export_png()
    export_html()
    print("Completed: Simulation of Spatial Distribution of BTS Status After Natural Disaster\n")

if __name__ == "__main__":
    run()
