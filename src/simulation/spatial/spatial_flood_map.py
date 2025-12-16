"""
spatial_flood_map.py

SPATIAL FLOOD VISUALIZATION
Simulation of Study Area and Flood Severity Map

Mô phỏng:
- Boundary khu vực nghiên cứu (Huế)
- Raster flood depth (độ sâu ngập)
- Xuất bản đồ tĩnh (PNG) cho báo cáo
- Xuất bản đồ tương tác (HTML) kiểu Google Maps

Author: Lợi Lưu
Generated & optimized for thesis-quality visualization
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import folium

# PATH CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOUNDARY_PATH = PROJECT_ROOT / "data/cleaned/hue_boundary_clean.geojson"
FLOOD_RASTER  = PROJECT_ROOT / "data/processed/flood/flood_depth_combined_clean.tif"

OUTPUT_DIR = PROJECT_ROOT / "outputs/simulation/spatial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUTPUT  = OUTPUT_DIR / "flood_map.png"
HTML_OUTPUT = OUTPUT_DIR / "flood_map.html"

# FLOOD LEVELS – MUST MATCH flood_simulation.py
# Actual flood depth values in raster
FLOOD_VALUES = [0.0, 0.2, 0.5, 1.0, 2.0]

# Color scheme (from light to dark)
FLOOD_COLORS = [
    (1, 1, 1, 0.0),          # 0.0 m (no flood)
    (198/255, 219/255, 239/255, 1.0),  # 0.2 m
    (158/255, 202/255, 225/255, 1.0),  # 0.5 m
    (107/255, 174/255, 214/255, 1.0),  # 1.0 m
    (33/255, 113/255, 181/255, 1.0),   # 2.0 m
]

# Boundaries for normalization
FLOOD_BOUNDS = [-0.01, 0.1, 0.35, 0.75, 1.5, 2.5]

# LOAD DATA
def load_boundary():
    return gpd.read_file(BOUNDARY_PATH)

def load_flood_raster():
    with rasterio.open(FLOOD_RASTER) as src:
        arr = src.read(1)
        bounds = src.bounds
        crs = src.crs
    return arr, bounds, crs

# EXPORT STATIC PNG (NO BASEMAP – STABLE)
def export_png():
    boundary = load_boundary()
    arr, bounds, crs = load_flood_raster()

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = ListedColormap(FLOOD_COLORS)
    norm = BoundaryNorm(FLOOD_BOUNDS, cmap.N)

    ax.imshow(
        arr,
        cmap=cmap,
        norm=norm,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
    )

    boundary.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.5
    )

    ax.set_title(
        "Figure: Study Area and Flood Severity Map",
        fontsize=14,
        pad=12
    )
    ax.set_axis_off()

    legend_items = [
        Patch(color=FLOOD_COLORS[1], label="0.2 m"),
        Patch(color=FLOOD_COLORS[2], label="0.5 m"),
        Patch(color=FLOOD_COLORS[3], label="1.0 m"),
        Patch(color=FLOOD_COLORS[4], label="2.0 m"),
    ]

    ax.legend(
        handles=legend_items,
        title="Flood Depth",
        loc="lower left",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] PNG saved at: {PNG_OUTPUT}")

# EXPORT INTERACTIVE HTML (GOOGLE SATELLITE)
def export_html():
    arr, bounds, _ = load_flood_raster()
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")

    # Encode flood values to index image
    idx = np.zeros(arr.shape, dtype=np.uint8)
    for i, val in enumerate(FLOOD_VALUES):
        idx[arr == val] = i

    overlay_png = OUTPUT_DIR / "flood_overlay_tmp.png"
    plt.imsave(overlay_png, idx, cmap=ListedColormap(FLOOD_COLORS))

    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
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

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(HTML_OUTPUT)

    print(f"[INFO] HTML saved at: {HTML_OUTPUT}")

# MAIN
def run():
    print("\nSpatial Food Map Simulation")
    export_png()
    export_html()
    print("Completed: Simulation of Study Area and Flood Severity Map!\n")

if __name__ == "__main__":
    run()
