"""
Flood Map Visualization – Hydrology Style Coloring (Custom Colors)
+ Overlay I_points.csv & J_sites.csv
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import folium
import rasterio.features as rfeatures

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RASTER_CLEAN = PROJECT_ROOT / "data" / "processed" / "flood" / "flood_depth_combined_B_clean.tif"
RASTER_RAW = PROJECT_ROOT / "data" / "processed" / "flood" / "flood_depth_combined_B.tif"
WATER_POLY = PROJECT_ROOT / "data" / "cleaned" / "water_hue_clean.geojson"
BOUNDARY_PATH = PROJECT_ROOT / "data" / "cleaned" / "hue_boundary_clean.geojson"

# NEW — file paths for I & J sets
I_POINTS_PATH = PROJECT_ROOT / "data" / "processed" / "position_I_J" / "I_points_B.csv"
J_SITES_PATH  = PROJECT_ROOT / "data" / "processed" / "position_I_J" / "J_sites_B.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUTPUT = OUTPUT_DIR / "visualization_all.png"
HTML_OUTPUT = OUTPUT_DIR / "visualization_all.html"


# ---------------------------------------------------------
# LOAD RASTER
# ---------------------------------------------------------
def _load_raster():
    raster_path = RASTER_CLEAN if RASTER_CLEAN.exists() else RASTER_RAW

    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        bounds = src.bounds
        crs = src.crs
        transform = src.transform

    print(f"[INFO] Loaded raster: {raster_path}")
    print("Unique values:", np.unique(arr))
    return arr, bounds, transform, crs


# ---------------------------------------------------------
# LOAD WATER MASK
# ---------------------------------------------------------
def _load_water_mask(shape_hw, transform):
    water_gdf = gpd.read_file(WATER_POLY)
    if water_gdf.empty:
        return np.zeros(shape_hw, dtype=np.uint8)

    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in water_gdf.geometry],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask


# ---------------------------------------------------------
# LOAD BOUNDARY
# ---------------------------------------------------------
def _load_boundary():
    gdf = gpd.read_file(BOUNDARY_PATH)
    return gdf.to_crs(epsg=4326)


# ---------------------------------------------------------
# LOAD I_points & J_sites
# ---------------------------------------------------------
def load_points():
    print("[INFO] Loading I_points.csv & J_sites.csv ...")

    df_I = pd.read_csv(I_POINTS_PATH)
    df_J = pd.read_csv(J_SITES_PATH)

    # Chuyển sang GeoDataFrame để dễ xử lý
    gdf_I = gpd.GeoDataFrame(
        df_I,
        geometry=gpd.points_from_xy(df_I.longitude, df_I.latitude),
        crs="EPSG:4326"
    )

    gdf_J = gpd.GeoDataFrame(
        df_J,
        geometry=gpd.points_from_xy(df_J.longitude, df_J.latitude),
        crs="EPSG:4326"
    )

    return gdf_I, gdf_J


# ---------------------------------------------------------
# CUSTOM FLOOD COLOR MAP
# ---------------------------------------------------------
HYDRO_COLORS = [
    (1, 1, 1, 0),                      # 0.0m transparent
    (21/255, 134/255, 125/255, 1),     # 0.2m
    (31/255, 120/255, 149/255, 1),     # 0.5m
    (33/255, 66/255, 155/255, 1),      # 1.0m
    (27/255, 42/255, 111/255, 1),      # 2.0m
]

WATER_COLOR = (33/255, 139/255, 231/255, 1)


# ---------------------------------------------------------
# HTML EXPORT (with I & J points)
# ---------------------------------------------------------
def _export_html(arr, bounds, boundary, water_mask, gdf_I, gdf_J):

    flood_levels = [0.0, 0.2, 0.5, 1.0, 2.0]

    index_arr = np.zeros(arr.shape, dtype=np.uint8)
    for i, lvl in enumerate(flood_levels):
        index_arr[arr == lvl] = i

    overlay_png = OUTPUT_DIR / "overlay_flood_B.png"
    plt.imsave(overlay_png, index_arr, cmap=ListedColormap(HYDRO_COLORS))

    water_png = OUTPUT_DIR / "overlay_water_B.png"
    plt.imsave(water_png,
               np.where(water_mask == 1, 1, np.nan),
               cmap=ListedColormap([WATER_COLOR]))

    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    # Flood overlay
    folium.raster_layers.ImageOverlay(
        name="Flood Depth",
        image=str(overlay_png),
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.60
    ).add_to(m)

    # Water overlay
    folium.raster_layers.ImageOverlay(
        name="Rivers & Lakes",
        image=str(water_png),
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.90
    ).add_to(m)

    # Boundary
    folium.GeoJson(
        boundary,
        name="Boundary",
        style_function=lambda x: {"color": "red", "weight": 2, "fill": False}
    ).add_to(m)

    # ------------------------------
    # ADD I POINTS (RED MARKERS)
    # ------------------------------
    for _, row in gdf_I.iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=4,
            color="#ff0000",
            fill=True,
            fill_color="#ff0000",
            popup=(
                f"<b>I Point</b><br>"
                f"ID: {row.site_id}<br>"
                f"Pop: {float(row['pop']):.0f}<br>"
                f"Priority: {row.priority_category} ({row.priority_weight})"
            )
        ).add_to(m)

    # ------------------------------
    # ADD J SITES (ORANGE MARKERS)
    # ------------------------------
    for _, row in gdf_J.iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=4,
            color="#ffaa00",
            fill=True,
            fill_color="#ffaa00",
            popup=(
                f"<b>J Site</b><br>"
                f"ID: {row.site_id}<br>"
                f"Linked I: {row.i_ref}<br>"
                f"Pop: {float(row['pop']):.0f}<br>"
                f"Priority: {row.priority_category} ({row.priority_weight})<br>"
                f"Slope: {row.slope}<br>"
                f"In water: {row.in_water}"
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(HTML_OUTPUT)

    print("[INFO] HTML saved →", HTML_OUTPUT)


# ---------------------------------------------------------
# PUBLIC ENTRY
# ---------------------------------------------------------
def run_map_visualization_all():
    print("\n=== FLOOD MAP VISUALIZATION (WITH I & J POINTS) START ===")

    arr, bounds, transform, crs = _load_raster()
    boundary = _load_boundary()
    water_mask = _load_water_mask(arr.shape, transform)
    gdf_I, gdf_J = load_points()

    _export_html(arr, bounds, boundary, water_mask, gdf_I, gdf_J)

    print("=== FLOOD MAP VISUALIZATION COMPLETE ===\n")
