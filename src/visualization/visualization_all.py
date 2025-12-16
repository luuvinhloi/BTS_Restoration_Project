"""
Unified Visualization: Flood + I/J Points + BTS Active / Power Outage / Failed
Author: Lợi Lưu – 2025
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import folium
import rasterio.features as rfeatures

# ============================================================
# CONFIG PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Flood raster
RASTER_CLEAN = PROJECT_ROOT / "data/processed/flood/flood_depth_combined_clean.tif"
RASTER_RAW   = PROJECT_ROOT / "data/processed/flood/flood_depth_combined.tif"
WATER_POLY   = PROJECT_ROOT / "data/cleaned/water_hue_clean.geojson"
BOUNDARY     = PROJECT_ROOT / "data/cleaned/hue_boundary_clean.geojson"

# I / J datasets
I_POINTS  = PROJECT_ROOT / "data/processed/position_I_J/I_points.csv"
J_SITES   = PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"

# BTS datasets
ACTIVE_CSV = PROJECT_ROOT / "data/processed/damage_bts/active_bts.csv"
FAILED_CSV = PROJECT_ROOT / "data/processed/damage_bts/failed_bts.csv"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_OUTPUT = OUTPUT_DIR / "visualization_combined.html"


# ============================================================
# LOAD RASTER
# ============================================================
def load_raster():
    raster_path = RASTER_CLEAN if RASTER_CLEAN.exists() else RASTER_RAW
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
    return arr, bounds, transform, crs


# ============================================================
# LOAD WATER MASK
# ============================================================
def load_water_mask(shape_hw, transform):
    if not WATER_POLY.exists():
        return np.zeros(shape_hw, dtype=np.uint8)

    gdf = gpd.read_file(WATER_POLY)
    mask = rfeatures.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=shape_hw, transform=transform, fill=0, dtype=np.uint8
    )
    return mask


# ============================================================
# LOAD I / J POINTS
# ============================================================
def load_IJ_points():
    df_I = pd.read_csv(I_POINTS)
    df_J = pd.read_csv(J_SITES)

    gdf_I = gpd.GeoDataFrame(
        df_I, geometry=gpd.points_from_xy(df_I.longitude, df_I.latitude), crs="EPSG:4326"
    )
    gdf_J = gpd.GeoDataFrame(
        df_J, geometry=gpd.points_from_xy(df_J.longitude, df_J.latitude), crs="EPSG:4326"
    )
    return gdf_I, gdf_J


# ============================================================
# LOAD BTS DATA
# ============================================================
def load_bts():
    df_active = pd.read_csv(ACTIVE_CSV)

    df_failed = pd.read_csv(FAILED_CSV)
    df_failed_failed = df_failed[df_failed["status"] == "failed"]
    df_failed_power  = df_failed[df_failed["status"] == "power_outage"]

    gdf_active = gpd.GeoDataFrame(
        df_active, geometry=gpd.points_from_xy(df_active.longitude, df_active.latitude), crs="EPSG:4326"
    )
    gdf_failed = gpd.GeoDataFrame(
        df_failed_failed, geometry=gpd.points_from_xy(df_failed_failed.longitude, df_failed_failed.latitude), crs="EPSG:4326"
    )
    gdf_power = gpd.GeoDataFrame(
        df_failed_power, geometry=gpd.points_from_xy(df_failed_power.longitude, df_failed_power.latitude), crs="EPSG:4326"
    )

    return gdf_active, gdf_power, gdf_failed


# ============================================================
# BTS POPUP BUILDER
# ============================================================
def build_bts_popup(row):
    status_text = {
        "active": "<span style='color:green;'>HOẠT ĐỘNG</span>",
        "power_outage": "<span style='color:#e6b800;'>MẤT NGUỒN ĐIỆN</span>",
        "failed": "<span style='color:red;'>HƯ HỎNG NẶNG</span>",
    }.get(row.get("status", ""), "N/A")

    html = f"""
    <b>THÔNG TIN TRẠM BTS</b><br>
    <b>ID:</b> {row.get("site_id","N/A")}<br>
    <b>Loại:</b> {row.get("bts_type","N/A")}<br>
    <b>Trạng thái:</b> {status_text}<br>
    <hr>
    <b>Lat:</b> {row.get("latitude"):.6f}<br>
    <b>Lon:</b> {row.get("longitude"):.6f}<br>
    """
    return folium.Popup(html, max_width=350)


# ============================================================
# ADD BTS LAYER
# ============================================================
def add_bts_layer(gdf, map_obj, name, color):
    fg = folium.FeatureGroup(name=name, show=False)
    for _, row in gdf.iterrows():
        popup = build_bts_popup(row)
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3.5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9
        ).add_child(popup).add_to(fg)
    fg.add_to(map_obj)


# ============================================================
# MAIN FUNCTION
# ============================================================
def run_visualization_combined():
    print("\n=== Combined Flood + I/J + BTS Visualization ===")

    arr, bounds, transform, crs = load_raster()
    water_mask = load_water_mask(arr.shape, transform)
    boundary = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    gdf_I, gdf_J = load_IJ_points()
    gdf_active, gdf_power, gdf_failed = load_bts()

    # Create flood map
    flood_levels = [0.0, 0.2, 0.5, 1.0, 2.0]
    flood_colors = [
        (1,1,1,0),
        (21/255,134/255,125/255,1),
        (31/255,120/255,149/255,1),
        (33/255,66/255,155/255,1),
        (27/255,42/255,111/255,1),
    ]
    idx = np.zeros(arr.shape, dtype=np.uint8)
    for i, lvl in enumerate(flood_levels):
        idx[arr == lvl] = i

    flood_png = OUTPUT_DIR / "overlay_flood.png"
    plt.imsave(flood_png, idx, cmap=ListedColormap(flood_colors))

    water_png = OUTPUT_DIR / "overlay_water.png"
    plt.imsave(water_png, np.where(water_mask==1,1,np.nan),
               cmap=ListedColormap([(33/255,139/255,231/255,1)]))

    # Map center
    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    # Flood overlays
    folium.raster_layers.ImageOverlay(
        name="Flood Depth", image=str(flood_png),
        bounds=[[bounds.bottom, bounds.left],[bounds.top, bounds.right]],
        opacity=0.55
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        name="Water Bodies", image=str(water_png),
        bounds=[[bounds.bottom, bounds.left],[bounds.top, bounds.right]],
        opacity=0.8
    ).add_to(m)

    # Boundary
    folium.GeoJson(
        boundary,
        name="Boundary",
        style_function=lambda x: {"color":"red","weight":2,"fillOpacity":0}
    ).add_to(m)

    # I Points
    fg_I = folium.FeatureGroup(name="I Points (Demand)", show=False)
    for _, row in gdf_I.iterrows():
        popup = f"<b>I Point</b><br>ID: {row.site_id}<br>Population: {row['pop']}"
        folium.Marker(
            location=[row.latitude, row.longitude],
            icon=folium.Icon(color="red", icon="info-sign"),
            popup=popup
        ).add_to(fg_I)
    fg_I.add_to(m)

    # J Sites
    fg_J = folium.FeatureGroup(name="J Sites (Candidate Locations)", show=False)
    for _, row in gdf_J.iterrows():
        popup = f"<b>J Site</b><br>ID: {row.site_id}<br>Pop: {row['pop']}"
        folium.Marker(
            location=[row.latitude, row.longitude],
            icon=folium.Icon(color="orange", icon="flag"),
            popup=popup
        ).add_to(fg_J)
    fg_J.add_to(m)

    # BTS layers
    add_bts_layer(gdf_active, m, "BTS Active", "green")
    add_bts_layer(gdf_power,  m, "BTS Power Outage", "#e6b800")
    add_bts_layer(gdf_failed, m, "BTS Failed", "red")

    # ---------- LAYER CONTROL COLLAPSED ----------
    folium.LayerControl(collapsed=True).add_to(m)

    # Save HTML
    m.save(HTML_OUTPUT)
    print("Saved HTML →", HTML_OUTPUT)
    print("=== DONE ===\n")


if __name__ == "__main__":
    run_visualization_combined()
