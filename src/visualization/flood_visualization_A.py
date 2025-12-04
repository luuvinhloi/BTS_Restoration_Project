"""
Flood Map Visualization – Hydrology Style Coloring (Custom Colors)
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
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

RASTER_CLEAN = PROJECT_ROOT / "data" / "processed" / "flood" / "flood_depth_combined_clean.tif"
RASTER_RAW = PROJECT_ROOT / "data" / "processed" / "flood" / "flood_depth_combined.tif"
WATER_POLY = PROJECT_ROOT / "data" / "cleaned" / "water_hue_clean.geojson"

BOUNDARY_PATH = PROJECT_ROOT / "data" / "cleaned" / "hue_boundary_clean.geojson"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flood"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUTPUT = OUTPUT_DIR / "flood_map_A.png"
HTML_OUTPUT = OUTPUT_DIR / "flood_map_A.html"


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

    mask = rfeatures.rasterize(
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
# CUSTOM FLOOD COLOR MAP (THEME YOU REQUESTED)
# ---------------------------------------------------------

HYDRO_COLORS = [
    (1, 1, 1, 0),                      # 0.0m transparent
    (21/255, 134/255, 125/255, 1),     # 0.2m  #15867d
    (31/255, 120/255, 149/255, 1),     # 0.5m  #1f7895
    (33/255, 66/255, 155/255, 1),      # 1.0m  #21429b
    (27/255, 42/255, 111/255, 1),      # 2.0m  #1b2a6f
]

WATER_COLOR = (33/255, 139/255, 231/255, 1)  # #218be7


# ---------------------------------------------------------
# PNG EXPORT
# ---------------------------------------------------------
def _export_png(arr, bounds, boundary, water_mask):

    cmap = ListedColormap(HYDRO_COLORS)
    norm = BoundaryNorm([-0.1, 0.1, 0.35, 0.75, 1.5, 3.0], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(arr, cmap=cmap, norm=norm)

    # Water overlay
    ax.imshow(
        np.where(water_mask == 1, 1, np.nan),
        cmap=ListedColormap([WATER_COLOR]),
        alpha=0.9
    )

    boundary.boundary.plot(ax=ax, color="red", linewidth=1.5)
    ax.set_axis_off()

    legend_items = [
        Patch(color=HYDRO_COLORS[i], label=f"Ngập {lvl} m")
        for i, lvl in enumerate([0.0, 0.2, 0.5, 1.0, 2.0]) if lvl != 0.0
    ]
    legend_items.append(Patch(color=WATER_COLOR, label="Sông, hồ tự nhiên"))

    ax.legend(handles=legend_items, loc="lower left", title="Chú thích")

    plt.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()
    print("[INFO] PNG saved →", PNG_OUTPUT)


# ---------------------------------------------------------
# HTML EXPORT
# ---------------------------------------------------------
def _export_html(arr, bounds, boundary, water_mask):

    flood_levels = [0.0, 0.2, 0.5, 1.0, 2.0]

    index_arr = np.zeros(arr.shape, dtype=np.uint8)
    for i, lvl in enumerate(flood_levels):
        index_arr[arr == lvl] = i

    overlay_png = OUTPUT_DIR / "overlay_flood_A.png"
    plt.imsave(overlay_png, index_arr, cmap=ListedColormap(HYDRO_COLORS))

    water_png = OUTPUT_DIR / "overlay_water_A.png"
    plt.imsave(water_png,
               np.where(water_mask == 1, 1, np.nan),
               cmap=ListedColormap([WATER_COLOR]))

    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB Positron")

    # Flood overlay
    folium.raster_layers.ImageOverlay(
        name="Flood Depth",
        image=str(overlay_png),
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.65
    ).add_to(m)

    # Water overlay
    folium.raster_layers.ImageOverlay(
        name="Rivers & Lakes",
        image=str(water_png),
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.95
    ).add_to(m)

    # Boundary
    folium.GeoJson(
        boundary,
        name="Boundary",
        style_function=lambda x: {"color": "red", "weight": 2, "fill": False}
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(HTML_OUTPUT)

    print("[INFO] HTML saved →", HTML_OUTPUT)


# ---------------------------------------------------------
# PUBLIC ENTRY
# ---------------------------------------------------------
def run_flood_map_visualization():
    print("\n=== FLOOD MAP VISUALIZATION (CUSTOM HYDRO COLORS) START ===")

    arr, bounds, transform, crs = _load_raster()
    boundary = _load_boundary()
    water_mask = _load_water_mask(arr.shape, transform)

    _export_png(arr, bounds, boundary, water_mask)
    _export_html(arr, bounds, boundary, water_mask)

    print("=== FLOOD MAP VISUALIZATION COMPLETE ===\n")
