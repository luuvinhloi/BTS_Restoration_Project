import os
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_RASTER = PROJECT_ROOT / "data" / "processed" / "flood" / "flood_depth_combined.tif"
BOUNDARY_PATH = PROJECT_ROOT / "data" / "cleaned" / "hue_boundary_clean.geojson"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flood"
OUTPUT_IMG = OUTPUT_DIR / "flood_map_visualization.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading flood raster...")
with rasterio.open(INPUT_RASTER) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds

print("Loading boundary...")
boundary = gpd.read_file(BOUNDARY_PATH)

# IMPORTANT: convert boundary CRS → raster CRS for masking
boundary = boundary.to_crs(raster_crs)

# ============================================================
# CLIP RASTER USING BOUNDARY
# ============================================================

print("Clipping raster by boundary...")
with rasterio.open(INPUT_RASTER) as src:
    geometry = [boundary.geometry.unary_union]
    clipped_array, clipped_transform = mask(src, geometry, crop=True)
    clipped_array = clipped_array[0]   # (1, H, W) → (H, W)

# ============================================================
# PREPARE COLOR MAP FOR FLOOD DEPTH
# ============================================================

colors = [
    (1.0, 1.0, 1.0, 0),        # 0.0m – transparent
    (0.65, 0.80, 0.91, 1.0),  # 0.2m – light blue
    (0.12, 0.47, 0.71, 1.0),  # 0.5m – blue
    (0.13, 0.37, 0.64, 1.0),  # 1.0m – deeper blue
    (0.03, 0.11, 0.35, 1.0)   # 2.0m – darkest blue
]

cmap = ListedColormap(colors)

# Convert boundary to Web Mercator (EPSG:3857) for basemap
boundary_web = boundary.to_crs(epsg=3857)

# Convert raster bounds to EPSG:3857 for plotting
with rasterio.open(INPUT_RASTER) as src:
    bounds_3857 = transform_bounds(src.crs, "EPSG:3857", *src.bounds)

extent = (
    bounds_3857[0],  # minX
    bounds_3857[2],  # maxX
    bounds_3857[1],  # minY
    bounds_3857[3],  # maxY
)

# ============================================================
# PLOT FLOOD MAP
# ============================================================

print("Rendering map...")

fig, ax = plt.subplots(figsize=(12, 10))

# Draw flood raster
ax.imshow(
    clipped_array,
    cmap=cmap,
    extent=extent,
    interpolation="nearest",
    vmin=0,
    vmax=2
)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)

# Draw boundary outline
boundary_web.plot(
    ax=ax,
    facecolor="none",
    edgecolor="red",
    linewidth=1.2
)

# ============================================================
# LEGEND
# ============================================================

legend_patches = [
    mpatches.Patch(color=colors[1], label="Ngập 0.2 m"),
    mpatches.Patch(color=colors[2], label="Ngập 0.5 m"),
    mpatches.Patch(color=colors[3], label="Ngập 1.0 m"),
    mpatches.Patch(color=colors[4], label="Ngập 2.0 m"),
]

ax.legend(handles=legend_patches, title="Mức độ ngập", loc="lower left")

ax.set_title("Bản đồ mô phỏng ngập lụt – TP Huế", fontsize=16)
ax.set_axis_off()

# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight")
print(f"Visualization saved to: {OUTPUT_IMG}")

plt.show()
