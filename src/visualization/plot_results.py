# Hiển thị vùng phủ, trạm
"""
Produce static plots and simple summary tables (matplotlib + geopandas).
"""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

def plot_sites_and_population(i_csv, j_csv, chosen_site_ids, boundary_geojson, out_png):
    pts = pd.read_csv(i_csv)
    sites = pd.read_csv(j_csv)
    boundary = gpd.read_file(boundary_geojson)
    fig, ax = plt.subplots(figsize=(10,10))
    boundary.plot(ax=ax, color='none', edgecolor='black')

    # Dò tên cột hợp lệ (tự động nhận diện x/y hoặc lon/lat)
    x_col = 'x' if 'x' in pts.columns else 'longitude'
    y_col = 'y' if 'y' in pts.columns else 'latitude'
    sx_col = 'x' if 'x' in sites.columns else 'longitude'
    sy_col = 'y' if 'y' in sites.columns else 'latitude'
    id_col = 'id' if 'id' in sites.columns else 'site_id'

    # Vẽ các điểm
    ax.scatter(pts[x_col], pts[y_col], s=5, alpha=0.6, label='population points')
    ax.scatter(sites[sx_col], sites[sy_col], s=60, c='orange', marker='^', label='candidate sites')

    # Các site được chọn
    chosen = sites[sites[id_col].isin(chosen_site_ids)]
    ax.scatter(chosen[sx_col], chosen[sy_col], s=120, c='red', marker='*', label='chosen COW')

    ax.legend()
    plt.title("Population points and chosen COW sites")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

def main():
    # load result summary
    res = json.load(open(PROJECT_ROOT / "outputs" / "results" / "milp_solution_summary.json"))
    chosen_ids = res['chosen_site_ids']
    plot_sites_and_population(PROJECT_ROOT / "data" / "processed" / "I_points.csv",
                              PROJECT_ROOT / "data" / "processed" / "J_sites.csv",
                              chosen_ids,
                              PROJECT_ROOT / "data" / "raw" / "hue_boundary.geojson",
                              PROJECT_ROOT / "outputs" / "maps" / "milp_map.png")

if __name__ == "__main__":
    main()
