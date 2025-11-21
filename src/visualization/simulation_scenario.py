"""
Simulation Scenario (Unified)
Tích hợp mô phỏng cho cả hai phương pháp:
    - MILP (Phương pháp 1)
    - GA–PSO (Phương pháp 2)

Chức năng:
    - Đọc kết quả tối ưu từ thư mục outputs
    - Hiển thị bản đồ Folium trực quan
    - Hiển thị bản đồ tĩnh (matplotlib)
"""

import folium
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from folium.plugins import MarkerCluster, Fullscreen
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

# 1. BẢN ĐỒ TƯƠNG TÁC - DÙNG CHUNG CHO CẢ 2 PHƯƠNG PHÁP
def create_interactive_map(method: str):
    """Sinh bản đồ tương tác hiển thị kết quả tối ưu."""
    boundary = gpd.read_file(DATA_DIR / "raw" / "hue_boundary.geojson")
    centroid = boundary.geometry.unary_union.centroid

    I_points = pd.read_csv(DATA_DIR / "processed" / "I_points.csv")
    J_sites = pd.read_csv(DATA_DIR / "processed" / "J_sites.csv")
    COWs = pd.read_csv(DATA_DIR / "raw" / "cow_dataset.csv")
    BTS = pd.read_csv(DATA_DIR / "raw" / "bts_ga.csv")

    if method == "MILP":
        result_path = OUT_DIR / "results" / "milp_solution_summary.json"
        if not result_path.exists():
            logging.error("Không tìm thấy kết quả MILP.")
            return None
        result = json.load(open(result_path))
        chosen_site_ids = set(result.get("chosen_site_ids", []))
        map_name = "milp_dashboard.html"
    else:
        result_path = OUT_DIR / "results" / "ga_pso_summary.json"
        if not result_path.exists():
            logging.error("Không tìm thấy kết quả GA–PSO.")
            return None
        result = json.load(open(result_path))
        assign_path = OUT_DIR / "results" / "ga_pso_assignments.csv"
        assign_df = pd.read_csv(assign_path)

        # Đọc danh sách site_id được chọn
        # Nếu cột assigned_site_id là chuỗi (dạng "J_00023" ...), dùng trực tiếp
        if "assigned_site_id" in assign_df.columns and assign_df["assigned_site_id"].dtype == object:
            chosen_site_ids = set(assign_df["assigned_site_id"].dropna().astype(str).unique().tolist())

        # Nếu cột chỉ là số index (1,2,3,...) thì map sang site_id trong J_sites
        else:
            chosen_site_ids = set()
            if "assigned_site_index" in assign_df.columns:
                for idx in assign_df["assigned_site_index"].dropna().astype(int).unique():
                    if 1 <= idx <= len(J_sites):
                        chosen_site_ids.add(str(J_sites.iloc[idx - 1]["site_id"]))
            elif "assigned_site_id" in assign_df.columns:
                for idx in assign_df["assigned_site_id"].dropna().astype(int).unique():
                    if 1 <= idx <= len(J_sites):
                        chosen_site_ids.add(str(J_sites.iloc[idx - 1]["site_id"]))

        map_name = "ga_pso_dashboard.html"

    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=9, tiles="CartoDB positron")
    Fullscreen().add_to(m)
    folium.GeoJson(boundary.to_json(), name="Boundary",
                   style_function=lambda x: {"color": "black", "weight": 2}).add_to(m)

    # Dân cư
    pop_layer = folium.FeatureGroup(name="Population Demand", show=True)
    for _, row in I_points.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(2, min(6, row.get("pop", 1) / 50)),
            color="blue",
            fill=True,
            fill_opacity=0.4,
            popup=f"<b>Pop:</b> {row.get('pop', 0):.1f}<br>"
                  f"<b>Priority:</b> {row.get('priority_category', 'normal')}<br>"
                  f"<b>Weight:</b> {row.get('priority_weight', 1.0)}"
        ).add_to(pop_layer)
    pop_layer.add_to(m)

    # BTS
    bts_layer = folium.FeatureGroup(name="BTS Stations", show=False)
    for _, row in BTS.iterrows():
        color = "green" if row.get("status", "active") == "active" else "gray"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"<b>BTS:</b> {row['site_id']} ({row.get('status', 'unknown')})"
        ).add_to(bts_layer)
    bts_layer.add_to(m)

    # Candidate sites
    cand_layer = folium.FeatureGroup(name="Candidate Sites", show=False)
    for _, row in J_sites.iterrows():
        color = "orange"
        if row["site_id"] in chosen_site_ids or row.get("i_ref") in chosen_site_ids:
            color = "red"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"<b>Site ID:</b> {row['site_id']}<br>"
                  f"<b>Slope:</b> {row.get('slope', 0):.1f}°<br>"
                  f"<b>Dist to road:</b> {row.get('dist_to_road_m', 0):.0f} m"
        ).add_to(cand_layer)
    cand_layer.add_to(m)

    # Chosen COWs + Coverage buffer
    chosen_layer = folium.FeatureGroup(name=f"Chosen COWs ({method})", show=True)
    for sid in chosen_site_ids:
        j_row = J_sites[J_sites["site_id"] == sid]
        if j_row.empty:
            j_row = J_sites[J_sites["i_ref"] == sid]
        if j_row.empty:
            continue

        lat, lon = float(j_row.iloc[0]["latitude"]), float(j_row.iloc[0]["longitude"])
        cow = COWs.sample(1).iloc[0]
        radius = float(cow.get("coverage_radius_m", 3000))
        cost = float(cow.get("cost_vnd", 0))
        speed = float(cow.get("speed_kmh", 40))
        cow_type = cow.get("type", "N/A")

        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color="red",
            fill=False,
            opacity=0.3,
            weight=2
        ).add_to(chosen_layer)

        popup_html = (
            f"<b>COW Site:</b> {sid}<br>"
            f"<b>Coverage:</b> {radius:.0f} m<br>"
            f"<b>Speed:</b> {speed:.0f} km/h<br>"
            f"<b>Cost:</b> ${cost:,.0f}<br>"
            f"<b>Type:</b> {cow_type}"
        )
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="red", icon="star"),
            popup=popup_html
        ).add_to(chosen_layer)
    chosen_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    map_out = OUT_DIR / "maps"
    map_out.mkdir(parents=True, exist_ok=True)
    out_path = map_out / map_name
    m.save(str(out_path))
    logging.info(f"Saved interactive map to {out_path}")
    return m


# 2. BẢN ĐỒ TĨNH - VẼ VÙNG PHỦ, CÁC SITE
def plot_static_result(method: str):
    """Sinh bản đồ tĩnh so sánh giữa MILP và GA–PSO."""
    i_csv = DATA_DIR / "processed" / "I_points.csv"
    j_csv = DATA_DIR / "processed" / "J_sites.csv"
    boundary_path = DATA_DIR / "raw" / "hue_boundary.geojson"

    pts = pd.read_csv(i_csv)
    sites = pd.read_csv(j_csv)
    boundary = gpd.read_file(boundary_path)

    if method == "MILP":
        res_path = OUT_DIR / "results" / "milp_solution_summary.json"
        res = json.load(open(res_path))
        chosen_ids = res.get("chosen_site_ids", [])
        title = "MILP Optimization Result"
        out_png = OUT_DIR / "maps" / "milp_static.png"
    else:
        assign_path = OUT_DIR / "results" / "ga_pso_assignments.csv"
        if not assign_path.exists():
            logging.error("Không tìm thấy file gán GA–PSO.")
            return

        assign_df = pd.read_csv(assign_path)

        # Xác định danh sách site_id
        if "assigned_site_id" in assign_df.columns and assign_df["assigned_site_id"].dtype == object:
            chosen_ids = assign_df["assigned_site_id"].dropna().astype(str).unique().tolist()
        else:
            chosen_ids = []
            if "assigned_site_index" in assign_df.columns:
                for idx in assign_df["assigned_site_index"].dropna().astype(int).unique():
                    if 1 <= idx <= len(sites):
                        chosen_ids.append(str(sites.iloc[idx - 1]["site_id"]))
            elif "assigned_site_id" in assign_df.columns:
                for idx in assign_df["assigned_site_id"].dropna().astype(int).unique():
                    if 1 <= idx <= len(sites):
                        chosen_ids.append(str(sites.iloc[idx - 1]["site_id"]))

        title = "GA–PSO Optimization Result"
        out_png = OUT_DIR / "maps" / "ga_pso_static.png"

    fig, ax = plt.subplots(figsize=(10, 10))
    boundary.plot(ax=ax, color='none', edgecolor='black')
    ax.scatter(pts["longitude"], pts["latitude"], s=5, c='blue', label='Population points', alpha=0.5)
    ax.scatter(sites["longitude"], sites["latitude"], s=40, c='orange', label='Candidate sites', alpha=0.7)

    chosen = sites[sites["site_id"].isin(chosen_ids) | sites["i_ref"].isin(chosen_ids)]
    ax.scatter(chosen["longitude"], chosen["latitude"], s=120, c='red', marker='*', label='Chosen COW')

    ax.set_title(title)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    logging.info(f"Saved static result plot to {out_png}")


# 3. ENTRYPOINT GỌI TỪ MAIN
def run_simulation_scenario(method: str):
    """Tự động chạy mô phỏng phù hợp với phương pháp."""
    logging.info(f"Running simulation scenario for method: {method}")
    create_interactive_map(method)
    plot_static_result(method)
    logging.info("Simulation complete.")


if __name__ == "__main__":
    run_simulation_scenario("MILP")