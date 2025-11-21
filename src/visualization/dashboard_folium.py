"""
Interactive Map Dashboard using Folium
--------------------------------------
Hiển thị kết quả tối ưu vị trí triển khai COW:
- Boundary (Huế)
- Dân cư (I_points)
- BTS active / failed
- Candidate + chosen COW sites (với buffer phủ sóng)
"""

import folium
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
from folium.plugins import MarkerCluster, Fullscreen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

def create_interactive_map():
    # 1. Đọc dữ liệu
    boundary = gpd.read_file(DATA_DIR / "raw" / "hue_boundary.geojson")
    centroid = boundary.geometry.unary_union.centroid

    I_points = pd.read_csv(DATA_DIR / "processed" / "I_points.csv")
    J_sites = pd.read_csv(DATA_DIR / "processed" / "J_sites.csv")
    COWs = pd.read_csv(DATA_DIR / "raw" / "cow_dataset.csv")
    BTS = pd.read_csv(DATA_DIR / "raw" / "bts_ga.csv")

    result = json.load(open(OUT_DIR / "results" / "milp_solution_summary.json"))
    chosen_site_ids = set(result["chosen_site_ids"])

    # 2. Tạo map nền
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=9, tiles="CartoDB positron")
    Fullscreen().add_to(m)
    folium.GeoJson(boundary.to_json(), name="Boundary", style_function=lambda x: {"color": "black", "weight": 2}).add_to(m)

    # 3. Lớp dân cư (I_points)
    pop_layer = folium.FeatureGroup(name="Population Demand", show=True)
    for _, row in I_points.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(2, min(6, row.get("pop", 1) / 50)),
            color="blue",
            fill=True,
            fill_opacity=0.5,
            popup=f"<b>Pop:</b> {row.get('pop', 0):.1f}<br>"
                  f"<b>Priority:</b> {row.get('priority_category', 'normal')}<br>"
                  f"<b>Weight:</b> {row.get('priority_weight', 1.0)}"
        ).add_to(pop_layer)
    pop_layer.add_to(m)

    # 4. Lớp BTS (hoạt động và hư hỏng)
    bts_layer = folium.FeatureGroup(name="BTS Stations", show=False)
    for _, row in BTS.iterrows():
        color = "green" if row.get("status", "active") == "active" else "gray"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"<b>BTS ID:</b> {row['site_id']}<br>"
                  f"<b>Status:</b> {row.get('status', 'unknown')}"
        ).add_to(bts_layer)
    bts_layer.add_to(m)

    # 5. Lớp COW ứng viên
    cand_layer = folium.FeatureGroup(name="Candidate Sites", show=False)
    for _, row in J_sites.iterrows():
        color = "orange" if row["i_ref"] not in chosen_site_ids else "red"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"<b>Candidate from:</b> {row['i_ref']}<br>"
                  f"<b>Slope:</b> {row.get('slope', 0):.1f}°<br>"
                  f"<b>Dist to road:</b> {row.get('dist_to_road_m', 0):.0f} m"
        ).add_to(cand_layer)
    cand_layer.add_to(m)

    # 6. Lớp COW được chọn
    chosen_layer = folium.FeatureGroup(name="Chosen COWs", show=True)
    for site_id in chosen_site_ids:
        j_row = J_sites[J_sites["i_ref"] == site_id].head(1)
        if not j_row.empty:
            lat = float(j_row.iloc[0]["latitude"])
            lon = float(j_row.iloc[0]["longitude"])

            # chọn ngẫu nhiên COW tương ứng
            cow = COWs.sample(1).iloc[0]
            radius = float(cow.get("coverage_radius_m", 3000))
            speed = float(cow.get("speed_kmh", 40))
            cost = float(cow.get("cost_vnd", 0))
            cow_type = str(cow.get("cow_type", "N/A"))

            # buffer vùng phủ sóng
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color="red",
                fill=False,
                opacity=0.3,
                weight=2
            ).add_to(chosen_layer)

            # marker COW
            popup_html = (
                f"<b>COW Site:</b> {site_id}<br>"
                f"<b>Coverage:</b> {radius:.0f} m<br>"
                f"<b>Speed:</b> {speed} km/h<br>"
                f"<b>Cost:</b> ${cost:,.0f}<br>"
                f"<b>Type:</b> {cow_type}"
            )
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="star"),
                popup=popup_html
            ).add_to(chosen_layer)
    chosen_layer.add_to(m)

    # 7. Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # 8. Lưu bản đồ
    map_out = OUT_DIR / "maps"
    map_out.mkdir(parents=True, exist_ok=True)
    out_path = map_out / "milp_dashboard.html"
    m.save(str(out_path))
    print(f"Saved Folium interactive dashboard to: {out_path}")

    return m


if __name__ == "__main__":
    create_interactive_map()
