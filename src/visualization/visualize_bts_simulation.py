import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import Fullscreen
from folium import LayerControl
from pathlib import Path

# Config: Đường dẫn dữ liệu
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
OUT_DIR = PROJECT_ROOT / "outputs" / "maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_CSV = PROCESSED_DIR / "damage_bts" / "active_bts.csv"
FAILED_CSV = PROCESSED_DIR / "damage_bts" / "failed_bts.csv"
ACTIVE_GEO = PROCESSED_DIR / "damage_bts" / "active_bts.geojson"
FAILED_GEO = PROCESSED_DIR / "damage_bts" / "failed_bts.geojson"
HUE_BOUNDARY = CLEANED_DIR / "hue_boundary_clean.geojson"

OUTPUT_HTML = OUT_DIR / "bts_damage_simulation.html"

# Hàm load dữ liệu
def load_data():
    gdf_active = gpd.read_file(ACTIVE_GEO)
    gdf_failed = gpd.read_file(FAILED_GEO)
    boundary = gpd.read_file(HUE_BOUNDARY)

    df_active = pd.read_csv(ACTIVE_CSV)
    df_failed = pd.read_csv(FAILED_CSV)

    return gdf_active, gdf_failed, boundary, df_active, df_failed

# Hàm tạo popup thông tin
def build_popup(row):
    html = f"""
    <b><span style="color:black;">THÔNG TIN TRẠM BTS</span></b><br>
    <b>ID trạm:</b> {row.get('site_id', 'N/A')}<br>
    <b>Loại trạm:</b> {row.get('bts_type', 'N/A')}<br>
    <b>Mô hình nguồn điện:</b> {row.get('power_model_type', 'N/A')}<br>
    <b>Nguồn chính:</b> {row.get('power_source_main', 'N/A')}<br>
    <b>Dự phòng:</b> {row.get('power_backup_sources', 'N/A')}<br>
    <b>Thời gian backup:</b> {row.get('backup_duration_hr', 'N/A')} giờ<br>
    <b>Trạng thái:</b> <span style="color:red;">{
        "BỊ HƯ HỎNG" if row.get("status") == "failed" else "HOẠT ĐỘNG"
    }</span><br>
    <hr>
    <b>Lat:</b> {row.get('latitude', row.geometry.y):.6f}<br>
    <b>Lon:</b> {row.get('longitude', row.geometry.x):.6f}<br>
    """
    return folium.Popup(html, max_width=350)

# Hàm về lớp BTS
def add_bts_layer(gdf, name, color, map_obj, status_label):
    fg = folium.FeatureGroup(name=name, show=True)

    for _, row in gdf.iterrows():
        popup = build_popup(row)

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"{status_label} | Trạm ID: {row.get('site_id', 'N/A')}"
        ).add_child(popup).add_to(fg)

    fg.add_to(map_obj)

# Hàm main
def main():
    # Load dữ liệu
    active_gdf, failed_gdf, boundary, df_active, df_failed = load_data()

    # Thêm trường trạng thái
    active_gdf["status"] = "active"
    failed_gdf["status"] = "failed"

    # Tính tâm bản đồ
    center = boundary.geometry.iloc[0].centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="CartoDB positron")

    # Thêm chế độ toàn màn hình
    Fullscreen().add_to(m)


    # Vẽ ranh giới tỉnh
    folium.GeoJson(
        boundary,
        name="Ranh giới tỉnh Thừa Thiên Huế",
        style_function=lambda x: {"color": "blue", "weight": 2, "fillOpacity": 0}
    ).add_to(m)


    # Vẽ lớp BTS
    add_bts_layer(active_gdf, "BTS Hoạt Động", "green", m, "BTS HOẠT ĐỘNG")
    add_bts_layer(failed_gdf, "BTS Hư Hỏng Sau Bão", "red", m, "BTS BỊ HƯ HỎNG")

    # Bộ điều khiển lớp
    folium.LayerControl(collapsed=False).add_to(m)

    # Lưu file HTML
    m.save(OUTPUT_HTML)

    print("==> ĐÃ TẠO FILE MÔ PHỎNG:", OUTPUT_HTML)

if __name__ == "__main__":
    main()
