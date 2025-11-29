"""
Sinh vùng mất sóng (Generate Damage Scenario) Tạo các trạm BTS bị hư hỏng do thiên tai

Mục đích:
Mô phỏng thiệt hại do thiên tai bằng cách chọn ngẫu nhiên 85% các trạm BTS.
Các trạm còn lại vẫn hoạt động bình thường.

Kết quả đầu ra:
    - active_bts.geojson : các trạm BTS còn hoạt động
    - failed_bts.geojson : các trạm BTS bị hư hỏng
    - active_bts.csv     : dữ liệu thuộc tính của các trạm còn hoạt động
    - failed_bts.csv     : dữ liệu thuộc tính của các trạm bị hư hỏng
"""

import geopandas as gpd
import numpy as np
import random
from pathlib import Path
from src.utils.io_utils import read_csv, read_geojson, write_geojson
import os

# Đường dẫn dữ liệu gốc
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Hàm chọn ngẫu nhiên các trạm bị hư hỏng
def sample_failed_bts(bts_csv_path, damage_rate=0.65, seed=0):
    # Đọc dữ liệu BTS
    df = read_csv(bts_csv_path)

    # Kiểm tra tên cột chứa toạ độ
    lon_col = next((c for c in df.columns if c.lower() in ['lon','longitude','lng','x']), None)
    lat_col = next((c for c in df.columns if c.lower() in ['lat','latitude','y']), None)

    if lon_col is None or lat_col is None:
        raise ValueError("bts_ga.csv must contain lat/lon columns")

    # Chuyển thành GeoDataFrame để có thể ghi ra GeoJSON
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

    # Sinh ngẫu nhiên danh sách trạm bị hư hỏng
    np.random.seed(seed)
    N = len(gdf)
    k = int(np.round(damage_rate * N))
    failed_idx = np.random.choice(gdf.index, size=k, replace=False)

    # Tách dữ liệu thành 2 nhóm: active và failed
    failed_gdf = gdf.loc[failed_idx].copy()
    active_gdf = gdf.drop(index=failed_idx).copy()

    failed_gdf["status"] = "failed"
    active_gdf["status"] = "active"

    return active_gdf, failed_gdf

# Hàm chính xử lý & ghi file
def main(bts_csv, out_dir, damage_rate=0.85, seed=42):
    active_gdf, failed_gdf = sample_failed_bts(bts_csv, damage_rate, seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ghi file GeoJSON
    active_gdf.to_file(out_dir / "active_bts.geojson", driver="GeoJSON")
    failed_gdf.to_file(out_dir / "failed_bts.geojson", driver="GeoJSON")

    # Ghi file CSV
    active_gdf.drop(columns=["geometry"]).to_csv(out_dir / "active_bts.csv", index=False)
    failed_gdf.drop(columns=["geometry"]).to_csv(out_dir / "failed_bts.csv", index=False)

    # Thông tin log kết quả
    print("CREATED A SIMULATOR OF BTS STATION DAMAGE:")
    print(f"    Total number of stations: {len(active_gdf) + len(failed_gdf)}")
    print(f"    The station is still active: {len(active_gdf)} ({(1 - damage_rate) * 100:.1f}%)")
    print(f"    The station is damaged: {len(failed_gdf)} ({damage_rate * 100:.1f}%)")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bts_csv", default=str(DATA_DIR / "processed" / "bts_network" / "bts_ga.csv"))
    p.add_argument("--out_dir", default=str(DATA_DIR / "processed" / "damage_bts"))
    p.add_argument("--damage_rate", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(args.bts_csv, args.out_dir, args.damage_rate, args.seed)
