"""
BTS DAMAGE SCENARIO GENERATOR
Sinh dữ liệu BTS bị ảnh hưởng bởi thiên tai (bão, lũ, ngập lụt).

YÊU CẦU:
- 20% trạm ACTIVE và bắt buộc nằm ngoài vùng ngập.
- 15% trạm POWER_OUTAGE (mất nguồn), phân bố ngẫu nhiên toàn khu vực.
- 65% trạm FAILED (hư hỏng nặng), phân bố ngẫu nhiên toàn khu vực.

DỮ LIỆU ĐẦU VÀO:
- BTS CSV: BTS_Restoration_Project/data/processed/bts_network/bts_ga.csv
- Flood Raster (TIF): BTS_Restoration_Project/data/processed/flood/flood_depth_combined_B_clean.tif

DỮ LIỆU ĐẦU RA:
- active_bts.geojson / csv
- failed_bts.geojson / csv
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path

# Utility: Load BTS DataFrame → GeoDataFrame
def load_bts_dataframe(csv_path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)

    lon_col = next((c for c in df.columns if c.lower() in ["lon", "lng", "longitude", "x"]), None)
    lat_col = next((c for c in df.columns if c.lower() in ["lat", "latitude", "y"]), None)

    if lon_col is None or lat_col is None:
        raise ValueError("CSV must contain longitude/latitude columns!")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

# Utility: Extract flood depth values for each BTS
def extract_flood_depth(points_gdf: gpd.GeoDataFrame, raster_path: str) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        depths = []
        for geom in points_gdf.geometry:
            x, y = geom.x, geom.y
            try:
                row, col = src.index(x, y)
                value = src.read(1)[row, col]
            except Exception:
                value = np.nan
            depths.append(value)
        return np.array(depths)

# MAIN FUNCTION
def generate_bts_damage_dataset(
        bts_csv_path: str,
        flood_tif_path: str,
        output_dir: str,
        active_rate=0.20,
        power_outage_rate=0.15,
        failed_rate=0.65,
        seed=42
):
    np.random.seed(seed)

    # Load BTS
    bts = load_bts_dataframe(bts_csv_path)
    total = len(bts)
    print(f"Loaded {total} BTS stations.")

    # Extract flood depth
    print("Extracting flood depth...")
    flood_depth = extract_flood_depth(bts, flood_tif_path)
    bts["flooded"] = (flood_depth > 0).astype(int)

    # Filter non-flooded for ACTIVE
    non_flooded = bts[bts["flooded"] == 0]

    required_active = int(total * active_rate)
    required_power = int(total * power_outage_rate)
    required_failed = total - required_active - required_power

    if len(non_flooded) < required_active:
        raise ValueError(f"Không đủ trạm không ngập để phân bố ACTIVE ({len(non_flooded)}/{required_active})")

    # Select ACTIVE
    active_idx = np.random.choice(non_flooded.index, required_active, replace=False)
    active_bts = bts.loc[active_idx].copy()
    active_bts["status"] = "active"

    # Remaining are damaged
    remaining = bts.drop(index=active_idx)

    # POWER OUTAGE
    power_idx = np.random.choice(remaining.index, required_power, replace=False)
    power_bts = remaining.loc[power_idx].copy()
    power_bts["status"] = "power_outage"

    # FAILED
    remaining = remaining.drop(index=power_idx)
    failed_bts = remaining.copy()
    failed_bts["status"] = "failed"

    # Merge damaged BTS into one dataset
    failed_all = pd.concat([power_bts, failed_bts], axis=0)
    failed_all = failed_all.sort_index()

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export ACTIVE
    active_bts.to_file(output_path / "active_bts.geojson", driver="GeoJSON")
    active_bts.drop(columns=["geometry"]).to_csv(output_path / "active_bts.csv", index=False)

    # Export FAILED (power_outage + failed)
    failed_all.to_file(output_path / "failed_bts.geojson", driver="GeoJSON")
    failed_all.drop(columns=["geometry"]).to_csv(output_path / "failed_bts.csv", index=False)

    print("\n===== DAMAGE SCENARIO GENERATED SUCCESSFULLY =====")
    print(f"Total BTS: {total}")
    print(f"Active (20%): {len(active_bts)}")
    print(f"Power Outage (15%): {len(power_bts)}")
    print(f"Failed (65%): {len(failed_bts)}")
    print(f"FAILED TOTAL: {len(failed_all)}")
    print(f"Output saved to: {output_path.resolve()}")
    print("==================================================")

    return active_bts, failed_all

# CLI ENTRY POINT
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate BTS damage dataset")
    parser.add_argument("--bts_csv", required=False,
                        default="BTS_Restoration_Project/data/processed/bts_network/bts_ga.csv")
    parser.add_argument("--flood_tif", required=False,
                        default="BTS_Restoration_Project/data/processed/flood/flood_depth_combined_B_clean.tif")
    parser.add_argument("--out_dir", required=False,
                        default="BTS_Restoration_Project/data/processed/bts_damage")

    args = parser.parse_args()

    generate_bts_damage_dataset(
        bts_csv_path=args.bts_csv,
        flood_tif_path=args.flood_tif,
        output_dir=args.out_dir
    )
