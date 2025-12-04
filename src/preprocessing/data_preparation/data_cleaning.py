#!/usr/bin/env python3
"""
recreate_centers.py  — FIXED & IMPROVED VERSION

This version preserves all correct logic,
and fully fixes the issues in geometry handling,
duplicate handling, command-center detection,
and centroid conversion safety.

Author: ChatGPT
"""

from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely

# CONFIG
SCRIPT_DIR = Path(__file__).resolve().parent
IN_MEDICAL = SCRIPT_DIR / "medical_centers.geojson"
IN_COMMAND = SCRIPT_DIR / "command_centers.geojson"

OUT_MEDICAL = SCRIPT_DIR / "medical_centers_clean.geojson"
OUT_COMMAND = SCRIPT_DIR / "command_centers_clean.geojson"

TARGET_CRS = "EPSG:4326"

KEEP_KEYS = [
    "id", "osm_id", "name", "amenity", "healthcare",
    "operator", "source", "addr:city", "addr:street",
    "phone", "website"
]

COMMAND_AMENITIES = {
    "police", "fire_station", "townhall", "community_centre",
    "government", "emergency"
}
COMMAND_OFFICES = {
    "government", "fire_department", "police",
    "town_hall", "emergency"
}

# BASIC HELPERS
def standardize_gdf_read(path: Path):
    """Load, ensure geometry exists, ensure CRS=EPSG:4326."""
    if not path.exists():
        print(f"[WARN] Missing dataset: {path}")
        return None

    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return None

    if "geometry" not in gdf:
        print(f"[ERROR] No geometry column in file: {path}")
        return None

    if gdf.crs is None:
        print(f"[WARN] `{path.name}` has no CRS — assuming EPSG:4326.")
        gdf = gdf.set_crs(TARGET_CRS)

    if str(gdf.crs) != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    return gdf


def geom_to_point(geom):
    """Convert any geometry to a point. Point stays same, Polygon/MultiPolygon -> centroid."""
    if geom is None or geom.is_empty:
        return None

    gtype = geom.geom_type

    if gtype == "Point":
        return geom

    if gtype in ("Polygon", "MultiPolygon"):
        try:
            c = geom.centroid
            return Point(c.x, c.y)
        except:
            rp = geom.representative_point()
            return Point(rp.x, rp.y)

    # For line-based geometries that might appear in OSM:
    try:
        rp = geom.representative_point()
        return Point(rp.x, rp.y)
    except:
        return None


def pick_keep_attributes(row):
    """Extract attributes that matter."""
    out = {}
    for key in KEEP_KEYS:
        if key in row and row[key] is not None:
            out[key] = row[key]

    if "osm_id" not in out:
        if "osmid" in row and row["osmid"] is not None:
            out["osm_id"] = row["osmid"]
        elif "id" in row and row["id"] is not None:
            out["osm_id"] = row["id"]

    return out


def deduplicate_points(gdf, digits=6):
    """Remove duplicate points based on coordinate rounding."""
    seen = set()
    kept = []

    for idx, geom in enumerate(gdf.geometry):
        if not isinstance(geom, Point):
            continue

        key = (round(float(geom.x), digits), round(float(geom.y), digits))
        if key in seen:
            continue

        seen.add(key)
        kept.append(idx)

    return gdf.iloc[kept].reset_index(drop=True)

# MEDICAL CENTERS
def recreate_medical(in_path: Path, out_path: Path):
    gdf = standardize_gdf_read(in_path)
    if gdf is None:
        print("[ERROR] Medical centers file could not be loaded.")
        return False

    total = len(gdf)
    pts = []
    rows = []
    converted = 0
    kept_points = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        pt = geom_to_point(geom)
        if pt is None:
            continue

        rec = pick_keep_attributes(row)
        rec["derived"] = geom.geom_type in ("Polygon", "MultiPolygon")
        rec["original_geom"] = geom.geom_type
        rec["source_file"] = in_path.name

        rows.append(rec)
        pts.append(pt)

        if rec["derived"]:
            converted += 1
        else:
            kept_points += 1

    out = gpd.GeoDataFrame(rows, geometry=pts, crs=TARGET_CRS)

    before = len(out)
    out = deduplicate_points(out)
    after = len(out)

    print(f"[OK] medical_centers: total={total}, final={after}, converted_polygons={converted}, duplicates_removed={before - after}")
    out.to_file(out_path, driver="GeoJSON")
    print(f"      Output: {out_path}")
    return True

# COMMAND CENTERS
def detect_command_like(row):
    """Heuristic identification of command/emergency/government facilities."""
    amen = row.get("amenity")
    office = row.get("office")

    if amen and str(amen).lower() in COMMAND_AMENITIES:
        return True
    if office and str(office).lower() in COMMAND_OFFICES:
        return True

    for key in ["emergency", "police", "fire", "operator", "government"]:
        if key in row and row[key]:
            return True

    if "name" in row and row["name"]:
        name = str(row["name"]).lower()
        hints = ["ubnd", "ủy ban", "pccc", "fire", "police", "command", "chỉ huy"]
        if any(h in name for h in hints):
            return True

    return False


def recreate_command_centers(in_path: Path, out_path: Path):
    gdf = standardize_gdf_read(in_path)
    if gdf is None:
        print("[ERROR] Command centers file could not be loaded.")
        return False

    total = len(gdf)
    pts = []
    rows = []
    selected = 0

    # PHASE 1 — Try to detect real command/emergency POIs
    for _, row in gdf.iterrows():
        try:
            if detect_command_like(row):
                geom = row.geometry
                pt = geom_to_point(geom)
                if pt is None:
                    continue

                rec = pick_keep_attributes(row)
                rec["derived"] = geom.geom_type in ("Polygon", "MultiPolygon")
                rec["original_geom"] = geom.geom_type
                rec["source_file"] = in_path.name

                rows.append(rec)
                pts.append(pt)
                selected += 1
        except:
            continue

    # PHASE 2 — If detection fails, fallback
    fallback_used = False
    if selected == 0:
        fallback_used = True
        for _, row in gdf.iterrows():
            geom = row.geometry
            pt = geom_to_point(geom)
            if pt is None:
                continue

            rec = pick_keep_attributes(row)
            rec["derived"] = geom.geom_type in ("Polygon", "MultiPolygon")
            rec["original_geom"] = geom.geom_type
            rec["source_file"] = in_path.name
            rec["suggest_review"] = True  # Mark uncertain entries

            rows.append(rec)
            pts.append(pt)

    out = gpd.GeoDataFrame(rows, geometry=pts, crs=TARGET_CRS)

    # Deduplicate points
    before = len(out)
    out = deduplicate_points(out)
    after = len(out)

    if fallback_used:
        print("[OK] command_centers: No valid emergency/command POIs detected → using fallback centroid extraction.")
    else:
        print(f"[OK] command_centers: Detected {selected} valid command/emergency features.")

    print(f"      total_in={total}, written={after}, duplicates_removed={before - after}")
    out.to_file(out_path, driver="GeoJSON")
    print(f"      Output: {out_path}")

    return True


def main():
    print("=== RECREATING MEDICAL & COMMAND CENTER DATASETS ===")

    if IN_MEDICAL.exists():
        recreate_medical(IN_MEDICAL, OUT_MEDICAL)
    else:
        print(f"[SKIP] Missing medical dataset: {IN_MEDICAL}")

    if IN_COMMAND.exists():
        recreate_command_centers(IN_COMMAND, OUT_COMMAND)
    else:
        print(f"[SKIP] Missing command dataset: {IN_COMMAND}")

    print("=== DONE — Review *_clean.geojson results in QGIS ===")

if __name__ == "__main__":
    main()
