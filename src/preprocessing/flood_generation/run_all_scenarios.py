# run_all_scenarios.py
"""
Main orchestrator for realistic flood modeling A+B.
"""

import yaml
import logging
import os
from pathlib import Path
from datetime import datetime

from .dem_processing import process_dem_pipeline
from .river_flood_model import run_river_flood
from .pluvial_flood_model import run_pluvial_flood
from .combine_floods import combine_river_and_pluvial
from .depth_to_polygon import depth_raster_to_polygon
from .roads_flood_analysis import analyze_roads
from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_flood_simulation(config_path: str):

    cfg = load_config(config_path)

    dem_path = cfg["input"]["dem"]
    roads_path = cfg["input"]["roads"]
    outdir = cfg["output"]["outdir"]
    ensure_dir(outdir)

    logger.info("[FLOOD] Processing DEM...")
    dem_filled, flowdir, accum, dem_profile = process_dem_pipeline(
        dem_path, os.path.join(outdir, "dem_corrected.tif"),
        offset=cfg.get("dem_offset", 1.0)
    )

    results = {}

    for sc in cfg["scenarios"]:
        m = float(sc["m"])
        rainfall = float(sc.get("rainfall", 0.5))  # 0.5m default rainfall

        logger.info(f"[SCENARIO] Running scenario m={m}, rainfall={rainfall}")

        riv_out = os.path.join(outdir, f"river_{m}.tif")
        plu_out = os.path.join(outdir, f"pluvial_{m}.tif")
        cmb_out = os.path.join(outdir, f"combined_{m}.tif")
        poly_out = os.path.join(outdir, f"flood_{m}.geojson")
        roads_out = os.path.join(outdir, f"roads_{m}.geojson")

        # A) River flood
        river = run_river_flood(dem_filled, m, riv_out, dem_profile)

        # B) Pluvial flood
        pluvial = run_pluvial_flood(dem_filled, accum, rainfall, plu_out, dem_profile)

        # Combine A+B
        combined = combine_river_and_pluvial(river, pluvial, cmb_out, dem_profile)

        # Vectorize
        depth_raster_to_polygon(cmb_out, poly_out)

        # Roads
        analyze_roads(roads_path, cmb_out, roads_out, sampling_m=cfg.get("sampling_m", 5))

        # Summary
        summary = {
            "scenario": m,
            "rainfall": rainfall,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "outputs": {
                "river": riv_out,
                "pluvial": plu_out,
                "combined": cmb_out,
                "polygons": poly_out,
                "roads": roads_out
            }
        }

        results[str(m)] = summary

    save_json(results, os.path.join(outdir, "scenarios_index.json"))

    logger.info("[FLOOD] Completed all scenarios.")

    return results
