# ga_pso_solver.py
"""
GA-PSO hybrid solver for BTS restoration (COW deployment + backup power assignment)

UPDATED VERSION:
- Step 1: HARD CONSTRAINT – 100% coverage of outage area
- Step 2: Minimize deployment time
- Step 3: Minimize deployment cost
- OUTPUT STRUCTURE: UNCHANGED

Save at:
BTS_Restoration_Project/src/optimization/GA_PSO/ga_pso_solver.py
"""

import os
import math
import json
import random
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx
import rasterio
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm

# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ============================================================
# DEFAULT CONFIGURATION (UNCHANGED OUTPUT PATHS)
# ============================================================
DEFAULT_CONFIG = {
    "data_paths": {
        "J_sites": str(PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"),
        "cow_dataset": str(PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"),
        "backup_power": str(PROJECT_ROOT / "data/processed/backup_power/backup_power.csv"),
        "failed_bts": str(PROJECT_ROOT / "data/processed/damage_bts/failed_bts.csv"),
        "cow_to_J": str(PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites.csv"),
        "backup_to_bts": str(PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts.csv"),
        "roads_graphml": str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
        "flood_tif": str(PROJECT_ROOT / "data/processed/flood/flood_depth_combined_B_clean.tif"),
    },
    "ga_pso": {
        "runs": 1,
        "pop_size": 100,
        "max_iter": 200,
        "mutation_rate": 0.08,
        "ga_period": 10,
        "elitism": 0.1,

        # weights for step 2 & 3 (step 1 is HARD)
        "weights": {
            "w_time": 0.3,
            "w_cost": 0.7
        },

        "default_setup_time_h": 0.5,
        "budget_max": 1e19,
        "flood_depth_threshold_m": 0.5,

        # OUTPUT DIRECTORY (KEEP UNCHANGED)
        "output_dir": "BTS_Restoration_Project/outputs/results_ga_pso"
    },
    "random_seed": 42
}

# ============================================================
# CONFIG MERGE
# ============================================================
def merge_with_default(cfg: Dict) -> Dict:
    merged = deepcopy(DEFAULT_CONFIG)
    for k, v in cfg.items():
        if isinstance(v, dict) and k in merged:
            merged[k].update(v)
        else:
            merged[k] = v
    return merged

# ============================================================
# UTILITIES
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ============================================================
# DATA MODEL
# ============================================================
class DataModel:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        dp = cfg["data_paths"]

        # Load datasets
        self.J = pd.read_csv(dp["J_sites"])
        self.cow = pd.read_csv(dp["cow_dataset"])
        self.backups = pd.read_csv(dp["backup_power"])
        self.failed_bts = pd.read_csv(dp["failed_bts"])
        self.cow_to_J = pd.read_csv(dp["cow_to_J"])
        self.backup_to_bts = pd.read_csv(dp["backup_to_bts"])

        # Optional GIS
        self.roads = None
        self.flood_raster = None

        if os.path.exists(dp["roads_graphml"]):
            try:
                self.roads = nx.read_graphml(dp["roads_graphml"])
            except Exception as e:
                logging.warning("Cannot read roads graph: %s", e)

        if os.path.exists(dp["flood_tif"]):
            try:
                self.flood_raster = rasterio.open(dp["flood_tif"])
            except Exception as e:
                logging.warning("Cannot open flood raster: %s", e)

        self._prepare_indices()

    def _prepare_indices(self):
        self.J_ids = self.J["site_id"].tolist()
        self.cow_ids = self.cow["cow_id"].tolist()
        self.power_ids = self.backups["power_id"].tolist()

        coords = np.vstack([self.J["latitude"], self.J["longitude"]]).T
        try:
            self.J_kdtree = cKDTree(coords)
        except Exception:
            self.J_kdtree = None

    def sample_flood_depth(self, lat, lon) -> Optional[float]:
        if self.flood_raster is None:
            return None
        try:
            for val in self.flood_raster.sample([(lon, lat)]):
                return float(val[0])
        except Exception:
            return None

    def is_J_flooded(self, site_id) -> bool:
        row = self.J[self.J["site_id"] == site_id]
        if row.empty:
            return False
        depth = self.sample_flood_depth(row.iloc[0]["latitude"], row.iloc[0]["longitude"])
        if depth is None:
            return False
        return depth > self.cfg["ga_pso"]["flood_depth_threshold_m"]

# ============================================================
# INDIVIDUAL ENCODING
# ============================================================
class Individual:
    """
    cow_assign   : {cow_id -> J_site or None}
    power_assign : {bts_id -> power_id or None}
    """
    def __init__(self, cow_ids: List[str], bts_ids: List[str]):
        self.cow_assign = {cid: None for cid in cow_ids}
        self.power_assign = {bid: None for bid in bts_ids}
        self.fitness: Optional[float] = None
        self.metrics: Dict = {}

    def copy(self):
        new = Individual(list(self.cow_assign.keys()), list(self.power_assign.keys()))
        new.cow_assign = dict(self.cow_assign)
        new.power_assign = dict(self.power_assign)
        new.fitness = self.fitness
        new.metrics = dict(self.metrics)
        return new

# ============================================================
# GA-PSO SOLVER
# ============================================================
class GA_PSOSolver:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.params = cfg["ga_pso"]

        random.seed(cfg.get("random_seed", 42))
        np.random.seed(cfg.get("random_seed", 42))

        self.dm = DataModel(cfg)

        self.output_dir = Path(self.params["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Convenience lists ---
        self.cow_ids = self.dm.cow["cow_id"].tolist()
        self.J_ids = self.dm.J["site_id"].tolist()

        self.bts_to_power = (
            self.dm.failed_bts[self.dm.failed_bts["status"] == "power_outage"]["site_id"]
            .tolist()
        )
        self.power_ids = self.dm.backups["power_id"].tolist()

        # --- Precompute travel maps ---
        self.cow_travel_map = self._build_cow_travel_map()
        self.power_travel_map = self._build_power_travel_map()

        # --- Precompute coverage matrices ---
        self.cover_cow = self._build_cover_cow()
        self.cover_bts = self._build_cover_bts()

    # ========================================================
    # TRAVEL MAPS
    # ========================================================
    def _build_cow_travel_map(self):
        m = {}
        for _, r in self.dm.cow_to_J.iterrows():
            m[(r["cow_id"], r["site_id"])] = {
                "distance_km": float(r["distance_km"]),
                "travel_time_hr": float(r["travel_time_hr"]),
                "travel_cost_vnd": float(r["travel_cost_vnd"]),
            }
        return m

    def _build_power_travel_map(self):
        m = {}
        for _, r in self.dm.backup_to_bts.iterrows():
            m[(r["power_id"], r["bts_id"])] = {
                "distance_km": float(r["distance_km"]),
                "total_time_hr": float(r["total_time_hr"])
                if not pd.isna(r["total_time_hr"]) else np.inf,
                "total_cost_vnd": float(r["total_cost_vnd"])
                if not pd.isna(r["total_cost_vnd"]) else np.inf,
            }
        return m

    # ========================================================
    # COVERAGE MATRICES
    # ========================================================
    def _build_cover_cow(self):
        cover = {}
        cow_coords = {
            r["cow_id"]: (r["lat"], r["lon"], r["coverage_radius_m"])
            for _, r in self.dm.cow.iterrows()
        }

        for cow_id, (clat, clon, cov_m) in cow_coords.items():
            cover[cow_id] = {}
            for _, j in self.dm.J.iterrows():
                d_km = haversine_km(
                    clat, clon, float(j["latitude"]), float(j["longitude"])
                )
                cover[cow_id][j["site_id"]] = (d_km * 1000.0) <= cov_m
        return cover

    def _build_cover_bts(self):
        cover = {}
        for _, b in self.dm.failed_bts.iterrows():
            bts_id = b["site_id"]
            cover[bts_id] = {}
            for _, j in self.dm.J.iterrows():
                d_km = haversine_km(
                    float(b["latitude"]),
                    float(b["longitude"]),
                    float(j["latitude"]),
                    float(j["longitude"]),
                )
                cover[bts_id][j["site_id"]] = (
                    d_km * 1000.0 <= float(b["coverage_radius_m"])
                )
        return cover

    # ========================================================
    # HARD COVERAGE (STEP 1)
    # ========================================================
    def _compute_covered_J(self, ind: Individual) -> set:
        """
        Return set of J that are covered by:
        - deployed COW
        - restored BTS
        """
        covered = set()

        # --- COW coverage ---
        for cow_id, j_id in ind.cow_assign.items():
            if j_id is None:
                continue
            if self.cover_cow.get(cow_id, {}).get(j_id, False):
                if not self.dm.is_J_flooded(j_id):
                    covered.add(j_id)

        # --- BTS coverage ---
        for bts_id, power_id in ind.power_assign.items():
            if power_id is None:
                continue
            pinfo = self.power_travel_map.get((power_id, bts_id))
            if pinfo is None or pinfo["total_time_hr"] == np.inf:
                continue
            for j_id, ok in self.cover_bts.get(bts_id, {}).items():
                if ok:
                    covered.add(j_id)

        return covered

    def _repair_full_coverage(self, ind: Individual):
        """
        HARD CONSTRAINT:
        If any J is uncovered → force assign best feasible COW
        """
        covered = self._compute_covered_J(ind)
        uncovered = [j for j in self.J_ids if j not in covered]

        if not uncovered:
            return ind

        for j_id in uncovered:
            candidates = []
            for cow_id in self.cow_ids:
                if self.cover_cow.get(cow_id, {}).get(j_id, False):
                    if self.dm.is_J_flooded(j_id):
                        continue
                    tinfo = self.cow_travel_map.get((cow_id, j_id))
                    if tinfo:
                        candidates.append((cow_id, tinfo["travel_cost_vnd"]))

            if candidates:
                best_cow = min(candidates, key=lambda x: x[1])[0]
                ind.cow_assign[best_cow] = j_id

        return ind

    # ========================================================
    # INITIALIZATION
    # ========================================================
    def _init_individual_random(self) -> Individual:
        ind = Individual(self.cow_ids, self.bts_to_power)

        # Random COW assignment
        for cow_id in self.cow_ids:
            feasible = [
                j for j in self.J_ids
                if self.cover_cow[cow_id].get(j, False)
                and not self.dm.is_J_flooded(j)
            ]
            if feasible and random.random() < 0.6:
                ind.cow_assign[cow_id] = random.choice(feasible)

        # Random power assignment
        for bts_id in self.bts_to_power:
            if random.random() < 0.7:
                ind.power_assign[bts_id] = random.choice(self.power_ids)

        # HARD coverage repair
        ind = self._repair_full_coverage(ind)
        return ind

    def _init_population(self, N: int) -> List[Individual]:
        pop = []
        for _ in range(N):
            ind = self._init_individual_random()
            pop.append(ind)
        return pop

    # ========================================================
    # FITNESS EVALUATION (LEXICOGRAPHIC)
    # Step 1: HARD coverage = 100%
    # Step 2: Minimize deployment time
    # Step 3: Minimize deployment cost
    # ========================================================
    def evaluate_individual(self, ind: Individual):
        # ---------- STEP 1: HARD COVERAGE ----------
        covered_J = self._compute_covered_J(ind)
        total_J = len(self.J_ids)
        missing = total_J - len(covered_J)

        # If not fully covered → VERY LARGE PENALTY
        if missing > 0:
            ind.fitness = 1e9 + missing * 1e6
            ind.metrics = {
                "Rcov": len(covered_J) / max(total_J, 1),
                "missing_J": missing,
                "max_time_hr": np.inf,
                "total_cost_vnd": np.inf,
            }
            return ind.fitness

        # ---------- STEP 2 & 3 ----------
        setup_time = float(self.params.get("default_setup_time_h", 0.5))

        max_time = 0.0
        total_cost = 0.0
        covered_pop = 0.0

        # Population at J
        popJ = {
            r["site_id"]: float(r.get("pop", 0.0))
            for _, r in self.dm.J.iterrows()
        }

        # ---------- COW CONTRIBUTION ----------
        for cow_id, j_id in ind.cow_assign.items():
            if j_id is None:
                continue

            tinfo = self.cow_travel_map.get((cow_id, j_id))
            if not tinfo:
                continue

            # Time
            time_cow = tinfo["travel_time_hr"] + setup_time
            max_time = max(max_time, time_cow)

            # Cost
            crow = self.dm.cow[self.dm.cow["cow_id"] == cow_id].iloc[0]
            total_cost += float(crow.get("cost_vnd", 0.0))
            total_cost += float(tinfo.get("travel_cost_vnd", 0.0))

            # Coverage population
            covered_pop += popJ.get(j_id, 0.0)

        # ---------- POWER CONTRIBUTION ----------
        for bts_id, power_id in ind.power_assign.items():
            if power_id is None:
                continue

            pinfo = self.power_travel_map.get((power_id, bts_id))
            if not pinfo or pinfo["total_time_hr"] == np.inf:
                continue

            # Time
            max_time = max(max_time, pinfo["total_time_hr"])

            # Cost
            total_cost += float(pinfo.get("total_cost_vnd", 0.0))

            prow = self.dm.backups[self.dm.backups["power_id"] == power_id]
            if not prow.empty:
                total_cost += float(prow.iloc[0].get("cost_vnd_24h", 0.0))

            # Coverage population
            brow = self.dm.failed_bts[self.dm.failed_bts["site_id"] == bts_id]
            if not brow.empty:
                covered_pop += float(brow.iloc[0].get("pop_covered", 0.0))

        # ---------- LEXICOGRAPHIC FITNESS ----------
        # Since coverage is HARD, fitness only depends on time + cost
        w_time = self.params["weights"]["w_time"]
        w_cost = self.params["weights"]["w_cost"]

        fitness = w_time * max_time + w_cost * total_cost

        ind.fitness = fitness
        ind.metrics = {
            "Rcov": 1.0,
            "max_time_hr": max_time,
            "total_cost_vnd": total_cost,
            "covered_pop": covered_pop,
        }
        return ind.fitness

    # ========================================================
    # REPAIR OPERATOR (DO NOT BREAK COVERAGE)
    # ========================================================
    def repair(self, ind: Individual):
        # Ensure 1 COW per J
        site_to_cows = {}
        for cow_id, j_id in ind.cow_assign.items():
            if j_id is None:
                continue
            site_to_cows.setdefault(j_id, []).append(cow_id)

        for j_id, cows in site_to_cows.items():
            if len(cows) <= 1:
                continue
            best = min(
                cows,
                key=lambda c: self.cow_travel_map.get((c, j_id), {})
                .get("travel_cost_vnd", np.inf),
            )
            for c in cows:
                if c != best:
                    ind.cow_assign[c] = None

        # Enforce valid power routes
        for bts_id, power_id in ind.power_assign.items():
            if power_id is None:
                continue
            if (power_id, bts_id) not in self.power_travel_map:
                ind.power_assign[bts_id] = None

        # Re-apply HARD coverage
        ind = self._repair_full_coverage(ind)
        return ind

    # ========================================================
    # GA OPERATORS
    # ========================================================
    def crossover(self, a: Individual, b: Individual) -> Individual:
        child = a.copy()

        for cow_id in child.cow_assign:
            if random.random() < 0.5:
                child.cow_assign[cow_id] = b.cow_assign.get(cow_id)

        for bts_id in child.power_assign:
            if random.random() < 0.5:
                child.power_assign[bts_id] = b.power_assign.get(bts_id)

        return self.repair(child)

    def mutate(self, ind: Individual, rate: float):
        for cow_id in ind.cow_assign:
            if random.random() < rate:
                feasible = [
                    j for j in self.J_ids
                    if self.cover_cow[cow_id].get(j, False)
                    and not self.dm.is_J_flooded(j)
                ]
                ind.cow_assign[cow_id] = random.choice(feasible) if feasible else None

        for bts_id in ind.power_assign:
            if random.random() < rate:
                ind.power_assign[bts_id] = random.choice(self.power_ids)

        return self.repair(ind)

    # ========================================================
    # PSO UPDATE (DISCRETE)
    # ========================================================
    def pso_update(self, ind: Individual, pbest: Individual, gbest: Individual):
        child = ind.copy()

        for cow_id in child.cow_assign:
            if random.random() < 0.3:
                child.cow_assign[cow_id] = pbest.cow_assign.get(cow_id)
            if random.random() < 0.3:
                child.cow_assign[cow_id] = gbest.cow_assign.get(cow_id)

        for bts_id in child.power_assign:
            if random.random() < 0.3:
                child.power_assign[bts_id] = pbest.power_assign.get(bts_id)
            if random.random() < 0.3:
                child.power_assign[bts_id] = gbest.power_assign.get(bts_id)

        return self.repair(child)

    # ========================================================
    # SELECTION
    # ========================================================
    def _tournament_select(self, pop: List[Individual], k: int = 3) -> Individual:
        cand = random.sample(pop, k)
        return min(cand, key=lambda x: x.fitness)

    # ========================================================
    # MAIN OPTIMIZATION LOOP
    # ========================================================
    def run(self):
        N = int(self.params["pop_size"])
        max_iter = int(self.params["max_iter"])
        mutation_rate = float(self.params["mutation_rate"])
        elitism = float(self.params["elitism"])
        ga_period = int(self.params["ga_period"])

        n_elite = max(1, int(elitism * N))

        # -------- INIT POPULATION --------
        pop = self._init_population(N)

        for ind in pop:
            self.repair(ind)
            self.evaluate_individual(ind)

        # pbest & gbest
        pbest = [ind.copy() for ind in pop]
        gbest = min(pop, key=lambda x: x.fitness).copy()

        logging.info(
            "Initial best | fitness=%.4f | metrics=%s",
            gbest.fitness,
            gbest.metrics,
        )

        # -------- EVOLUTION LOOP --------
        for it in range(1, max_iter + 1):
            new_pop = []

            # --- Elitism ---
            elites = sorted(pop, key=lambda x: x.fitness)[:n_elite]
            new_pop.extend([e.copy() for e in elites])

            # --- Generate rest ---
            while len(new_pop) < N:
                a = self._tournament_select(pop)
                b = self._tournament_select(pop)

                idx_a = pop.index(a)
                a_pbest = pbest[idx_a]

                # PSO update
                child = self.pso_update(a, a_pbest, gbest)

                # GA crossover periodically
                if it % ga_period == 0 and random.random() < 0.8:
                    child = self.crossover(child, b)

                # Mutation
                child = self.mutate(child, mutation_rate)

                # Repair + evaluate
                child = self.repair(child)
                self.evaluate_individual(child)

                new_pop.append(child)

            pop = new_pop

            # --- Update pbest & gbest ---
            for i, ind in enumerate(pop):
                if ind.fitness < pbest[i].fitness:
                    pbest[i] = ind.copy()

            current_best = min(pop, key=lambda x: x.fitness)
            if current_best.fitness < gbest.fitness:
                gbest = current_best.copy()

            # --- Logging ---
            if it == 1 or it % max(1, max_iter // 10) == 0:
                logging.info(
                    "Iter %d/%d | best fitness=%.4f | time=%.2f | cost=%.2f",
                    it,
                    max_iter,
                    gbest.fitness,
                    gbest.metrics.get("max_time_hr", 0.0),
                    gbest.metrics.get("total_cost_vnd", 0.0),
                )

        # -------- FINAL SOLUTION --------
        gbest = self.repair(gbest)
        self.evaluate_individual(gbest)

        self._export_solution(gbest)
        return gbest

    # ========================================================
    # OUTPUT (KEEP STRUCTURE UNCHANGED)
    # ========================================================
    def _export_solution(self, sol: Individual):
        # -------- COW ASSIGNMENTS --------
        cow_rows = []
        for cow_id, site_id in sol.cow_assign.items():
            rec = {
                "cow_id": cow_id,
                "site_id": site_id if site_id else "",
            }
            if site_id and (cow_id, site_id) in self.cow_travel_map:
                rec.update(self.cow_travel_map[(cow_id, site_id)])
            cow_rows.append(rec)

        pd.DataFrame(cow_rows).to_csv(
            self.output_dir / "solution_cow_assignments_cov.csv",
            index=False,
        )

        # -------- POWER ASSIGNMENTS --------
        power_rows = []
        for bts_id, power_id in sol.power_assign.items():
            rec = {
                "bts_id": bts_id,
                "power_id": power_id if power_id else "",
            }
            if power_id and (power_id, bts_id) in self.power_travel_map:
                pinfo = self.power_travel_map[(power_id, bts_id)]
                prow = self.dm.backups[self.dm.backups["power_id"] == power_id]

                rec["total_cost_vnd"] = float(pinfo.get("total_cost_vnd", 0.0))
                rec["operating_cost_vnd_24h"] = (
                    float(prow.iloc[0].get("cost_vnd_24h", 0.0))
                    if not prow.empty else 0.0
                )
                rec["total_cost_all_vnd"] = (
                    rec["total_cost_vnd"] + rec["operating_cost_vnd_24h"]
                )

            power_rows.append(rec)

        pd.DataFrame(power_rows).to_csv(
            self.output_dir / "solution_power_assignments_cov.csv",
            index=False,
        )

        # -------- SUMMARY --------
        summary = {
            "fitness": sol.fitness,
            **sol.metrics,
        }

        with open(self.output_dir / "solution_summary_cov.json", "w") as f:
            json.dump(summary, f, indent=2)

        logging.info("Results exported to %s", self.output_dir)


# ============================================================
# CLI
# ============================================================
def run_from_config(config: Dict = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    cfg = DEFAULT_CONFIG if config is None else merge_with_default(config)

    solver = GA_PSOSolver(cfg)
    return solver.run()


if __name__ == "__main__":
    run_from_config()
