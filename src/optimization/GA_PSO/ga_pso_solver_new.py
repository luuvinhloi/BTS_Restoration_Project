# ga_pso_solver.py
"""
GA-PSO hybrid solver for BTS restoration (COW deployment + backup power assignment)

Save at:
BTS_Restoration_Project/src/optimization/GA_PSO/ga_pso_solver.py

References: design and fitness formulas follow the project report (Graduation_Project_Report_CE.pdf).
"""
import os
import math
import json
import random
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import rasterio
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# ga_pso_solver.py
# parents[0] = GA_PSO
# parents[1] = optimization
# parents[2] = src
# parents[3] = BTS_Restoration_Project

# Configuration / defaults
DEFAULT_CONFIG = {
    "data_paths": {
        "I_points": str(PROJECT_ROOT / "data/processed/position_I_J/I_points.csv"),
        "J_sites": str(PROJECT_ROOT / "data/processed/position_I_J/J_sites.csv"),
        "cow_dataset": str(PROJECT_ROOT / "data/processed/cow/cow_dataset.csv"),
        "backup_power": str(PROJECT_ROOT / "data/processed/backup_power/backup_power.csv"),
        "failed_bts": str(PROJECT_ROOT / "data/processed/damage_bts/failed_bts.csv"),
        "cow_to_J": str(PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites.csv"),
        "backup_to_bts": str(PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts.csv"),
        "roads_graphml": str(PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"),
        "flood_tif": str(PROJECT_ROOT / "data/processed/flood/flood_depth_combined_clean.tif"),
    },
    "ga_pso": {
        "runs": 1,
        "pop_size": 100,
        "max_iter": 200,
        "mutation_rate": 0.08,
        "ga_period": 10,
        "elitism": 0.1,
        "cost_per_km_factor": 0.2,
        "max_slope_deg": 15.0,
        "weights": {
            "w_time": 0.2,
            "w_cost": 0.4,
            "coverage_weight": 0.5
        },
        "budget_max": 1e9,
        "flood_depth_threshold_m": 0.5,
        "output_dir": "BTS_Restoration_Project/outputs/results_ga_pso_new"
    },
    "random_seed": 42
}

# CONFIG MERGE
def merge_with_default(cfg: Dict) -> Dict:
    merged = deepcopy(DEFAULT_CONFIG)
    for k, v in cfg.items():
        if isinstance(v, dict) and k in merged:
            merged[k].update(v)
        else:
            merged[k] = v
    return merged

# Utilities
def haversine_km(lat1, lon1, lat2, lon2):
    # compute Haversine distance in km
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2*R*math.asin(math.sqrt(a))

# Data loader
class DataModel:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        dp = cfg["data_paths"]
        # Load CSVs
        self.J = pd.read_csv(dp["J_sites"])
        self.I = pd.read_csv(dp["I_points"])
        self.cow = pd.read_csv(dp["cow_dataset"])
        self.backups = pd.read_csv(dp["backup_power"])
        self.failed_bts = pd.read_csv(dp["failed_bts"])
        self.cow_to_J = pd.read_csv(dp["cow_to_J"])
        self.backup_to_bts = pd.read_csv(dp["backup_to_bts"])
        # Optional graph and raster
        self.roads = None
        self.flood_raster = None
        if os.path.exists(dp["roads_graphml"]):
            try:
                self.roads = nx.read_graphml(dp["roads_graphml"])
            except Exception as e:
                logging.warning("Cannot read roads_graphml: %s", e)
        if os.path.exists(dp["flood_tif"]):
            try:
                self.flood_raster = rasterio.open(dp["flood_tif"])
            except Exception as e:
                logging.warning("Cannot open flood tif: %s", e)

        # Indexing and helper maps
        self._prepare_indices()

    def _prepare_indices(self):
        # Build lookups for J, cow bases, backups, BTS
        self.J_idx = {row["site_id"]: i for i, row in self.J.iterrows()}
        self.cow_idx = {row["cow_id"]: i for i, row in self.cow.iterrows()}
        self.backup_idx = {row["power_id"]: i for i, row in self.backups.iterrows()}
        self.bts_idx = {row["site_id"]: i for i, row in self.failed_bts.iterrows()}

        # kd-tree for J positions (for nearest non-flooded search)
        coords = np.vstack([self.J["latitude"].values, self.J["longitude"].values]).T
        try:
            self.J_kdtree = cKDTree(coords)
        except Exception:
            self.J_kdtree = None

    def sample_flood_depth(self, lat: float, lon: float) -> Optional[float]:
        if self.flood_raster is None:
            return None
        try:
            for val in self.flood_raster.sample([(lon, lat)]):
                return float(val[0])
        except Exception:
            return None

    def is_J_flooded(self, j_id) -> bool:
        row = self.J.loc[self.J["site_id"] == j_id]
        if row.empty:
            return False
        lat = float(row["latitude"].values[0])
        lon = float(row["longitude"].values[0])
        depth = self.sample_flood_depth(lat, lon)
        if depth is None:
            return False
        return depth > self.cfg["ga_pso"]["flood_depth_threshold_m"]

    def find_nearest_nonflood_J(self, lat, lon, radius_km=5.0):
        # return site_id of nearest J not flooded within radius_km, else None
        if self.J_kdtree is None:
            return None
        dist, idx = self.J_kdtree.query([lat, lon], k=50, distance_upper_bound=radius_km)
        if np.isscalar(idx):
            idx = [idx]
        for ii in idx:
            if ii >= len(self.J):
                continue
            sid = self.J.iloc[ii]["site_id"]
            if not self.is_J_flooded(sid) and not self.J.iloc[ii]["in_water"]:
                return sid
        return None

# Individual encoding and evaluation
class Individual:
    """
    Encoding:
      - cow_assign: dict {cow_id: site_id or None(=0)}  (0 means not deployed)
      - power_assign: dict {bts_id: power_id or None}
    """
    def __init__(self, cow_ids: List[str], bts_ids: List[str]):
        self.cow_assign = {cid: None for cid in cow_ids}
        self.power_assign = {bid: None for bid in bts_ids}
        # bookkeeping
        self.fitness: Optional[float] = None
        self.metrics: Dict = {}

    def copy(self):
        ind = Individual(list(self.cow_assign.keys()), list(self.power_assign.keys()))
        ind.cow_assign = dict(self.cow_assign)
        ind.power_assign = dict(self.power_assign)
        ind.fitness = self.fitness
        ind.metrics = dict(self.metrics)
        return ind

# GA-PSO Solver
class GA_PSOSolver:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        random.seed(cfg.get("random_seed", 42))
        np.random.seed(cfg.get("random_seed", 42))

        self.dm = DataModel(cfg)
        self.params = cfg["ga_pso"]

        self.output_dir = Path(self.params["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # convenience lists
        self.cow_ids = self.dm.cow["cow_id"].tolist()
        # only consider BTS that need power assignment (status==power_outage)
        self.bts_to_power = self.dm.failed_bts[self.dm.failed_bts["status"] == "power_outage"]["site_id"].tolist()
        # only consider J sites (for deployment) generally; but J may get filtered by feasibility
        self.J_ids = self.dm.J["site_id"].tolist()
        # priority weight per J (default = 1.0 if missing)
        self.J_priority = {
            row["site_id"]: float(row.get("priority_weight", 1.0))
            for _, row in self.dm.J.iterrows()
        }
        # list of available power units
        self.power_ids = self.dm.backups["power_id"].tolist()

        # Precompute lookups from travel CSVs for speed
        self.cow_travel_map = self._build_cow_travel_map()
        self.power_travel_map = self._build_power_travel_map()

        # coverage matrices precompute: cover_COW[cow_id][j] True/False
        self.cover_cow = self._build_cover_cow()
        self.cover_bts = self._build_cover_bts()

    # Precompute helpers
    def _build_cow_travel_map(self):
        m = {}
        for _, row in self.dm.cow_to_J.iterrows():
            try:
                dist = float(row["distance_km"])
                time = float(row["travel_time_hr"])
                cost = float(row["travel_cost_vnd"])
            except Exception as e:
                raise ValueError(
                    f"Invalid numeric value in cow_to_J_sites.csv "
                    f"at cow={row['cow_id']} site={row['site_id']}: {e}"
                )

            m[(row["cow_id"], row["site_id"])] = {
                "distance_km": dist,
                "travel_time_hr": time,
                "travel_cost_vnd": cost
            }
        return m

    def _build_power_travel_map(self):
        """
        map (power_id, bts_id) ->
          distance_km
          total_time_hr
          travel_cost_vnd   (CHI PHÍ DI CHUYỂN)
        """
        m = {}
        for _, row in self.dm.backup_to_bts.iterrows():
            try:
                dist = float(row["distance_km"])
                time = float(row["total_time_hr"]) if not pd.isna(row["total_time_hr"]) else np.inf
                travel_cost = float(row["travel_cost_vnd"]) if not pd.isna(row["travel_cost_vnd"]) else np.inf
            except Exception as e:
                raise ValueError(
                    f"Invalid numeric value in backup_to_failed_bts.csv "
                    f"at power={row['power_id']} bts={row['bts_id']}: {e}"
                )

            m[(row["power_id"], row["bts_id"])] = {
                "distance_km": dist,
                "total_time_hr": time,
                "travel_cost_vnd": travel_cost,
                "note": row.get("note", "")
            }
        return m

    def _build_cover_cow(self):
        cover = {}

        cow_coords = {
            row["cow_id"]: (row["lat"], row["lon"], row["coverage_radius_m"])
            for _, row in self.dm.cow.iterrows()
        }

        for c_id, (clat, clon, cov_m) in cow_coords.items():
            cover[c_id] = {}

            for _, jrow in self.dm.J.iterrows():
                j_id = jrow["site_id"]

                d_km = haversine_km(
                    clat, clon,
                    float(jrow["latitude"]),
                    float(jrow["longitude"])
                )

                cover[c_id][j_id] = (d_km * 1000.0) <= cov_m

        return cover

    def _build_cover_bts(self):
        cover = {}

        for _, brow in self.dm.failed_bts.iterrows():
            b_id = brow["site_id"]
            cover[b_id] = {}

            for _, jrow in self.dm.J.iterrows():
                j_id = jrow["site_id"]

                d_km = haversine_km(
                    float(brow["latitude"]),
                    float(brow["longitude"]),
                    float(jrow["latitude"]),
                    float(jrow["longitude"])
                )

                cover[b_id][j_id] = (d_km * 1000.0) <= float(brow["coverage_radius_m"])

        return cover

    def evaluate_stage1_power(self, ind: Individual):
        """
        Stage 1: chỉ xét power_assign
        Mục tiêu: maximize BTS coverage
        """
        covered_pop = 0.0

        for bts_id, power_id in ind.power_assign.items():
            if power_id is None:
                continue

            pinfo = self.power_travel_map.get((power_id, bts_id))
            if pinfo is None or pinfo["total_time_hr"] == np.inf:
                continue

            # duyệt các J mà BTS này phủ
            for j_id, can_cover in self.cover_bts.get(bts_id, {}).items():
                if not can_cover:
                    continue

                # population của J
                j_row = self.dm.J[self.dm.J["site_id"] == j_id].iloc[0]
                pop = float(j_row.get("pop", 0.0))
                w = self.J_priority.get(j_id, 1.0)

                covered_pop += pop * w

        total_pop = sum(
            float(row.get("pop", 0.0)) * self.J_priority.get(row["site_id"], 1.0)
            for _, row in self.dm.J.iterrows()
        )
        Rcov = covered_pop / max(total_pop, 1.0)

        # Stage 1: maximize coverage → minimize (1 - Rcov)
        ind.fitness = 1.0 - Rcov
        ind.metrics = {
            "stage": 1,
            "Rcov_BTS": Rcov,
            "covered_pop": covered_pop
        }
        return ind.fitness

    def compute_remaining_J(self, best_power_ind: Individual):
        covered_J = set()

        for bts_id, power_id in best_power_ind.power_assign.items():
            if power_id is None:
                continue  # CHỈ BTS ĐƯỢC CẤP NGUỒN

            pinfo = self.power_travel_map.get((power_id, bts_id))
            if pinfo is None or pinfo["total_time_hr"] == np.inf:
                continue  # KHÔNG KHẢ THI

            for j_id, can_cover in self.cover_bts.get(bts_id, {}).items():
                if can_cover:
                    covered_J.add(j_id)

        return [
            j for j in self.J_ids
            if j not in covered_J and not self.dm.is_J_flooded(j)
        ]

    def evaluate_stage2_cow(self, ind: Individual, J_remain: List[str]):
        """
        Stage 2:
          - Ràng buộc cứng: phủ TOÀN BỘ J_remain
          - Mục tiêu: min (T_deploy, Cost)
        """
        covered = set()
        max_time = 0.0
        total_cost = 0.0
        setup_time = float(self.params.get("default_setup_time_h", 0.5))

        used_J = set()
        for cow_id, j_id in ind.cow_assign.items():
            if j_id is None:
                continue
            if j_id in used_J:
                ind.fitness = np.inf
                ind.metrics = {"stage": 2, "feasible": False}
                return ind.fitness
            used_J.add(j_id)

        for cow_id, j_id in ind.cow_assign.items():
            if j_id is None:
                continue

            if j_id not in J_remain:
                ind.cow_assign[cow_id] = None

            tinfo = self.cow_travel_map.get((cow_id, j_id))
            if not tinfo:
                continue

            covered.add(j_id)

            time_cow = tinfo["travel_time_hr"] + setup_time
            max_time = max(max_time, time_cow)

            crow = self.dm.cow[self.dm.cow["cow_id"] == cow_id].iloc[0]
            total_cost += float(crow.get("cost_vnd", 0.0)) + tinfo["travel_cost_vnd"]

        # KHÔNG phủ đủ -> loại
        if set(J_remain) - covered:
            ind.fitness = np.inf
            ind.metrics = {"stage": 2, "feasible": False}
            return ind.fitness

        # lexicographic: time trước, cost sau
        ind.fitness = max_time + 1e-9 * total_cost
        ind.metrics = {
            "stage": 2,
            "T_deploy": max_time,
            "Cost": total_cost,
            "covered_J": len(covered)
        }
        return ind.fitness

    # Initialization
    def _enforce_power_unique(self, ind: Individual):
        usage = {}  # power_id -> list of (bts_id, travel_time)

        for bts_id, p_id in ind.power_assign.items():
            if p_id is None:
                continue

            t = self.power_travel_map.get((p_id, bts_id), {}).get("total_time_hr", np.inf)
            usage.setdefault(p_id, []).append((bts_id, t))

        # For each power_id, keep only the assignment with minimum travel_time
        for p_id, lst in usage.items():
            if len(lst) <= 1:
                continue

            # Sort by travel_time -> keep smallest
            lst_sorted = sorted(lst, key=lambda x: x[1])
            keep_bts, _ = lst_sorted[0]

            # Remove all others
            for bts_remove, _ in lst_sorted[1:]:
                ind.power_assign[bts_remove] = None

    def _init_individual_random(self) -> Individual:
        ind = Individual(self.cow_ids, self.bts_to_power)
        # Randomly assign some COWs to feasible J (or None)
        for cow_id in self.cow_ids:
            if random.random() < 0.5:
                # choose a feasible J that cow can cover and not flooded
                feasible_js = [j for j in self.J_ids if self.cover_cow.get(cow_id, {}).get(j, False) and not self.dm.is_J_flooded(j)]
                if feasible_js:
                    ind.cow_assign[cow_id] = random.choice(feasible_js)
                else:
                    ind.cow_assign[cow_id] = None
            else:
                ind.cow_assign[cow_id] = None
        # Randomly assign power: choose compatible power with sufficient capacity
        for bts_id in self.bts_to_power:
            # find candidate power units with P >= power_W
            brow = self.dm.failed_bts[self.dm.failed_bts["site_id"] == bts_id].iloc[0]
            demand_w = float(brow["power_W"])
            candidates = []
            for _, prow in self.dm.backups.iterrows():
                # Here backup file does not have direct power_W field; assume resource_amount ~ capacity_kW or derive from model name
                # Use simple heuristic: if model contains "10KW" or resource_amount numeric proportion
                cap_kw = None
                if "runtime_h" in prow and "resource_amount" in prow:
                    # fallback: approximate capacity from resource_amount column if meaningful
                    try:
                        cap_kw = float(prow["resource_amount"])  # heuristic
                    except Exception:
                        cap_kw = None
                # also accept all because we will check further in repair/feasibility (strict rule: only one type per BTS)
                candidates.append(prow["power_id"])
            if candidates and random.random() < 0.7:
                ind.power_assign[bts_id] = random.choice(candidates)
            else:
                ind.power_assign[bts_id] = None
        self._enforce_power_unique(ind)
        return ind

    def _init_population(self, N: int) -> List[Individual]:
        pop = []
        # half random, half heuristic (coverage-first)
        for _ in range(N):
            ind = self._init_individual_random()
            pop.append(ind)
        # one heuristic seed: greedy fill coverage
        greedy = Individual(self.cow_ids, self.bts_to_power)
        # assign COW greedily: pick COW->J that yields max pop_covered/travel_cost ratio using J pop weight
        j_pop = {
            row["site_id"]: float(row.get("pop", 0.0)) * self.J_priority.get(row["site_id"], 1.0)
            for _, row in self.dm.J.iterrows()
        }
        used_sites = set()
        for cow_id in self.cow_ids:
            candidates = [(j, j_pop.get(j, 0.0)) for j in self.J_ids if self.cover_cow.get(cow_id, {}).get(j, False) and not self.dm.is_J_flooded(j)]
            if not candidates:
                greedy.cow_assign[cow_id] = None
                continue
            # choose highest pop
            best_j = max(candidates, key=lambda x: x[1])[0]
            if best_j in used_sites:
                greedy.cow_assign[cow_id] = None
            else:
                greedy.cow_assign[cow_id] = best_j
                used_sites.add(best_j)
        # assign power greedily by nearest power unit using travel map
        for bts_id in self.bts_to_power:
            candidates = [(pid, self.power_travel_map.get((pid, bts_id), {}).get("total_time_hr", np.inf))
                          for pid in self.power_ids]
            candidates = [c for c in candidates if c[1] < np.inf]

            if candidates:
                chosen = min(candidates, key=lambda x: x[1])[0]
                greedy.power_assign[bts_id] = chosen
            else:
                greedy.power_assign[bts_id] = None

        # enforce unique AFTER all assignments
        self._enforce_power_unique(greedy)
        pop[0] = greedy
        return pop

    # Fitness evaluation
    def evaluate_individual(self, ind: Individual):
        # compute coverage
        covered_pop = 0.0
        popJ = {row["site_id"]: float(row.get("pop", 0.0)) for _, row in self.dm.J.iterrows()}

        # Coverage by COW
        covered_J = set()
        for cow_id, j_id in ind.cow_assign.items():
            if j_id and (cow_id, j_id) in self.cow_travel_map:
                if not self.dm.is_J_flooded(j_id):
                    covered_J.add(j_id)
        for j in covered_J:
            pop = popJ.get(j, 0.0)
            w = self.J_priority.get(j, 1.0)
            covered_pop += pop * w

        # Coverage by restored BTS
        for bts_id, power_id in ind.power_assign.items():
            if power_id is None:
                continue
            travel_info = self.power_travel_map.get((power_id, bts_id))
            if travel_info is None or travel_info.get("total_time_hr", float("inf")) == float("inf"):
                continue

            brow = self.dm.failed_bts[self.dm.failed_bts["site_id"] == bts_id].iloc[0]
            covered_pop += float(brow.get("pop_covered", 0.0)) * 1.0

        total_pop = sum(
            float(row.get("pop", 0.0)) * self.J_priority.get(row["site_id"], 1.0)
            for _, row in self.dm.J.iterrows()
        )
        Rcov = covered_pop / max(total_pop, 1.0)

        # TIME & COST FORMULAS
        setup_time = float(self.params.get("default_setup_time_h", 0.5))

        max_time = 0.0
        total_cost = 0.0

        # COW Deployment Time & Cost
        for cow_id, j_id in ind.cow_assign.items():
            if j_id is None:
                continue

            tinfo = self.cow_travel_map.get((cow_id, j_id))
            if not tinfo:
                continue

            # NEW TIME
            time_cow = tinfo["travel_time_hr"] + setup_time
            max_time = max(max_time, time_cow)

            # NEW COST
            crow = self.dm.cow[self.dm.cow["cow_id"] == cow_id].iloc[0]
            cost_cow_fixed = float(crow.get("cost_vnd", 0.0))
            travel_cost = tinfo["travel_cost_vnd"]

            total_cost += cost_cow_fixed + travel_cost

        # POWER Deployment Time & Cost
        for bts_id, power_id in ind.power_assign.items():
            if power_id is None:
                continue

            # Travel + deployment info
            pinfo = self.power_travel_map.get((power_id, bts_id))
            if not pinfo:
                continue

            # TIME
            time_power = pinfo["total_time_hr"]
            max_time = max(max_time, time_power)

            # COST
            # (1) Deployment + transport cost (from backup_to_failed_bts.csv)
            cost_deploy = float(pinfo.get("travel_cost_vnd", 0.0))

            # (2) Operating cost 24h (from backup_power.csv)
            prow = self.dm.backups[self.dm.backups["power_id"] == power_id]
            if not prow.empty:
                cost_operating = float(prow.iloc[0].get("cost_vnd_24h", 0.0))
            else:
                cost_operating = 0.0

            # (3) Total power cost
            total_cost += cost_deploy + cost_operating

        # Budget penalty
        budget_max = float(self.params.get("budget_max", 1e9))
        penalty = max(0.0, (total_cost - budget_max) / budget_max)

        Tmax = max(1.0, max_time)
        w_time = self.params["weights"]["w_time"]
        w_cost = self.params["weights"]["w_cost"]

        f = self.params["weights"]["coverage_weight"] * (1.0 - Rcov) + w_time * (max_time / Tmax) + w_cost * penalty

        ind.fitness = f
        ind.metrics = {
            "Rcov": Rcov,
            "max_time_hr": max_time,
            "total_cost_vnd": total_cost,
            "covered_pop": covered_pop
        }
        return ind.fitness

    def _is_cow_coverage_overlap(self, j1, j2, cow_id):
        """
        Check whether site j1 lies inside coverage of cow deployed at j2
        """
        row1 = self.dm.J[self.dm.J["site_id"] == j1].iloc[0]
        row2 = self.dm.J[self.dm.J["site_id"] == j2].iloc[0]

        crow = self.dm.cow[self.dm.cow["cow_id"] == cow_id].iloc[0]
        cov_m = float(crow["coverage_radius_m"])

        d_km = haversine_km(
            float(row1["latitude"]), float(row1["longitude"]),
            float(row2["latitude"]), float(row2["longitude"])
        )

        return (d_km * 1000.0) <= cov_m

    # Repair operators
    def repair(self, ind: Individual):
        # enforce 1 COW per site: if duplicates, keep one with best metric (heuristic: keep lower travel_cost)
        # build map site->list of cows
        site_to_cows = {}
        for cow_id, site in ind.cow_assign.items():
            if site is None:
                continue
            site_to_cows.setdefault(site, []).append(cow_id)
        for site, cows in site_to_cows.items():
            if len(cows) <= 1:
                continue
            # select cow with minimum travel cost
            best = min(cows, key=lambda c: self.cow_travel_map.get((c, site), {}).get("travel_cost_vnd", np.inf))
            for c in cows:
                if c != best:
                    ind.cow_assign[c] = None

        # ============================================================
        # NEW: enforce NON-OVERLAPPING COW coverage (CORRECT VERSION)
        # ============================================================

        # build active cows list
        active_cows = [(cid, jid) for cid, jid in ind.cow_assign.items() if jid is not None]

        occupied_J = set(j for _, j in active_cows)

        for cow_id, j_id in list(active_cows):

            # refresh active cows each iteration
            active_cows = [(cid, jid) for cid, jid in ind.cow_assign.items() if jid is not None]

            # check overlap with other cows
            overlapped = False
            for other_cow, other_j in active_cows:
                if other_cow == cow_id:
                    continue

                if self._is_cow_coverage_overlap(j_id, other_j, other_cow):
                    overlapped = True
                    break

            if not overlapped:
                continue

            # Try to relocate this COW
            feasible_js = []
            for j in self.J_ids:
                if j in occupied_J:
                    continue
                if self.dm.is_J_flooded(j):
                    continue
                if not self.cover_cow.get(cow_id, {}).get(j, False):
                    continue

                # ensure new J is not inside coverage of any existing COW
                ok = True
                for _, other_j in active_cows:
                    if self._is_cow_coverage_overlap(j, other_j, cow_id):
                        ok = False
                        break

                if ok:
                    feasible_js.append(j)

            if feasible_js:
                new_j = random.choice(feasible_js)
                ind.cow_assign[cow_id] = new_j
                occupied_J.add(new_j)
            else:
                # no feasible relocation → remove this COW
                ind.cow_assign[cow_id] = None

        # enforce power compatibility and single type per BTS
        for bts_id, power_id in ind.power_assign.items():
            if power_id is None:
                continue
            # check travel exists
            if (power_id, bts_id) not in self.power_travel_map:
                ind.power_assign[bts_id] = None
                continue
            # Here enforce that only one type is provided: backup dataset has 'type' column
            p_row = self.dm.backups[self.dm.backups["power_id"] == power_id]
            if p_row.empty:
                ind.power_assign[bts_id] = None
                continue
            p_type = p_row.iloc[0].get("type", "").upper()
            # ensure capacity >= demand if possible (best-effort)
            b_row = self.dm.failed_bts[self.dm.failed_bts["site_id"] == bts_id].iloc[0]
            demand_w = float(b_row["power_W"])
            # try simple check: if resource_amount present and numeric, use it
            try:
                cap = float(p_row.iloc[0].get("resource_amount", np.nan))
                if not math.isnan(cap) and cap < 1.0:
                    # if too small, unassign
                    ind.power_assign[bts_id] = None
            except Exception:
                pass
        # budget repair: if cost > budget, remove least efficient units
        self.evaluate_individual(ind)
        total_cost = ind.metrics["total_cost_vnd"]
        budget_max = float(self.params.get("budget_max", 1e9))
        if total_cost > budget_max:
            # compute effectivity eff = delta_coverage / cost for each enabled component
            eff_list = []
            # for each cow
            for cow_id, site in ind.cow_assign.items():
                if site is None:
                    continue
                # compute delta pop (approx) as J.pop
                jpop = float(self.dm.J[self.dm.J["site_id"] == site]["pop"].values[0]) if "pop" in self.dm.J.columns else 0.0
                cost = self.cow_travel_map.get((cow_id, site), {}).get("travel_cost_vnd", 0.0)
                w = self.J_priority.get(site, 1.0)
                eff = (jpop * w) / max(cost, 1.0)
                eff_list.append(("cow", cow_id, eff, cost))
            # for each power
            for bts_id, power_id in ind.power_assign.items():
                if power_id is None:
                    continue
                # delta pop = pop_covered of BTS
                bpop = float(self.dm.failed_bts[self.dm.failed_bts["site_id"] == bts_id]["pop_covered"].values[0])

                pinfo = self.power_travel_map.get((power_id, bts_id), {})
                cost_deploy = pinfo.get("travel_cost_vnd", 0.0)

                prow = self.dm.backups[self.dm.backups["power_id"] == power_id]
                cost_operating = float(prow.iloc[0].get("cost_vnd_24h", 0.0)) if not prow.empty else 0.0

                cost = max(cost_deploy + cost_operating, 1e-6)
                eff = bpop / cost

                eff_list.append(("power", bts_id, eff, cost))
            # sort ascending by eff (least effective first) and drop until cost<=budget
            eff_list.sort(key=lambda x: x[2])
            for typ, idv, eff, cost in eff_list:
                if total_cost <= budget_max:
                    break
                if typ == "cow":
                    ind.cow_assign[idv] = None
                else:
                    ind.power_assign[idv] = None
                total_cost -= cost
        self._enforce_power_unique(ind)
        return ind

    # Genetic operators
    def crossover(self, a: Individual, b: Individual) -> Individual:
        # uniform crossover
        child = a.copy()
        # cows
        for c in child.cow_assign.keys():
            if random.random() < 0.5:
                child.cow_assign[c] = b.cow_assign.get(c)
        # power
        for p in child.power_assign.keys():
            if random.random() < 0.5:
                child.power_assign[p] = b.power_assign.get(p)
        self._enforce_power_unique(child)
        return child

    def mutate(self, ind: Individual, mutation_rate: float):
        # mutate cow assignment: with prob mutation_rate change to another feasible site or None
        for cow_id in ind.cow_assign.keys():
            if random.random() < mutation_rate:
                feasible_js = [j for j in self.J_ids if self.cover_cow.get(cow_id, {}).get(j, False) and not self.dm.is_J_flooded(j)]
                if feasible_js:
                    ind.cow_assign[cow_id] = random.choice(feasible_js + [None])
                else:
                    ind.cow_assign[cow_id] = None
        # mutate power assignment: swap to another power or None
        for bts_id in ind.power_assign.keys():
            if random.random() < mutation_rate:
                ind.power_assign[bts_id] = random.choice(self.power_ids + [None])
        self._enforce_power_unique(ind)
        return ind

    # PSO-style discrete update (simple)
    def pso_update(self, ind: Individual, pbest: Individual, gbest: Individual, w_p=0.3, w_g=0.3):
        # For each gene, with prob w_p copy from pbest, with prob w_g copy from gbest
        child = ind.copy()
        for cow_id in child.cow_assign.keys():
            if random.random() < w_p:
                child.cow_assign[cow_id] = pbest.cow_assign.get(cow_id)
            if random.random() < w_g:
                child.cow_assign[cow_id] = gbest.cow_assign.get(cow_id)
        for bts_id in child.power_assign.keys():
            if random.random() < w_p:
                child.power_assign[bts_id] = pbest.power_assign.get(bts_id)
            if random.random() < w_g:
                child.power_assign[bts_id] = gbest.power_assign.get(bts_id)
        self._enforce_power_unique(child)
        return child

    # Main optimize loop
    def run_ga_pso(self):
        """
        GA-PSO 2 giai đoạn:
          Stage 1: Power only (maximize BTS coverage)
          Stage 2: COW only (cover all remaining, min time -> cost)
        """

        N = int(self.params["pop_size"])
        max_iter = int(self.params["max_iter"])
        mutation_rate = float(self.params["mutation_rate"])
        elitism = float(self.params["elitism"])
        n_elite = max(1, int(elitism * N))

        # =========================
        # STAGE 1 – POWER ONLY
        # =========================
        pop = self._init_population(N)

        # Disable COW genes
        for ind in pop:
            ind.cow_assign = {cid: None for cid in self.cow_ids}
            self.repair(ind)
            self.evaluate_stage1_power(ind)

        pbest = [ind.copy() for ind in pop]
        gbest = min(pop, key=lambda x: x.fitness).copy()

        for _ in range(max_iter):
            new_pop = sorted(pop, key=lambda x: x.fitness)[:n_elite]

            while len(new_pop) < N:
                a = self._tournament_select(pop)
                b = self._tournament_select(pop)

                child = self.pso_update(a, pbest[pop.index(a)], gbest)
                child = self.crossover(child, b)
                child = self.mutate(child, mutation_rate)

                # Disable COW again
                child.cow_assign = {cid: None for cid in self.cow_ids}

                self.repair(child)
                self.evaluate_stage1_power(child)
                new_pop.append(child)

            pop = new_pop
            for i, ind in enumerate(pop):
                if ind.fitness < pbest[i].fitness:
                    pbest[i] = ind.copy()
            gbest = min(pop, key=lambda x: x.fitness).copy()

        best_power = gbest.copy()

        # =========================
        # STAGE 2 – COW ONLY
        # =========================
        J_remain = self.compute_remaining_J(best_power)

        pop = self._init_population(N)

        # Fix power_assign, only optimize COW
        for ind in pop:
            ind.power_assign = dict(best_power.power_assign)
            self.repair(ind)
            self.evaluate_stage2_cow(ind, J_remain)

        pbest = [ind.copy() for ind in pop]
        gbest = min(pop, key=lambda x: x.fitness).copy()

        for _ in range(max_iter):
            new_pop = sorted(pop, key=lambda x: x.fitness)[:n_elite]

            while len(new_pop) < N:
                a = self._tournament_select(pop)
                b = self._tournament_select(pop)

                child = self.pso_update(a, pbest[pop.index(a)], gbest)
                child = self.crossover(child, b)
                child = self.mutate(child, mutation_rate)

                # Fix power_assign
                child.power_assign = dict(best_power.power_assign)

                self.repair(child)
                self.evaluate_stage2_cow(child, J_remain)
                new_pop.append(child)

            pop = new_pop
            for i, ind in enumerate(pop):
                if ind.fitness < pbest[i].fitness:
                    pbest[i] = ind.copy()
            gbest = min(pop, key=lambda x: x.fitness).copy()

        final_solution = gbest.copy()
        self._export_solution(final_solution)
        return final_solution

    # Selection
    def _tournament_select(self, pop: List[Individual], k=3):
        cand = random.sample(pop, k)
        return min(cand, key=lambda x: x.fitness)

    # Output
    def _export_solution(self, sol: Individual):
        # EXPORT COW ASSIGNMENTS
        cow_rows = []
        setup_time = float(self.params.get("default_setup_time_h", 0.5))

        for cow_id, site_id in sol.cow_assign.items():

            # Xuất kết quả COW ĐƯỢC TRIỂN KHAI
            if site_id is None:
                continue

            key = (cow_id, site_id)
            if key not in self.cow_travel_map:
                continue

            tinfo = self.cow_travel_map[key]

            crow = self.dm.cow[self.dm.cow["cow_id"] == cow_id].iloc[0]

            distance_km = float(tinfo["distance_km"])
            travel_time_hr = float(tinfo["travel_time_hr"])
            travel_cost_vnd = float(tinfo["travel_cost_vnd"])
            cost_vnd = float(crow.get("cost_vnd", 0.0))
            total_time_hr = travel_time_hr + setup_time
            total_cost_vnd = travel_cost_vnd + cost_vnd

            cow_rows.append({
                "cow_id": cow_id,
                "site_id": site_id,
                "distance_km": distance_km,
                "travel_time_hr": travel_time_hr,
                "total_time_hr": total_time_hr,
                "travel_cost_vnd": travel_cost_vnd,
                "cost_vnd": cost_vnd,
                "total_cost_vnd": total_cost_vnd
            })

        df_cow_out = pd.DataFrame(
            cow_rows,
            columns=[
                "cow_id",
                "site_id",
                "distance_km",
                "travel_time_hr",
                "total_time_hr",
                "travel_cost_vnd",
                "cost_vnd",
                "total_cost_vnd"
            ]
        )

        df_cow_out.to_csv(
            self.output_dir / "solution_cow_assignments.csv",
            index=False
        )

        # EXPORT POWER ASSIGNMENTS
        power_rows = []

        for bts_id, power_id in sol.power_assign.items():

            # Xuất kết quả BTS ĐƯỢC CẤP NGUỒN
            if power_id is None:
                continue

            key = (power_id, bts_id)
            if key not in self.power_travel_map:
                continue

            pinfo = self.power_travel_map[key]

            total_time_hr = float(pinfo["total_time_hr"])

            # travel_cost_vnd = chi phí di chuyển
            travel_cost_vnd = float(pinfo["travel_cost_vnd"])

            prow = self.dm.backups[self.dm.backups["power_id"] == power_id].iloc[0]
            operating_cost_vnd_24h = float(prow.get("cost_vnd_24h", 0.0))

            total_cost_vnd = travel_cost_vnd + operating_cost_vnd_24h

            power_rows.append({
                "bts_id": bts_id,
                "power_id": power_id,
                "total_time_hr": total_time_hr,
                "operating_cost_vnd_24h": operating_cost_vnd_24h,
                "travel_cost_vnd": travel_cost_vnd,
                "total_cost_vnd": total_cost_vnd
            })

        df_power_out = pd.DataFrame(
            power_rows,
            columns=[
                "bts_id",
                "power_id",
                "total_time_hr",
                "operating_cost_vnd_24h",
                "travel_cost_vnd",
                "total_cost_vnd"
            ]
        )

        df_power_out.to_csv(
            self.output_dir / "solution_power_assignments.csv",
            index=False
        )

        # summary
        summary = {
            "fitness": sol.fitness,
            **sol.metrics
        }
        with open(self.output_dir / "solution_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logging.info("Solution exported to %s", str(self.output_dir))

# CLI
def run_from_config(config: Dict = None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    if config is None:
        cfg = DEFAULT_CONFIG
    else:
        cfg = merge_with_default(config)

    solver = GA_PSOSolver(cfg)
    best = solver.run_ga_pso()
    return best

if __name__ == "__main__":
    run_from_config()
