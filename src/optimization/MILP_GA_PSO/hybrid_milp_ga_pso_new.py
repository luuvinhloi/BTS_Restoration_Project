"""
Hybrid MILP + GA-PSO solver module for BTS restoration
Save path (user): BTS_Restoration_Project/src/optimization/GA_PSO/hybrid_milp_ga_pso.py

This file is a cleaned, single-copy version of your hybrid solver.
"""

import os
import json
import math
import random
import time
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Set

import pandas as pd
import numpy as np
import networkx as nx
import pulp
from pathlib import Path

try:
    import rasterio
    from rasterio import features
except Exception:
    rasterio = None

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# hybrid_milp_ga_pso.py
# parents[0] = MILP_GA_PSO
# parents[1] = optimization
# parents[2] = src
# parents[3] = BTS_Restoration_Project

DATA_ROOT = PROJECT_ROOT / "data" / "processed"

PATHS = {
    'J_sites': DATA_ROOT / "position_I_J" / "J_sites.csv",
    'cow_dataset': DATA_ROOT / "cow" / "cow_dataset.csv",
    'backup_power': DATA_ROOT / "backup_power" / "backup_power.csv",
    'failed_bts': DATA_ROOT / "damage_bts" / "failed_bts.csv",
    'flood_tif': DATA_ROOT / "flood" / "flood_depth_combined_clean.tif",
    'roads_graph': DATA_ROOT / "road" / "roads_flooded.graphml",
    'cow_travel': DATA_ROOT / "travel_cost" / "cow_to_J_sites.csv",
    'power_travel': DATA_ROOT / "travel_cost" / "backup_to_failed_bts.csv"
}

BUDGET_MAX = 1e9  # 1 tỷ VNĐ
ALPHA = 1.0
BETA = 0.01
GAMMA = 0.000001

# Data structures
COW = namedtuple('COW', 'cow_id base_id base_name type lat lon coverage_radius_m power_kw speed_kmh endurance_hr cost_vnd assigned_region')
POWER = namedtuple('POWER', 'base_id power_id lat lon base_name type model runtime_h cost_vnd_24h resource_amount')
BTS = namedtuple('BTS', 'site_id latitude longitude utm_x utm_y pop_covered pop_unique_covered overlap_ratio_network total_unique_pop_network elevation_m slope_deg neighbour_weight dist_to_school_m dist_to_hospital_m dist_to_road_m dist_to_residential_m dist_to_industrial_m site_accessibility_score antenna_height_m region_type bts_type coverage_radius_m power_W flooded status')
JSITE = namedtuple('JSITE', 'site_id i_ref latitude longitude pop priority_category priority_weight slope dist_to_road_m in_water')

# Utility
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def get_priority_weight(J_df, site_id):
    row = J_df[J_df['site_id'] == site_id]
    if not row.empty and 'priority_weight' in row.columns:
        try:
            return float(row.iloc[0]['priority_weight'])
        except Exception:
            return 1.0
    return 1.0

# Data Loading
def load_all():
    print("Loading datasets...")
    J_df = pd.read_csv(PATHS['J_sites'])
    cow_df = pd.read_csv(PATHS['cow_dataset'])
    power_df = pd.read_csv(PATHS['backup_power'])
    bts_df = pd.read_csv(PATHS['failed_bts'])
    cow_travel_df = pd.read_csv(PATHS['cow_travel'])
    power_travel_df = pd.read_csv(PATHS['power_travel'])

    G = None
    if os.path.exists(PATHS['roads_graph']):
        try:
            G = nx.read_graphml(PATHS['roads_graph'])
            print('Loaded roads graph', G.number_of_nodes(), 'nodes')
        except Exception as e:
            print('Could not load roads graph:', e)

    flood_src = None
    if rasterio and os.path.exists(PATHS['flood_tif']):
        try:
            flood_src = rasterio.open(PATHS['flood_tif'])
            print('Loaded flood raster')
        except Exception as e:
            print('Could not open flood raster:', e)

    return {
        'J_df': J_df,
        'cow_df': cow_df,
        'power_df': power_df,
        'bts_df': bts_df,
        'cow_travel_df': cow_travel_df,
        'power_travel_df': power_travel_df,
        'graph': G,
        'flood_src': flood_src
    }

# Coverage matrices
def build_cover_matrices(bts_df, J_df, cow_df):
    print('Building coverage matrices...')
    cover_cow = defaultdict(dict)
    cover_bts = defaultdict(dict)

    # cows
    if {'cow_id', 'lat', 'lon', 'coverage_radius_m'}.issubset(cow_df.columns):
        for _, r in cow_df.iterrows():
            c_id = r['cow_id']
            clat, clon, Rm = r['lat'], r['lon'], r['coverage_radius_m']
            for _, j in J_df.iterrows():
                d = haversine_km(clat, clon, j['latitude'], j['longitude']) * 1000
                cover_cow[c_id][j['site_id']] = 1 if d <= Rm else 0

    # bts
    if {'latitude', 'longitude', 'coverage_radius_m', 'site_id'}.issubset(bts_df.columns):
        for _, b in bts_df.iterrows():
            for _, j in J_df.iterrows():
                d = haversine_km(b['latitude'], b['longitude'], j['latitude'], j['longitude']) * 1000
                cover_bts[b['site_id']][j['site_id']] = 1 if d <= b['coverage_radius_m'] else 0

    return cover_cow, cover_bts

# MILP presolve
def milp_presolve(data):
    J_df = data['J_df']
    cow_df = data['cow_df']
    power_df = data['power_df']
    bts_df = data['bts_df']
    cow_travel = data['cow_travel_df']
    power_travel = data['power_travel_df']

    feasible_cow_ids = set()
    if 'cow_id' in cow_df.columns:
        for _, cow in cow_df.iterrows():
            if not cow_travel[cow_travel['cow_id'] == cow['cow_id']].empty:
                feasible_cow_ids.add(cow['cow_id'])

    feasible_power_ids = set()
    if {'resource_amount', 'power_id'}.issubset(power_df.columns):
        feasible_power_ids = set(power_df[power_df['resource_amount'] > 0]['power_id'])

    reduced_J = [j['site_id'] for _, j in J_df.iterrows() if not bool(j.get('in_water', False))]

    seeds = []
    outage_bts = bts_df[bts_df.get('status', '') == 'power_outage'] if 'status' in bts_df.columns else pd.DataFrame()

    sorted_J = J_df.sort_values('pop', ascending=False).head(200) if 'pop' in J_df.columns else J_df

    for _ in [0, 1]:
        sol = {'cows': {}, 'powers': {}}
        used_powers = set()

        for _, j in sorted_J.iterrows():
            j_id = j['site_id']
            cand = cow_travel[cow_travel['site_id'] == j_id] if 'site_id' in cow_travel.columns else pd.DataFrame()
            if cand.empty:
                continue
            c = cand.sort_values('travel_cost_vnd').iloc[0] if 'travel_cost_vnd' in cand.columns else cand.iloc[0]
            if c['cow_id'] in feasible_cow_ids:
                sol['cows'][c['cow_id']] = j_id

        for _, row in outage_bts.iterrows():
            b_id = row['site_id']
            cand = power_travel[power_travel['bts_id'] == b_id] if 'bts_id' in power_travel.columns else pd.DataFrame()
            if cand.empty:
                continue
            for _, p in cand.sort_values('total_cost_vnd').iterrows() if 'total_cost_vnd' in cand.columns else cand.iterrows():
                if p['power_id'] in feasible_power_ids and p['power_id'] not in used_powers:
                    sol['powers'][b_id] = p['power_id']
                    used_powers.add(p['power_id'])
                    break

        seeds.append(sol)

    print(f"MILP presolve: {len(seeds)} seed solutions generated.")
    return reduced_J, feasible_cow_ids, feasible_power_ids, seeds

# =========================
# Helper functions (NEW)
# =========================

def compute_P_remain(data, restored_bts_ids, cover_bts):
    """
    Compute remaining uncovered population sites after BTS restoration
    """
    J_df = data['J_df']
    P_remain = set(J_df['site_id'].tolist())

    for bts_id in restored_bts_ids:
        covered_sites = cover_bts.get(bts_id, {})
        for j, v in covered_sites.items():
            if v == 1 and j in P_remain:
                P_remain.remove(j)

    return P_remain


def check_full_coverage(sol, P_remain, cover_cow):
    """
    Check if COW solution covers all remaining population sites
    """
    covered = set()
    for cow, site in sol.get('cows', {}).items():
        for j, v in cover_cow.get(cow, {}).items():
            if v == 1:
                covered.add(j)

    return P_remain.issubset(covered)

def check_cow_overlap(sol, cover_cow):
    """
    Return list of cows that are inside coverage of another cow
    """
    cows = list(sol.get('cows', {}).items())  # (cow_id, site_id)
    overlapped = set()

    for i, (c1, s1) in enumerate(cows):
        for j, (c2, s2) in enumerate(cows):
            if i == j:
                continue
            # if site s1 is covered by cow c2
            if cover_cow.get(c2, {}).get(s1, 0) == 1:
                overlapped.add(c1)

    return list(overlapped)

def relocate_overlapping_cows(sol, cover_cow, reduced_J, P_remain=None):
    """
    Relocate overlapping COWs to uncovered & non-overlapping J sites
    """
    cows = sol.get('cows', {})
    used_sites = set(cows.values())

    overlapped = check_cow_overlap(sol, cover_cow)
    if not overlapped:
        return sol

    for cow_id in overlapped:
        # try to find a new J site
        candidate_J = list(P_remain) if P_remain else list(reduced_J)

        for j in candidate_J:
            if j in used_sites:
                continue

            # check j is NOT covered by any other cow
            conflict = False
            for other_cow, other_site in cows.items():
                if other_cow == cow_id:
                    continue
                if cover_cow.get(other_cow, {}).get(j, 0) == 1:
                    conflict = True
                    break

            if not conflict:
                cows[cow_id] = j
                used_sites.add(j)
                break

        # if cannot relocate → remove this cow
        else:
            cows.pop(cow_id, None)

    sol['cows'] = cows
    return sol


# GA-PSO Implementation
class HybridGAPSO:
    def __init__(self, data, cover_cow, cover_bts, reduced_J, feasible_cows, feasible_powers, seeds,
                 pop_size=80, max_iter=150, elite_n=5):

        self.data = data
        self.cover_cow = cover_cow
        self.cover_bts = cover_bts
        self.reduced_J = set(reduced_J)

        self.feasible_cows = list(feasible_cows)
        self.feasible_powers = list(feasible_powers)
        self.seeds = seeds or []

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.elite_n = elite_n

        # Preload travel costs for speed (use .get to avoid KeyError)
        self.cow_travel = {
            (row.get('cow_id'), row.get('site_id')): (
                row.get('distance_km', np.nan),
                row.get('travel_time_hr', np.nan),
                row.get('travel_cost_vnd', np.nan),
            )
            for _, row in data['cow_travel_df'].iterrows()
        }

        self.power_travel = {
            (row.get('power_id'), row.get('bts_id')): (
                row.get('distance_km', np.nan),
                row.get('total_time_hr', np.nan),
                row.get('total_cost_vnd', np.nan),
            )
            for _, row in data['power_travel_df'].iterrows()
        }

        self.population = []
        self.fitnesses = []
        self.pbest = []
        self.pbest_f = []
        self.gbest = None
        self.gbest_f = -1e18

    def _enforce_power_unique(self, sol):
        usage = defaultdict(list)
        for bts_id, p in list(sol.get('powers', {}).items()):
            if p is None:
                continue
            key = (p, bts_id)
            tt = self.power_travel.get(key, (np.inf, np.inf, np.inf))[1]
            cost = self.power_travel.get(key, (np.inf, np.inf, np.inf))[2]
            usage[p].append((bts_id, tt, cost))

        for pid, lst in usage.items():
            if len(lst) <= 1:
                continue
            lst = sorted(lst, key=lambda x: (x[1], x[2]))
            keep = lst[0][0]
            for bts_remove, _, _ in lst[1:]:
                sol['powers'].pop(bts_remove, None)

    def initialize(self):
        print('Initializing GA-PSO population...')

        seed_count = max(0, int(self.pop_size * 0.5))
        seed_count = min(seed_count, len(self.seeds)) if self.seeds else 0

        for i in range(seed_count):
            s = self.seeds[i % len(self.seeds)]
            sol = {'cows': dict(s.get('cows', {})), 'powers': dict(s.get('powers', {}))}
            self._enforce_power_unique(sol)
            self.population.append(sol)

        outage_bts = self.data['bts_df'][self.data['bts_df'].get('status', '') == 'power_outage']['site_id'].tolist() if 'status' in self.data['bts_df'].columns else []

        while len(self.population) < self.pop_size:
            sol = {'cows': {}, 'powers': {}}

            # COW deployment
            if self.feasible_cows:
                for cow in random.sample(self.feasible_cows, k=min(len(self.feasible_cows), 10)):
                    if random.random() < 0.3:
                        sol['cows'][cow] = random.choice(list(self.reduced_J))

            # Power assignment
            used = set()
            if outage_bts:
                for b in random.sample(outage_bts, k=min(10, len(outage_bts))):
                    if random.random() < 0.5 and self.feasible_powers:
                        p = random.choice(self.feasible_powers)
                        if p not in used:
                            sol['powers'][b] = p
                            used.add(p)

            self._enforce_power_unique(sol)
            self.population.append(sol)

        # evaluate
        if not self.population:
            self.population = [{'cows': {}, 'powers': {}}]
        self.fitnesses = [0.0 for _ in self.population]
        self.pbest = [dict(sol) for sol in self.population]
        self.pbest_f = list(self.fitnesses)
        self.gbest = self.population[0]
        self.gbest_f = -1e18

        self.pbest = [dict(sol) for sol in self.population]
        self.pbest_f = list(self.fitnesses)
        idx = int(np.argmax(self.fitnesses))
        self.gbest = self.population[idx]
        self.gbest_f = self.fitnesses[idx]
        print('Initial best fitness =', self.gbest_f)

    def evaluate(self, sol, mode='bts', P_remain=None):
        """
        mode = 'bts'  → only BTS restoration (maximize coverage)
        mode = 'cow'  → only COW deployment (lexicographic: time → cost)
        """

        J_df = self.data['J_df']
        bts_df = self.data['bts_df']

        # =========================
        # PHASE 1: BTS restoration
        # =========================
        if mode == 'bts':
            covered = 0.0
            for bts_id, p in sol.get('powers', {}).items():
                row = bts_df[bts_df['site_id'] == bts_id]
                if not row.empty and 'pop_covered' in row.columns:
                    covered += float(row.iloc[0]['pop_covered']) * float(
                        row.iloc[0].get('neighbour_weight', 1.0)
                    )

            total_pop = float(bts_df['pop_covered'].sum()) + 1e-9
            return covered / total_pop

        # =========================
        # PHASE 2: COW deployment
        # =========================
        if mode == 'cow':
            # HARD CONSTRAINT: must cover all remaining areas
            if P_remain is not None:
                if not check_full_coverage(sol, P_remain, self.cover_cow):
                    return -1e9  # infeasible

            max_t = 0.0
            total_cost = 0.0

            for cow, site in sol.get('cows', {}).items():
                key = (cow, site)
                info = self.cow_travel.get(key)
                if not info:
                    return -1e9

                tt = 0.0 if pd.isna(info[1]) else float(info[1])
                cost_c = 0.0 if pd.isna(info[2]) else float(info[2])

                max_t = max(max_t, tt + 0.5)

                try:
                    cow_fixed = float(
                        self.data['cow_df'][self.data['cow_df']['cow_id'] == cow]['cost_vnd'].iloc[0]
                    )
                except Exception:
                    cow_fixed = 0.0

                total_cost += cost_c + cow_fixed
            weighted_covered = 0.0
            covered_J = set()

            for cow, site in sol.get('cows', {}).items():
                for j, v in self.cover_cow.get(cow, {}).items():
                    if v == 1 and j not in covered_J:
                        pop = float(J_df[J_df['site_id'] == j]['pop'].sum())
                        pw = get_priority_weight(J_df, j)
                        weighted_covered += pop * pw
                        covered_J.add(j)

            # Lexicographic → encode as tuple
            return weighted_covered - (max_t * 1e6 + total_cost)

    def repair(self, sol):
        cost_items = []
        total_cost = 0.0

        for cow_id, site_id in list(sol.get('cows', {}).items()):
            if site_id not in self.reduced_J:
                sol['cows'].pop(cow_id, None)
                continue
            key = (cow_id, site_id)
            info = self.cow_travel.get(key)
            if not info:
                sol['cows'].pop(cow_id, None)
                continue
            travel_cost = 0.0 if pd.isna(info[2]) else float(info[2])
            try:
                cow_fixed = float(self.data['cow_df'][self.data['cow_df']['cow_id'] == cow_id]['cost_vnd'].iloc[0])
            except Exception:
                cow_fixed = 0.0
            cost = cow_fixed + travel_cost
            pop = float(self.data['J_df'][self.data['J_df']['site_id'] == site_id]['pop'].sum()) if 'pop' in self.data['J_df'].columns else 0.0
            eff = pop / (cost + 1e-9)
            cost_items.append(('cow', cow_id, site_id, cost, eff))
            total_cost += cost

        for bts_id, p_id in list(sol.get('powers', {}).items()):
            key = (p_id, bts_id)
            info = self.power_travel.get(key)
            if not info:
                sol['powers'].pop(bts_id, None)
                continue
            pop = float(
                self.data['bts_df'][self.data['bts_df']['site_id'] == bts_id]['pop_covered'].sum()) if 'pop_covered' in \
                                                                                                       self.data[
                                                                                                           'bts_df'].columns else 0.0
            # deployment cost
            cost_deploy = 0.0 if pd.isna(info[2]) else float(info[2])

            # operating cost
            prow = self.data['power_df'][self.data['power_df']['power_id'] == p_id]
            if not prow.empty:
                cost_operating = float(prow.iloc[0].get('cost_vnd_24h', 0.0))
            else:
                cost_operating = 0.0

            cost = cost_deploy + cost_operating
            eff = pop / (cost + 1e-9)

            cost_items.append(('power', bts_id, p_id, cost, eff))
            total_cost += cost

        if total_cost > BUDGET_MAX:
            cost_items.sort(key=lambda x: x[4])
            for item in cost_items:
                if total_cost <= BUDGET_MAX:
                    break
                kind, id1, id2, c, _ = item
                if kind == 'cow' and id1 in sol.get('cows', {}):
                    sol['cows'].pop(id1, None)
                    total_cost -= c
                elif kind == 'power' and id1 in sol.get('powers', {}):
                    sol['powers'].pop(id1, None)
                    total_cost -= c

        self._enforce_power_unique(sol)

        # NEW: enforce non-overlapping COWs (only in COW phase)
        if sol.get('cows'):
            sol = relocate_overlapping_cows(
                sol,
                self.cover_cow,
                self.reduced_J,
                P_remain=getattr(self, '_P_remain', None)
            )

        return sol

    def crossover(self, a, b):
        child = {'cows': {}, 'powers': {}}
        for cow in set(a.get('cows', {}).keys()) | set(b.get('cows', {}).keys()):
            if random.random() < 0.5:
                if cow in a.get('cows', {}):
                    child['cows'][cow] = a['cows'][cow]
            else:
                if cow in b.get('cows', {}):
                    child['cows'][cow] = b['cows'][cow]

        for bt in set(a.get('powers', {}).keys()) | set(b.get('powers', {}).keys()):
            if random.random() < 0.5:
                if bt in a.get('powers', {}):
                    child['powers'][bt] = a['powers'][bt]
            else:
                if bt in b.get('powers', {}):
                    child['powers'][bt] = b['powers'][bt]

        self._enforce_power_unique(child)
        return child

    def mutate(self, sol, p_mut=0.05, mode='bts'):
        # =====================
        # PHA 1: BTS ONLY
        # =====================
        if mode == 'bts':
            # NO COW
            sol['cows'] = {}

            outage_bts = self.data['bts_df'][
                self.data['bts_df'].get('status', '') == 'power_outage'
                ]['site_id'].tolist() if 'status' in self.data['bts_df'].columns else []

            if random.random() < p_mut and outage_bts:
                b = random.choice(outage_bts)
                possible = [p for p in self.feasible_powers
                            if p not in sol.get('powers', {}).values()]
                if possible:
                    sol['powers'][b] = random.choice(possible)

            self._enforce_power_unique(sol)
            return sol

        # =====================
        # PHA 2: COW ONLY
        # =====================
        if mode == 'cow':
            sol['powers'] = {}

            if random.random() < p_mut and self.feasible_cows:
                cow = random.choice(self.feasible_cows)
                sol['cows'][cow] = random.choice(list(self.reduced_J))

            return sol

    def pso_update(self, sol, pbest, gbest, mode='bts', w_p=0.2, w_g=0.2):
        child = {'cows': {}, 'powers': {}}

        if mode == 'bts':
            for b in set(sol.get('powers', {})) | set(pbest.get('powers', {})) | set(gbest.get('powers', {})):
                r = random.random()
                if r < w_p and b in pbest.get('powers', {}):
                    child['powers'][b] = pbest['powers'][b]
                elif r < w_p + w_g and b in gbest.get('powers', {}):
                    child['powers'][b] = gbest['powers'][b]
                elif b in sol.get('powers', {}):
                    child['powers'][b] = sol['powers'][b]
            return child

        if mode == 'cow':
            for c in set(sol.get('cows', {})) | set(pbest.get('cows', {})) | set(gbest.get('cows', {})):
                r = random.random()
                if r < w_p and c in pbest.get('cows', {}):
                    child['cows'][c] = pbest['cows'][c]
                elif r < w_p + w_g and c in gbest.get('cows', {}):
                    child['cows'][c] = gbest['cows'][c]
                elif c in sol.get('cows', {}):
                    child['cows'][c] = sol['cows'][c]
            return child

    def run(self, mode='bts', P_remain=None):
        self._P_remain = P_remain
        self.initialize()
        stagn = 0

        for it in range(self.max_iter):
            new_pop = []
            new_f = []

            ranked = sorted(zip(self.population, self.fitnesses), key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in ranked[:self.elite_n]]

            new_pop.extend(elites)
            new_f.extend([x[1] for x in ranked[:self.elite_n]])

            while len(new_pop) < self.pop_size:
                idx = random.randrange(len(self.population))
                parent = self.population[idx]

                child = self.pso_update(parent, self.pbest[idx], self.gbest, mode=mode)
                if random.random() < 0.8:
                    mate = random.choice(self.population)
                    child = self.crossover(child, mate)

                child = self.mutate(child, mode=mode)
                child = self.repair(child)
                f = self.evaluate(child, mode=mode, P_remain=P_remain)

                new_pop.append(child)
                new_f.append(f)

            self.population = new_pop
            self.fitnesses = new_f

            for i, f in enumerate(new_f):
                if f > self.pbest_f[i]:
                    self.pbest[i] = self.population[i]
                    self.pbest_f[i] = f
                if f > self.gbest_f:
                    self.gbest = self.population[i]
                    self.gbest_f = f
                    stagn = 0

            stagn += 1
            if it % 10 == 0:
                print(f"Iter {it}: gbest_f = {self.gbest_f:.6f}")

            if stagn > 50:
                print('Stopping due to stagnation.')
                break

        return self.gbest, self.gbest_f

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# Metrics + Exporter
def _build_travel_maps(data):
    cow_travel_map = {}
    for _, r in data['cow_travel_df'].iterrows():
        cow_travel_map[(r.get('cow_id'), r.get('site_id'))] = {
            'distance_km': safe_float(r.get('distance_km')),
            'travel_time_hr': safe_float(r.get('travel_time_hr')),
            'travel_cost_vnd': safe_float(r.get('travel_cost_vnd'))
        }

    power_travel_map = {}
    for _, r in data['power_travel_df'].iterrows():
        power_travel_map[(r.get('power_id'), r.get('bts_id'))] = {
            'distance_km': float(r.get('distance_km', np.nan)),
            'total_time_hr': float(r.get('total_time_hr', np.nan)) if not pd.isna(r.get('total_time_hr', None)) else np.nan,
            'total_cost_vnd': float(r.get('total_cost_vnd', np.nan)) if not pd.isna(r.get('total_cost_vnd', None)) else np.nan,
            'note': r.get('note', '')
        }
    return cow_travel_map, power_travel_map

def compute_solution_metrics(data, sol):
    J_df = data['J_df']
    bts_df = data['bts_df']
    cow_travel_map, power_travel_map = _build_travel_maps(data)

    covered = 0.0
    total_cost = 0.0
    max_t = 0.0

    # cows
    for cow, site in sol.get('cows', {}).items():
        if site is None or site == '':
            continue
        row = J_df[J_df['site_id'] == site]
        if not row.empty and 'pop' in row.columns:
            pw = get_priority_weight(J_df, site)
            covered += float(row.iloc[0].get('pop', 0.0)) * pw
        info = cow_travel_map.get((cow, site))
        if info:
            tt = 0.0 if pd.isna(info['travel_time_hr']) else info['travel_time_hr']
            cost_c = 0.0 if pd.isna(info['travel_cost_vnd']) else info['travel_cost_vnd']
            max_t = max(max_t, tt)
            total_cost += cost_c
        try:
            cow_fixed = float(data['cow_df'][data['cow_df']['cow_id'] == cow]['cost_vnd'].iloc[0])
            total_cost += cow_fixed
        except Exception:
            pass

    # powers
    for bts_id, p in sol.get('powers', {}).items():
        if p is None or p == '':
            continue
        row = bts_df[bts_df['site_id'] == bts_id]
        if not row.empty and 'pop_covered' in row.columns:
            covered += float(row.iloc[0].get('pop_covered', 0.0))
        info = power_travel_map.get((p, bts_id))
        if info:
            tt = 0.0 if pd.isna(info['total_time_hr']) else info['total_time_hr']
            max_t = max(max_t, tt)
            cost_deploy = 0.0 if pd.isna(info['total_cost_vnd']) else info['total_cost_vnd']

            prow = data['power_df'][data['power_df']['power_id'] == p]
            if not prow.empty:
                cost_operating = float(prow.iloc[0].get('cost_vnd_24h', 0.0))
            else:
                cost_operating = 0.0

            total_cost += cost_deploy + cost_operating

    total_pop = float((J_df['pop'].sum() if 'pop' in J_df.columns else 0.0) + (bts_df['pop_covered'].sum() if 'pop_covered' in bts_df.columns else 0.0))
    Rcov = covered / (total_pop + 1e-9)

    Tnorm = max_t / (24.0 + 1e-9)
    Cost_pen = max(0.0, (total_cost - BUDGET_MAX) / (BUDGET_MAX + 1e-9))
    fitness = ALPHA * Rcov - BETA * Tnorm - GAMMA * Cost_pen

    return {
        'fitness': fitness,
        'Rcov': Rcov,
        'max_time_hr': max_t,
        'total_cost_vnd': total_cost,
        'covered_pop': covered
    }

def milp_local_refinement(data, sol, cover_cow, cover_bts, time_limit_sec=60, top_k_neighbors=3):
    try:
        J_df = data['J_df']
        bts_df = data['bts_df']
        cow_df = data['cow_travel_df']
        power_df = data['power_travel_df']

        # Build lookups
        cow_lookup = {}
        for _, r in cow_df.iterrows():
            cow_id = r.get('cow_id')
            site_id = r.get('site_id')
            try:
                travel_time = float(r.get('travel_time_hr', 0.0)) + 0.5
            except Exception:
                travel_time = 0.5
            try:
                travel_cost = float(r.get('travel_cost_vnd', 0.0))
            except Exception:
                travel_cost = 0.0
            try:
                cow_fixed = float(data['cow_df'][data['cow_df']['cow_id'] == cow_id]['cost_vnd'].iloc[0])
            except Exception:
                cow_fixed = 0.0
            cow_lookup[(cow_id, site_id)] = (travel_time, travel_cost + cow_fixed)

        power_lookup = {}
        for _, r in power_df.iterrows():
            pid = r.get('power_id')
            bid = r.get('bts_id')
            try:
                tt = float(r.get('total_time_hr', 0.0))
            except Exception:
                tt = 0.0

            # deployment cost
            cc_deploy = float(r.get('total_cost_vnd', 0.0))

            prow = data['power_df'][data['power_df']['power_id'] == pid]
            if not prow.empty:
                cc_operating = float(prow.iloc[0].get('cost_vnd_24h', 0.0))
            else:
                cc_operating = 0.0

            power_lookup[(pid, bid)] = (tt, cc_deploy + cc_operating)

        # Candidate neighborhoods
        cow_candidates = {}
        if 'cow_id' in cow_df.columns and 'site_id' in cow_df.columns:
            for cow_id in cow_df['cow_id'].unique():
                entries = cow_df[cow_df['cow_id'] == cow_id]
                if 'travel_cost_vnd' in entries.columns:
                    entries = entries.sort_values('travel_cost_vnd')
                cand = entries.head(top_k_neighbors)['site_id'].tolist() if not entries.empty else []
                cur = sol.get('cows', {}).get(cow_id)
                if cur and cur not in cand:
                    cand.insert(0, cur)
                cow_candidates[cow_id] = list(dict.fromkeys(cand))

        power_candidates = {}
        outage = bts_df[bts_df.get('status', '') == 'power_outage']['site_id'].tolist() if 'status' in bts_df.columns else []
        for b in outage:
            entries = power_df[power_df['bts_id'] == b] if 'bts_id' in power_df.columns else pd.DataFrame()
            if 'total_cost_vnd' in entries.columns:
                entries = entries.sort_values('total_cost_vnd')
            cand = entries.head(top_k_neighbors)['power_id'].tolist() if not entries.empty else []
            cur = sol.get('powers', {}).get(b)
            if cur and cur not in cand:
                cand.insert(0, cur)
            power_candidates[b] = list(dict.fromkeys(cand))

        if not cow_candidates and not power_candidates:
            return sol

        # Build MILP
        model = pulp.LpProblem('local_refinement', pulp.LpMaximize)
        x = {}
        z = {}

        for cow, sites in cow_candidates.items():
            for s in sites:
                x[(cow, s)] = pulp.LpVariable(f'x_cow_{cow}_{s}', cat='Binary')

        for bts_id, p_list in power_candidates.items():
            for p in p_list:
                z[(p, bts_id)] = pulp.LpVariable(f'z_pow_{p}_{bts_id}', cat='Binary')

        # constraints
        for cow in cow_candidates:
            model += pulp.lpSum(x[(cow, s)] for s in cow_candidates[cow]) <= 1
        for b in power_candidates:
            model += pulp.lpSum(z[(p, b)] for p in power_candidates[b]) <= 1

        all_powers = set(p for (p, b) in z.keys())
        for p in all_powers:
            model += pulp.lpSum(z[(pp, b)] for (pp, b) in z.keys() if pp == p) <= 1

        # objective coverage
        cov_terms = []
        for (cow, s), var in x.items():
            if cover_cow.get(cow, {}).get(s, 0) == 1:
                pop = float(J_df[J_df['site_id'] == s]['pop'].sum()) if 'pop' in J_df.columns else 0.0
                pw = float(J_df[J_df['site_id'] == s]['priority_weight'].iloc[0]) \
                    if 'priority_weight' in J_df.columns else 1.0
                cov_terms.append(pop * pw * var)
        for (p, b), var in z.items():
            pop = float(bts_df[bts_df['site_id'] == b]['pop_covered'].sum()) if 'pop_covered' in bts_df.columns else 0.0
            cov_terms.append(pop * var)

        covered_expr = pulp.lpSum(cov_terms)
        total_pop = float((J_df['pop'].sum() if 'pop' in J_df.columns else 0.0) + (bts_df['pop_covered'].sum() if 'pop_covered' in bts_df.columns else 0.0)) + 1e-9

        Tmax = pulp.LpVariable('Tmax', lowBound=0)
        for (cow, s), var in x.items():
            tt = cow_lookup.get((cow, s), (np.nan, np.nan))[0]
            if not pd.isna(tt):
                model += Tmax >= tt * var
        for (p, b), var in z.items():
            tt = power_lookup.get((p, b), (np.nan, np.nan))[0]
            if not pd.isna(tt):
                model += Tmax >= tt * var

        # cost
        cost_terms = []
        for (cow, s), var in x.items():
            cost = cow_lookup.get((cow, s), (np.nan, np.nan))[1]
            if not pd.isna(cost):
                cost_terms.append(cost * var)
        for (p, b), var in z.items():
            cost = power_lookup.get((p, b), (np.nan, np.nan))[1]
            if not pd.isna(cost):
                cost_terms.append(cost * var)

        total_cost_expr = pulp.lpSum(cost_terms)
        cost_pen = (total_cost_expr - BUDGET_MAX) / (BUDGET_MAX + 1e-9)

        model += ALPHA * (covered_expr / total_pop) - BETA * (Tmax / 24.0) - GAMMA * cost_pen

        model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=False))
        if pulp.LpStatus[model.status] not in ['Optimal', 'Feasible']:
            return sol

        refined = {'cows': {}, 'powers': {}}
        for (cow, s), var in x.items():
            val = var.value()
            if val is not None and val > 0.5:
                refined['cows'][cow] = s
        for (p, b), var in z.items():
            val = var.value()
            if val is not None and val > 0.5:
                refined['powers'][b] = p

        if not refined['cows'] and not refined['powers']:
            return sol
        return refined

    except Exception as e:
        print('MILP refine error:', e)
        return sol

def export_solution_files(data, sol, output_dir=None, prefix='solution'):
    if output_dir is None:
        output_dir = os.path.join('BTS_Restoration_Project', 'outputs', 'results_hybrid_new')
    os.makedirs(output_dir, exist_ok=True)

    cow_travel_map, power_travel_map = _build_travel_maps(data)

    cow_rows = []
    SETUP_TIME_HR = 0.5

    for cow_id, site_id in sorted(
            sol.get('cows', {}).items(),
            key=lambda x: x[0] if isinstance(x[0], (int, str)) else str(x[0])
    ):
        rec = {
            'cow_id': cow_id,
            'site_id': site_id if site_id is not None else ''
        }

        info = cow_travel_map.get((cow_id, site_id), {})

        travel_time = info.get('travel_time_hr', 0.0)
        travel_cost = info.get('travel_cost_vnd', 0.0)

        # fixed cow cost
        crow = data['cow_df'][data['cow_df']['cow_id'] == cow_id]
        if not crow.empty:
            cost_vnd = float(crow.iloc[0].get('cost_vnd', 0.0))
        else:
            cost_vnd = 0.0

        rec.update({
            'distance_km': info.get('distance_km', ''),
            'travel_time_hr': travel_time,
            'travel_cost_vnd': travel_cost,
            'total_time_hr': (
                float(travel_time) + SETUP_TIME_HR
                if travel_time != '' and not pd.isna(travel_time)
                else ''
            ),
            'total_cost_vnd': (
                float(travel_cost) + cost_vnd
                if travel_cost != '' and not pd.isna(travel_cost)
                else ''
            )
        })

        cow_rows.append(rec)

    df_cow = pd.DataFrame(
        cow_rows,
        columns=[
            'cow_id',
            'site_id',
            'distance_km',
            'travel_time_hr',
            'travel_cost_vnd',
            'total_time_hr',
            'total_cost_vnd'
        ]
    )

    cow_csv = os.path.join(output_dir, f'{prefix}_cow_assignments.csv')
    df_cow.to_csv(cow_csv, index=False)
    print('Wrote', cow_csv)

    power_rows = []
    for bts_id, p_id in sorted(sol.get('powers', {}).items(),
                               key=lambda x: x[0] if isinstance(x[0], (int, str)) else str(x[0])):

        info = power_travel_map.get((p_id, bts_id), {})

        # operating cost
        prow = data['power_df'][data['power_df']['power_id'] == p_id]
        if not prow.empty:
            cost_vnd_24h = float(prow.iloc[0].get('cost_vnd_24h', 0.0))
        else:
            cost_vnd_24h = 0.0

        travel_cost_vnd = info.get('total_cost_vnd', 0.0)

        rec = {
            'bts_id': bts_id,
            'power_id': p_id if p_id is not None else '',
            'distance_km': info.get('distance_km', ''),
            'total_time_hr': info.get('total_time_hr', ''),
            'total_travel_cost_vnd': travel_cost_vnd,
            'total_cost_vnd': travel_cost_vnd + cost_vnd_24h,
            'note': info.get('note', '')
        }

        power_rows.append(rec)

    df_power = pd.DataFrame(
        power_rows,
        columns=[
            'bts_id',
            'power_id',
            'distance_km',
            'total_time_hr',
            'total_travel_cost_vnd',
            'total_cost_vnd',
            'note'
        ]
    )

    power_csv = os.path.join(output_dir, f'{prefix}_power_assignments.csv')
    df_power.to_csv(power_csv, index=False)
    print('Wrote', power_csv)

    metrics = compute_solution_metrics(data, sol)
    summary_path = os.path.join(output_dir, f'{prefix}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Wrote', summary_path)

    return {'cow_csv': cow_csv, 'power_csv': power_csv, 'summary_json': summary_path, 'metrics': metrics}

# run_hybrid
def run_hybrid(max_iter=300, top_k=5, export_outputs=True):
    data = load_all()
    cover_cow, cover_bts = build_cover_matrices(
        data['bts_df'], data['J_df'], data['cow_df']
    )
    reduced_J, feasible_cows, feasible_powers, seeds = milp_presolve(data)

    # ==================================================
    # PHASE 1: GA-PSO — BTS RESTORATION
    # ==================================================
    ga_bts = HybridGAPSO(
        data, cover_cow, cover_bts,
        reduced_J, feasible_cows, feasible_powers,
        seeds, pop_size=80, max_iter=max_iter
    )

    # Disable COW entirely
    for s in ga_bts.seeds:
        s['cows'] = {}

    best_bts, best_bts_f = ga_bts.run(mode='bts')
    print('[PHASE 1] BTS fitness =', best_bts_f)

    restored_bts = set(best_bts.get('powers', {}).keys())

    # ==================================================
    # COMPUTE REMAINING UNCOVERED AREAS
    # ==================================================
    P_remain = compute_P_remain(data, restored_bts, cover_bts)
    print('[INFO] Remaining uncovered areas:', len(P_remain))

    # ==================================================
    # PHASE 2: GA-PSO — COW DEPLOYMENT
    # ==================================================
    ga_cow = HybridGAPSO(
        data, cover_cow, cover_bts,
        reduced_J, feasible_cows, feasible_powers,
        seeds=[], pop_size=80, max_iter=max_iter
    )

    best_cow, best_cow_f = ga_cow.run(
        mode='cow',
        P_remain=P_remain
    )
    print('[PHASE 2] COW fitness =', best_cow_f)

    # ==================================================
    # MERGE FINAL SOLUTION
    # ==================================================
    final_sol = {
        'powers': best_bts.get('powers', {}),
        'cows': best_cow.get('cows', {})
    }

    if export_outputs:
        export_solution_files(
            data, final_sol,
            output_dir=os.path.join(
                'BTS_Restoration_Project',
                'outputs',
                'results_hybrid_new'
            )
        )

    return final_sol

if __name__ == '__main__':
    run_hybrid(max_iter=300, top_k=5, export_outputs=True)
