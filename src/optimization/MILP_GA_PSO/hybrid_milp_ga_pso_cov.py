# =====================================================
# Hybrid MILP + GA-PSO solver module for BTS restoration
# (Step 1 HARD 100% COVERAGE – Step 2,3 unchanged)
# =====================================================

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
except Exception:
    rasterio = None

# --------------------------- Configuration ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

PATHS = {
    'J_sites': DATA_ROOT / "position_I_J" / "J_sites_new.csv",
    'cow_dataset': DATA_ROOT / "cow" / "cow_dataset.csv",
    'backup_power': DATA_ROOT / "backup_power" / "backup_power.csv",
    'failed_bts': DATA_ROOT / "damage_bts" / "failed_bts.csv",
    'flood_tif': DATA_ROOT / "flood" / "flood_depth_combined_B_clean.tif",
    'roads_graph': DATA_ROOT / "road" / "roads_flooded.graphml",
    'cow_travel': DATA_ROOT / "travel_cost" / "cow_to_J_sites_new.csv",
    'power_travel': DATA_ROOT / "travel_cost" / "backup_to_failed_bts_new.csv"
}

# ==========================
# Lexicographic Weights
# ==========================
ALPHA = 1.0        # coverage (HARD)
BETA = 0.01       # time
GAMMA = 0.000001  # cost

BUDGET_MAX = 1e9  # VNĐ

# --------------------------- Data Structures ---------------------------

COW = namedtuple('COW', 'cow_id lat lon coverage_radius_m cost_vnd')
POWER = namedtuple('POWER', 'power_id cost_vnd_24h resource_amount')
BTS = namedtuple('BTS', 'site_id latitude longitude pop_covered status')
JSITE = namedtuple('JSITE', 'site_id latitude longitude pop in_water')

# --------------------------- Utilities ---------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# --------------------------- Data Loading ---------------------------

def load_all():
    print("Loading datasets...")

    data = {
        'J_df': pd.read_csv(PATHS['J_sites']),
        'cow_df': pd.read_csv(PATHS['cow_dataset']),
        'power_df': pd.read_csv(PATHS['backup_power']),
        'bts_df': pd.read_csv(PATHS['failed_bts']),
        'cow_travel_df': pd.read_csv(PATHS['cow_travel']),
        'power_travel_df': pd.read_csv(PATHS['power_travel'])
    }

    if os.path.exists(PATHS['roads_graph']):
        try:
            data['graph'] = nx.read_graphml(PATHS['roads_graph'])
        except Exception:
            data['graph'] = None
    else:
        data['graph'] = None

    if rasterio and os.path.exists(PATHS['flood_tif']):
        try:
            data['flood_src'] = rasterio.open(PATHS['flood_tif'])
        except Exception:
            data['flood_src'] = None
    else:
        data['flood_src'] = None

    return data

# --------------------------- Coverage Matrices ---------------------------

def build_cover_matrices(bts_df, J_df, cow_df):
    """
    cover_cow[cow_id][j_site] = 1 nếu COW phủ được J
    cover_bts[bts_id][j_site] = 1 nếu BTS phủ được J
    """
    cover_cow = defaultdict(dict)
    cover_bts = defaultdict(dict)

    for _, cow in cow_df.iterrows():
        for _, j in J_df.iterrows():
            d = haversine_km(
                cow['lat'], cow['lon'],
                j['latitude'], j['longitude']
            ) * 1000
            cover_cow[cow['cow_id']][j['site_id']] = int(d <= cow['coverage_radius_m'])

    for _, bts in bts_df.iterrows():
        for _, j in J_df.iterrows():
            d = haversine_km(
                bts['latitude'], bts['longitude'],
                j['latitude'], j['longitude']
            ) * 1000
            cover_bts[bts['site_id']][j['site_id']] = int(d <= bts['coverage_radius_m'])

    return cover_cow, cover_bts

# --------------------------- MILP Presolve ---------------------------

def milp_presolve(data):
    """
    Mục tiêu: tạo seed solutions PHỦ HẾT J ngay từ đầu
    """
    J_df = data['J_df']
    cow_travel = data['cow_travel_df']
    power_travel = data['power_travel_df']
    bts_df = data['bts_df']

    reduced_J = set(J_df[J_df.get('in_water', False) == False]['site_id'])

    feasible_cows = set(cow_travel['cow_id'].unique())
    feasible_powers = set(data['power_df'][data['power_df']['resource_amount'] > 0]['power_id'])

    seeds = []

    # Greedy FULL COVER seed
    sol = {'cows': {}, 'powers': {}}

    for j in reduced_J:
        cand = cow_travel[cow_travel['site_id'] == j]
        if not cand.empty:
            best = cand.sort_values('travel_cost_vnd').iloc[0]
            sol['cows'][best['cow_id']] = j

    outage = bts_df[bts_df.get('status', '') == 'power_outage']['site_id']
    used_power = set()

    for b in outage:
        cand = power_travel[power_travel['bts_id'] == b]
        if not cand.empty:
            for _, r in cand.sort_values('total_cost_vnd').iterrows():
                if r['power_id'] not in used_power:
                    sol['powers'][b] = r['power_id']
                    used_power.add(r['power_id'])
                    break

    seeds.append(sol)

    print("MILP presolve: generated FULL-COVERAGE seed")

    return reduced_J, feasible_cows, feasible_powers, seeds

# =====================================================
# GA–PSO Core (PART 2)
# Step 1: HARD 100% COVERAGE
# Step 2,3: giữ nguyên
# =====================================================

class HybridGAPSO:
    def __init__(self, data, cover_cow, cover_bts,
                 reduced_J, feasible_cows, feasible_powers, seeds,
                 pop_size=80, max_iter=200, elite_n=5):

        self.data = data
        self.cover_cow = cover_cow
        self.cover_bts = cover_bts

        self.J_all = set(reduced_J)
        self.feasible_cows = list(feasible_cows)
        self.feasible_powers = list(feasible_powers)
        self.seeds = seeds or []

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.elite_n = elite_n

        # travel maps
        self.cow_travel = {
            (r['cow_id'], r['site_id']): (
                r.get('distance_km', np.nan),
                r.get('travel_time_hr', np.nan),
                r.get('travel_cost_vnd', np.nan),
            )
            for _, r in data['cow_travel_df'].iterrows()
        }

        self.power_travel = {
            (r['power_id'], r['bts_id']): (
                r.get('distance_km', np.nan),
                r.get('total_time_hr', np.nan),
                r.get('total_cost_vnd', np.nan),
            )
            for _, r in data['power_travel_df'].iterrows()
        }

        self.population = []
        self.fitnesses = []
        self.pbest = []
        self.pbest_f = []
        self.gbest = None
        self.gbest_f = -1e18

    # --------------------------------------------------
    # Utility: J covered by solution
    # --------------------------------------------------
    def _covered_J_sites(self, sol):
        covered = set()

        for cow, site in sol.get('cows', {}).items():
            if site is not None:
                covered.add(site)

        for bts_id, p in sol.get('powers', {}).items():
            if p is None:
                continue
            for j in self.cover_bts.get(bts_id, {}):
                if self.cover_bts[bts_id][j] == 1:
                    covered.add(j)

        return covered

    # --------------------------------------------------
    # HARD COVERAGE FITNESS
    # --------------------------------------------------
    def evaluate(self, sol):
        J_df = self.data['J_df']
        bts_df = self.data['bts_df']

        covered_J = self._covered_J_sites(sol)
        uncovered_J = self.J_all - covered_J

        # 🚨 HARD CONSTRAINT
        if uncovered_J:
            return -1e12  # tuyệt đối loại nghiệm chưa phủ hết

        # =============================
        # Step 2 + Step 3 (giữ nguyên)
        # =============================
        covered_pop = 0.0
        for _, j in J_df.iterrows():
            covered_pop += float(j.get('pop', 0.0))

        for _, b in bts_df.iterrows():
            covered_pop += float(b.get('pop_covered', 0.0))

        total_pop = covered_pop + 1e-9
        Rcov = 1.0  # đã phủ 100%

        max_t = 0.0
        total_cost = 0.0

        # COW
        for cow, site in sol.get('cows', {}).items():
            key = (cow, site)
            info = self.cow_travel.get(key)
            if info:
                tt = 0.0 if pd.isna(info[1]) else float(info[1])
                cost = 0.0 if pd.isna(info[2]) else float(info[2])
                max_t = max(max_t, tt + 0.5)
                total_cost += cost

            try:
                fixed = float(self.data['cow_df']
                              [self.data['cow_df']['cow_id'] == cow]['cost_vnd'].iloc[0])
                total_cost += fixed
            except Exception:
                pass

        # POWER
        for bts_id, p in sol.get('powers', {}).items():
            key = (p, bts_id)
            info = self.power_travel.get(key)
            if info:
                tt = 0.0 if pd.isna(info[1]) else float(info[1])
                cost = 0.0 if pd.isna(info[2]) else float(info[2])
                max_t = max(max_t, tt)
                total_cost += cost

            prow = self.data['power_df'][self.data['power_df']['power_id'] == p]
            if not prow.empty:
                total_cost += float(prow.iloc[0].get('cost_vnd_24h', 0.0))

        Tnorm = max_t / 24.0
        Cost_pen = max(0.0, (total_cost - BUDGET_MAX) / (BUDGET_MAX + 1e-9))

        fitness = ALPHA * Rcov - BETA * Tnorm - GAMMA * Cost_pen
        return fitness

    # --------------------------------------------------
    # REPAIR: BẮT BUỘC PHỦ HẾT J
    # --------------------------------------------------
    def repair(self, sol):
        covered = self._covered_J_sites(sol)
        missing = list(self.J_all - covered)

        # Vá coverage bằng COW rẻ nhất
        for j in missing:
            cand = self.data['cow_travel_df']
            cand = cand[cand['site_id'] == j]
            if cand.empty:
                continue
            best = cand.sort_values('travel_cost_vnd').iloc[0]
            sol.setdefault('cows', {})[best['cow_id']] = j

        return sol

    # --------------------------------------------------
    # Enforce unique power usage
    # --------------------------------------------------
    def _enforce_power_unique(self, sol):
        used = {}
        for bts_id, p in list(sol.get('powers', {}).items()):
            if p is None:
                continue
            if p not in used:
                used[p] = bts_id
            else:
                # keep the one with smaller travel time
                t1 = self.power_travel.get((p, bts_id), (np.inf, np.inf))[1]
                t2 = self.power_travel.get((p, used[p]), (np.inf, np.inf))[1]
                if t1 < t2:
                    sol['powers'].pop(used[p], None)
                    used[p] = bts_id
                else:
                    sol['powers'].pop(bts_id, None)

    # --------------------------------------------------
    # INITIALIZATION – ALWAYS FULL COVERAGE
    # --------------------------------------------------
    def initialize(self):
        print("Initializing GA-PSO population (FULL COVERAGE)...")

        # 1) Seed solutions (already full coverage)
        for s in self.seeds:
            sol = {
                'cows': dict(s.get('cows', {})),
                'powers': dict(s.get('powers', {}))
            }
            sol = self.repair(sol)
            self._enforce_power_unique(sol)
            self.population.append(sol)

        # 2) Randomized full-coverage solutions
        while len(self.population) < self.pop_size:
            sol = {'cows': {}, 'powers': {}}

            # ensure each J is covered
            for j in self.J_all:
                cand = self.data['cow_travel_df']
                cand = cand[cand['site_id'] == j]
                if cand.empty:
                    continue
                if random.random() < 0.7:
                    best = cand.sort_values('travel_cost_vnd').iloc[0]
                else:
                    best = cand.sample(1).iloc[0]
                sol['cows'][best['cow_id']] = j

            # power assignment
            outage = self.data['bts_df']
            outage = outage[outage.get('status', '') == 'power_outage']
            used = set()

            for _, r in outage.iterrows():
                if random.random() < 0.7:
                    cand = self.data['power_travel_df']
                    cand = cand[cand['bts_id'] == r['site_id']]
                    if cand.empty:
                        continue
                    for _, c in cand.sort_values('total_cost_vnd').iterrows():
                        if c['power_id'] not in used:
                            sol['powers'][r['site_id']] = c['power_id']
                            used.add(c['power_id'])
                            break

            sol = self.repair(sol)
            self._enforce_power_unique(sol)
            self.population.append(sol)

        # Evaluate
        self.fitnesses = [self.evaluate(sol) for sol in self.population]
        self.pbest = [dict(sol) for sol in self.population]
        self.pbest_f = list(self.fitnesses)

        idx = int(np.argmax(self.fitnesses))
        self.gbest = self.population[idx]
        self.gbest_f = self.fitnesses[idx]

        print("Initial best fitness:", self.gbest_f)

    # --------------------------------------------------
    # Crossover (coverage-safe)
    # --------------------------------------------------
    def crossover(self, a, b):
        child = {'cows': {}, 'powers': {}}

        for j in self.J_all:
            if random.random() < 0.5:
                for cow, site in a.get('cows', {}).items():
                    if site == j:
                        child['cows'][cow] = j
                        break
            else:
                for cow, site in b.get('cows', {}).items():
                    if site == j:
                        child['cows'][cow] = j
                        break

        for bts in set(a.get('powers', {})) | set(b.get('powers', {})):
            if random.random() < 0.5 and bts in a.get('powers', {}):
                child['powers'][bts] = a['powers'][bts]
            elif bts in b.get('powers', {}):
                child['powers'][bts] = b['powers'][bts]

        child = self.repair(child)
        self._enforce_power_unique(child)
        return child

    # --------------------------------------------------
    # Mutation (coverage-safe)
    # --------------------------------------------------
    def mutate(self, sol, p_mut=0.05):
        if random.random() < p_mut:
            j = random.choice(list(self.J_all))
            cand = self.data['cow_travel_df']
            cand = cand[cand['site_id'] == j]
            if not cand.empty:
                pick = cand.sample(1).iloc[0]
                sol['cows'][pick['cow_id']] = j

        outage = self.data['bts_df']
        outage = outage[outage.get('status', '') == 'power_outage']

        if random.random() < p_mut and not outage.empty:
            r = outage.sample(1).iloc[0]
            cand = self.data['power_travel_df']
            cand = cand[cand['bts_id'] == r['site_id']]
            if not cand.empty:
                sol['powers'][r['site_id']] = cand.sample(1).iloc[0]['power_id']

        sol = self.repair(sol)
        self._enforce_power_unique(sol)
        return sol

    # --------------------------------------------------
    # PSO update (coverage-safe)
    # --------------------------------------------------
    def pso_update(self, sol, pbest, gbest, w_p=0.3, w_g=0.3):
        child = {'cows': {}, 'powers': {}}

        for j in self.J_all:
            r = random.random()
            if r < w_p:
                for cow, site in pbest.get('cows', {}).items():
                    if site == j:
                        child['cows'][cow] = j
                        break
            elif r < w_p + w_g:
                for cow, site in gbest.get('cows', {}).items():
                    if site == j:
                        child['cows'][cow] = j
                        break
            else:
                for cow, site in sol.get('cows', {}).items():
                    if site == j:
                        child['cows'][cow] = j
                        break

        for bts in set(sol.get('powers', {})) | set(pbest.get('powers', {})) | set(gbest.get('powers', {})):
            r = random.random()
            if r < w_p and bts in pbest.get('powers', {}):
                child['powers'][bts] = pbest['powers'][bts]
            elif r < w_p + w_g and bts in gbest.get('powers', {}):
                child['powers'][bts] = gbest['powers'][bts]
            elif bts in sol.get('powers', {}):
                child['powers'][bts] = sol['powers'][bts]

        child = self.repair(child)
        self._enforce_power_unique(child)
        return child

    # --------------------------------------------------
    # RUN LOOP
    # --------------------------------------------------
    def run(self):
        self.initialize()
        stagn = 0

        for it in range(self.max_iter):
            ranked = sorted(zip(self.population, self.fitnesses),
                            key=lambda x: x[1], reverse=True)

            new_pop = [x[0] for x in ranked[:self.elite_n]]
            new_fit = [x[1] for x in ranked[:self.elite_n]]

            while len(new_pop) < self.pop_size:
                idx = random.randrange(len(self.population))
                parent = self.population[idx]

                child = self.pso_update(parent, self.pbest[idx], self.gbest)
                if random.random() < 0.7:
                    mate = random.choice(self.population)
                    child = self.crossover(child, mate)

                child = self.mutate(child)
                f = self.evaluate(child)

                new_pop.append(child)
                new_fit.append(f)

            self.population = new_pop
            self.fitnesses = new_fit

            for i, f in enumerate(new_fit):
                if f > self.pbest_f[i]:
                    self.pbest[i] = self.population[i]
                    self.pbest_f[i] = f
                if f > self.gbest_f:
                    self.gbest = self.population[i]
                    self.gbest_f = f
                    stagn = 0

            stagn += 1
            if it % 10 == 0:
                print(f"Iter {it}: best fitness = {self.gbest_f:.6f}")

            if stagn > 50:
                print("Stopping due to stagnation.")
                break

        return self.gbest, self.gbest_f

# =====================================================
# MILP LOCAL REFINEMENT – HARD COVERAGE
# =====================================================

def milp_local_refinement(data, sol, cover_cow, cover_bts,
                          time_limit_sec=60, top_k_neighbors=3):
    try:
        J_df = data['J_df']
        bts_df = data['bts_df']
        cow_travel = data['cow_travel_df']
        power_travel = data['power_travel_df']

        J_all = set(J_df[J_df.get('in_water', False) == False]['site_id'])

        # ---------------------------
        # Candidate neighborhoods
        # ---------------------------
        cow_cand = defaultdict(list)
        for cow_id in cow_travel['cow_id'].unique():
            rows = cow_travel[cow_travel['cow_id'] == cow_id]
            rows = rows.sort_values('travel_cost_vnd')
            cow_cand[cow_id] = rows.head(top_k_neighbors)['site_id'].tolist()

        power_cand = defaultdict(list)
        outage = bts_df[bts_df.get('status', '') == 'power_outage']
        for _, r in outage.iterrows():
            rows = power_travel[power_travel['bts_id'] == r['site_id']]
            rows = rows.sort_values('total_cost_vnd')
            power_cand[r['site_id']] = rows.head(top_k_neighbors)['power_id'].tolist()

        # ---------------------------
        # Build MILP
        # ---------------------------
        model = pulp.LpProblem("local_refine_full_coverage", pulp.LpMaximize)

        x = {(c, j): pulp.LpVariable(f"x_{c}_{j}", cat="Binary")
             for c, js in cow_cand.items() for j in js}

        z = {(p, b): pulp.LpVariable(f"z_{p}_{b}", cat="Binary")
             for b, ps in power_cand.items() for p in ps}

        # ---------------------------
        # Constraints
        # ---------------------------
        # Each cow at most one J
        for c in cow_cand:
            model += pulp.lpSum(x[(c, j)] for j in cow_cand[c]) <= 1

        # Each BTS at most one power
        for b in power_cand:
            model += pulp.lpSum(z[(p, b)] for p in power_cand[b]) <= 1

        # Power uniqueness
        for p in set(p for p, _ in z):
            model += pulp.lpSum(z[(p, b)] for (pp, b) in z if pp == p) <= 1

        # 🚨 HARD COVERAGE CONSTRAINT
        for j in J_all:
            model += (
                pulp.lpSum(x[(c, j)]
                           for (c, jj) in x if jj == j) +
                pulp.lpSum(z[(p, b)]
                           for (p, b) in z if cover_bts.get(b, {}).get(j, 0) == 1)
                >= 1
            )

        # ---------------------------
        # Objective: Step 2 + Step 3
        # ---------------------------
        Tmax = pulp.LpVariable("Tmax", lowBound=0)
        cost_terms = []

        for (c, j), var in x.items():
            r = cow_travel[(cow_travel['cow_id'] == c) &
                           (cow_travel['site_id'] == j)].iloc[0]
            model += Tmax >= (r['travel_time_hr'] + 0.5) * var
            cost_terms.append((r['travel_cost_vnd']) * var)

        for (p, b), var in z.items():
            r = power_travel[(power_travel['power_id'] == p) &
                             (power_travel['bts_id'] == b)].iloc[0]
            model += Tmax >= r['total_time_hr'] * var
            cost_terms.append(r['total_cost_vnd'] * var)

        model += -BETA * (Tmax / 24.0) - GAMMA * pulp.lpSum(cost_terms)

        model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=False))

        if pulp.LpStatus[model.status] not in ["Optimal", "Feasible"]:
            return sol

        refined = {'cows': {}, 'powers': {}}
        for (c, j), var in x.items():
            if var.value() and var.value() > 0.5:
                refined['cows'][c] = j

        for (p, b), var in z.items():
            if var.value() and var.value() > 0.5:
                refined['powers'][b] = p

        return refined

    except Exception as e:
        print("MILP refinement error:", e)
        return sol

def _build_travel_maps(data):
    cow_travel_map = {}
    for _, r in data['cow_travel_df'].iterrows():
        cow_travel_map[(r.get('cow_id'), r.get('site_id'))] = {
            'distance_km': float(r.get('distance_km', np.nan)),
            'travel_time_hr': float(r.get('travel_time_hr', np.nan)),
            'travel_cost_vnd': float(r.get('travel_cost_vnd', np.nan))
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
            covered += float(row.iloc[0].get('pop', 0.0))
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


def export_solution_files(data, sol, output_dir=None, prefix='solution'):
    if output_dir is None:
        output_dir = os.path.join('BTS_Restoration_Project', 'outputs', 'results_hybrid')
    os.makedirs(output_dir, exist_ok=True)

    cow_travel_map, power_travel_map = _build_travel_maps(data)

    cow_rows = []
    for cow_id, site_id in sorted(sol.get('cows', {}).items(), key=lambda x: x[0] if isinstance(x[0], (int, str)) else str(x[0])):
        rec = {'cow_id': cow_id, 'site_id': site_id if site_id is not None else ''}
        info = cow_travel_map.get((cow_id, site_id), {})
        rec.update({'distance_km': info.get('distance_km', ''), 'travel_time_hr': info.get('travel_time_hr', ''), 'travel_cost_vnd': info.get('travel_cost_vnd', '')})
        cow_rows.append(rec)

    df_cow = pd.DataFrame(cow_rows, columns=['cow_id', 'site_id', 'distance_km', 'travel_time_hr', 'travel_cost_vnd'])
    cow_csv = os.path.join(output_dir, f'{prefix}_cow_assignments_new.csv')
    df_cow.to_csv(cow_csv, index=False)
    print('Wrote', cow_csv)

    power_rows = []
    for bts_id, p_id in sorted(sol.get('powers', {}).items(),
                               key=lambda x: x[0] if isinstance(x[0], (int, str)) else str(x[0])):

        info = power_travel_map.get((p_id, bts_id), {})

        # operating cost
        prow = data['power_df'][data['power_df']['power_id'] == p_id]
        if not prow.empty:
            cost_operating = float(prow.iloc[0].get('cost_vnd_24h', 0.0))
        else:
            cost_operating = 0.0

        cost_deploy = info.get('total_cost_vnd', 0.0)

        rec = {
            'bts_id': bts_id,
            'power_id': p_id if p_id is not None else '',
            'distance_km': info.get('distance_km', ''),
            'total_time_hr': info.get('total_time_hr', ''),
            'total_cost_vnd': cost_deploy,
            'operating_cost_vnd_24h': cost_operating,
            'total_cost_all_vnd': cost_deploy + cost_operating,
            'note': info.get('note', '')
        }

        power_rows.append(rec)

    df_power = pd.DataFrame(power_rows, columns=['bts_id', 'power_id', 'distance_km', 'total_time_hr', 'total_cost_vnd', 'note'])
    power_csv = os.path.join(output_dir, f'{prefix}_power_assignments_new.csv')
    df_power.to_csv(power_csv, index=False)
    print('Wrote', power_csv)

    metrics = compute_solution_metrics(data, sol)
    summary_path = os.path.join(output_dir, f'{prefix}_summary_new.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Wrote', summary_path)

    return {'cow_csv': cow_csv, 'power_csv': power_csv, 'summary_json': summary_path, 'metrics': metrics}


# =====================================================
# RUN HYBRID – KEEP OUTPUT STRUCTURE
# =====================================================

def run_hybrid_cov(max_iter=300, top_k=5, export_outputs=True):
    data = load_all()

    cover_cow, cover_bts = build_cover_matrices(
        data['bts_df'], data['J_df'], data['cow_df']
    )

    reduced_J, feasible_cows, feasible_powers, seeds = milp_presolve(data)

    ga = HybridGAPSO(
        data, cover_cow, cover_bts,
        reduced_J, feasible_cows, feasible_powers, seeds,
        pop_size=80, max_iter=max_iter, elite_n=5
    )

    best, best_f = ga.run()
    print("GA–PSO best fitness:", best_f)

    ranked = sorted(zip(ga.population, ga.fitnesses),
                    key=lambda x: x[1], reverse=True)

    refined_best = None
    refined_f = -1e18

    for sol, _ in ranked[:top_k]:
        cand = milp_local_refinement(data, sol, cover_cow, cover_bts)
        f = ga.evaluate(cand)
        if f > refined_f:
            refined_best = cand
            refined_f = f

    print("Refined best fitness:", refined_f)

    final_sol = refined_best if refined_best else best

    if export_outputs:
        export_solution_files(
            data,
            final_sol,
            output_dir=os.path.join(
                "BTS_Restoration_Project", "outputs", "results_hybrid"
            )
        )

    return {
        "ga_best": best,
        "ga_best_f": best_f,
        "refined_best": refined_best,
        "refined_best_f": refined_f
    }


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_hybrid(max_iter=300, top_k=5, export_outputs=True)
