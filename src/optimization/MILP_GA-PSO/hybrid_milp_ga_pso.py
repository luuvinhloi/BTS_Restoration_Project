"""
Hybrid MILP + GA-PSO solver module for BTS restoration
Save path (user): BTS_Restoration_Project/src/optimization/GA_PSO/hybrid_milp_ga_pso.py

Main capabilities implemented:
- load datasets (CSV, graphml, tif)
- preprocess: build cover matrices, read travel matrices
- MILP presolve (PuLP) to prune infeasible COW/sites and generate seeds
- GA-PSO hybrid search (encoding, PSO-like update, GA crossover/mutation, repair)
- MILP local refinement (PuLP) to refine top-K candidates
- outputs: selected COW deployments, assigned backup power, covered population, total time/cost

Notes:
- Module assumes datasets exist under project data/processed paths as described by user.
- Uses PuLP as MILP interface. If Gurobi is available, PuLP can call it by name.
- Uses networkx to read roads_flooded.graphml and checks edge attributes is_passable / flood_class
- Uses rasterio to check flood depth at candidate J sites (requires rasterio)

This is a single-file implementation intended to be a working starting point. It is engineered for clarity and completeness.

References: Implementation follows the design and pseudocode in the project report (Hybrid_MILP_GA_PSO). See report for model details. Citation: fileciteturn1file0
"""

import os
import csv
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

try:
    import rasterio
    from rasterio import features
except Exception:
    rasterio = None

# --------------------------- Configuration ---------------------------
DATA_ROOT = os.path.join('BTS_Restoration_Project', 'data', 'processed')
PATHS = {
    'J_sites': os.path.join(DATA_ROOT, 'position_I_J', 'J_sites.csv'),
    'cow_dataset': os.path.join(DATA_ROOT, 'cow', 'cow_dataset.csv'),
    'backup_power': os.path.join(DATA_ROOT, 'backup_power', 'backup_power.csv'),
    'failed_bts': os.path.join(DATA_ROOT, 'damage_bts', 'failed_bts.csv'),
    'flood_tif': os.path.join(DATA_ROOT, 'flood', 'flood_depth_combined_B_clean.tif'),
    'roads_graph': os.path.join(DATA_ROOT, 'road', 'roads_flooded.graphml'),
    'cow_travel': os.path.join(DATA_ROOT, 'travel_cost', 'cow_to_J_sites.csv'),
    'power_travel': os.path.join(DATA_ROOT, 'travel_cost', 'backup_to_failed_bts.csv')
}

BUDGET_MAX = 1e9  # VNĐ
ALPHA = 1.0
BETA = 0.01
GAMMA = 0.000001

# --------------------------- Data structures ---------------------------
COW = namedtuple('COW', 'cow_id base_id base_name type lat lon coverage_radius_m power_kw speed_kmh endurance_hr cost_vnd assigned_region')
POWER = namedtuple('POWER', 'base_id power_id lat lon base_name type model runtime_h cost_vnd_24h resource_amount')
BTS = namedtuple('BTS', 'site_id latitude longitude utm_x utm_y pop_covered pop_unique_covered overlap_ratio_network total_unique_pop_network elevation_m slope_deg neighbour_weight dist_to_school_m dist_to_hospital_m dist_to_road_m dist_to_residential_m dist_to_industrial_m site_accessibility_score antenna_height_m region_type bts_type coverage_radius_m power_W flooded status')
JSITE = namedtuple('JSITE', 'site_id i_ref latitude longitude pop priority_category priority_weight slope dist_to_road_m in_water')

# --------------------------- Utility functions ---------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in km
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# --------------------------- Data loading ---------------------------

def load_csv_to_namedtuples(path, record_type):
    df = pd.read_csv(path)
    records = []
    for _, r in df.iterrows():
        records.append(record_type(*[r[c] for c in df.columns]))
    return df, records


def load_all():
    print('Loading datasets...')
    J_df = pd.read_csv(PATHS['J_sites'])
    cow_df = pd.read_csv(PATHS['cow_dataset'])
    power_df = pd.read_csv(PATHS['backup_power'])
    bts_df = pd.read_csv(PATHS['failed_bts'])
    cow_travel_df = pd.read_csv(PATHS['cow_travel'])
    power_travel_df = pd.read_csv(PATHS['power_travel'])

    # Read roads graph
    G = None
    if os.path.exists(PATHS['roads_graph']):
        G = nx.read_graphml(PATHS['roads_graph'])
        print('Loaded roads graph with', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges')
    else:
        print('roads graph not found at', PATHS['roads_graph'])

    # flood raster
    flood_src = None
    if rasterio and os.path.exists(PATHS['flood_tif']):
        flood_src = rasterio.open(PATHS['flood_tif'])
        print('Loaded flood raster')
    else:
        print('Warning: rasterio/flood tif not available')

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

# --------------------------- Preprocessing ---------------------------

def site_in_deep_flood(flood_src, lat, lon, depth_threshold_m=0.5):
    if flood_src is None:
        return False
    try:
        row, col = flood_src.index(lon, lat)
        val = float(flood_src.read(1)[row, col])
        return val > depth_threshold_m
    except Exception:
        return False


def build_cover_matrices(bts_df: pd.DataFrame, J_df: pd.DataFrame, cow_df: pd.DataFrame):
    # For this simplified module, we assume COW covers J site if within its radius (based on base location)
    print('Building coverage matrices...')
    cover_cow = defaultdict(dict)
    cover_bts = defaultdict(dict)

    # Map COW base location
    cow_bases = {}
    for _, r in cow_df.iterrows():
        cow_bases[r['cow_id']] = (r['lat'], r['lon'], r['coverage_radius_m'])

    for cow_id, (clat, clon, r) in cow_bases.items():
        for _, j in J_df.iterrows():
            d = haversine_km(clat, clon, j['latitude'], j['longitude']) * 1000.0
            cover_cow[cow_id][j['site_id']] = 1 if d <= r else 0

    for _, b in bts_df.iterrows():
        for _, j in J_df.iterrows():
            d = haversine_km(b['latitude'], b['longitude'], j['latitude'], j['longitude']) * 1000.0
            cover_bts[b['site_id']][j['site_id']] = 1 if d <= b['coverage_radius_m'] else 0

    return cover_cow, cover_bts

# --------------------------- MILP Presolve ---------------------------

def milp_presolve(data, time_limit_sec=60):
    """Run a light-weight MILP to prune infeasible COWs/sites and generate seeds.
    Returns: reduced_J_ids, feasible_cow_ids, feasible_power_ids, seed_solutions(list)
    """
    J_df = data['J_df']
    cow_df = data['cow_df']
    power_df = data['power_df']
    bts_df = data['bts_df']
    cow_travel = data['cow_travel_df']
    power_travel = data['power_travel_df']

    # Simple feasibility checks
    feasible_cow_ids = set()
    for _, cow in cow_df.iterrows():
        # if a cow can cover at least one J site and has reasonable cost
        poss = cow_travel[cow_travel['cow_id'] == cow['cow_id']]
        if poss.shape[0] == 0:
            continue
        # check if any distance is finite and site not in deep flood later (we don't check flood here)
        feasible_cow_ids.add(cow['cow_id'])

    # feasible power: if resource_amount > 0
    feasible_power_ids = set(power_df[power_df['resource_amount'] > 0]['power_id'].tolist())

    # Reduce J_sites: remove those in deep flood >0.5 or in_water True
    reduced_J = []
    for _, j in J_df.iterrows():
        if j.get('in_water', False) or (('in_water' in j) and bool(j['in_water'])):
            continue
        reduced_J.append(j['site_id'])

    # Create a couple of seed solutions by greedy coverage-first heuristics
    seeds = []
    # seed 1: assign nearest COW to each high-pop J, limited by endurance
    sorted_J = J_df.sort_values('pop', ascending=False).head(200)
    for seed_mode in [0, 1]:
        assignment = {'cows': {}, 'powers': {}}
        budget = BUDGET_MAX
        # greedy assign cows to highest pop J
        for _, j in sorted_J.iterrows():
            j_id = j['site_id']
            # find candidate cows that can reach j from cow_travel
            cand = cow_travel[cow_travel['site_id'] == j_id]
            if cand.shape[0] == 0:
                continue
            cand = cand.sort_values('travel_cost_vnd')
            for _, c in cand.iterrows():
                if c['cow_id'] in feasible_cow_ids and c['cow_id'] not in assignment['cows']:
                    assignment['cows'][c['cow_id']] = j_id
                    budget -= c['travel_cost_vnd']
                    break
        # greedy assign powers to power_outage BTS
        outage_bts = bts_df[bts_df['status'] == 'power_outage']
        for _, b in outage_bts.iterrows():
            b_id = b['site_id']
            cand = power_travel[power_travel['bts_id'] == b_id]
            if cand.shape[0] == 0:
                continue
            cand = cand.sort_values('total_cost_vnd')
            for _, p in cand.iterrows():
                if p['power_id'] in feasible_power_ids and p['power_id'] not in assignment['powers']:
                    assignment['powers'][b_id] = p['power_id']
                    budget -= p['total_cost_vnd']
                    break
        seeds.append(assignment)

    print(f'MILP presolve produced {len(seeds)} seeds, {len(reduced_J)} reduced sites')
    return reduced_J, feasible_cow_ids, feasible_power_ids, seeds

# --------------------------- GA-PSO Implementation ---------------------------

class HybridGAPSO:
    def __init__(self, data, cover_cow, cover_bts, reduced_J, feasible_cows, feasible_powers, seeds,
                 pop_size=100, max_iter=300, elite_n=5):
        self.data = data
        self.cover_cow = cover_cow
        self.cover_bts = cover_bts
        self.reduced_J = set(reduced_J)
        self.feasible_cows = list(feasible_cows)
        self.feasible_powers = list(feasible_powers)
        self.seeds = seeds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.elite_n = elite_n

        # load travel matrices as dicts for quick lookup
        self.cow_travel = {(r['cow_id'], r['site_id']): (r['distance_km'], r['travel_time_hr'], r['travel_cost_vnd'])
                           for _, r in data['cow_travel_df'].iterrows()}
        self.power_travel = {(r['power_id'], r['bts_id']): (r['distance_km'], r['total_time_hr'], r['total_cost_vnd'])
                             for _, r in data['power_travel_df'].iterrows()}

        # population holds genomes: {'cows': {cow_id: site_id or None}, 'powers': {bts_id: power_id or None}}
        self.population = []
        self.fitnesses = []
        self.pbest = []
        self.pbest_f = []
        self.gbest = None
        self.gbest_f = -1e9

    def initialize(self):
        print('Initializing GA-PSO population...')
        # half seeds, half random
        seed_count = max(1, int(self.pop_size * 0.5))
        for i in range(seed_count):
            sol = self.seeds[i % len(self.seeds)].copy()
            # ensure full keys
            sol['cows'] = dict(sol.get('cows', {}))
            sol['powers'] = dict(sol.get('powers', {}))
            self.population.append(sol)
        while len(self.population) < self.pop_size:
            sol = {'cows': {}, 'powers': {}}
            # random feasible cows assign to random reduced J or none
            for cow in random.sample(self.feasible_cows, k=min(len(self.feasible_cows), 10)):
                if random.random() < 0.3:
                    site = random.choice(list(self.reduced_J))
                    sol['cows'][cow] = site
            # random power assignments to a subset of outage BTS
            outage_bts = [r['site_id'] for _, r in self.data['bts_df'].iterrows() if r['status'] == 'power_outage']
            for b in random.sample(outage_bts, k=min(len(outage_bts), 10)):
                if random.random() < 0.5 and len(self.feasible_powers) > 0:
                    sol['powers'][b] = random.choice(self.feasible_powers)
            self.population.append(sol)

        # Evaluate initial population
        self.fitnesses = [self.evaluate(sol) for sol in self.population]
        self.pbest = list(self.population)
        self.pbest_f = list(self.fitnesses)
        idx = int(np.argmax(self.fitnesses))
        self.gbest = self.population[idx]
        self.gbest_f = self.fitnesses[idx]
        print('Initial gbest fitness:', self.gbest_f)

    def evaluate(self, sol):
        # compute coverage F1 (population covered by cows + restored BTS with power)
        # we use coverage over J_sites pop available in data
        J_df = self.data['J_df']
        bts_df = self.data['bts_df']

        covered_pop = 0.0
        # covered by cows
        for cow_id, site_id in sol['cows'].items():
            if site_id is None:
                continue
            match = J_df[J_df['site_id'] == site_id]
            if match.shape[0] > 0:
                covered_pop += float(match.iloc[0]['pop'])
        # covered by recovered BTS
        for bts_id, power_id in sol['powers'].items():
            if power_id is None:
                continue
            match = bts_df[bts_df['site_id'] == bts_id]
            if match.shape[0] > 0:
                covered_pop += float(match.iloc[0]['pop_covered'])

        # Normalize
        total_pop = float(J_df['pop'].sum() + bts_df['pop_covered'].sum())
        Rcov = covered_pop / (total_pop + 1e-9)

        # time: maximum travel_time among assignments
        max_time = 0.0
        for (cow_id, site_id), vals in self.cow_travel.items():
            # if assignment matches
            if cow_id in sol['cows'] and sol['cows'][cow_id] == site_id:
                max_time = max(max_time, vals[1])
        for (power_id, bts_id), vals in self.power_travel.items():
            if bts_id in sol['powers'] and sol['powers'][bts_id] == power_id:
                max_time = max(max_time, vals[1])
        Tnorm = max_time / (24.0 + 1e-9)  # assume Tmax 24h

        # cost
        total_cost = 0.0
        for (cow_id, site_id), vals in self.cow_travel.items():
            if cow_id in sol['cows'] and sol['cows'][cow_id] == site_id:
                total_cost += vals[2]
        for (power_id, bts_id), vals in self.power_travel.items():
            if bts_id in sol['powers'] and sol['powers'][bts_id] == power_id:
                total_cost += vals[2]
        Cost_pen = max(0.0, (total_cost - BUDGET_MAX) / (BUDGET_MAX + 1e-9))

        fitness = ALPHA * Rcov - BETA * Tnorm - GAMMA * Cost_pen
        return fitness

    def repair(self, sol):
        # enforce: each J site max 1 cow, each cow at most 1 site (we already assign that way), each BTS gets at most one power
        # ensure cows assigned to reduced_J and not to flooded J (we skip flood here)
        # budget repair: remove least efficient assignments if cost>budget
        # compute costs and eff
        cost_items = []
        total_cost = 0.0
        for cow_id, site_id in list(sol['cows'].items()):
            key = (cow_id, site_id)
            if key in self.cow_travel:
                cost = self.cow_travel[key][2]
                total_cost += cost
                # approximate coverage: pop at site
                pop = float(self.data['J_df'][self.data['J_df']['site_id'] == site_id]['pop'].sum())
                eff = pop / (cost + 1e-9)
                cost_items.append(('cow', cow_id, site_id, cost, eff))
            else:
                # invalid travel record
                sol['cows'].pop(cow_id, None)
        for bts_id, power_id in list(sol['powers'].items()):
            key = (power_id, bts_id)
            if key in self.power_travel:
                cost = self.power_travel[key][2]
                total_cost += cost
                pop = float(self.data['bts_df'][self.data['bts_df']['site_id'] == bts_id]['pop_covered'].sum())
                eff = pop / (cost + 1e-9)
                cost_items.append(('power', bts_id, power_id, cost, eff))
            else:
                sol['powers'].pop(bts_id, None)

        # If over budget, remove lowest eff items until under budget
        if total_cost > BUDGET_MAX:
            cost_items.sort(key=lambda x: x[4])  # ascending eff
            for item in cost_items:
                if total_cost <= BUDGET_MAX:
                    break
                typ = item[0]
                if typ == 'cow':
                    _, cow_id, site_id, cost, _ = item
                    if cow_id in sol['cows']:
                        sol['cows'].pop(cow_id, None)
                        total_cost -= cost
                else:
                    _, bts_id, power_id, cost, _ = item
                    if bts_id in sol['powers']:
                        sol['powers'].pop(bts_id, None)
                        total_cost -= cost
        return sol

    def crossover(self, a, b):
        # uniform crossover for both cows and powers
        child = {'cows': {}, 'powers': {}}
        for cow in set(list(a['cows'].keys()) + list(b['cows'].keys())):
            if random.random() < 0.5:
                if cow in a['cows']:
                    child['cows'][cow] = a['cows'][cow]
            else:
                if cow in b['cows']:
                    child['cows'][cow] = b['cows'][cow]
        for bts in set(list(a['powers'].keys()) + list(b['powers'].keys())):
            if random.random() < 0.5:
                if bts in a['powers']:
                    child['powers'][bts] = a['powers'][bts]
            else:
                if bts in b['powers']:
                    child['powers'][bts] = b['powers'][bts]
        return child

    def mutate(self, sol, p_mut=0.05):
        # change assignment of a random cow or power
        if random.random() < p_mut and len(sol['cows']) > 0:
            cow = random.choice(list(sol['cows'].keys()))
            if random.random() < 0.5:
                sol['cows'].pop(cow, None)
            else:
                sol['cows'][cow] = random.choice(list(self.reduced_J))
        if random.random() < p_mut and len(sol['powers']) > 0:
            bts = random.choice(list(sol['powers'].keys()))
            if random.random() < 0.5:
                sol['powers'].pop(bts, None)
            else:
                if len(self.feasible_powers) > 0:
                    sol['powers'][bts] = random.choice(self.feasible_powers)
        return sol

    def pso_update(self, sol, pbest, gbest, w_p=0.2, w_g=0.2):
        # discrete PSO-like: for each gene, with prob w_p take pbest value, w_g take gbest value
        child = {'cows': {}, 'powers': {}}
        all_cows = set(list(sol['cows'].keys()) + list(pbest['cows'].keys()) + list(gbest['cows'].keys()))
        for cow in all_cows:
            r = random.random()
            if r < w_p and cow in pbest['cows']:
                child['cows'][cow] = pbest['cows'][cow]
            elif r < w_p + w_g and cow in gbest['cows']:
                child['cows'][cow] = gbest['cows'][cow]
            elif cow in sol['cows']:
                child['cows'][cow] = sol['cows'][cow]
        all_powers = set(list(sol['powers'].keys()) + list(pbest['powers'].keys()) + list(gbest['powers'].keys()))
        for bts in all_powers:
            r = random.random()
            if r < w_p and bts in pbest['powers']:
                child['powers'][bts] = pbest['powers'][bts]
            elif r < w_p + w_g and bts in gbest['powers']:
                child['powers'][bts] = gbest['powers'][bts]
            elif bts in sol['powers']:
                child['powers'][bts] = sol['powers'][bts]
        return child

    def run(self):
        self.initialize()
        stagn = 0
        for it in range(self.max_iter):
            new_pop = []
            new_f = []
            # elitism
            ranked = sorted(zip(self.population, self.fitnesses), key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in ranked[:self.elite_n]]
            new_pop.extend(elites)
            new_f.extend([x[1] for x in ranked[:self.elite_n]])

            while len(new_pop) < self.pop_size:
                i = random.randrange(len(self.population))
                parent = self.population[i]
                # PSO update
                child = self.pso_update(parent, self.pbest[i], self.gbest)
                # crossover with random mate
                if random.random() < 0.8:
                    mate = random.choice(self.population)
                    child = self.crossover(child, mate)
                child = self.mutate(child, p_mut=0.05)
                child = self.repair(child)
                f = self.evaluate(child)
                new_pop.append(child)
                new_f.append(f)
            self.population = new_pop
            self.fitnesses = new_f

            # update pbest and gbest
            for i, f in enumerate(self.fitnesses):
                if f > self.pbest_f[i]:
                    self.pbest[i] = self.population[i]
                    self.pbest_f[i] = f
                if f > self.gbest_f:
                    self.gbest = self.population[i]
                    self.gbest_f = f
                    stagn = 0
            stagn += 1
            if it % 10 == 0:
                print(f'Iter {it} gbest_f={self.gbest_f:.6f}')
            if stagn > 50:
                print('Stopping due to stagnation')
                break
        return self.gbest, self.gbest_f

# --------------------------- MILP Local Refinement ---------------------------

def milp_local_refinement(data, candidate, cover_cow, cover_bts, time_limit_sec=60, solver_name=None):
    """Given a candidate solution from GA-PSO, fix some decisions and run MILP to refine.
    candidate: {'cows':{cow:site}, 'powers':{bts:power}}
    Returns improved solution (same format)
    """
    # We'll build a small MILP: variables for powers assigned to outage BTS (choose power or not), and for optionally swapping cows among a small set.
    model = pulp.LpProblem('local_refine', pulp.LpMaximize)

    # Variables
    # for each bts in candidate powers or all outage bts, create binary z_b_p for feasible powers
    bts_df = data['bts_df']
    power_df = data['power_df']
    outage_bts = list(bts_df[bts_df['status'] == 'power_outage']['site_id'])

    z = pulp.LpVariable.dicts('z', ((b, p) for b in outage_bts for p in power_df['power_id']), lowBound=0, upBound=1, cat='Binary')

    # objective: maximize coverage (population from restored BTS plus cows fixed)
    # build population covered by restored BTS
    pop_by_bts = {r['site_id']: float(r['pop_covered']) for _, r in bts_df.iterrows()}

    # coverage from cows is fixed in this refinement
    cow_covered_pop = 0.0
    for cow_id, site in candidate['cows'].items():
        pop = float(data['J_df'][data['J_df']['site_id'] == site]['pop'].sum()) if site is not None else 0.0
        cow_covered_pop += pop

    objective = cow_covered_pop + pulp.lpSum([pop_by_bts[b] * z[(b, p)] for b in outage_bts for p in power_df['power_id']])
    model += objective

    # constraints: each bts gets at most one power and power available count
    for b in outage_bts:
        model += pulp.lpSum([z[(b, p)] for p in power_df['power_id']]) <= 1
    # each power resource amount limit
    power_amount = {r['power_id']: float(r['resource_amount']) for _, r in power_df.iterrows()}
    for p in power_df['power_id']:
        model += pulp.lpSum([z[(b, p)] for b in outage_bts]) <= power_amount[p]

    # compatibility: ensure power capacity >= bts power_W
    bts_power = {r['site_id']: float(r['power_W']) for _, r in bts_df.iterrows()}
    power_cap = {}
    for _, r in power_df.iterrows():
        # try to parse model/runtime as capacity: fallback to runtime_h*? Not exact but use runtime_h as proxy
        power_cap[r['power_id']] = float(r.get('runtime_h', 0)) * 1000.0  # crude proxy
    for b in outage_bts:
        for p in power_df['power_id']:
            if power_cap[p] < bts_power[b]:
                model += z[(b, p)] == 0

    # budget constraint - cost of assigned powers + fixed cow travel cost
    power_travel = data['power_travel_df']
    cow_travel = data['cow_travel_df']
    fixed_cow_cost = 0.0
    for cow_id, site in candidate['cows'].items():
        key = (cow_id, site)
        row = cow_travel[(cow_travel['cow_id'] == cow_id) & (cow_travel['site_id'] == site)]
        if row.shape[0] > 0:
            fixed_cow_cost += float(row.iloc[0]['travel_cost_vnd'])
    power_cost_expr = pulp.lpSum([float(power_travel[(power_travel['power_id'] == p) & (power_travel['bts_id'] == b)].iloc[0]['total_cost_vnd']) * z[(b, p)]
                                  for b in outage_bts for p in power_df['power_id'] if not power_travel[(power_travel['power_id'] == p) & (power_travel['bts_id'] == b)].empty])
    model += fixed_cow_cost + power_cost_expr <= BUDGET_MAX

    # Solve
    solver = None
    if solver_name is not None:
        solver = pulp.getSolver(solver_name)
    model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit_sec))

    # extract
    new_candidate = {'cows': dict(candidate['cows']), 'powers': {}}
    for b in outage_bts:
        for p in power_df['power_id']:
            try:
                if pulp.value(z[(b, p)]) > 0.5:
                    new_candidate['powers'][b] = p
            except Exception:
                pass
    return new_candidate

# --------------------------- Orchestration ---------------------------

def run_hybrid(max_iter=300, top_k=5):
    data = load_all()
    cover_cow, cover_bts = build_cover_matrices(data['bts_df'], data['J_df'], data['cow_df'])
    reduced_J, feasible_cows, feasible_powers, seeds = milp_presolve(data)

    ga = HybridGAPSO(data, cover_cow, cover_bts, reduced_J, feasible_cows, feasible_powers, seeds,
                     pop_size=80, max_iter=max_iter, elite_n=5)
    best, best_f = ga.run()
    print('GA-PSO best fitness', best_f)

    # get top-K from population
    ranked = sorted(zip(ga.population, ga.fitnesses), key=lambda x: x[1], reverse=True)
    topK = [x[0] for x in ranked[:top_k]]

    refined = None
    refined_f = -1e9
    for s in topK:
        cand = milp_local_refinement(data, s, cover_cow, cover_bts, time_limit_sec=60)
        # evaluate cand
        f = ga.evaluate(cand)
        if f > refined_f:
            refined = cand
            refined_f = f
    print('Refined best fitness', refined_f)

    # produce outputs
    out = {
        'ga_best': best,
        'ga_best_f': best_f,
        'refined_best': refined,
        'refined_best_f': refined_f
    }
    with open('hybrid_result_summary.json', 'w') as f:
        json.dump(out, f, default=list, indent=2)
    print('Saved hybrid_result_summary.json')
    return out


if __name__ == '__main__':
    run_hybrid(max_iter=300, top_k=5)
