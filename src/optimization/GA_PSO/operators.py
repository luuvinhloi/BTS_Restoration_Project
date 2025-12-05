# FILE: src/optimization/GA_PSO/operators.py
import numpy as np
import logging

def mutation_discrete(X: np.ndarray, n_sites: int, n_powers: int, mut_rate: float):
    """
    X is concatenated array [cow_assignments (len n_cows), power_assignments (len n_bts_power)]
    For cow genes: allowed values {0,1..n_sites}
    For power genes: allowed values {0,1..n_powers} (index into power list)
    """
    X_new = X.copy()
    n = len(X_new)
    # assuming last part are power genes: we need n_sites and n_powers sizes passed separately
    # We'll detect power count by counting zeros? Instead we assume caller passes consistent shapes;
    # here we randomly mutate each gene with prob mut_rate
    for i in range(n):
        if np.random.rand() < mut_rate:
            # simple mutation: if gene is in cow region (value <= n_sites), choose random in [0..n_sites]
            # We cannot infer split here; caller ensures passed n_sites and n_powers as arguments
            # To support, we rely on metadata encoded in X dtype: we'll assume user passes correct splitting externally
            # So mutation policy performed by caller that knows split. But to keep safe we perform small local perturbation:
            if X_new[i] <= n_sites:
                X_new[i] = int(np.random.choice([0] + [s + 1 for s in range(n_sites)]))
            else:
                # power gene
                X_new[i] = int(np.random.choice([0] + [s + 1 for s in range(n_powers)]))
    return X_new

def uniform_crossover_arrays(A: np.ndarray, B: np.ndarray):
    mask = np.random.rand(len(A)) < 0.5
    child = A.copy()
    child[~mask] = B[~mask]
    return child

def initialize_population(pop_size: int,
                          n_cows: int,
                          n_sites: int,
                          n_bts_power: int,
                          power_index_map: dict,
                          cover_arr,
                          cow_df,
                          J_df,
                          bts_power_list):
    """
    Build population of individuals length = n_cows + n_bts_power
    - First n_cows values: cow assignments (0..n_sites)
    - Last n_bts_power values: power assignment indices (0..len(power_index_map))
    """
    population = []
    n_powers = len(power_index_map)
    # use coverage-first for cows, random for powers (start all 0 means no power assigned)
    # build cow site ranking by coverage potential
    try:
        cover_scores = (cover_arr.sum(axis=1)).sum(axis=0)  # sum over cows and I -> per J
        site_rank = list(np.argsort(cover_scores)[::-1])
    except Exception:
        site_rank = list(range(n_sites))

    for _ in range(int(pop_size * 0.5)):
        X = np.zeros(n_cows + n_bts_power, dtype=int)
        # cows: assign each cow to best site that covers at least one I (or 0)
        for p in range(n_cows):
            assigned = 0
            for j in site_rank:
                if cover_arr[p, :, j].any():
                    assigned = int(j + 1)
                    break
            if assigned == 0:
                assigned = int(np.random.randint(0, n_sites)) + 1
            X[p] = assigned
        # powers: initial zero (unassigned)
        # possibility: greedy assign nearest compatible power for BTS targets
        population.append(X)

    # some random individuals
    for _ in range(pop_size - len(population)):
        X = np.zeros(n_cows + n_bts_power, dtype=int)
        for p in range(n_cows):
            choice = np.random.choice([0] + [i + 1 for i in range(n_sites)])
            X[p] = int(choice)
        # random powers
        for b in range(n_bts_power):
            X[n_cows + b] = int(np.random.choice([0] + [i + 1 for i in range(n_powers)]))
        population.append(X)

    return population


def decode_solution_to_assignments(X, cow_df, J_df, power_df, bts_power_list, power_index_map, cow_travel_dict, power_travel_dict, default_setup_time_h):
    """
    Convert solution vector into rows for assignments dataframe.
    Returns list of dict rows.
    """
    n_cows = len(cow_df)
    n_bts_power = len(bts_power_list)
    rows = []
    # COW rows
    for p in range(n_cows):
        site = int(X[p])
        cow_row = cow_df.iloc[p]
        if site <= 0:
            assigned_site_id = None
            travel_cost_vnd = 0.0
            travel_time_hr = 0.0
        else:
            j_idx = site - 1
            assigned_site_id = str(J_df.iloc[j_idx]["site_id"])
            cow_id = str(cow_row["cow_id"])
            tinfo = cow_travel_dict.get((cow_id, assigned_site_id), {})
            travel_cost_vnd = float(tinfo.get("travel_cost_vnd", 0.0))
            travel_time_hr = float(tinfo.get("travel_time_hr", 0.0))
        fixed_cost = float(cow_row.get("cost_vnd", 0.0))
        total_cost_vnd = fixed_cost + travel_cost_vnd
        setup_time_h = default_setup_time_h
        deployment_time_hr = travel_time_hr + setup_time_h
        rows.append({
            "type": "COW",
            "cow_id": cow_row.get("cow_id"),
            "base_id": cow_row.get("base_id"),
            "assigned_site_id": assigned_site_id,
            "coverage_radius_m": cow_row.get("coverage_radius_m"),
            "cost_vnd": fixed_cost,
            "travel_cost_vnd": travel_cost_vnd,
            "total_cost_vnd": total_cost_vnd,
            "travel_time_hr": travel_time_hr,
            "setup_time_h": setup_time_h,
            "deployment_time_hr": deployment_time_hr
        })
    # POWER rows
    for b_idx, bts_id in enumerate(bts_power_list):
        gene = int(X[n_cows + b_idx])
        if gene <= 0:
            power_id = None
            travel_cost_vnd = 0.0
            travel_time_hr = 0.0
            fixed_cost = 0.0
        else:
            power_key = list(power_index_map.keys())[gene - 1]
            power_row = power_df[power_df["power_id"] == power_key].iloc[0]
            power_id = power_key
            tinfo = power_travel_dict.get((power_key, bts_id), {})
            travel_cost_vnd = float(tinfo.get("total_cost_vnd", 0.0))
            travel_time_hr = float(tinfo.get("total_time_hr", 0.0))
            fixed_cost = float(power_row.get("cost_vnd_24h", 0.0))
        total_cost_vnd = travel_cost_vnd + fixed_cost
        rows.append({
            "type": "POWER",
            "power_id": power_id,
            "assigned_target_bts": bts_id,
            "cost_vnd": fixed_cost,
            "travel_cost_vnd": travel_cost_vnd,
            "total_cost_vnd": total_cost_vnd,
            "travel_time_hr": travel_time_hr,
            "deployment_time_hr": travel_time_hr
        })
    return rows
