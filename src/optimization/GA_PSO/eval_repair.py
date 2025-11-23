# FILE: src/optimization/GA_PSO/eval_repair.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

def compute_coverage_from_solution(X: np.ndarray, cover_arr: np.ndarray, I_df: pd.DataFrame):
    """
    Use exact population weights on I_df['pop'].
    X: array length n_cows, values in {0,1..n_J}
    cover_arr: shape (n_cows, n_I, n_J)
    returns: coverage_ratio, covered_pop, covered_mask (bool array length n_I)
    """
    n_I = cover_arr.shape[1]
    covered = np.zeros(n_I, dtype=bool)
    for p, site in enumerate(X):
        site = int(site)
        if site <= 0:
            continue
        j_idx = site - 1
        if 0 <= j_idx < cover_arr.shape[2]:
            covered = covered | cover_arr[p, :, j_idx]
    pop_vec = I_df["pop"].values if "pop" in I_df.columns else np.ones(n_I)
    covered_pop = float((pop_vec * covered).sum())
    total_pop = float(pop_vec.sum()) if pop_vec.sum() > 0 else 0.0
    coverage_ratio = (covered_pop / total_pop) if total_pop > 0 else 0.0
    return coverage_ratio, covered_pop, covered

def compute_cost_and_time_from_solution(X: np.ndarray, cow_df: pd.DataFrame, J_df: pd.DataFrame, travel_dict: Dict[Tuple[str, str], dict], default_setup_time_h: float):
    total_cost = 0.0
    makespan = 0.0
    for p, site in enumerate(X):
        site = int(site)
        if site <= 0:
            continue
        j_idx = site - 1
        cow_id = str(cow_df.iloc[p]["cow_id"])
        site_id = str(J_df.iloc[j_idx]["site_id"])
        travel = float(travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
        travel_t = float(travel_dict.get((cow_id, site_id), {}).get("travel_time_hr", 0.0))
        fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
        total_cost += (fixed + travel)
        makespan = max(makespan, travel_t + default_setup_time_h)
    return total_cost, makespan

def evaluate_solution(X, cover_arr, cow_df, I_df, J_df, travel_dict, default_setup_time_h, budget_max, time_max_norm, w_time, w_cost):
    cov_ratio, covered_pop, covered_mask = compute_coverage_from_solution(X, cover_arr, I_df)
    total_cost, makespan = compute_cost_and_time_from_solution(X, cow_df, J_df, travel_dict, default_setup_time_h)
    # normalized time
    time_term = (makespan / max(1e-9, time_max_norm))
    cost_pen = max(0.0, (total_cost - budget_max) / max(1.0, budget_max))
    f = (1.0 - cov_ratio) + w_time * time_term + w_cost * cost_pen
    used = int((np.array(X) > 0).sum())
    return float(f), float(cov_ratio), float(total_cost), float(makespan), int(used), covered_mask

def repair_solution(X: np.ndarray,
                    cover_arr: np.ndarray,
                    cow_df: pd.DataFrame,
                    J_df: pd.DataFrame,
                    travel_dict: Dict[Tuple[str, str], dict],
                    depot_capacity_map: dict,
                    I_df: pd.DataFrame,
                    budget_max: float,
                    default_setup_time_h: float,
                    budget_hard: bool = True,
                    enforce_site_unique: bool = True):
    """
    Repairs X in place and returns new array.
    Steps:
      - site uniqueness (keep cow with largest marginal pop_gain / cost)
      - enforce depot capacities (pop-weighted)
      - enforce budget (if budget_hard): greedily keep highest score until cost <= budget_max
    Uses I_df['pop'] to compute pop_gain exactly.
    """
    X = X.copy()
    n_cows = len(X)
    n_I = cover_arr.shape[1]
    n_J = cover_arr.shape[2]
    pop_vec = I_df["pop"].values if "pop" in I_df.columns else np.ones(n_I)

    # Helper: compute marginal pop gain for cow p at its assigned j
    def marginal_pop_for(p, j_idx):
        if j_idx < 0 or j_idx >= n_J:
            return 0.0
        mask = cover_arr[p, :, j_idx]
        return float((pop_vec * mask).sum())

    # Step 1: enforce site uniqueness
    if enforce_site_unique:
        site_map = {}
        for p in range(n_cows):
            site = int(X[p])
            if site <= 0:
                continue
            site_map.setdefault(site - 1, []).append(p)
        for j_idx, cows_assigned in site_map.items():
            if len(cows_assigned) <= 1:
                continue
            scored = []
            for p in cows_assigned:
                pop_gain = marginal_pop_for(p, j_idx)
                cow_id = str(cow_df.iloc[p]["cow_id"])
                site_id = str(J_df.iloc[j_idx]["site_id"])
                travel_cost = float(travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
                fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
                cost = travel_cost + fixed
                score = pop_gain / (cost + 1e-9)
                scored.append((p, score))
            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            keep = scored_sorted[0][0]
            for p, _ in scored_sorted[1:]:
                X[p] = 0

    # Step 2: enforce depot capacities
    base_map = {}
    for p in range(n_cows):
        site = int(X[p])
        if site <= 0:
            continue
        base_id = str(cow_df.iloc[p].get("base_id", ""))
        base_map.setdefault(base_id, []).append(p)
    for base_id, assigned in base_map.items():
        cap = int(depot_capacity_map.get(str(base_id), len(assigned)))
        if len(assigned) <= cap:
            continue
        scored = []
        for p in assigned:
            # benefit = marginal pop
            site = int(X[p])
            j_idx = site - 1
            pop_gain = marginal_pop_for(p, j_idx)
            fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
            scored.append((p, pop_gain, fixed))
        # remove those with smallest pop_gain, tie-break by higher cost
        scored_sorted = sorted(scored, key=lambda x: (x[1], -x[2]))
        to_remove = scored_sorted[:(len(assigned) - cap)]
        for p, _, _ in to_remove:
            X[p] = 0

    # Step 3: enforce budget hard if required
    def total_cost_of_X(X_arr):
        tot = 0.0
        for p in range(n_cows):
            site = int(X_arr[p])
            if site <= 0:
                continue
            j_idx = site - 1
            cow_id = str(cow_df.iloc[p]["cow_id"])
            site_id = str(J_df.iloc[j_idx]["site_id"])
            travel_cost = float(travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
            fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
            tot += (fixed + travel_cost)
        return tot

    if budget_hard:
        tot_cost = total_cost_of_X(X)
        if tot_cost > budget_max:
            scored = []
            for p in range(n_cows):
                site = int(X[p])
                if site <= 0:
                    continue
                j_idx = site - 1
                pop_gain = marginal_pop_for(p, j_idx)
                cow_id = str(cow_df.iloc[p]["cow_id"])
                site_id = str(J_df.iloc[j_idx]["site_id"])
                travel_cost = float(travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
                fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
                cost = travel_cost + fixed
                score = pop_gain / (cost + 1e-9)
                scored.append((p, score, pop_gain, cost))
            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            X_new = np.zeros_like(X)
            running = 0.0
            for p, _, _, cost in scored_sorted:
                if running + cost <= budget_max:
                    X_new[p] = X[p]
                    running += cost
                else:
                    X_new[p] = 0
            X = X_new

    return X
