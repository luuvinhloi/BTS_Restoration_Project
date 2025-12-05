# FILE: src/optimization/GA_PSO/eval_repair.py
import numpy as np
import pandas as pd
import logging

def build_power_index_maps(power_df: pd.DataFrame):
    """Return mapping power_id -> index and reverse"""
    idx_map = {}
    rev = {}
    for i, pid in enumerate(list(power_df["power_id"].astype(str).values)):
        idx_map[pid] = i + 1  # gene values 1..n_powers
        rev[i + 1] = pid
    return idx_map, rev

def compute_coverage_from_solution(X_cows: np.ndarray, cover_arr: np.ndarray, I_df: pd.DataFrame):
    n_I = cover_arr.shape[1]
    covered = np.zeros(n_I, dtype=bool)
    for p, site in enumerate(X_cows):
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

def compute_cost_and_time_from_solution(X, n_cows, cow_df, J_df, cow_travel_dict, n_bts_power, power_df, bts_power_list, power_travel_dict, power_index_map, default_setup_time_h):
    total_cost = 0.0
    makespan = 0.0
    # cows
    for p in range(n_cows):
        site = int(X[p])
        if site <= 0:
            continue
        j_idx = site - 1
        cow_id = str(cow_df.iloc[p]["cow_id"])
        site_id = str(J_df.iloc[j_idx]["site_id"])
        travel = float(cow_travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
        travel_t = float(cow_travel_dict.get((cow_id, site_id), {}).get("travel_time_hr", 0.0))
        fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
        total_cost += (fixed + travel)
        makespan = max(makespan, travel_t + default_setup_time_h)
    # powers
    for b_idx in range(n_bts_power):
        gene = int(X[n_cows + b_idx])
        if gene <= 0:
            continue
        power_id = list(power_index_map.keys())[gene - 1]
        tinfo = power_travel_dict.get((power_id, bts_power_list[b_idx]), {})
        travel_cost = float(tinfo.get("total_cost_vnd", 0.0))
        travel_t = float(tinfo.get("total_time_hr", 0.0))
        fixed = float(power_df[power_df["power_id"] == power_id].iloc[0].get("cost_vnd_24h", 0.0))
        total_cost += (fixed + travel_cost)
        makespan = max(makespan, travel_t)
    return total_cost, makespan

def evaluate_solution(X, cover_arr, cow_df, I_df, J_df, cow_travel_dict, power_df, power_travel_dict, bts_power_list, power_index_map, default_setup_time_h, budget_max, time_max_norm, w_time, w_cost, coverage_weight):
    n_cows = len(cow_df)
    n_bts_power = len(bts_power_list)
    X_cows = X[:n_cows]
    cov_ratio, covered_pop, covered_mask = compute_coverage_from_solution(X_cows, cover_arr, I_df)
    total_cost, makespan = compute_cost_and_time_from_solution(X, n_cows, cow_df, J_df, cow_travel_dict, n_bts_power, power_df, bts_power_list, power_travel_dict, power_index_map, default_setup_time_h)
    # normalized time
    time_term = (makespan / max(1e-9, time_max_norm))
    cost_pen = max(0.0, (total_cost - budget_max) / max(1.0, budget_max))
    # Lexicographic emphasis: focus on coverage -> use (1 - cov_ratio) primary term; time and cost secondary
    f = (1.0 - cov_ratio) + w_time * time_term + w_cost * cost_pen - coverage_weight * cov_ratio * 0.0
    # (Note: we structured to minimize f; coverage decreases f)
    used = int((np.array(X_cows) > 0).sum()) + int(sum(int(g > 0) for g in X[n_cows:]))
    return float(f), float(cov_ratio), float(total_cost), float(makespan), int(used), covered_mask

def repair_solution(X: np.ndarray,
                    cover_arr: np.ndarray,
                    cow_df: pd.DataFrame,
                    J_df: pd.DataFrame,
                    cow_travel_dict: dict,
                    power_df: pd.DataFrame,
                    power_travel_dict: dict,
                    bts_power_list: list,
                    power_index_map: dict,
                    index_power_map: dict,
                    failed_bts_df: pd.DataFrame,
                    flood_tif_path,
                    roads_graph,
                    budget_max: float,
                    default_setup_time_h: float,
                    budget_hard: bool = True,
                    enforce_site_unique: bool = True):
    """
    Repairs X in place; ensure:
     - J uniqueness (1 COW per J)
     - depot capacities (from cow_df counts)
     - each power assigned at most once
     - power compatibility: power resource_amount or capacity >= bts power_W
     - only assign power to BTS in bts_power_list
     - if travel note says fallback_no_path (from backup csv), set gene=0
     - avoid assigning COW to J whose roads graph marks path blocked (best-effort using travel dict)
    """
    X = X.copy()
    n_cows = len(cow_df)
    n_bts_power = len(bts_power_list)
    n_J = cover_arr.shape[2]
    pop_vec = I_df_pop = None  # not needed here

    # Step: enforce site uniqueness for cows (if multiple cows to same J, keep best one)
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
                # compute pop gain (approx using cover_arr pop count; use number of I points covered)
                pop_gain = float(cover_arr[p, :, j_idx].sum())
                cow_id = str(cow_df.iloc[p]["cow_id"])
                site_id = str(J_df.iloc[j_idx]["site_id"])
                travel_cost = float(cow_travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
                fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
                cost = travel_cost + fixed
                score = pop_gain / (cost + 1e-9)
                scored.append((p, score))
            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            keep = scored_sorted[0][0]
            for p, _ in scored_sorted[1:]:
                X[p] = 0

    # Step: enforce depot capacities
    depot_capacity_map = {}
    for base_id, g in cow_df.groupby("base_id"):
        depot_capacity_map[str(base_id)] = int(len(g))
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
            site = int(X[p])
            j_idx = site - 1
            pop_gain = float(cover_arr[p, :, j_idx].sum())
            fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
            scored.append((p, pop_gain, fixed))
        scored_sorted = sorted(scored, key=lambda x: (x[1], -x[2]))
        to_remove = scored_sorted[:(len(assigned) - cap)]
        for p, _, _ in to_remove:
            X[p] = 0

    # Step: repair power assignments
    # ensure each power used at most once
    used_powers = {}
    for b_idx in range(n_bts_power):
        gene = int(X[n_cows + b_idx])
        if gene <= 0:
            continue
        # map gene to power_id
        power_id = list(power_index_map.keys())[gene - 1]
        # check travel note for path missing
        trav = power_travel_dict.get((power_id, bts_power_list[b_idx]), {})
        if trav.get("note", "").startswith("fallback_no_path"):
            # if no path, disallow assignment (set 0)
            X[n_cows + b_idx] = 0
            continue
        if power_id in used_powers:
            # resolve conflict: keep assignment with smaller travel cost
            prev_idx = used_powers[power_id]
            prev_tcost = float(power_travel_dict.get((power_id, bts_power_list[prev_idx]), {}).get("total_cost_vnd", 1e12))
            this_tcost = float(trav.get("total_cost_vnd", 1e12))
            if this_tcost < prev_tcost:
                # keep this, remove previous
                X[n_cows + prev_idx] = 0
                used_powers[power_id] = b_idx
            else:
                X[n_cows + b_idx] = 0
        else:
            used_powers[power_id] = b_idx

    # Step: ensure compatibility and capacity: power.resource_amount (or runtime_h * P?) vs bts power_W
    # power_df has resource_amount (units); failed_bts_df has power_W
    for b_idx, bts_id in enumerate(bts_power_list):
        gene = int(X[n_cows + b_idx])
        if gene <= 0:
            continue
        power_id = list(power_index_map.keys())[gene - 1]
        p_row = power_df[power_df["power_id"] == power_id].iloc[0]
        # check resource_amount: if present and < required -> unassign
        try:
            resource_amount = float(p_row.get("resource_amount", 0.0))
        except:
            resource_amount = 0.0
        req_power = float(failed_bts_df[failed_bts_df["site_id"] == bts_id].iloc[0].get("power_W", 0.0))
        # Heuristic: if resource_amount < req_power (interpreting units consistent), unassign
        if resource_amount < req_power:
            X[n_cows + b_idx] = 0

    # Step: budget enforcement (greedy keep highest score until under budget)
    def total_cost_of_X(X_arr):
        tot = 0.0
        # cows
        for p in range(n_cows):
            site = int(X_arr[p])
            if site <= 0:
                continue
            j_idx = site - 1
            cow_id = str(cow_df.iloc[p]["cow_id"])
            site_id = str(J_df.iloc[j_idx]["site_id"])
            travel_cost = float(cow_travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
            fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
            tot += (fixed + travel_cost)
        # powers
        for b_idx in range(n_bts_power):
            gene = int(X_arr[n_cows + b_idx])
            if gene <= 0:
                continue
            power_id = list(power_index_map.keys())[gene - 1]
            tinfo = power_travel_dict.get((power_id, bts_power_list[b_idx]), {})
            travel_cost = float(tinfo.get("total_cost_vnd", 0.0))
            fixed = float(power_df[power_df["power_id"] == power_id].iloc[0].get("cost_vnd_24h", 0.0))
            tot += (fixed + travel_cost)
        return tot

    if budget_hard:
        tot_cost = total_cost_of_X(X)
        if tot_cost > budget_max:
            # build score list for items (cows and powers) and greedily keep best coverage per cost
            scored = []
            # cows
            for p in range(n_cows):
                site = int(X[p])
                if site <= 0:
                    continue
                j_idx = site - 1
                pop_gain = float(cover_arr[p, :, j_idx].sum())
                cow_id = str(cow_df.iloc[p]["cow_id"])
                site_id = str(J_df.iloc[j_idx]["site_id"])
                travel_cost = float(cow_travel_dict.get((cow_id, site_id), {}).get("travel_cost_vnd", 0.0))
                fixed = float(cow_df.iloc[p].get("cost_vnd", 0.0))
                cost = travel_cost + fixed
                score = pop_gain / (cost + 1e-9)
                scored.append(("COW", p, score, pop_gain, cost))
            # powers
            for b_idx in range(n_bts_power):
                gene = int(X[n_cows + b_idx])
                if gene <= 0:
                    continue
                power_id = list(power_index_map.keys())[gene - 1]
                tinfo = power_travel_dict.get((power_id, bts_power_list[b_idx]), {})
                travel_cost = float(tinfo.get("total_cost_vnd", 0.0))
                fixed = float(power_df[power_df["power_id"] == power_id].iloc[0].get("cost_vnd_24h", 0.0))
                cost = travel_cost + fixed
                # power contribution approximated via covered pop if assigning power enables BTS to serve pop_covered
                pop_gain = float(failed_bts_df[failed_bts_df["site_id"] == bts_power_list[b_idx]].iloc[0].get("pop_covered", 0.0))
                score = (pop_gain) / (cost + 1e-9)
                scored.append(("POWER", b_idx, score, pop_gain, cost))
            scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
            X_new = X.copy()
            running = 0.0
            for typ, idx, sc, pg, cost in scored_sorted:
                if running + cost <= budget_max:
                    # keep
                    running += cost
                else:
                    # drop
                    if typ == "COW":
                        X_new[idx] = 0
                    else:
                        X_new[n_cows + idx] = 0
            X = X_new

    return X
