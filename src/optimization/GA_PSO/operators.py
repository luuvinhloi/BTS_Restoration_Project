# FILE: src/optimization/GA_PSO/operators.py
import numpy as np
from typing import List
import logging

def mutation_discrete(X: np.ndarray, n_sites: int, mut_rate: float):
    X_new = X.copy()
    for i in range(len(X_new)):
        if np.random.rand() < mut_rate:
            X_new[i] = int(np.random.choice([0] + [s + 1 for s in range(n_sites)]))
    return X_new

def uniform_crossover_arrays(A: np.ndarray, B: np.ndarray):
    mask = np.random.rand(len(A)) < 0.5
    child = A.copy()
    child[~mask] = B[~mask]
    return child

def initialize_population(pop_size: int, n_cows: int, n_sites: int, cover_arr, cow_df, J_df):
    """
    50% coverage-first, 20% greedy marginal, rest random.
    cover_arr: (n_cows, n_I, n_J)
    """
    population = []
    n_cover_score = (cover_arr.sum(axis=0)).sum(axis=0) if hasattr(cover_arr, "shape") else None
    if n_cover_score is None:
        site_rank = list(range(n_sites))
    else:
        site_rank = list(np.argsort(n_cover_score)[::-1])

    # coverage-first
    for _ in range(int(pop_size * 0.5)):
        X = np.zeros(n_cows, dtype=int)
        for p in range(n_cows):
            assigned = 0
            for j in site_rank:
                if cover_arr[p, :, j].any():
                    assigned = int(j + 1)
                    break
            if assigned == 0:
                assigned = int(np.random.randint(0, n_sites)) + 1
            X[p] = assigned
        population.append(X)

    # greedy marginal
    for _ in range(int(pop_size * 0.2)):
        X = np.zeros(n_cows, dtype=int)
        covered_mask = np.zeros(cover_arr.shape[1], dtype=bool)
        for p in range(n_cows):
            best_j = 0
            best_gain = -1.0
            for j in site_rank:
                cov_idxs = np.where(cover_arr[p, :, j])[0]
                if cov_idxs.size == 0:
                    continue
                marginal = np.sum(~covered_mask[cov_idxs])
                if marginal > best_gain:
                    best_gain = marginal
                    best_j = int(j + 1)
            X[p] = best_j if best_j > 0 else int(np.random.randint(0, n_sites)) + 1
            if X[p] > 0:
                covered_mask = covered_mask | cover_arr[p, :, X[p] - 1]
        population.append(X)

    # random
    for _ in range(pop_size - len(population)):
        X = np.zeros(n_cows, dtype=int)
        for p in range(n_cows):
            choice = np.random.choice([0] + [i + 1 for i in range(n_sites)])
            X[p] = int(choice)
        population.append(X)
    return population
