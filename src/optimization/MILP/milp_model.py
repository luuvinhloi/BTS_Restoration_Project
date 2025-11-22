# src/optimization/milp_model.py
"""
Enhanced MILP Model for COW Deployment Optimization (Fixed budget type issue)
    - Hỗ trợ đa loại COW (small / standard / heavy)
    - Hàm mục tiêu gồm (1 - R_cov) + w_time * (T/T_max) + w_dist * (TravelCost/TravelCost_max) + w_cost * penalty
    - Ép kiểu budget_max và cost_vnd sang float để tránh lỗi kiểu dữ liệu Pyomo
"""

import pyomo.environ as pyo
import numpy as np


def build_milp_model(I_points, J_sites, COWs, cover, travel_time, travel_cost, params):
    """
    Inputs:
      I_points : list[dict] - các điểm dân cư (pop)
      J_sites  : list[dict] - vị trí ứng viên
      COWs     : list[dict] - thông tin từng xe COW
      cover    : np.ndarray (N,M,K) - ma trận phủ sóng
      travel_time : np.ndarray (K,M) - thời gian di chuyển
      travel_cost : np.ndarray (K,M) - chi phí di chuyển
      params   : dict — chứa weights, M_max, budget_max,...
    """

    N, M, K = len(I_points), len(J_sites), len(COWs)
    model = pyo.ConcreteModel()

    # Tập chỉ mục
    model.I = pyo.RangeSet(0, N - 1)
    model.J = pyo.RangeSet(0, M - 1)
    model.K = pyo.RangeSet(0, K - 1)

    # Biến quyết định
    model.x = pyo.Var(model.K, model.J, within=pyo.Binary)
    model.y = pyo.Var(model.I, model.J, within=pyo.Binary)
    model.z = pyo.Var(model.I, within=pyo.Binary)
    model.penalty = pyo.Var(within=pyo.NonNegativeReals)

    # Trọng số
    w_time = float(params["weights"].get("w_time", 0.05))
    w_cost = float(params["weights"].get("w_cost", 0.1))
    w_dist = float(params["weights"].get("w_dist", 0.05))

    # Đảm bảo budget_max là số
    try:
        budget_max = float(params.get("budget_max", 5e8))
    except Exception:
        budget_max = 5e8

    # Ràng buộc phạt ngân sách
    def penalty_constraint(m):
        total_cost = sum(float(COWs[k].get("cost_vnd", 0.0)) * m.x[k, j] for k in m.K for j in m.J)
        return m.penalty >= (total_cost - budget_max) / budget_max

    model.c_penalty = pyo.Constraint(rule=penalty_constraint)

    # Hàm mục tiêu
    def objective_rule(m):
        total_pop = sum(float(I_points[i].get("pop", 0)) for i in m.I)
        covered_pop = sum(float(I_points[i].get("pop", 0)) * m.z[i] for i in m.I)
        R_cov = covered_pop / (total_pop + 1e-9)

        total_time = sum(float(travel_time[k, j]) * m.x[k, j] for k in m.K for j in m.J)
        total_travel_cost = sum(float(travel_cost[k, j]) * m.x[k, j] for k in m.K for j in m.J)

        T_max = max(1.0, float(np.sum(travel_time)))
        TravelCost_max = max(1.0, float(np.sum(travel_cost)))

        return (1 - R_cov) + w_time * (total_time / T_max) + w_dist * (total_travel_cost / TravelCost_max) + w_cost * m.penalty

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Các ràng buộc logic
    def one_site_per_cow_rule(m, k):
        return sum(m.x[k, j] for j in m.J) <= 1
    model.c_one_site_per_cow = pyo.Constraint(model.K, rule=one_site_per_cow_rule)

    def one_cow_per_site_rule(m, j):
        return sum(m.x[k, j] for k in m.K) <= 1
    model.c_one_cow_per_site = pyo.Constraint(model.J, rule=one_cow_per_site_rule)

    def serve_only_if_deployed_rule(m, i, j):
        return m.y[i, j] <= sum(m.x[k, j] for k in m.K)
    model.c_serve_only_if_deployed = pyo.Constraint(model.I, model.J, rule=serve_only_if_deployed_rule)

    def cover_rule(m, i, j):
        if not np.any(cover[i, j, :]):
            return m.y[i, j] == 0
        return pyo.Constraint.Skip
    model.c_cover = pyo.Constraint(model.I, model.J, rule=cover_rule)

    def one_site_per_area_rule(m, i):
        return sum(m.y[i, j] for j in m.J) <= 1
    model.c_one_site_per_area = pyo.Constraint(model.I, rule=one_site_per_area_rule)

    def z_def_rule(m, i):
        return m.z[i] <= sum(m.y[i, j] for j in m.J)
    model.c_zdef = pyo.Constraint(model.I, rule=z_def_rule)

    def Mmax_rule(m):
        return sum(m.x[k, j] for k in m.K for j in m.J) <= params.get("M_max", len(COWs))
    model.c_Mmax = pyo.Constraint(rule=Mmax_rule)

    def budget_rule(m):
        return sum(float(COWs[k].get("cost_vnd", 0.0)) * m.x[k, j] for k in m.K for j in m.J) <= budget_max
    model.c_budget = pyo.Constraint(rule=budget_rule)

    def endurance_rule(m, k, j):
        endurance = float(COWs[k].get("endurance_hr", 0.0))
        if float(travel_time[k, j]) > endurance:
            return m.x[k, j] == 0
        return pyo.Constraint.Skip
    model.c_endurance = pyo.Constraint(model.K, model.J, rule=endurance_rule)

    # Giới hạn theo loại COW
    cow_types = list(set(cow.get("type", "default") for cow in COWs))
    type_to_indices = {t: [k for k, cow in enumerate(COWs) if cow.get("type", "default") == t] for t in cow_types}
    max_per_type = {t: len(idxs) for t, idxs in type_to_indices.items()}

    def type_limit_rule(m, t):
        idxs = type_to_indices[t]
        return sum(m.x[k, j] for k in idxs for j in m.J) <= max_per_type[t]
    model.c_type_limit = pyo.Constraint(cow_types, rule=type_limit_rule)

    return model
