from math import radians, sin, cos, asin, sqrt
import pulp


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def build_base_problem(I_points, J_sites, BTS_failed, cows, backup_powers,
                       cow_travel, backup_travel, params, solver_name="CBC"):

    prob = pulp.LpProblem("MILP_Full", pulp.LpMinimize)

    I_ids = [i["site_id"] for i in I_points]
    J_ids = [j["site_id"] for j in J_sites]
    B_ids = [b["site_id"] for b in BTS_failed]
    C_ids = [c["cow_id"] for c in cows]
    G_ids = [g["power_id"] for g in backup_powers]

    budget_max = float(params.get("budget_max", 5e8))
    M_max = int(params.get("M_max", len(cows)))
    setup_time_h = float(params.get("default_setup_time_h", 0.5))
    endurance_mode = params.get("endurance_mode", "none")
    required_broadcast_time_h = float(params.get("required_broadcast_time_h", 0.0))

    I_map = {i["site_id"]: i for i in I_points}
    J_map = {j["site_id"]: j for j in J_sites}
    B_map = {b["site_id"]: b for b in BTS_failed}
    C_map = {c["cow_id"]: c for c in cows}
    G_map = {g["power_id"]: g for g in backup_powers}

    # BTS required power (kW)
    bts_required_kw = {}
    for b in BTS_failed:
        try:
            bts_required_kw[b["site_id"]] = float(b.get("power_W", 0.0)) / 1000.0
        except:
            bts_required_kw[b["site_id"]] = 0.0

    # Power source capacity (kW)
    g_power_kw = {}
    for g in backup_powers:
        if "power_kw" in g and g["power_kw"] not in (None, ""):
            g_power_kw[g["power_id"]] = float(g.get("power_kw"))
        else:
            if "resource_amount" in g:
                try:
                    g_power_kw[g["power_id"]] = float(g["resource_amount"])
                except:
                    g_power_kw[g["power_id"]] = 10.0
            else:
                typ = str(g.get("type", "")).upper()
                g_power_kw[g["power_id"]] = 10.0 if "GEN" in typ else 2.0

    x = pulp.LpVariable.dicts("x", (C_ids, J_ids), 0, 1, pulp.LpBinary)
    w_cow = pulp.LpVariable.dicts("w_cow", (C_ids, I_ids), 0, 1, pulp.LpBinary)
    w_bts = pulp.LpVariable.dicts("w_bts", (B_ids, I_ids), 0, 1, pulp.LpBinary)
    z = pulp.LpVariable.dicts("z", (G_ids, B_ids), 0, 1, pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", B_ids, 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", I_ids, 0, 1, pulp.LpBinary)
    T_max = pulp.LpVariable("T_max", lowBound=0)

    # Constraint 1: each COW at most one J
    for c in C_ids:
        prob += pulp.lpSum([x[c][j] for j in J_ids]) <= 1

    # Constraint 2: each J at most one COW
    for j in J_ids:
        prob += pulp.lpSum([x[c][j] for c in C_ids]) <= 1

    # Constraint 3: each I at most one provider
    for i in I_ids:
        prob += pulp.lpSum([w_bts[b][i] for b in B_ids]) + pulp.lpSum([w_cow[c][i] for c in C_ids]) <= 1

    # Constraint 4: y links to w
    for i in I_ids:
        prob += y[i] <= pulp.lpSum([w_bts[b][i] for b in B_ids]) + pulp.lpSum([w_cow[c][i] for c in C_ids])
        prob += pulp.lpSum([w_bts[b][i] for b in B_ids]) + pulp.lpSum([w_cow[c][i] for c in C_ids]) <= y[i] * 1e6

    # Constraint 5: BTS coverage
    for b in B_ids:
        bi = B_map[b]
        for i in I_ids:
            ii = I_map[i]
            dkm = haversine_km(bi["latitude"], bi["longitude"], ii["latitude"], ii["longitude"])
            covers = dkm * 1000 <= float(bi.get("coverage_radius_m", params.get("default_R", 3000)))
            if not covers:
                prob += w_bts[b][i] == 0
            else:
                prob += w_bts[b][i] <= u[b]

    # Constraint 6: COW coverage
    for c in C_ids:
        crow = C_map[c]
        radius = float(crow.get("coverage_radius_m", params.get("default_R", 3000)))
        for i in I_ids:
            ii = I_map[i]
            coverable = []
            for j in J_ids:
                jj = J_map[j]
                dkm = haversine_km(ii["latitude"], ii["longitude"], jj["latitude"], jj["longitude"])
                if dkm * 1000 <= radius:
                    coverable.append(j)
            if not coverable:
                prob += w_cow[c][i] == 0
            else:
                prob += w_cow[c][i] <= pulp.lpSum([x[c][j] for j in coverable])

    # Constraint 7: each BTS gets ≤1 source
    for b in B_ids:
        prob += pulp.lpSum([z[g][b] for g in G_ids]) <= 1
        prob += u[b] <= pulp.lpSum([z[g][b] for g in G_ids])

    # Constraint 8: Power satisfaction
    for b in B_ids:
        prob += pulp.lpSum([z[g][b] * g_power_kw[g] for g in G_ids]) >= bts_required_kw[b] * u[b]

    # Constraint 9: Endurance (optional)
    if endurance_mode in ("travel_plus_setup", "total"):
        for c in C_ids:
            endur = float(C_map[c].get("endurance_hr", 0))
            for j in J_ids:
                travel = float(cow_travel.get((c, j), {}).get("travel_time_hr", 0))
                needed = travel + setup_time_h
                if endurance_mode == "total":
                    needed += required_broadcast_time_h
                if needed > endur:
                    prob += x[c][j] == 0

    # Constraint 10: Budget
    cost_terms = []
    for c in C_ids:
        cow_cost = float(C_map[c].get("cost_vnd", 0))
        for j in J_ids:
            travel = float(cow_travel.get((c, j), {}).get("travel_cost_vnd", 0))
            cost_terms.append((cow_cost + travel) * x[c][j])

    for g in G_ids:
        g_cost = float(G_map[g].get("cost_vnd_24h", 0))
        for b in B_ids:
            travel = float(backup_travel.get((g, b), {}).get("total_cost_vnd", 0))
            cost_terms.append((g_cost + travel) * z[g][b])

    prob += pulp.lpSum(cost_terms) <= budget_max

    # Constraint 11: M_max
    prob += pulp.lpSum([x[c][j] for c in C_ids for j in J_ids]) <= M_max

    # Constraint 12: T_max
    for c in C_ids:
        for j in J_ids:
            t = float(cow_travel.get((c, j), {}).get("travel_time_hr", 0))
            prob += T_max >= (t + setup_time_h) * x[c][j]

    for g in G_ids:
        for b in B_ids:
            t = float(backup_travel.get((g, b), {}).get("total_time_hr", 0))
            prob += T_max >= t * z[g][b]

    return prob, {
        "x": x,
        "w_cow": w_cow,
        "w_bts": w_bts,
        "z": z,
        "u": u,
        "y": y,
        "T_max": T_max,
        "I_ids": I_ids,
        "J_ids": J_ids,
        "B_ids": B_ids,
        "C_ids": C_ids,
        "G_ids": G_ids,
        "maps": {"I": I_map, "J": J_map, "B": B_map, "C": C_map, "G": G_map},
    }
