"""
spatial_routes_map.py

REALISTIC ROUTE SIMULATION FOR DEPLOYMENT

Realistic Deployment Routes under Flood Conditions

Outputs (per method):
- MILP
- GA_PSO
- MILP_GA_PSO

Author: Lợi Lưu
"""

from pathlib import Path
import networkx as nx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.ops import linemerge
from matplotlib.lines import Line2D
import folium

# PATH CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRAPH_PATH    = PROJECT_ROOT / "data/processed/road/roads_flooded.graphml"
BOUNDARY_PATH = PROJECT_ROOT / "data/cleaned/hue_boundary_clean.geojson"
J_SITES_PATH  = PROJECT_ROOT / "data/processed/position_I_J/J_sites_B.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/simulation_B/spatial/routes"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "MILP": {
        "cow": PROJECT_ROOT / "outputs/milp_runs/milp_gurobi/assignments_cow_GUROBI.csv",
        "power": PROJECT_ROOT / "outputs/milp_runs/milp_gurobi/assignments_power_GUROBI.csv",
        "cow_lookup": PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites.csv",
        "power_lookup": PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts.csv",
        "out": OUTPUT_ROOT / "milp"
    },
    "GA_PSO": {
        "cow": PROJECT_ROOT / "outputs/results_ga_pso_B/solution_cow_assignments.csv",
        "power": PROJECT_ROOT / "outputs/results_ga_pso_B/solution_power_assignments.csv",
        "cow_lookup": PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites_B.csv",
        "power_lookup": PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts_B.csv",
        "out": OUTPUT_ROOT / "ga_pso"
    },
    "MILP_GA_PSO": {
        "cow": PROJECT_ROOT / "outputs/results_hybrid_B/solution_cow_assignments.csv",
        "power": PROJECT_ROOT / "outputs/results_hybrid_B/solution_power_assignments.csv",
        "cow_lookup": PROJECT_ROOT / "data/processed/travel_cost/cow_to_J_sites_B.csv",
        "power_lookup": PROJECT_ROOT / "data/processed/travel_cost/backup_to_failed_bts_B.csv",
        "out": OUTPUT_ROOT / "hybrid"
    }
}

# GRAPH UTILS
def load_graph():
    G = nx.read_graphml(GRAPH_PATH)
    for _, d in G.nodes(data=True):
        d["x"] = float(d["x"])
        d["y"] = float(d["y"])
    return G

def nearest_node(G, lon, lat):
    return min(
        G.nodes,
        key=lambda n: (G.nodes[n]["x"] - lon) ** 2 + (G.nodes[n]["y"] - lat) ** 2
    )

def extract_linestring(G, path):
    segments = []
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)

        if isinstance(edge_data, dict) and any(isinstance(vv, dict) for vv in edge_data.values()):
            edge_attr = next(iter(edge_data.values()))
        else:
            edge_attr = edge_data

        geom = edge_attr.get("geometry") if isinstance(edge_attr, dict) else None

        if geom:
            from shapely import wkt
            segments.append(wkt.loads(geom))
        else:
            p1 = (G.nodes[u]["x"], G.nodes[u]["y"])
            p2 = (G.nodes[v]["x"], G.nodes[v]["y"])
            segments.append(LineString([p1, p2]))

    return linemerge(segments) if segments else None

def load_all_bases(cow_lookup, power_lookup):
    cow = pd.read_csv(cow_lookup)[["base_id", "base_lat", "base_lon"]]
    power = pd.read_csv(power_lookup)[["base_id", "base_lat", "base_lon"]]
    return pd.concat([cow, power]).drop_duplicates("base_id")

# ROUTE SIMULATION
def simulate_routes(cfg):
    G = load_graph()
    boundary = gpd.read_file(BOUNDARY_PATH)
    j_sites = pd.read_csv(J_SITES_PATH)

    cow_assign   = pd.read_csv(cfg["cow"])
    cow_lookup   = pd.read_csv(cfg["cow_lookup"])
    power_assign = pd.read_csv(cfg["power"])
    power_lookup = pd.read_csv(cfg["power_lookup"])

    cow_routes, power_routes = [], []
    j_points, bts_points = [], []

    # COW to J
    for _, r in cow_assign.iterrows():
        site_id = r["site_id"]
        j = j_sites[j_sites["site_id"] == site_id]
        lk = cow_lookup[cow_lookup["site_id"] == site_id]

        if j.empty or lk.empty:
            continue

        j = j.iloc[0]
        lk = lk.iloc[0]

        try:
            path = nx.shortest_path(
                G,
                nearest_node(G, lk.base_lon, lk.base_lat),
                nearest_node(G, j.longitude, j.latitude),
                weight="length_m"
            )
            geom = extract_linestring(G, path)
            if geom:
                cow_routes.append(geom)
                j_points.append((j.latitude, j.longitude))
        except:
            continue

    # POWER to BTS
    for _, r in power_assign.iterrows():
        lk = power_lookup[
            (power_lookup["power_id"] == r["power_id"]) &
            (power_lookup["bts_id"] == r["bts_id"])
        ]
        if lk.empty:
            continue

        lk = lk.iloc[0]

        try:
            path = nx.shortest_path(
                G,
                nearest_node(G, lk.base_lon, lk.base_lat),
                nearest_node(G, lk.bts_lon, lk.bts_lat),
                weight="length_m"
            )
            geom = extract_linestring(G, path)
            if geom:
                power_routes.append(geom)
                bts_points.append((lk.bts_lat, lk.bts_lon))
        except:
            continue

    return boundary, cow_routes, power_routes, j_points, bts_points

# EXPORT MAPS
def export_maps(method, cfg):
    out = cfg["out"]
    out.mkdir(parents=True, exist_ok=True)

    boundary, cow_routes, power_routes, js, btss = simulate_routes(cfg)
    bases = load_all_bases(cfg["cow_lookup"], cfg["power_lookup"])

    # PNG
    fig, ax = plt.subplots(figsize=(13, 11))

    boundary.boundary.plot(ax=ax, color="black", linewidth=2.2, zorder=5)

    G = load_graph()
    for u, v in G.edges():
        ax.plot(
            [G.nodes[u]["x"], G.nodes[v]["x"]],
            [G.nodes[u]["y"], G.nodes[v]["y"]],
            color="#dddddd", linewidth=0.3
        )

    for g in cow_routes:
        x, y = g.xy
        ax.plot(x, y, color="#1f77b4", lw=2.5)

    for g in power_routes:
        x, y = g.xy
        ax.plot(x, y, color="#d62728", lw=2)

    ax.scatter(
        bases.base_lon, bases.base_lat,
        marker="s", s=90, c="black",
        edgecolor="white", zorder=6
    )

    ax.legend(handles=[
        Line2D([0],[0], color="#1f77b4", lw=3, label="COW → J"),
        Line2D([0],[0], color="#d62728", lw=3, label="Power → BTS"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor="black", markersize=8, label="BASE")
    ], loc="lower left")

    ax.set_title(f"Figure: Realistic Deployment Routes - ({method})")
    ax.set_axis_off()
    plt.savefig(out / "routes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # HTML
    center = [bases.base_lat.mean(), bases.base_lon.mean()]
    m = folium.Map(center, zoom_start=11, tiles="OpenStreetMap")

    folium.GeoJson(
        boundary.to_crs("EPSG:4326"),
        name="Hue Boundary",
        style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0}
    ).add_to(m)

    fg_cow = folium.FeatureGroup(name="COW to J routes")
    fg_power = folium.FeatureGroup(name="Power to BTS routes")
    fg_base = folium.FeatureGroup(name="BASE")
    fg_j = folium.FeatureGroup(name="J deployed")
    fg_bts = folium.FeatureGroup(name="Powered BTS")

    for g in cow_routes:
        folium.PolyLine([(y,x) for x,y in g.coords], color="#1f77b4", weight=4).add_to(fg_cow)

    for g in power_routes:
        folium.PolyLine([(y,x) for x,y in g.coords], color="#d62728", weight=3).add_to(fg_power)

    for _, r in bases.iterrows():
        folium.Marker(
            [r.base_lat, r.base_lon],
            icon=folium.Icon(icon="home", prefix="fa", color="black"),
            popup=f"BASE {r.base_id}"
        ).add_to(fg_base)

    for lat, lon in js:
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html="<div style='font-size:20px;color:#4981f3;'>★</div>"
            )
        ).add_to(fg_j)

    for lat, lon in btss:
        folium.Marker(
            [lat, lon],
            icon=folium.Icon(color="green", icon="flash")
        ).add_to(fg_bts)

    for fg in [fg_cow, fg_power, fg_base, fg_j, fg_bts]:
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out / "routes.html")

    print(f"[DONE] {method}: Saved PNG & HTML at {out}")

# MAIN
def run():
    print("\nSpatial Routes Map Simulation (MULTI-METHOD)")
    for name, cfg in METHODS.items():
        export_maps(name, cfg)
    print("Completed: Simulation of Realistic Deployment Routes\n")


if __name__ == "__main__":
    run()
