#!/usr/bin/env python3
"""
Route Visualizer for COW Deployments & Backup Power Deployment
BTS Restoration Project – 2025

Features:
1. Loads assignment results (COW → J, Power → BTS)
2. Loads travel cost lookup tables
3. Loads road network graph (roads_flooded.graphml)
4. Extracts REAL routes via NetworkX shortest paths
5. Creates a SINGLE GeoJSON file (all_routes.geojson)
6. Creates an interactive Folium map where:
   - Clicking a J marker highlights its COW route
   - Clicking a BTS marker highlights its POWER route

Author: Lợi Lưu – Optimized by ChatGPT (2025)
"""

import os
import json
import folium
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from pathlib import Path


# ======================================================================
# 1. LOAD GRAPH
# ======================================================================

def load_graph(graphml_path):
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML does not exist: {graphml_path}")

    print(f"[INFO] Loading graph: {graphml_path}")
    G = nx.read_graphml(graphml_path)

    # Normalize coordinate types
    for n, data in G.nodes(data=True):
        try:
            data["x"] = float(data["x"])
            data["y"] = float(data["y"])
        except:
            pass

    return G


# ======================================================================
# 2. Extract geometry from path
# ======================================================================

def extract_route_geometry(G, path):
    segments = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        if v not in G[u]:
            print(f"[WARN] Missing edge {u} → {v}")
            continue

        edge_data = G[u][v]

        # If multigraph
        if isinstance(edge_data, dict) and len(edge_data) > 0:
            e = edge_data[list(edge_data.keys())[0]]
        else:
            e = edge_data

        geom = e.get("geometry")

        if geom is None:
            # fallback use straight line
            p1 = (G.nodes[u]["x"], G.nodes[u]["y"])
            p2 = (G.nodes[v]["x"], G.nodes[v]["y"])
            segments.append(LineString([p1, p2]))
        else:
            try:
                from shapely import wkt
                segments.append(wkt.loads(geom))
            except:
                pass

    if not segments:
        return None

    merged = linemerge(MultiLineString(segments))
    return merged


# ======================================================================
# 3. Convert geometry to GeoJSON feature
# ======================================================================

def geometry_to_feature(geometry, properties):
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": list(geometry.coords)
        },
        "properties": properties
    }


# ======================================================================
# 4. Build INTERACTIVE MAP (Click marker → highlight route)
# ======================================================================

def build_interactive_map(all_features, markers, out_html):
    """
    all_features: list of GeoJSON features (routes)
    markers: list of marker definitions
    """

    # Base map
    m = folium.Map(location=[16.47, 107.6], zoom_start=11, tiles="OpenStreetMap")

    # Add markers and JS click event
    for mk in markers:
        folium.Marker(
            [mk["lat"], mk["lon"]],
            popup=mk["name"],
            tooltip="Click to show route",
            icon=folium.Icon(color=mk["color"])
        ).add_to(m)

    # ==================================================================
    # JavaScript system to dynamically load & display routes on click
    # ==================================================================

    routes_js = json.dumps(all_features)

    custom_js = f"""
        <script>
        var allRoutes = {routes_js};
        var drawnRoute = null;

        function showRoute(id, type) {{
            if (drawnRoute !== null) {{
                map.removeLayer(drawnRoute);
            }}

            let feature = allRoutes.find(
                r => r.properties.id === id && r.properties.route_type === type
            );

            if (!feature) {{
                alert("Route not found for " + id);
                return;
            }}

            drawnRoute = L.geoJSON(feature, {{
                style: function() {{
                    return {{ color: feature.properties.route_type === "COW" ? "blue" : "red", weight: 5 }};
                }}
            }}).addTo(map);
        }}

        // Attach to marker clicks
        {''.join([
            f'''
            var el = document.getElementsByClassName("leaflet-marker-icon")[{idx}];
            el.addEventListener("click", function() {{
                showRoute("{mk['id']}", "{mk['type']}");
            }});
            '''
            for idx, mk in enumerate(markers)
        ])}
        </script>
    """

    m.get_root().html.add_child(folium.Element(custom_js))

    # Save map
    m.save(out_html)
    print(f"[INFO] Interactive map saved → {out_html}")


# ======================================================================
# 5. MAIN SIMULATION
# ======================================================================

def simulate_routes(graphml_path, cow_assign_csv, cow_lookup_csv,
                    power_assign_csv, power_lookup_csv):

    G = load_graph(graphml_path)

    df_cow_assign = pd.read_csv(cow_assign_csv)
    df_cow_lookup = pd.read_csv(cow_lookup_csv)

    df_power_assign = pd.read_csv(power_assign_csv)
    df_power_lookup = pd.read_csv(power_lookup_csv)

    all_features = []      # For all_routes.geojson
    marker_list = []       # Marker definitions for map

    # ======================================================
    # COW ROUTES
    # ======================================================

    for _, row in df_cow_assign.iterrows():
        cow = row["cow_id"]
        site = row["site_id"]

        lookup = df_cow_lookup[
            (df_cow_lookup["cow_id"] == cow) &
            (df_cow_lookup["site_id"] == site)
        ].iloc[0]

        slat, slon = lookup["base_lat"], lookup["base_lon"]
        elat, elon = lookup["site_lat"], lookup["site_lon"]

        # Snap to nearest node
        start_node = min(G.nodes, key=lambda n: (G.nodes[n]["x"]-slon)**2 + (G.nodes[n]["y"]-slat)**2)
        end_node   = min(G.nodes, key=lambda n: (G.nodes[n]["x"]-elon)**2 + (G.nodes[n]["y"]-elat)**2)

        path = nx.shortest_path(G, start_node, end_node, weight="length_m")
        geom = extract_route_geometry(G, path)

        feature = geometry_to_feature(geom, {
            "route_type": "COW",
            "id": cow,
            "dest": site
        })

        all_features.append(feature)

        marker_list.append({
            "id": cow,
            "type": "COW",
            "lat": elat,
            "lon": elon,
            "name": f"COW {cow} → {site}",
            "color": "blue"
        })


    # ======================================================
    # POWER ROUTES
    # ======================================================

    for _, row in df_power_assign.iterrows():
        bts = row["bts_id"]
        power = row["power_id"]

        lookup = df_power_lookup[
            (df_power_lookup["power_id"] == power) &
            (df_power_lookup["bts_id"] == bts)
        ].iloc[0]

        slat, slon = lookup["base_lat"], lookup["base_lon"]
        elat, elon = lookup["bts_lat"], lookup["bts_lon"]

        start_node = min(G.nodes, key=lambda n: (G.nodes[n]["x"]-slon)**2 + (G.nodes[n]["y"]-slat)**2)
        end_node   = min(G.nodes, key=lambda n: (G.nodes[n]["x"]-elon)**2 + (G.nodes[n]["y"]-elat)**2)

        try:
            path = nx.shortest_path(G, start_node, end_node, weight="length_m")
        except:
            print(f"[WARN] No route for POWER {power} → {bts}")
            continue

        geom = extract_route_geometry(G, path)

        feature = geometry_to_feature(geom, {
            "route_type": "POWER",
            "id": power,
            "dest": bts
        })

        all_features.append(feature)

        marker_list.append({
            "id": power,
            "type": "POWER",
            "lat": elat,
            "lon": elon,
            "name": f"POWER {power} → {bts}",
            "color": "red"
        })


    # ======================================================
    # SAVE GEOJSON (ONE SINGLE FILE)
    # ======================================================

    out_geo = {
        "type": "FeatureCollection",
        "features": all_features
    }

    out_geo_path = "BTS_Restoration_Project/outputs/routes/all_routes.geojson"
    os.makedirs(os.path.dirname(out_geo_path), exist_ok=True)

    with open(out_geo_path, "w") as f:
        json.dump(out_geo, f, indent=2)

    print(f"[INFO] Saved all routes → {out_geo_path}")

    # ======================================================
    # BUILD INTERACTIVE MAP
    # ======================================================

    build_interactive_map(
        all_features=all_features,
        markers=marker_list,
        out_html="BTS_Restoration_Project/outputs/routes/routes_map.html"
    )


# ======================================================================
# MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    BASE = Path("BTS_Restoration_Project")

    simulate_routes(
        graphml_path=BASE / "data/processed/road/roads_flooded.graphml",
        cow_assign_csv=BASE / "outputs/results_ga_pso/solution_cow_assignments.csv",
        cow_lookup_csv=BASE / "data/processed/travel_cost/cow_to_J_sites.csv",
        power_assign_csv=BASE / "outputs/results_ga_pso/solution_power_assignments.csv",
        power_lookup_csv=BASE / "data/processed/travel_cost/backup_to_failed_bts.csv"
    )

    print("\n[INFO] Route simulation completed.")
