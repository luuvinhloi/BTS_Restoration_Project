# generate_bts_network.py

import os
import math
import random
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy as rio_xy
from rasterstats import zonal_stats
import osmnx as ox
from sklearn.cluster import DBSCAN
from deap import base, creator, tools, algorithms
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

#  CONFIG
BOUNDARY_GEOJSON = "data/cleaned/hue_boundary_clean.geojson"
POP_RASTER = "data/cleaned/pop_hue_clean.tif"
DEM_RASTER = "data/cleaned/elev_hue_clean.tif"
SLOPE_RASTER = "data/cleaned/slope_hue_clean.tif"

OUTPUT_DIR = "data/processed/bts_network"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Targets
NUM_CANDIDATES = 20000
NUM_BTS = 1000
MAX_CLUSTER_CAND = 8000
SEED = 42

# GA / clustering
COVERAGE_RADIUS_DEFAULT = 1500
DBSCAN_EPS_M = 100
GA_POPSIZE = 100
GA_NGEN = 100

# Heuristics
ALPHA_NEIGHBOR = 0.4
NEIGH_PENALTY_SCALE = 1000.0

POP_RASTER_IS_DENSITY = None

random.seed(SEED)
np.random.seed(SEED)

# Globals
POP_ARR_CANDIDATES = None
NEIGH_ARR = None
INDICES = None

# Utility functions
def compute_pixel_area_km2_from_raster(src):
    import pyproj
    width = src.width
    height = src.height
    transform = src.transform
    row = height // 2
    col = width // 2
    lon, lat = rio_xy(transform, row, col, offset='center')
    lon_r, lat_r = rio_xy(transform, row, col+1, offset='center')
    lon_d, lat_d = rio_xy(transform, row+1, col, offset='center')
    proj = pyproj.Transformer.from_crs("epsg:4326", "epsg:32648", always_xy=True)
    x_c, y_c = proj.transform(lon, lat)
    x_r, y_r = proj.transform(lon_r, lat_r)
    x_d, y_d = proj.transform(lon_d, lat_d)
    dx = abs(x_r - x_c)
    dy = abs(y_d - y_c)
    return (dx * dy) / 1e6

def detect_raster_density_or_count(src):
    band = src.read(1, masked=True)
    data = band.compressed() if hasattr(band, "compressed") else band[~band.mask]
    if data.size == 0:
        raise RuntimeError("Raster invalid.")
    mean_val = float(data.mean())
    pixel_area_km2 = compute_pixel_area_km2_from_raster(src)
    implied_density = mean_val / pixel_area_km2
    if implied_density > 50:
        return False, dict(mean=mean_val, implied_density=implied_density, pixel_area_km2=pixel_area_km2)
    else:
        return True, dict(mean=mean_val, implied_density=implied_density, pixel_area_km2=pixel_area_km2)

def ensure_within_boundary(gdf, boundary):
    return gdf[gdf.geometry.within(boundary.unary_union)].copy().reset_index(drop=True)

def nearest_distance(point, gdf_targets):
    if gdf_targets is None or gdf_targets.empty:
        return float('inf')
    return float(gdf_targets.distance(point).min())

# GA helpers
def pool_init(pop_arr, neigh_arr, indices):
    global POP_ARR_CANDIDATES, NEIGH_ARR, INDICES
    POP_ARR_CANDIDATES = np.array(pop_arr)
    NEIGH_ARR = np.array(neigh_arr)
    INDICES = list(indices)

def fitness_of_indices(idx_list):
    idxs = np.array(idx_list, dtype=int)
    pop_sum = POP_ARR_CANDIDATES[idxs].sum()
    neigh_mean = NEIGH_ARR[idxs].mean() if len(idxs) else 0.0
    score = pop_sum - ALPHA_NEIGHBOR * neigh_mean * NEIGH_PENALTY_SCALE
    return (score,)

def evaluate_individual(ind):
    return fitness_of_indices(ind)

def init_individual_top():
    return creator.Individual(random.sample(INDICES, NUM_BTS))

def mutate_swap_top(individual, indpb=0.2):
    if random.random() < indpb:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return (individual,)

def cx_twopoint_unique_top(ind1, ind2):
    a, b = sorted(random.sample(range(len(ind1)), 2))
    s1, s2 = ind1[a:b], ind2[a:b]
    ind1[a:b], ind2[a:b] = s2[:], s1[:]
    for ind in (ind1, ind2):
        seen = set()
        unused = [i for i in INDICES if i not in ind]
        for pos in range(len(ind)):
            if ind[pos] in seen:
                if unused:
                    ind[pos] = unused.pop()
            else:
                seen.add(ind[pos])
    return ind1, ind2

# Domain-specific helpers
def get_power_w(bts_type):
    mapping = {
        "4G_remote": 1200,
        "4G_macro": 5000,
        "5G_small": 250,
        "5G_macro": 10000
    }
    return mapping.get(bts_type, 5000)

def coverage_radius_by_type(t):
    if t is None:
        return COVERAGE_RADIUS_DEFAULT
    t = t.lower()
    return {
        "5g_small": 300,
        "5g_macro": 1000,
        "4g_remote": 2500,
        "4g_macro": 1500
    }.get(t, COVERAGE_RADIUS_DEFAULT)

def assign_bts_type(row):
    region = row.get('region_type','rural')
    popcov = row.get('pop_covered',0)
    dist_res = row.get('dist_to_residential_m', 1e9)
    dist_ind = row.get('dist_to_industrial_m', 1e9)
    slope = row.get('slope_deg', 0)

    if region=='urban' or dist_ind<1000 or dist_res<300:
        if popcov>=600: return "5G_macro"
        elif popcov>=300: return "5G_small"
        else: return "4G_macro"

    if region=='rural':
        if popcov>=400 and dist_res<500:
            return "5G_small"
        else:
            return "4G_macro"

    if region=='mountain' or slope>15:
        return "4G_remote"

    return "4G_macro"

# Unique coverage helpers
def compute_unique_coverage_generic(selected_gdf, pop_raster, raster_is_density, nodata):
    if selected_gdf.empty:
        return dict(total_pop=0.0, area_km2=0.0, overlap_ratio=0.0)

    sel_proj = selected_gdf.to_crs(epsg=32648)
    buffers = [g.buffer(r) for g, r in zip(sel_proj.geometry, sel_proj['coverage_radius_m'])]

    total_area_km2 = sum(b.area for b in buffers)/1e6
    union_geom = unary_union(buffers)
    if union_geom.is_empty:
        return dict(total_pop=0.0, area_km2=0.0, overlap_ratio=0.0)

    geoms = [union_geom] if union_geom.geom_type!="GeometryCollection" else [g for g in union_geom.geoms if not g.is_empty]
    union_area_km2 = sum(g.area for g in geoms)/1e6
    overlap_ratio = (total_area_km2-union_area_km2)/total_area_km2 * 100.0 if total_area_km2>0 else 0.0

    union_gdf = gpd.GeoDataFrame(geometry=geoms, crs=sel_proj.crs).to_crs(epsg=4326)

    if raster_is_density:
        stats = zonal_stats(union_gdf, pop_raster, stats="mean", nodata=nodata, all_touched=True)
        mean_density = np.mean([s.get("mean",0) for s in stats])
        total_pop = mean_density * union_area_km2
    else:
        stats = zonal_stats(union_gdf, pop_raster, stats="sum", nodata=nodata, all_touched=True)
        total_pop = sum(s.get("sum",0) for s in stats)

    return dict(total_pop=total_pop, area_km2=union_area_km2, overlap_ratio=overlap_ratio)

def compute_per_site_unique_pop(selected_gdf, pop_raster, raster_is_density, nodata):
    if selected_gdf.empty:
        return []

    sel_proj = selected_gdf.to_crs(epsg=32648)
    buffers = [g.buffer(r) for g,r in zip(sel_proj.geometry, sel_proj['coverage_radius_m'])]
    results = []

    for i,b in enumerate(buffers):
        others = [buffers[j] for j in range(len(buffers)) if j!=i]
        union_others = unary_union(others) if others else None
        unique_geom = b if (union_others is None or union_others.is_empty) else b.difference(union_others)
        if unique_geom.is_empty:
            results.append(0.0)
            continue
        gdf_u = gpd.GeoDataFrame(geometry=[unique_geom], crs=sel_proj.crs).to_crs(epsg=4326)
        if raster_is_density:
            stats = zonal_stats(gdf_u, pop_raster, stats="mean", nodata=nodata, all_touched=True)
            area_km2 = unique_geom.area/1e6
            pop = stats[0].get("mean",0)*area_km2
        else:
            stats = zonal_stats(gdf_u, pop_raster, stats="sum", nodata=nodata, all_touched=True)
            pop = stats[0].get("sum",0)
        results.append(pop)
    return results

# MAIN PIPELINE
def main():
    print("     Load boundary & raster...")
    boundary = gpd.read_file(BOUNDARY_GEOJSON).to_crs(epsg=4326)

    with rasterio.open(POP_RASTER) as src:
        pop_meta = src.meta.copy()
        nodata = src.nodata if src.nodata is not None else -99999
        out_img, out_transform = mask(src, boundary.geometry, crop=True)
        pop_arr = out_img[0]
        pop_transform = out_transform

    with rasterio.open(POP_RASTER) as src:
        if POP_RASTER_IS_DENSITY is None:
            raster_is_density, info = detect_raster_density_or_count(src)
            print("      Raster detection info:", info)
        else:
            raster_is_density = bool(POP_RASTER_IS_DENSITY)
            print("Using override POP_RASTER_IS_DENSITY =", raster_is_density)

    # 2) Sample candidate points
    print("     Sampling candidates...")
    rows, cols = pop_arr.shape
    flat = pop_arr.flatten().astype(float)
    flat[flat<0] = 0
    valid = np.where(flat>0)[0]
    if len(valid)==0:
        raise RuntimeError("No positive raster cells.")

    probs = flat[valid]/flat[valid].sum()
    chosen_idxs = np.random.choice(valid, size=NUM_CANDIDATES, replace=True, p=probs)

    cand = []
    for idx in chosen_idxs:
        r = idx//cols
        c = idx%cols
        lon, lat = rio_xy(pop_transform, r, c, offset='center')
        cand.append((lon,lat))

    cand_gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in cand], crs='EPSG:4326')
    cand_gdf = ensure_within_boundary(cand_gdf, boundary)
    print("      ->", len(cand_gdf), "candidates inside boundary")

    # clustering
    cand_proj = cand_gdf.to_crs(epsg=32648)
    coords = np.vstack([cand_proj.geometry.x, cand_proj.geometry.y]).T
    db = DBSCAN(eps=DBSCAN_EPS_M, min_samples=1).fit(coords)

    cand_proj['cluster'] = db.labels_
    clusters = cand_proj.dissolve(by='cluster', aggfunc='mean').reset_index(drop=True)
    clusters['geometry'] = clusters.geometry.centroid
    clusters = clusters.to_crs(epsg=4326)
    clusters['lon'] = clusters.geometry.x
    clusters['lat'] = clusters.geometry.y

    if len(clusters)>MAX_CLUSTER_CAND:
        clusters = clusters.sample(n=MAX_CLUSTER_CAND, random_state=SEED)

    cand_gdf = gpd.GeoDataFrame(clusters[['lon','lat']], geometry=gpd.points_from_xy(clusters.lon, clusters.lat), crs='EPSG:4326')
    print("      -> after clustering:", len(cand_gdf))

    # 3) Load OSM features
    print("     Loading OSM features...")
    poly = boundary.geometry.unary_union
    roads = ox.features_from_polygon(poly, {"highway": True})
    roads = roads[roads.geometry.type.isin(["LineString","MultiLineString"])].to_crs(epsg=32648)

    amen = ox.features_from_polygon(poly, {"amenity": ["hospital","clinic","school"]})
    amen = amen[amen.geometry.type.isin(["Point","MultiPoint"])].to_crs(epsg=32648)

    landuse = ox.features_from_polygon(poly, {"landuse": ["residential","industrial"]}).to_crs(epsg=32648)

    cand_proj = cand_gdf.to_crs(epsg=32648)
    print("      > computing distances...")

    cand_proj['dist_to_road_m'] = cand_proj.geometry.apply(lambda p: nearest_distance(p, roads))
    cand_proj['dist_to_hospital_m'] = cand_proj.geometry.apply(lambda p: nearest_distance(p, amen[amen['amenity'].isin(['hospital','clinic'])]))
    cand_proj['dist_to_school_m'] = cand_proj.geometry.apply(lambda p: nearest_distance(p, amen[amen['amenity']=='school']))

    cand_gdf = cand_proj.to_crs(epsg=4326)
    cand_gdf['dist_to_road_m'] = cand_proj['dist_to_road_m']
    cand_gdf['dist_to_hospital_m'] = cand_proj['dist_to_hospital_m']
    cand_gdf['dist_to_school_m'] = cand_proj['dist_to_school_m']

    # 4) Pop covered default radius
    print("     Pop_covered initial...")
    cand_proj = cand_gdf.to_crs(epsg=32648)
    cand_proj['coverage_radius_m'] = COVERAGE_RADIUS_DEFAULT
    buff = cand_proj.copy()
    buff['geometry'] = cand_proj.geometry.buffer(COVERAGE_RADIUS_DEFAULT)
    buff_wgs = buff.to_crs(epsg=4326)

    if raster_is_density:
        stats = zonal_stats(buff_wgs, POP_RASTER, stats="mean", nodata=nodata, all_touched=True)
        areas = buff.geometry.area/1e6
        popcov = [(s.get('mean',0)*a) for s,a in zip(stats, areas)]
    else:
        stats = zonal_stats(buff_wgs, POP_RASTER, stats="sum", nodata=nodata, all_touched=True)
        popcov = [s.get('sum',0) for s in stats]

    cand_gdf['pop_covered'] = np.round(popcov,2)

    # DEM/slope
    if os.path.exists(DEM_RASTER) or os.path.exists(SLOPE_RASTER):
        print("      Extracting DEM/slope...")
        def sample_raster(raster, gdf):
            vals = []
            with rasterio.open(raster) as src:
                for geom in gdf.to_crs(src.crs).geometry:
                    for v in src.sample([(geom.x, geom.y)]):
                        vals.append(v[0])
            return vals

        if os.path.exists(DEM_RASTER):
            cand_gdf['elevation_m'] = sample_raster(DEM_RASTER, cand_gdf)
        else:
            cand_gdf['elevation_m'] = np.nan

        if os.path.exists(SLOPE_RASTER):
            cand_gdf['slope_deg'] = sample_raster(SLOPE_RASTER, cand_gdf)
        else:
            cand_gdf['slope_deg'] = np.nan
    else:
        cand_gdf['elevation_m'] = np.nan
        cand_gdf['slope_deg'] = np.nan

    # 5) neighbour weight
    print("     neighbour weight...")
    cand_gdf['neighbour_weight'] = cand_gdf.apply(
        lambda r: (30 if r['dist_to_school_m']<50 else (15 if r['dist_to_school_m']<100 else 0)) +
                  (20 if r['dist_to_hospital_m']<100 else 0) +
                  (5 if r['dist_to_road_m']<5 else 0),
        axis=1
    )

    # 5.5) Attributes
    print("      adding attributes...")
    cand_gdf = cand_gdf.reset_index(drop=True)
    cand_gdf['site_id'] = ['BTS_%05d' % (i+1) for i in range(len(cand_gdf))]
    cand_gdf['latitude'] = cand_gdf.geometry.y
    cand_gdf['longitude'] = cand_gdf.geometry.x

    cand_proj = cand_gdf.to_crs(epsg=32648)
    cand_gdf['utm_x'] = cand_proj.geometry.x
    cand_gdf['utm_y'] = cand_proj.geometry.y

    # landuse distances
    if not landuse.empty:
        landuse_proj = landuse
        cand_proj = cand_gdf.to_crs(epsg=32648)
        cand_proj['dist_to_residential_m'] = cand_proj.geometry.apply(
            lambda p: nearest_distance(p, landuse_proj[landuse_proj['landuse']=='residential'])
        )
        cand_proj['dist_to_industrial_m'] = cand_proj.geometry.apply(
            lambda p: nearest_distance(p, landuse_proj[landuse_proj['landuse']=='industrial'])
        )
        cand_gdf['dist_to_residential_m'] = cand_proj['dist_to_residential_m']
        cand_gdf['dist_to_industrial_m'] = cand_proj['dist_to_industrial_m']
    else:
        cand_gdf['dist_to_residential_m'] = np.nan
        cand_gdf['dist_to_industrial_m'] = np.nan

    cand_gdf['site_accessibility_score'] = 1 - (cand_gdf['dist_to_road_m']/2000).clip(0,1)

    def pick_antenna_height(row):
        if row['dist_to_residential_m'] < 500:
            return round(np.random.normal(25,4),1)
        else:
            return round(np.random.normal(30,5),1)

    cand_gdf['antenna_height_m'] = cand_gdf.apply(pick_antenna_height, axis=1)

    # region
    cand_gdf['region_type'] = cand_gdf.apply(
        lambda r: 'mountain' if (not math.isnan(r['slope_deg']) and r['slope_deg']>15)
                  else ('urban' if r['dist_to_residential_m']<500 else 'rural'),
        axis=1
    )

    # STEP 6: bts type + coverage + power_W
    cand_gdf['bts_type'] = cand_gdf.apply(assign_bts_type, axis=1)
    cand_gdf['coverage_radius_m'] = cand_gdf['bts_type'].apply(coverage_radius_by_type)
    cand_gdf['power_W'] = cand_gdf['bts_type'].apply(get_power_w)

    # recompute pop covered with actual radii
    print("     recomputing pop_covered...")
    cand_proj = cand_gdf.to_crs(epsg=32648)
    buff2 = cand_proj.copy()
    buff2['geometry'] = [g.buffer(r) for g,r in zip(cand_proj.geometry, cand_proj['coverage_radius_m'])]
    buff2_wgs = buff2.to_crs(epsg=4326)

    if raster_is_density:
        st2 = zonal_stats(buff2_wgs, POP_RASTER, stats="mean", nodata=nodata, all_touched=True)
        areas = buff2.geometry.area/1e6
        popcov2 = [(s.get('mean',0)*a) for s,a in zip(st2, areas)]
    else:
        st2 = zonal_stats(buff2_wgs, POP_RASTER, stats="sum", nodata=nodata, all_touched=True)
        popcov2 = [s.get('sum',0) for s in st2]

    cand_gdf['pop_covered'] = np.round(popcov2,2)

    # STEP 7 greedy
    print("     greedy selection...")
    def greedy(df, k):
        tmp = df.copy().reset_index(drop=True)
        tmp['score'] = tmp['pop_covered']/(1+ALPHA_NEIGHBOR*tmp['neighbour_weight'].fillna(0))
        return tmp.nlargest(k,'score').copy()

    chosen_greedy = greedy(cand_gdf, NUM_BTS)
    chosen_greedy.to_file(os.path.join(OUTPUT_DIR,"bts_greedy.geojson"), driver="GeoJSON")

    # STEP 8 GA
    print("     GA selection...")
    indices = list(range(len(cand_gdf)))
    pop_arr = cand_gdf['pop_covered'].values
    neigh_arr = cand_gdf['neighbour_weight'].values
    global INDICES
    INDICES = indices

    if not hasattr(creator,"FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator,"Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual_top)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cx_twopoint_unique_top)
    toolbox.register("mutate", mutate_swap_top)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual)

    with multiprocessing.Pool(initializer=pool_init, initargs=(pop_arr, neigh_arr, indices)) as pool:
        toolbox.register("map", pool.map)
        pop = toolbox.population(n=GA_POPSIZE)
        hof = tools.HallOfFame(5)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=GA_NGEN,
                            stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    chosen_ga = cand_gdf.iloc[list(best)].copy()
    chosen_ga.to_file(os.path.join(OUTPUT_DIR,"bts_ga.geojson"), driver="GeoJSON")

    # STEP 9 coverage evaluation
    print("     coverage evaluation...")
    result_ga = compute_unique_coverage_generic(chosen_ga, POP_RASTER, raster_is_density, nodata)
    result_greedy = compute_unique_coverage_generic(chosen_greedy, POP_RASTER, raster_is_density, nodata)

    print("      > computing total population boundary...")
    boundary_stats = zonal_stats(boundary, POP_RASTER,
                                 stats="sum" if not raster_is_density else "mean",
                                 nodata=nodata, all_touched=True)
    if raster_is_density:
        mean_den = boundary_stats[0].get('mean',0)
        area = boundary.to_crs(epsg=32648).geometry.area.sum()/1e6
        total_pop = mean_den*area
    else:
        total_pop = boundary_stats[0].get('sum',0)

    print("\nRESULTS:")
    print(" - Total pop:", total_pop)
    print(" - GA unique:", result_ga["total_pop"])
    print(" - Greedy unique:", result_greedy["total_pop"])

    # pop_unique per site
    print("      per-site unique pop...")
    chosen_ga['pop_unique_covered'] = compute_per_site_unique_pop(chosen_ga, POP_RASTER, raster_is_density, nodata)
    chosen_greedy['pop_unique_covered'] = compute_per_site_unique_pop(chosen_greedy, POP_RASTER, raster_is_density, nodata)

    # STEP 10 export standardized v3
    print("     exporting output_v3...")

    chosen_ga['overlap_ratio_network'] = result_ga['overlap_ratio']
    chosen_ga['total_unique_pop_network'] = result_ga['total_pop']
    chosen_greedy['overlap_ratio_network'] = result_greedy['overlap_ratio']
    chosen_greedy['total_unique_pop_network'] = result_greedy['total_pop']

    required_cols = [
        "site_id","latitude","longitude","utm_x","utm_y",
        "pop_covered","pop_unique_covered",
        "overlap_ratio_network","total_unique_pop_network",
        "elevation_m","slope_deg","neighbour_weight",
        "dist_to_school_m","dist_to_hospital_m","dist_to_road_m",
        "dist_to_residential_m","dist_to_industrial_m",
        "site_accessibility_score","antenna_height_m",
        "region_type","bts_type","coverage_radius_m",
        "power_W"
    ]

    for df in [chosen_ga, chosen_greedy]:
        for c in required_cols:
            if c not in df:
                df[c] = np.nan

    chosen_ga[required_cols].to_csv(os.path.join(OUTPUT_DIR,"bts_ga.csv"), index=False)
    chosen_greedy[required_cols].to_csv(os.path.join(OUTPUT_DIR,"bts_greedy.csv"), index=False)

    summary = {
        "province": "Thừa Thiên Huế",
        "total_population": float(total_pop),
        "GA": {
            "unique_population": float(result_ga["total_pop"]),
            "coverage_ratio_percent": float(100*result_ga['total_pop']/total_pop) if total_pop>0 else 0,
            "overlap_ratio_percent": float(result_ga["overlap_ratio"])
        },
        "Greedy": {
            "unique_population": float(result_greedy["total_pop"]),
            "coverage_ratio_percent": float(100*result_greedy['total_pop']/total_pop) if total_pop>0 else 0,
            "overlap_ratio_percent": float(result_greedy["overlap_ratio"])
        }
    }

    with open(os.path.join(OUTPUT_DIR,"network_summary.json"),"w",encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Done. All outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
