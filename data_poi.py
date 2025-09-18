# OUT_PATH = "roads_with_population.geojson"
# 1) Frontage（20m）：front20_total、front20_per100m、front20_<l1>、front20_<l1>_per100m
# 2) Density（100/300/800m）：dens{R}_total_kmp2、dens{R}_{l1}_kmp2
# 3) distance（m）：dist_{l1}_m（transport/retail/food/education/health）

import numpy as np
import geopandas as gpd
import pandas as pd

ROADS_PATH = "roads_with_population.geojson"      # road（Line/MultiLine）
POI_PATH   = "poi.geojson"                         # POI（Point），若无 l1 列会自动映射
OUT_PATH   = "roads_with_poi_feats.geojson"
CRS_METRIC = 32647
FRONTAGE_DIST = 20
DENSITY_BUFFERS = (100, 300, 800)
NEAREST_L1 = ["transport", "retail", "food", "education", "health"]

def get_str(row, key):
    v = row.get(key)
    return "" if v is None else str(v).strip().lower()

def map_to_l1_row(row):
    amenity = get_str(row, "amenity")
    shop    = get_str(row, "shop")
    tourism = get_str(row, "tourism")
    leisure = get_str(row, "leisure")
    office  = get_str(row, "office")
    pt      = get_str(row, "public_transport")
    railway = get_str(row, "railway")
    landuse = get_str(row, "landuse")
    manmade = get_str(row, "man_made")
    category= get_str(row, "category")
    fclass  = get_str(row, "fclass")

    tags = {amenity, shop, tourism, leisure, office, pt, railway, landuse, manmade, category, fclass}
    tags.discard("")

    if {"bus_station","bus_stop","taxi","parking","ferry_terminal","bicycle_rental","tram_stop","subway_entrance"} & tags \
       or pt in {"stop_position","platform"} or railway in {"station","stop","halt","subway","light_rail"}:
        return "transport"
    if shop or amenity == "marketplace" or category in {"retail","supermarket","convenience"}:
        return "retail"
    if amenity in {"restaurant","cafe","fast_food","bar","pub","food_court"} or category in {"food","restaurant","cafe"}:
        return "food"
    if amenity in {"school","college","university","kindergarten","language_school"} or category == "education":
        return "education"
    if amenity in {"hospital","clinic","pharmacy","doctors","dentist"} or category in {"health","hospital","clinic"}:
        return "health"
    if tourism in {"hotel","hostel","guest_house"} or category in {"lodging","hotel"}:
        return "lodging"
    if tourism in {"attraction","museum","gallery"} or amenity == "place_of_worship" or category in {"tourism","culture","temple"}:
        return "tourism_culture"
    if leisure in {"park","pitch","fitness_centre","stadium","sports_centre","swimming_pool"} or category in {"leisure","sport"}:
        return "leisure_sport"
    if landuse == "industrial" or manmade in {"works"} or amenity == "warehouse" or category in {"industrial","logistics","warehouse"}:
        return "industry_logistics"
    if amenity in {"bank","atm","post_office","townhall"} or office or category in {"office","government","bank"}:
        return "office_gov"
    if amenity in {"hairdresser","car_repair","laundry","courier","beauty_salon","veterinary"} or category == "services":
        return "services"
    return "other"

def add_per100m(df, count_col, length_col="length_m"):
    per_col = f"{count_col}_per100m"
    df[per_col] = (100.0 * df[count_col] / df[length_col]).replace([np.inf, -np.inf], 0.0)
    df[per_col] = df[per_col].fillna(0.0)
    return per_col

roads = gpd.read_file(ROADS_PATH)
poi   = gpd.read_file(POI_PATH)

if "road_id" not in roads.columns:
    roads["road_id"] = np.arange(len(roads))

orig_crs_roads = roads.crs
roads_m = roads.to_crs(epsg=CRS_METRIC)
poi_m   = poi.to_crs(epsg=CRS_METRIC)

# road length
roads_m["length_m"] = roads_m.geometry.length.replace({np.inf: np.nan})
roads_m["length_m"] = roads_m["length_m"].fillna(0.0)

if "l1" not in poi_m.columns:
    poi_m["l1"] = poi_m.apply(map_to_l1_row, axis=1)

# ========== 1) Frontage（20 m） ==========
buf20 = roads_m[["road_id","geometry"]].copy()
buf20["geometry"] = buf20.buffer(FRONTAGE_DIST)
front_join = gpd.sjoin(poi_m[["geometry","l1"]], buf20, predicate="within", how="left")
front_tab = (front_join.groupby(["road_id","l1"]).size()
             .unstack(fill_value=0))
front_tab = front_tab.add_prefix("front20_")
roads_m = roads_m.merge(front_tab, on="road_id", how="left").fillna(0)
roads_m["front20_total"] = roads_m.filter(like="front20_").sum(axis=1)
_ = add_per100m(roads_m, "front20_total", "length_m")
for c in roads_m.filter(like="front20_").columns:
    if c == "front20_total":
        continue
    add_per100m(roads_m, c, "length_m")

# ========== 2) density（100/300/800 m） ==========
for R in DENSITY_BUFFERS:
    buf = roads_m[["road_id","geometry"]].copy()
    buf["geometry"] = buf.buffer(R)
    buf_area_km2 = (buf.geometry.area / 1e6).rename("area_km2")
    roads_m = roads_m.merge(buf_area_km2, left_on="road_id", right_index=True, how="left", suffixes=("",""))
    jj = gpd.sjoin(poi_m[["geometry","l1"]], buf, predicate="within", how="left").dropna(subset=["road_id"])
    tab = (jj.groupby(["road_id","l1"]).size()
             .unstack(fill_value=0))
    dens = tab.div(roads_m.set_index("road_id")["area_km2"], axis=0).fillna(0)
    dens.columns = [f"dens{R}_{c}_kmp2" for c in dens.columns]
    dens[f"dens{R}_total_kmp2"] = dens.sum(axis=1)
    roads_m = roads_m.merge(dens, left_on="road_id", right_index=True, how="left").fillna(0)
    roads_m = roads_m.drop(columns=["area_km2"])

# ========== 3) distance ==========
for cat in NEAREST_L1:
    sub = poi_m[poi_m["l1"] == cat][["geometry"]].copy()
    col = f"dist_{cat}_m"
    if len(sub) == 0:
        roads_m[col] = np.nan
        continue
    try:
        nn = gpd.sjoin_nearest(
            roads_m[["road_id","geometry"]],
            sub,
            how="left",
            distance_col=col
        )[["road_id", col]]
        roads_m = roads_m.drop(columns=[col], errors="ignore").merge(nn, on="road_id", how="left")
    except TypeError:
        union = sub.unary_union
        roads_m[col] = roads_m.geometry.apply(lambda g: g.distance(union) if union else np.nan)

# ========== output ==========
roads_out = roads_m.to_crs(orig_crs_roads)
num_cols = [c for c in roads_out.columns if c not in roads.columns or c in ("length_m",)]
for c in num_cols:
    if c == "road_id":
        continue
    if c != "road_id":
        try:
            roads_out[c] = roads_out[c].astype(float)
        except Exception:
            pass

roads_out.to_file(OUT_PATH, driver="GeoJSON")
print("Saved:", OUT_PATH)
