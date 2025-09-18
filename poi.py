# OUT_PATH    = "roads_with_population.geojson"
# roads_poi_features.py
# ------------------------------------------------------------
# 功能：
# 1) Frontage（20m）：front20_total、front20_per100m、front20_<l1>、front20_<l1>_per100m
# 2) 多尺度密度（100/300/800m）：dens{R}_total_kmp2、dens{R}_{l1}_kmp2
# 3) 最近距离（m）：dist_{l1}_m（transport/retail/food/education/health）
#
# 依赖：geopandas, shapely, numpy
# ------------------------------------------------------------

import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ========== 路径与参数 ==========
ROADS_PATH = "roads_with_population.geojson"   # 道路（Line/MultiLine）
POI_PATH   = "poi.geojson"                         # POI（Point），若无 l1 列会自动映射
OUT_PATH   = "roads_with_poi_feats.geojson"                   # 输出
CRS_METRIC = 32647                                            # 米制投影（曼谷 UTM 47N）
FRONTAGE_DIST = 20                                            # 沿街距离（m）
DENSITY_BUFFERS = (100, 300, 800)                             # 多尺度缓冲（m）
NEAREST_L1 = ["transport", "retail", "food", "education", "health"]  # 最近距离的关键类别

# ========== 工具函数 ==========
def _get_str(row, key):
    v = row.get(key)
    return "" if v is None else str(v).strip().lower()

def map_to_l1_row(row):
    """将 OSM/多源标签映射到 12 类 L1（若已存在 l1 字段则不使用此函数）"""
    amenity = _get_str(row, "amenity")
    shop    = _get_str(row, "shop")
    tourism = _get_str(row, "tourism")
    leisure = _get_str(row, "leisure")
    office  = _get_str(row, "office")
    pt      = _get_str(row, "public_transport")
    railway = _get_str(row, "railway")
    landuse = _get_str(row, "landuse")
    manmade = _get_str(row, "man_made")
    category= _get_str(row, "category")
    fclass  = _get_str(row, "fclass")

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

# ========== 读取数据 ==========
roads = gpd.read_file(ROADS_PATH)
poi   = gpd.read_file(POI_PATH)

# 准备 road_id
if "road_id" not in roads.columns:
    roads["road_id"] = np.arange(len(roads))

# 记录原始 CRS；统一投影到米制后做空间计算
orig_crs_roads = roads.crs
roads_m = roads.to_crs(epsg=CRS_METRIC)
poi_m   = poi.to_crs(epsg=CRS_METRIC)

# 道路长度
roads_m["length_m"] = roads_m.geometry.length.replace({np.inf: np.nan})
roads_m["length_m"] = roads_m["length_m"].fillna(0.0)

# POI L1（若无则映射）
if "l1" not in poi_m.columns:
    poi_m["l1"] = poi_m.apply(map_to_l1_row, axis=1)

# ========== 1) Frontage（20 m） ==========
buf20 = roads_m[["road_id","geometry"]].copy()
buf20["geometry"] = buf20.buffer(FRONTAGE_DIST)

# 连接：POI 落在 buf20 内
front_join = gpd.sjoin(poi_m[["geometry","l1"]], buf20, predicate="within", how="left")
# 各 L1 计数
front_tab = (front_join.groupby(["road_id","l1"]).size()
             .unstack(fill_value=0))
# 列名：front20_<l1>
front_tab = front_tab.add_prefix("front20_")

# 合并回道路
roads_m = roads_m.merge(front_tab, on="road_id", how="left").fillna(0)

# 总数 & 每100m
roads_m["front20_total"] = roads_m.filter(like="front20_").sum(axis=1)
_ = add_per100m(roads_m, "front20_total", "length_m")

# 各 L1 每100m
for c in roads_m.filter(like="front20_").columns:
    if c == "front20_total":
        continue
    add_per100m(roads_m, c, "length_m")

# ========== 2) 多尺度密度（100/300/800 m） ==========
for R in DENSITY_BUFFERS:
    buf = roads_m[["road_id","geometry"]].copy()
    buf["geometry"] = buf.buffer(R)
    # 每条路缓冲面积（m² → km²）
    buf_area_km2 = (buf.geometry.area / 1e6).rename("area_km2")
    roads_m = roads_m.merge(buf_area_km2, left_on="road_id", right_index=True, how="left", suffixes=("",""))

    # 统计 POI（按 L1）
    jj = gpd.sjoin(poi_m[["geometry","l1"]], buf, predicate="within", how="left").dropna(subset=["road_id"])
    tab = (jj.groupby(["road_id","l1"]).size()
             .unstack(fill_value=0))
    # 列名：dens{R}_{l1}_kmp2
    dens = tab.div(roads_m.set_index("road_id")["area_km2"], axis=0).fillna(0)
    dens.columns = [f"dens{R}_{c}_kmp2" for c in dens.columns]

    # 总密度
    dens[f"dens{R}_total_kmp2"] = dens.sum(axis=1)

    # 合并
    roads_m = roads_m.merge(dens, left_on="road_id", right_index=True, how="left").fillna(0)

    # 清理临时面积列（避免被下一轮覆盖干扰）
    roads_m = roads_m.drop(columns=["area_km2"])

# ========== 3) 最近距离（到关键 L1） ==========
# 优先用 sjoin_nearest；若版本不支持 distance_col，则回退为 unary_union 距离
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
        # 兼容旧版 GeoPandas：没有 distance_col 参数
        union = sub.unary_union
        roads_m[col] = roads_m.geometry.apply(lambda g: g.distance(union) if union else np.nan)

# ========== 输出 ==========
roads_out = roads_m.to_crs(orig_crs_roads)  # 转回原始 CRS
# 类型安全：所有新数值列转为 float
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
print("✅ Saved:", OUT_PATH)
