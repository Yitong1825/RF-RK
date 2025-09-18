# # roads_pop_buffer_mean.py
# import os
# import numpy as np
# import geopandas as gpd
# from shapely.geometry import mapping
# import rasterio
# from rasterio.mask import mask
#
# # Khwaeng(subdistricts).geojson
# # ========== 参数 ==========
# ROADS_PATH = "roads_2k.geojson"   # 道路 LineString
# RASTER_PATH = "pop.tif"      # 人口密度 TIFF
# OUT_PATH   = "roads_pop.geojson"                         # 结果输出
# # assign_population_by_zone_then_roads.py
# assign_population_by_zone_then_roads_integrated.py
# -------------------------------------------------
# 依赖: geopandas, shapely, rasterio, numpy
# 用法: 修改最上面的路径常量后直接运行

import numpy as np
import geopandas as gpd
from shapely.geometry import mapping, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask

# ========= 路径（请按需替换） =========
RASTER_PATH = "pop.tif"         # 人口密度栅格 (tif)
ZONES_PATH  = "Khwaeng(subdistricts).geojson"     # 分区边界 (Polygon/MultiPolygon)
ROADS_PATH  = "roads_2k.geojson"     # 道路 (LineString/MultiLineString)
OUT_PATH    = "roads_with_population.geojson"                    # 输出

# ========= 参数 =========
METRIC_EPSG = 32647                # 用于长度计算的米制投影（曼谷 UTM 47N）
POP_COL     = "population"         # 输出字段名
ADD_ROAD_ID = True                 # 如无 road_id 字段则自动创建

# ========= 几何清洗工具 =========
def keep_polygonal_part(geom):
    """保留几何中的多边形成分；若无多边形则返回 None"""
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt in ("Polygon", "MultiPolygon"):
        return geom
    if hasattr(geom, "geoms"):  # GeometryCollection 等
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        return unary_union(polys)
    return None

def to_singlepart_polygons(gdf):
    """将 Polygon/MultiPolygon 统一为单一 Polygon（explode），并修复自相交"""
    g = gdf.copy()
    g["geometry"] = g.geometry.apply(keep_polygonal_part)
    g = g[~g.geometry.isna() & ~g.geometry.is_empty].copy()
    # 仅对面数据用 buffer(0) 修复
    g["geometry"] = g.geometry.buffer(0)
    g = g.explode(index_parts=False, ignore_index=True)
    g = g[g.geometry.type.eq("Polygon")].copy()
    return g

def to_singlepart_lines(gdf):
    """将 LineString/MultiLineString 统一为单一 LineString"""
    g = gdf.copy()
    g = g[g.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    g = g.explode(index_parts=False, ignore_index=True)
    g = g[g.geometry.type.eq("LineString")].copy()
    return g

# ========= 1) 读入数据 =========
zones = gpd.read_file(ZONES_PATH)
roads = gpd.read_file(ROADS_PATH)

if ADD_ROAD_ID and "road_id" not in roads.columns:
    roads["road_id"] = np.arange(len(roads))

orig_crs_roads = roads.crs
orig_crs_zones = zones.crs

# ========= 2) 将分区 + 道路临时转到“栅格 CRS”（不重投影栅格本身） =========
with rasterio.open(RASTER_PATH) as src:
    raster_crs = src.crs

zones_r = zones.to_crs(raster_crs)
roads_r = roads.to_crs(raster_crs)

# ========= 3) 给每个分区赋“人口密度” =========
#   逻辑：
#   - 先用 representative_point() 采样一个像元值
#   - 若值为 NoData，则回退为分区内像元均值；仍无值则置 0
def sample_raster_at_point(src, x, y, nodata_val):
    val = list(src.sample([(x, y)]))[0][0]
    if nodata_val is not None and val == nodata_val:
        return None
    if np.isnan(val):
        return None
    return float(val)

def mean_in_polygon(src, poly, nodata_val):
    try:
        out, _ = mask(src, [mapping(poly)], crop=True, nodata=nodata_val)
        arr = out[0].astype("float64")
        if nodata_val is not None:
            valid = arr[arr != nodata_val]
        else:
            valid = arr[~np.isnan(arr)]
        return float(np.nanmean(valid)) if valid.size else None
    except Exception:
        return None

with rasterio.open(RASTER_PATH) as src:
    nodata = src.nodata
    zone_vals = []
    for geom in zones_r.geometry:
        if geom is None or geom.is_empty:
            zone_vals.append(0.0); continue
        p = geom.representative_point()
        v = sample_raster_at_point(src, p.x, p.y, nodata)
        if v is None:
            v = mean_in_polygon(src, geom, nodata)
            if v is None:
                v = 0.0
        zone_vals.append(v)

zones_r[POP_COL] = np.array(zone_vals, dtype=float)

# ========= 4) 清洗几何类型，避免 overlay 报“混合几何” =========
zones_r_slim = to_singlepart_polygons(zones_r[[POP_COL, "geometry"]])
roads_r_single = to_singlepart_lines(roads_r[["road_id", "geometry"]] if "road_id" in roads_r.columns else roads_r[["geometry"]])

# ========= 5) 道路 × 分区 叠加，得到道路分段继承分区 population =========
segments = gpd.overlay(
    roads_r_single,
    zones_r_slim,
    how="intersection",
    keep_geom_type=True
)

# 若没有任何相交（极端情况），直接置 0 输出
if len(segments) == 0:
    roads_out = roads.copy()
    roads_out[POP_COL] = 0.0
    roads_out.to_file(OUT_PATH, driver="GeoJSON")
    print("✅ 无相交；已输出所有道路 population=0:", OUT_PATH)
else:
    # ========= 6) 在米制投影下按“分段长度”做加权均值 =========
    segments_m = segments.to_crs(epsg=METRIC_EPSG)
    segments_m["len_m"] = segments_m.geometry.length

    # 某些环境下 overlay 结果可能混入 Point/Polygon（罕见），过滤掉
    segments_m = segments_m[segments_m.geometry.type.eq("LineString")].copy()

    # 路段缺 road_id 的情况（很少见）：为其生成临时 ID
    if "road_id" not in segments_m.columns:
        segments_m["road_id"] = np.arange(len(segments_m))

    grp = segments_m.groupby("road_id", dropna=False)
    wmean = (grp.apply(lambda df: (df[POP_COL] * df["len_m"]).sum() / df["len_m"].sum())
             .rename(POP_COL)
             .astype(float))

    # ========= 7) 合并回原始道路；未相交的置 0 =========
    roads_out = roads.copy()
    roads_out = roads_out.merge(wmean.to_frame(), on="road_id", how="left")
    roads_out[POP_COL] = roads_out[POP_COL].fillna(0.0).astype(float)

    # ========= 8) 保存（转回原始道路 CRS） =========
    roads_out = roads_out.to_crs(orig_crs_roads)
    roads_out.to_file(OUT_PATH, driver="GeoJSON")
    print("✅ 完成：道路人口密度已按分区长度占比赋值 →", OUT_PATH)
