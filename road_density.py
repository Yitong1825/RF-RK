# add_road_density_circle_noargs.py
# ------------------------------------------------------------
# 在原始道路数据中添加“道路密度”（km/km²）
# 定义：以每条道路的“质心”为中心，半径 R 的圆形邻域内，道路总长度 / 圆面积
# 依赖：geopandas shapely numpy pandas
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

# ======= 手动参数（按需修改） =======
IN_ROADS    = "roads_with_poi_feats.geojson"    # 输入道路文件
OUT_ROADS   = "roads_with_density.geojson"  # 输出文件
CRS_METRIC  = 32647       # 米制坐标系（曼谷 UTM 47N）。换地区请改为当地合适EPSG
RADIUS_M    = 300.0       # 圆形邻域半径（米）
INCLUDE_SELF = True       # 是否把自身道路长度计入邻域总长度
# ===================================

# 1) 读取与投影到米制
roads = gpd.read_file(IN_ROADS)
if roads.crs is None:
    raise ValueError("输入道路缺少 CRS。请先在 GIS 中设置正确坐标系。")
orig_crs = roads.crs
roads = roads[roads.geometry.notna()].copy()

roads_m = roads.to_crs(epsg=CRS_METRIC)

# 仅保留线要素并拆分 MultiLine 为单线
roads_m = roads_m[roads_m.geometry.type.isin(["LineString", "MultiLineString"])].copy()
try:
    roads_m = roads_m.explode(index_parts=False).reset_index(drop=True)
except TypeError:
    roads_m = roads_m.explode(ignore_index=True)

# 每段线唯一ID与长度
roads_m["seg_id"] = np.arange(len(roads_m))
roads_m["seg_len_m"] = roads_m.geometry.length

# 2) 以质心为中心构建圆形邻域
centers = roads_m[["seg_id", "geometry"]].copy()
centers["geometry"] = centers.geometry.centroid
buffers = centers.copy()
buffers["geometry"] = buffers.geometry.buffer(RADIUS_M)

# 3) 空间连接：道路线段 ∩ 每条路的圆形邻域
left  = roads_m[["seg_id", "geometry"]].copy()
right = buffers[["seg_id", "geometry"]].rename(columns={"seg_id": "buf_id"})  # 保持活动几何列名为 'geometry'

pairs = gpd.sjoin(left, right, how="inner", predicate="intersects")

# 兼容不同版本的列名（有的会生成 *_left/_right 或 index_right）
if "seg_id_left" in pairs.columns:
    pairs = pairs.rename(columns={"seg_id_left": "seg_id"})
if "buf_id_right" in pairs.columns:
    pairs = pairs.rename(columns={"buf_id_right": "buf_id"})
if "buf_id" not in pairs.columns and "index_right" in pairs.columns:
    pairs = pairs.merge(
        right.reset_index()[["index", "buf_id"]].rename(columns={"index": "index_right"}),
        on="index_right", how="left"
    )

# 把邻域几何并回（命名为 buf_geom）用于裁剪长度
pairs = pairs.merge(
    buffers[["seg_id", "geometry"]].rename(columns={"seg_id": "buf_id", "geometry": "buf_geom"}),
    on="buf_id", how="left"
)

# 4) 计算线段与圆形邻域的相交长度
def clip_len(row):
    try:
        return row["geometry"].intersection(row["buf_geom"]).length
    except Exception:
        return 0.0

pairs["clip_len_m"] = pairs.apply(clip_len, axis=1)

# 可选：去掉自身
if not INCLUDE_SELF:
    pairs.loc[pairs["seg_id"] == pairs["buf_id"], "clip_len_m"] = 0.0

# 5) 按 buf_id（每条路的圆）汇总：总长度（米）
sum_len_m = pairs.groupby("buf_id", as_index=True)["clip_len_m"].sum()

# 6) 密度 = (圆内总长度 km) / (圆面积 km²)
area_km2 = (np.pi * (RADIUS_M**2)) / 1e6
density_km_per_km2 = (sum_len_m / 1000.0) / area_km2  # km / km²

# 7) 写回道路表
roads_m["road_length_in_R_m"] = roads_m["seg_id"].map(sum_len_m).fillna(0.0)
roads_m[f"road_density_R{int(RADIUS_M)}_km_per_km2"] = (
    roads_m["seg_id"].map(density_km_per_km2).fillna(0.0).astype(float)
)

# 同时给一个通用列名（方便后续脚本直接用）
roads_m["road_density"] = roads_m[f"road_density_R{int(RADIUS_M)}_km_per_km2"]

# 8) 保存（转回原始CRS）
out = roads_m.to_crs(orig_crs)
out.to_file(OUT_ROADS)
print(f"✅ 已输出：{OUT_ROADS}")
print(f"   新增字段：road_length_in_R_m, road_density_R{int(RADIUS_M)}_km_per_km2, road_density")
print(f"   参数：R={RADIUS_M} m, include_self={INCLUDE_SELF}, EPSG={CRS_METRIC}")
