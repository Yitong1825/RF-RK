# ordinary kriging
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiLineString
from pykrige.ok import OrdinaryKriging

# === 文件路径 ===
csv_path = "osm_id_with_aadt.csv"           # 包含 osm_id, aadt, lat, lon
geojson_path = "roads_split.geojson"     # 包含所有道路的 GeoJSON 文件
output_path = "roads_kriging_result.geojson"

# === 1. 读取 GeoJSON，记录道路 osm_id 和坐标 ===
gdf_all = gpd.read_file(geojson_path)
gdf_all["osm_id"] = gdf_all["osm_id"].astype("Int64")
gdf_all = gdf_all.to_crs(epsg=32647)
print(f"道路总数：{len(gdf_all)}")

# === 2. 读取 CSV，按 osm_id 匹配到 GeoJSON，获取坐标，同时在 gdf_all 中剔除 ===
df_obs = pd.read_csv(csv_path)
df_obs["osm_id"] = df_obs["osm_id"].astype("Int64")

# 匹配部分
gdf_matched = gdf_all[gdf_all["osm_id"].isin(df_obs["osm_id"])].copy()
gdf_matched = gdf_matched.merge(df_obs[["osm_id", "aadt"]], on="osm_id", how="left")
print(f"匹配道路总数：{len(gdf_matched )}")
# 剩余（未观测）部分
gdf_unmatched = gdf_all[~gdf_all["osm_id"].isin(df_obs["osm_id"])].copy()
print(f"未匹配道路总数：{len(gdf_unmatched )}")
# === 3. 提取插值坐标 ===
def get_center(geom):
    if geom.is_empty:
        return None
    if isinstance(geom, MultiLineString):
        geom = max(geom.geoms, key=lambda g: g.length)
    return geom.interpolate(0.5, normalized=True)

gdf_matched["centroid"] = gdf_matched.geometry.apply(get_center)
gdf_unmatched["centroid"] = gdf_unmatched.geometry.apply(get_center)

gdf_matched = gdf_matched[gdf_matched["centroid"].notnull()]
gdf_unmatched = gdf_unmatched[gdf_unmatched["centroid"].notnull()]

# 克里金插值输入输出点
x_known = gdf_matched["centroid"].x.values
y_known = gdf_matched["centroid"].y.values
z_known = gdf_matched["aadt"].astype(float).values
print(x_known[:10])
print(y_known[:10])
print(z_known[:10])
x_unknown = gdf_unmatched["centroid"].x.values
y_unknown = gdf_unmatched["centroid"].y.values
print(x_unknown[:10])
print(y_unknown[:10])
print(f"已知点数量: {len(x_known)}，未知待插值点数量: {len(x_unknown)}")
print("AADT 分布:")
print("最小值：", np.min(z_known), "最大值：", np.max(z_known))
print("唯一值：", np.unique(z_known))

# === 4. 执行克里金插值 ===
if len(x_known) >= 3:
    OK = OrdinaryKriging(
        x_known, y_known, z_known,
        variogram_model="exponential",
        variogram_parameters={"sill": 80000, "range": 800, "nugget": 300},
        verbose=False, enable_plotting=False
    )
    z_pred, _ = OK.execute("points", x_unknown, y_unknown)
    gdf_unmatched["aadt"] = np.round(z_pred).astype(int)
    print("插值完成")
else:
    print("error")

# 合并结果
gdf_result = pd.concat([gdf_matched.drop(columns="centroid"),
                        gdf_unmatched.drop(columns="centroid")], ignore_index=True)
gdf_result["aadt"] = gdf_result["aadt"].astype("Int64")
gdf_result.to_file(output_path, driver="GeoJSON", encoding="utf-8")
print(f"最终结果输出到：{output_path}")
print(z_pred[:10])


original_geojson = geojson_path
kriging_result_geojson = output_path
final_output_geojson = "roads_with_aadt.geojson"

# 读取原始道路数据
gdf_original = gpd.read_file(original_geojson)
gdf_original["osm_id"] = gdf_original["osm_id"].astype("Int64")

# 读取带 AADT 的插值结果
gdf_kriged = gpd.read_file(kriging_result_geojson)
gdf_kriged["osm_id"] = gdf_kriged["osm_id"].astype("Int64")

# 提取 osm_id 和 aadt 字段用于合并
gdf_aadt = gdf_kriged[["osm_id", "aadt"]]

# 合并结果到原始文件中
gdf_merged = gdf_original.merge(gdf_aadt, on="osm_id", how="left")

# 输出新的 GeoJSON
gdf_merged.to_file(final_output_geojson, driver="GeoJSON", encoding="utf-8")
print(f"已将 AADT 添加到原始道路数据中：{final_output_geojson}")