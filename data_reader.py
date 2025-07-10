import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree
from rapidfuzz import fuzz
from shapely.geometry import Point

# =============================== 1. Read excel data ===================================
folder_path = "D:/Kriging/excel_data/"  # ← 修改为你的路径
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
df_all = []

for file in excel_files:
    try:
        df = pd.read_excel(file, header=3)

        # 清洗处理
        df["road_name"] = df["Unnamed: 2"].ffill()
        df["latlon"] = df["Unnamed: 13"].ffill()
        df_valid = df[df["แต่ละถนน"].notna()].copy()
        df_valid = df_valid[["road_name", "แต่ละถนน", "latlon"]]
        df_valid.columns = ["road_name", "aadt", "latlon"]
        # df_valid[["coords_lat", "coords_lon"]] = df_valid["latlon"].str.extract(r"(\d+\.\d+)\s+(\d+\.\d+)")
        # df_valid[["coords_lat", "coords_lon"]] = df_valid[["coords_lat", "coords_lon"]].astype(float)

        # 将 latlon 列转换为字符串并用正则提取经纬度（兼容逗号或空格分隔）
        df_valid[["coords_lat", "coords_lon"]] = df_valid["latlon"].astype(str).str.extract(r"(\d+\.\d+)[,\s]+(\d+\.\d+)")

        # 转换为浮点数（添加 errors='ignore' 可避免因异常值报错）
        df_valid[["coords_lat", "coords_lon"]] = df_valid[["coords_lat", "coords_lon"]].astype(float, errors="ignore")

        df_missing_coords = df_valid[df_valid[["coords_lat", "coords_lon"]].isna().any(axis=1)].copy()
        df_missing_coords.to_excel("rows_missing_coords.xlsx", index=False)
        print("🚫 已导出缺坐标行：rows_missing_coords.xlsx")

        df_valid["source_file"] = os.path.basename(file)
        df_all.append(df_valid)

    except Exception as e:
        print(f"error：{file}，problem：{e}")

df_combined = pd.concat(df_all, ignore_index=True)
print("✅ Reading excel data done.")

# =============================== 2. Read geojson road data ===================================
gdf_roads = gpd.read_file("road_json.geojson")

# 清洗字段（统一小写、去空格）
gdf_roads["name"] = gdf_roads["name"].astype(str).str.strip().str.lower()
df_combined["road_name"] = df_combined["road_name"].astype(str).str.strip().str.lower()

# =============================== 3. Build BallTree based on geo coords ==========================
# 使用道路中点
# gdf_roads["geometry"] = gdf_roads.geometry.centroid
gdf_roads = gdf_roads.to_crs(epsg=32647)
gdf_roads["geometry"] = gdf_roads.geometry.centroid

gdf_roads["lat_rad"] = np.radians(gdf_roads.geometry.y)
gdf_roads["lon_rad"] = np.radians(gdf_roads.geometry.x)

# 转换坐标为弧度
df_combined["lat_rad"] = np.radians(df_combined["coords_lat"])
df_combined["lon_rad"] = np.radians(df_combined["coords_lon"])

# 构建 BallTree
tree = BallTree(np.c_[gdf_roads["lat_rad"], gdf_roads["lon_rad"]], metric="haversine")

# =============================== 4. 距离+名称匹配逻辑 ==========================
# 投影为平面坐标系以保证距离计算正确
from shapely.geometry import Point

from shapely.geometry import LineString, MultiLineString

# 如果是 MultiLineString，则提取每段 LineString 的端点
def extract_endpoints(geom):
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        return [Point(coords[0]), Point(coords[-1])]
    elif isinstance(geom, MultiLineString):
        points = []
        for line in geom.geoms:
            coords = list(line.coords)
            points.append(Point(coords[0]))
            points.append(Point(coords[-1]))
        return points
    else:
        return []

# 构造 endpoint_records
endpoint_records = []

for idx, row in gdf_roads.iterrows():
    endpoints = extract_endpoints(row.geometry)
    for pt in endpoints:
        endpoint_records.append({
            "name": row["name"],
            "geometry": pt,
            "road_index": idx
        })
print(gdf_roads.shape)
print(gdf_roads.geometry.type.value_counts())

# 转为 GeoDataFrame
# 构建端点数据 DataFrame
df_endpoints = pd.DataFrame(endpoint_records)

# 检查是否包含 'geometry' 列
if "geometry" not in df_endpoints.columns:

    raise ValueError("❌ 'geometry' 列未成功创建，请检查 endpoint_records 内容！")

# 构建 GeoDataFrame
gdf_endpoints = gpd.GeoDataFrame(df_endpoints, geometry="geometry", crs="EPSG:4326")
gdf_endpoints = gdf_endpoints.to_crs(epsg=32647)


# ====== 2. 将观测点也转换为 GeoDataFrame，投影 CRS 相同 ======
df_combined["geometry"] = df_combined.apply(
    lambda row: Point(row["coords_lon"], row["coords_lat"]), axis=1
)
gdf_points = gpd.GeoDataFrame(df_combined, geometry="geometry", crs="EPSG:4326")
gdf_points = gdf_points.to_crs(epsg=32647)

# ====== 3. 匹配逻辑：找 500 米内端点 + 名称包含关系 ======
matched_rows = []
search_radius = 500  # 单位：米

for _, point_row in gdf_points.iterrows():
    point = point_row.geometry
    keyword = point_row["road_name"]

    # 查找 500 米内的端点
    gdf_endpoints["distance"] = gdf_endpoints.geometry.distance(point)
    nearby = gdf_endpoints[gdf_endpoints["distance"] <= search_radius].copy()

    if nearby.empty:
        continue

    # 名称包含匹配
    nearby["match_score"] = nearby["name"].apply(
        lambda x: 1 if keyword in x or x in keyword else 0
    )
    matches = nearby[nearby["match_score"] > 0]

    if not matches.empty:
        best = matches.sort_values("distance").iloc[0]
        combined = best.drop(columns=["distance"]).to_dict()
        combined.update(point_row.drop("geometry").to_dict())
        matched_rows.append(combined)



# =============================== 5. 输出结果 ==========================
df_matched_all = pd.DataFrame(matched_rows)
df_matched_all.to_excel("matched_roads_balltree.xlsx", index=False)
print("✅ 匹配完成，结果保存为 matched_roads_balltree.xlsx")

# =============================== 6. 找出未匹配的数据条 ==========================
# 创建唯一标识：road_name + lat + lon
df_combined["match_key"] = df_combined["road_name"].astype(str) + "_" + df_combined["coords_lat"].astype(str) + "_" + df_combined["coords_lon"].astype(str)
df_matched_all["match_key"] = df_matched_all["road_name"].astype(str) + "_" + df_matched_all["coords_lat"].astype(str) + "_" + df_matched_all["coords_lon"].astype(str)

# 找出未匹配项
unmatched_mask = ~df_combined["match_key"].isin(df_matched_all["match_key"])
df_unmatched = df_combined[unmatched_mask].copy()

# 输出数量与示例
print(f"🚫 未匹配记录数：{len(df_unmatched)}")
print(df_unmatched[["road_name", "latlon", "source_file"]].head())

# 可选：保存为 Excel 文件
df_unmatched.to_excel("unmatched_records_balltree.xlsx", index=False)
print("✅ 未匹配数据已保存为 unmatched_records_balltree.xlsx")
