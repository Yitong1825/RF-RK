import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from rapidfuzz import fuzz
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

# =============================== 1. 读取 Excel 数据 ===================================
folder_path = "D:/Kriging/excel_data/"  # ← 替换为你实际的路径
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
df_all = []

for file in excel_files:
    try:
        df = pd.read_excel(file, header=3)

        df["road_name"] = df["Unnamed: 2"].ffill()
        df["latlon"] = df["Unnamed: 13"].ffill()
        df_valid = df[df["แต่ละถนน"].notna()].copy()
        df_valid = df_valid[["road_name", "แต่ละถนน", "latlon"]]
        df_valid.columns = ["road_name", "aadt", "latlon"]

        df_valid[["coords_lat", "coords_lon"]] = df_valid["latlon"].astype(str).str.extract(r"(\d+\.\d+)[,\s]+(\d+\.\d+)")
        df_valid[["coords_lat", "coords_lon"]] = df_valid[["coords_lat", "coords_lon"]].astype(float, errors="ignore")

        df_valid["source_file"] = os.path.basename(file)
        df_all.append(df_valid)

    except Exception as e:
        print(f"error: {file} | problem: {e}")

df_combined = pd.concat(df_all, ignore_index=True)
print("Excel 数据读取完成")

# =============================== 2. 读取 GeoJSON 道路数据 ===================================
gdf_roads = gpd.read_file("road_json.geojson")
gdf_roads = gdf_roads.to_crs(epsg=32647)  # 投影为 UTM Zone 47N

gdf_roads = gdf_roads[gdf_roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
print("道路条数：", len(gdf_roads))
# 字段清洗
def clean_text(text):
    return str(text).strip().lower().replace('\u200b', '')

gdf_roads["name"] = gdf_roads["name"].apply(clean_text)
df_combined["road_name"] = df_combined["road_name"].apply(clean_text)

# =============================== 3. 提取道路端点 ===================================
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
    return []

endpoint_records = []
for idx, row in gdf_roads.iterrows():
    endpoints = extract_endpoints(row.geometry)
    for pt in endpoints:
        endpoint_records.append({
            "name": row["name"],
            "geometry": pt,
            "road_index": idx
        })

if not endpoint_records:
    raise ValueError("没有成功提取任何端点，请检查 geometry 类型是否为 LineString 或 MultiLineString")

gdf_endpoints = gpd.GeoDataFrame(endpoint_records, geometry="geometry", crs=gdf_roads.crs)

# =============================== 4. 转换观测点为 GeoDataFrame ===============================
df_combined["geometry"] = df_combined.apply(
    lambda row: Point(row["coords_lon"], row["coords_lat"]), axis=1
)
gdf_points = gpd.GeoDataFrame(df_combined, geometry="geometry", crs="EPSG:4326")
gdf_points = gdf_points.to_crs(epsg=32647)

# =============================== 5. 匹配：距离+名称包含 ===============================
matched_rows = []
search_radius = 15  # 单位：米

for _, point_row in gdf_points.iterrows():
    point = point_row.geometry
    keyword = point_row["road_name"]

    gdf_endpoints["distance"] = gdf_endpoints.geometry.distance(point)
    nearby = gdf_endpoints[gdf_endpoints["distance"] <= search_radius].copy()

    if nearby.empty:
        continue

    # nearby["match_score"] = nearby["name"].apply(
    #     lambda x: 1 if keyword in x or x in keyword else 0
    # )
    # matches = nearby[nearby["match_score"] > 0]
    # 使用 partial_ratio 模糊匹配，得分范围 0-100
    nearby["match_score"] = nearby["name"].apply(
        lambda x: fuzz.partial_ratio(keyword, x)
    )
    # 设置一个匹配得分阈值（比如 60），根据需要调整
    matches = nearby[nearby["match_score"] >= 60]

    if not matches.empty:
        best = matches.sort_values("distance").iloc[0]
        combined = best.drop(columns=["distance"]).to_dict()
        combined.update(point_row.drop(columns=["geometry"]).to_dict())
        matched_rows.append(combined)

# =============================== 6. 输出匹配结果 ===============================
df_matched_all = pd.DataFrame(matched_rows)
df_matched_all.to_excel("matched_roads_endpoints.xlsx", index=False)
print("匹配完成，结果保存为 matched_roads_endpoints.xlsx")

# =============================== 7. 找出未匹配项 ===============================
df_combined["match_key"] = df_combined["road_name"].astype(str) + "_" + df_combined["coords_lat"].astype(str) + "_" + df_combined["coords_lon"].astype(str)
df_matched_all["match_key"] = df_matched_all["road_name"].astype(str) + "_" + df_matched_all["coords_lat"].astype(str) + "_" + df_matched_all["coords_lon"].astype(str)

df_unmatched = df_combined[~df_combined["match_key"].isin(df_matched_all["match_key"])].copy()
print(f"未匹配记录数：{len(df_unmatched)}")
# print(df_unmatched[["road_name", "latlon", "source_file"]].head())

df_unmatched.to_excel("unmatched_records_endpoints.xlsx", index=False)
print("未匹配记录保存为 unmatched_records_endpoints.xlsx")



# =============================== 8. 克里金插值 ===============================

from shapely.geometry import LineString
# 1. 筛选出有效的观测点（即已匹配并具有数值的点）
gdf_matched = gpd.GeoDataFrame(df_matched_all, geometry=gpd.points_from_xy(df_matched_all["coords_lon"], df_matched_all["coords_lat"]), crs="EPSG:4326")
gdf_matched = gdf_matched.to_crs(epsg=32647)  # 投影为 UTM 47N（用于空间插值）

gdf_valid = gdf_matched[gdf_matched["aadt"].notna()].copy()
gdf_valid["aadt"] = gdf_valid["aadt"].astype(float)

# 2. 提取坐标与值
gdf_valid["x"] = gdf_valid.geometry.x
gdf_valid["y"] = gdf_valid.geometry.y
gdf_valid["z"] = gdf_valid["aadt"]

# 3. 构建 Kriging 模型
OK = OrdinaryKriging(
    gdf_valid["x"],
    gdf_valid["y"],
    gdf_valid["z"],
    variogram_model="linear",  # 可选：'linear', 'spherical', 'gaussian', 'exponential'
    verbose=False,
    enable_plotting=False
)

# 4. 构建插值网格
gridx = np.linspace(gdf_valid["x"].min(), gdf_valid["x"].max(), 200)
gridy = np.linspace(gdf_valid["y"].min(), gdf_valid["y"].max(), 200)
z_interp, ss = OK.execute("grid", gridx, gridy)
# 重新读取道路数据（EPSG:32647 投影后）
# gdf_roads_interp = gdf_roads.copy()
gdf_roads_interp = gdf_roads[gdf_roads.geometry.type == "LineString"].copy()

# 对每条道路采样若干点并进行克里金预测
def interpolate_line_aadt(geom, n_points=20):
    if not isinstance(geom, LineString):
        return np.nan
    sampled_points = [geom.interpolate(float(i) / (n_points - 1), normalized=True) for i in range(n_points)]
    xs = [pt.x for pt in sampled_points]
    ys = [pt.y for pt in sampled_points]
    z_interp, _ = OK.execute("points", xs, ys)
    return float(np.nanmean(z_interp))

print("正在对道路进行 AADT 克里金插值")

# 添加插值字段
gdf_roads_interp["kriged_aadt"] = gdf_roads_interp["geometry"].apply(interpolate_line_aadt)
# 转换为浮点数（非常关键）
gdf_roads_interp["kriged_aadt"] = gdf_roads_interp["kriged_aadt"].astype("float64")
# 导出结果为 shapefile 或 GeoJSON

# gdf_roads_interp.to_file("road_with_kriged_aadt1.gpkg", driver="GPKG", layer="kriged_roads")
gdf_roads_interp.to_file("road_with_kriged_aadt.shp", driver="ESRI Shapefile", index=False)

print(gdf_roads_interp["kriged_aadt"])
print("克里金插值结果已保存为 road_with_kriged_aadt.gpkg")





















# # 5. 可视化插值结果
# plt.figure(figsize=(10, 8))
# plt.contourf(gridx, gridy, z_interp, cmap="viridis", levels=20)
# plt.scatter(gdf_valid["x"], gdf_valid["y"], c=gdf_valid["aadt"], cmap="Reds", edgecolor="k", s=20)
# plt.colorbar(label="Interpolated AADT")
# plt.title("Kriging Interpolation of AADT")
# plt.xlabel("X (UTM)")
# plt.ylabel("Y (UTM)")
# plt.tight_layout()
# plt.savefig("kriging_result.png", dpi=300)
# plt.show()




