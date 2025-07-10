import pandas as pd
import glob
import os
import geopandas as gpd
from rapidfuzz import process, fuzz
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import from_origin

# =============================== 1. Read excel data ===================================
# 设置包含所有 Excel 文件的文件夹路径
folder_path = "D:/Kriging/excel_data/"  # ← 修改为你的路径
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

# 用于收集所有文件处理后的结果
df_all = []

# 遍历每个 Excel 文件
for file in excel_files:
    try:
        df = pd.read_excel(file, header=3)
        # 清洗处理
        df["road_name"] = df["Unnamed: 2"].ffill()
        df["latlon"] = df["Unnamed: 13"].ffill()
        df_valid = df[df["แต่ละถนน"].notna()].copy()
        df_valid = df_valid[["road_name", "แต่ละถนน", "latlon"]]
        df_valid.columns = ["road_name", "aadt", "latlon"]
        df_valid[["coords_lat", "coords_lon"]] = df_valid["latlon"].str.extract(r"(\d+\.\d+)\s+(\d+\.\d+)")
        df_valid[["coords_lat", "coords_lon"]] = df_valid[["coords_lat", "coords_lon"]].astype(float)

        # 加入来源信息（文件名）
        df_valid["source_file"] = os.path.basename(file)

        # 收集结果
        df_all.append(df_valid)

    except Exception as e:
        print(f"error：{file}，problem：{e}")

# 合并所有结果为一个总表
df_combined = pd.concat(df_all, ignore_index=True)
print("reading excel data done")

# 展示前几行结果
########################################################################################
# =============================== 2. Read geojson road data ===================================
gdf_roads = gpd.read_file("road_json.geojson")

########################################################################################
# =============================== 3. Merge data ===================================
# 保证为字符串并清洗空格
gdf_roads["name"] = gdf_roads["name"].astype(str).str.strip().str.lower()
df_combined["road_name"] = df_combined["road_name"].astype(str).str.strip().str.lower()

# 构建匹配结果列表
matched_rows = []

for _, excel_row in df_combined.iterrows():
    keyword = excel_row["road_name"]
    # 包含关系匹配：只要 name 包含 road_name
    matches = gdf_roads[gdf_roads["name"].str.contains(keyword, na=False)]

    for _, geo_row in matches.iterrows():
        combined = geo_row.to_dict()
        combined.update(excel_row.to_dict())
        matched_rows.append(combined)

# 转为 GeoDataFrame
gdf_merged = gpd.GeoDataFrame(matched_rows, geometry="geometry", crs=gdf_roads.crs)
print("merged")
# Excel 中所有唯一的道路名称数量
total_road_names = df_combined["road_name"].nunique()

# 实际成功匹配的道路名称数量（去重）
matched_road_names = gdf_merged["road_name"].nunique()

# 输出匹配情况
print(f"📊 Excel 中道路总数：{total_road_names}")
print(f"✅ 成功匹配的道路数：{matched_road_names}")
print(f"🔎 匹配成功率：{matched_road_names / total_road_names:.2%}")

# print(gdf_merged[["name", "aadt"]])
# # Output lines that are not NaN:
# print(gdf_merged[gdf_merged["aadt"].notna()][["name", "aadt"]])


########################################################################################
# =============================== 3. Kriging ===================================
# Select the rows with AADT data and extract the geometric center points
gdf_valid = gdf_merged[gdf_merged["aadt"].notna()].copy()
gdf_valid = gdf_valid.to_crs(gdf_valid.estimate_utm_crs())

# Take the center point
gdf_valid["geometry"] = gdf_valid.geometry.centroid

# Extract the coordinates for Krigin
gdf_valid["x"] = gdf_valid.geometry.x
gdf_valid["y"] = gdf_valid.geometry.y
gdf_valid["z"] = gdf_valid["aadt"].astype(float)

# 3. Construct the Kriging model
OK = OrdinaryKriging(
    gdf_valid["x"],
    gdf_valid["y"],
    gdf_valid["z"],
    variogram_model="linear",  # 可选：'linear', 'gaussian', 'spherical', 'exponential'
    verbose=False,
    enable_plotting=False
)
print("Kriging done")
# 4. Construct interpolation grids (based on the data range)
grid_x = np.linspace(gdf_valid["x"].min(), gdf_valid["x"].max(), 100)
grid_y = np.linspace(gdf_valid["y"].min(), gdf_valid["y"].max(), 100)
grid_z, _ = OK.execute("grid", grid_x, grid_y)

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.contourf(grid_x, grid_y, grid_z, cmap="viridis")
plt.scatter(gdf_valid["x"], gdf_valid["y"], c=gdf_valid["z"], edgecolor='k', cmap="viridis", s=40)
plt.colorbar(label="Interpolated AADT")
plt.title("Kriging Interpolation of AADT")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()