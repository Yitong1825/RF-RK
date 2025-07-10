import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from rapidfuzz import fuzz
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
        print(f"❌ error: {file} | problem: {e}")

df_combined = pd.concat(df_all, ignore_index=True)
print("✅ Excel 数据读取完成")

# =============================== 2. 读取 GeoJSON 道路数据 ===================================
gdf_roads = gpd.read_file("road_json.geojson")
gdf_roads = gdf_roads.to_crs(epsg=32647)  # 投影为 UTM Zone 47N

# 字段清洗
gdf_roads["name"] = gdf_roads["name"].astype(str).str.strip().str.lower()
df_combined["road_name"] = df_combined["road_name"].astype(str).str.strip().str.lower()

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
    raise ValueError("❌ 没有成功提取任何端点，请检查 geometry 类型是否为 LineString 或 MultiLineString")

gdf_endpoints = gpd.GeoDataFrame(endpoint_records, geometry="geometry", crs=gdf_roads.crs)

# =============================== 4. 转换观测点为 GeoDataFrame ===============================
df_combined["geometry"] = df_combined.apply(
    lambda row: Point(row["coords_lon"], row["coords_lat"]), axis=1
)
gdf_points = gpd.GeoDataFrame(df_combined, geometry="geometry", crs="EPSG:4326")
gdf_points = gdf_points.to_crs(epsg=32647)

# =============================== 5. 匹配：距离+名称包含 ===============================
matched_rows = []
search_radius = 1000  # 单位：米

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
print("✅ 匹配完成，结果保存为 matched_roads_endpoints.xlsx")

# =============================== 7. 找出未匹配项 ===============================
df_combined["match_key"] = df_combined["road_name"].astype(str) + "_" + df_combined["coords_lat"].astype(str) + "_" + df_combined["coords_lon"].astype(str)
df_matched_all["match_key"] = df_matched_all["road_name"].astype(str) + "_" + df_matched_all["coords_lat"].astype(str) + "_" + df_matched_all["coords_lon"].astype(str)

df_unmatched = df_combined[~df_combined["match_key"].isin(df_matched_all["match_key"])].copy()
print(f"🚫 未匹配记录数：{len(df_unmatched)}")
# print(df_unmatched[["road_name", "latlon", "source_file"]].head())

df_unmatched.to_excel("unmatched_records_endpoints.xlsx", index=False)
print("✅ 未匹配记录保存为 unmatched_records_endpoints.xlsx")

