# 该文件建立于7月10日，为了解决道路端点匹配不上的问题：使用道路整体几何匹配，进行了道路名标准化/正则化处理-但没有提升
import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from rapidfuzz import fuzz
import unicodedata
import re

# =============================== 通用清洗函数 ===================================
def clean_text(text):
    # 标准化字符串（统一格式、去不可见字符、转小写）
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
    return text.strip().lower()

# =============================== 读取 Excel 数据函数 ===================================
def load_excel_data(folder_path):
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
            df_valid["road_name"] = df_valid["road_name"].apply(clean_text)
            df_all.append(df_valid)
        except Exception as e:
            print(f"error: {file} | problem: {e}")
    df_combined = pd.concat(df_all, ignore_index=True)
    df_combined = df_combined[df_combined["coords_lat"].notna() & df_combined["coords_lon"].notna()]
    return df_combined

# =============================== 读取道路数据函数 ===================================
def load_road_data(geojson_path):
    gdf_roads = gpd.read_file(geojson_path, encoding="utf-8-sig")
    gdf_roads = gdf_roads.to_crs(epsg=32647)
    gdf_roads = gdf_roads[gdf_roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    gdf_roads["name"] = gdf_roads["name"].apply(clean_text)
    gdf_roads = gdf_roads[~gdf_roads.geometry.is_empty & gdf_roads.geometry.notna()]
    return gdf_roads

# =============================== 执行匹配函数 ===================================
def match_points_to_roads(df_combined, gdf_roads, search_radius=50, match_threshold=60):
    # 使用点到道路距离 + 名称模糊匹配逻辑，返回匹配结果 DataFrame
    df_combined["geometry"] = df_combined.apply(lambda row: Point(row["coords_lon"], row["coords_lat"]), axis=1)
    gdf_points = gpd.GeoDataFrame(df_combined, geometry="geometry", crs="EPSG:4326")
    gdf_points = gdf_points.to_crs(epsg=32647)
    gdf_points = gdf_points[~gdf_points.geometry.is_empty & gdf_points.geometry.notna()]

    matched_rows = []
    for _, point_row in gdf_points.iterrows():
        point = point_row.geometry
        keyword = point_row["road_name"]
        gdf_roads["distance"] = gdf_roads.geometry.distance(point)
        nearby = gdf_roads[gdf_roads["distance"] <= search_radius].copy()
        if nearby.empty:
            continue
        # 使用 rapidfuzz 模糊匹配得分
        nearby["match_score"] = nearby["name"].apply(lambda x: fuzz.partial_ratio(keyword, x))
        matches = nearby[nearby["match_score"] >= match_threshold]
        if not matches.empty:
            best = matches.sort_values("distance").iloc[0]
            combined = best.drop(columns=["distance"]).to_dict()
            combined.update(point_row.drop(columns=["geometry"]).to_dict())
            matched_rows.append(combined)
    return pd.DataFrame(matched_rows)

# =============================== 主函数逻辑 ===================================
if __name__ == "__main__":
    folder_path = "D:/Kriging/excel_data/"
    geojson_path = "road_json.geojson"

    df_combined = load_excel_data(folder_path)
    print("Excel 数据读取完成")

    gdf_roads = load_road_data(geojson_path)
    print("道路数据读取完成，共 {} 条".format(len(gdf_roads)))

    df_matched_all = match_points_to_roads(df_combined, gdf_roads)
    df_matched_all.to_excel("matched_roads_distance_based.xlsx", index=False)
    print("匹配完成，结果保存为 matched_roads_distance_based.xlsx")

    # 处理未匹配数据
    df_combined["match_key"] = df_combined["road_name"].astype(str) + "_" + df_combined["coords_lat"].astype(str) + "_" + df_combined["coords_lon"].astype(str)
    df_matched_all["match_key"] = df_matched_all["road_name"].astype(str) + "_" + df_matched_all["coords_lat"].astype(str) + "_" + df_matched_all["coords_lon"].astype(str)
    df_unmatched = df_combined[~df_combined["match_key"].isin(df_matched_all["match_key"])].copy()
    df_unmatched.to_excel("unmatched_records_distance_based.xlsx", index=False)
    print(f"未匹配记录数：{len(df_unmatched)} ，已保存 unmatched_records_distance_based.xlsx")
