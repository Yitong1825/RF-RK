#Read Excel and GeoJSON files, match them, and output a CSV file that labels the OSM IDs and AADT values of the roads with records.
import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from rapidfuzz import fuzz
import unicodedata
import re


# =============================== text clean ===================================
def clean_text(text):
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
    return text.strip().lower()

# =============================== read Excel  ===================================
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
            df_valid[["coords_lat", "coords_lon"]] = (df_valid["latlon"].astype(str)
                                                       .str.extract(r"(\d+\.\d+)[,\s]+(\d+\.\d+)"))
            df_valid[["coords_lat", "coords_lon"]] = (df_valid[["coords_lat", "coords_lon"]]
                                                       .astype(float, errors="ignore"))
            df_valid["source_file"] = os.path.basename(file)
            df_valid["road_name"] = df_valid["road_name"].apply(clean_text)
            df_all.append(df_valid)
        except Exception as e:
            print(f"error: {file} | problem: {e}")
    df_combined = pd.concat(df_all, ignore_index=True)
    df_combined = df_combined[df_combined["coords_lat"].notna() & df_combined["coords_lon"].notna()]
    return df_combined

# =============================== read road(OSM) data ===================================
def load_road_data(geojson_path):
    gdf_roads = gpd.read_file(geojson_path, encoding="utf-8-sig")
    gdf_roads = gdf_roads.to_crs(epsg=32647)
    gdf_roads = gdf_roads[gdf_roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    gdf_roads["name"] = gdf_roads["name"].apply(clean_text)
    gdf_roads = gdf_roads[~gdf_roads.geometry.is_empty & gdf_roads.geometry.notna()]

    if "osm_id" in gdf_roads.columns:
        gdf_roads["osm_id"] = gdf_roads["osm_id"].astype("Int64")
    elif "@id" in gdf_roads.columns:
        gdf_roads["osm_id"] = gdf_roads["@id"].str.extract(r'(\d+)').astype("Int64")
    else:
        print("can't find osm_id")

    return gdf_roads

# =============================== matching function ===================================
def match_points_to_roads(df_combined, gdf_roads, search_radius=50, match_threshold=60):
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
        nearby["match_score"] = nearby["name"].apply(lambda x: fuzz.partial_ratio(keyword, x))
        matches = nearby[nearby["match_score"] >= match_threshold]
        if not matches.empty:
            best = matches.sort_values("distance").iloc[0]
            combined = best.drop(columns=["distance"]).to_dict()
            combined["osm_id"] = best.get("osm_id", None)  # 若无该字段则为 None
            combined.update(point_row.drop(columns=["geometry"]).to_dict())
            matched_rows.append(combined)

    return pd.DataFrame(matched_rows)

# =============================== add attributes and save to new GeoJSON file ===================================
def copy_and_update_geojson(geojson_path, df_matched_all, output_path):
    # 1. 读取原始道路数据
    gdf = gpd.read_file(geojson_path, encoding="utf-8-sig")
    gdf["name"] = gdf["name"].apply(clean_text)

    # 2. 处理匹配数据
    df = df_matched_all.copy()
    df["road_name"] = df["road_name"].apply(clean_text)

    # 3. 提取时间：从 source_file 中提取如 2024-03-18
    df["date"] = df["source_file"].apply(
        lambda x: re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", x).group(0) if pd.notna(x) and re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", x) else None
    )

    # 4. 合并：根据道路名称进行左连接
    gdf["aadt"] = np.nan
    gdf["date"] = None
    gdf = gdf.merge(df[["road_name", "aadt", "date"]], how="left", left_on="name", right_on="road_name")

    # 5. 如果 merge 后存在重复列，清理
    gdf = gdf.drop(columns=["road_name"])

    # 6. 保存到新文件
    gdf.to_file(output_path, driver="GeoJSON", encoding="utf-8")
    print(f"A new file containing the AADT values and dates has been generated：{output_path}")


# =============================== main ===================================
if __name__ == "__main__":
    folder_path = "D:/Kriging/excel_data/"
    geojson_path = "road_json.geojson"

    df_combined = load_excel_data(folder_path)
    print("Excel data reading completed.")

    gdf_roads = load_road_data(geojson_path)
    print(f"Data has been read successfully. There are a total of {len(gdf_roads)} records.")

    df_matched_all = match_points_to_roads(df_combined, gdf_roads)
    df_matched_all.to_excel("matched_roads_distance_based.xlsx", index=False)
    print("Match completed. Result saved as matched_roads_distance_based.xlsx")

    df_combined["match_key"] = df_combined["road_name"].astype(str) + "_" + df_combined["coords_lat"].astype(str) + "_" + df_combined["coords_lon"].astype(str)
    df_matched_all["match_key"] = df_matched_all["road_name"].astype(str) + "_" + df_matched_all["coords_lat"].astype(str) + "_" + df_matched_all["coords_lon"].astype(str)
    df_unmatched = df_combined[~df_combined["match_key"].isin(df_matched_all["match_key"])].copy()
    df_unmatched.to_excel("unmatched_records_distance_based.xlsx", index=False)
    print(f"Unmatched records：{len(df_unmatched)}，已保存 unmatched_records_distance_based.xlsx")


# 导出用于插值的 CSV：osm_id + aadt
if "osm_id" in df_matched_all.columns:
    df_matched_all_export = df_matched_all[["osm_id", "aadt"]].dropna(subset=["osm_id", "aadt"])
    df_matched_all_export.to_csv("osm_id_with_aadt.csv", index=False)
    print("File osm_id_with_aadt.csv exported")
else:
    print("error: can not export osm_id_with_aadt.csv")
