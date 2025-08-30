import geopandas as gpd
import pandas as pd

# === 输入路径 ===
geojson_path = "roads_split.geojson"
csv_path = "osm_id_with_aadt.csv"
output_geojson_path = "matched_observed_roads.geojson"

# === 读取数据 ===
gdf = gpd.read_file(geojson_path)
df = pd.read_csv(csv_path)

# 确保 osm_id 数据类型一致
gdf["osm_id"] = gdf["osm_id"].astype("Int64")
df["osm_id"] = df["osm_id"].astype("Int64")

# === 匹配 ===
gdf_matched = gdf.merge(df, on="osm_id", how="inner")

# === 输出 ===
gdf_matched.to_file(output_geojson_path, driver="GeoJSON", encoding="utf-8")

print(f"匹配成功道路数：{len(gdf_matched)}")
print(f"输出地图已保存至：{output_geojson_path}")
