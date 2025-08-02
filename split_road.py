# The road, which is stored as "multilinestring", needs to be split.
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString

# === 1. 加载数据 ===
gdf_roads = gpd.read_file("road_json.geojson")
df_obs = pd.read_csv("osm_id_with_aadt.csv")
df_obs["osm_id"] = df_obs["osm_id"].astype("Int64")
gdf_roads["osm_id"] = gdf_roads["osm_id"].astype("Int64")

# === 2. 拆分 MultiLineString ===
exploded = []
for _, row in gdf_roads.iterrows():
    geom = row.geometry
    if isinstance(geom, LineString):
        exploded.append(row)
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            new_row = row.copy()
            new_row.geometry = part
            exploded.append(new_row)

gdf_split = gpd.GeoDataFrame(exploded, crs=gdf_roads.crs).reset_index(drop=True)

# === 3. 检测每个 osm_id 出现次数 ===
osm_id_counts = gdf_split["osm_id"].value_counts()

# === 4. 标记 CSV 中哪些 osm_id 被拆分了 ===
df_obs["is_split"] = df_obs["osm_id"].apply(lambda x: osm_id_counts.get(x, 0) > 1)

# === 5. 输出检查结果 ===
split_count = df_obs["is_split"].sum()
total = len(df_obs)
print(f"在 CSV 中，有 {split_count} / {total} 条道路在 GeoJSON 中被拆分成了多个段。")

# 构建新的 GeoDataFrame
gdf_exploded = gpd.GeoDataFrame(exploded, crs=gdf_roads.crs)

# 可选：为每一段新建唯一 ID（如 original_id + segment_no）
gdf_exploded = gdf_exploded.reset_index(drop=True)
gdf_exploded["segment_id"] = gdf_exploded.index + 1

# 保存结果
gdf_exploded.to_file("roads_split.geojson", driver="GeoJSON")
print("道路已拆分为独立段落，保存在 roads_split_segments.geojson")