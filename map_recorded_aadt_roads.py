import geopandas as gpd
import pandas as pd

# === path ===
geojson_path = "roads_split.geojson"
csv_path = "osm_id_with_aadt.csv"
output_geojson_path = "matched_observed_roads.geojson"

gdf = gpd.read_file(geojson_path)
df = pd.read_csv(csv_path)

gdf["osm_id"] = gdf["osm_id"].astype("Int64")
df["osm_id"] = df["osm_id"].astype("Int64")

# === match ===
gdf_matched = gdf.merge(df, on="osm_id", how="inner")

gdf_matched.to_file(output_geojson_path, driver="GeoJSON", encoding="utf-8")

print(f"matched roads number：{len(gdf_matched)}")
print(f"out put path：{output_geojson_path}")
