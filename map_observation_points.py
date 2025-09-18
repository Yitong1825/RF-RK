import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import re
import os

# === path ===
excel_path = "matched_roads_distance_based.xlsx"   # Excel
sheet_name = "Sheet1"
out_folder = "output_shp"
out_name_wgs84 = "matched_points_wgs84.shp"
out_name_utm47n = "matched_points_utm47n.shp"

os.makedirs(out_folder, exist_ok=True)

# === data reading ===
df = pd.read_excel(excel_path, sheet_name=sheet_name)

lon_col, lat_col = None, None
for c in df.columns:
    if c.lower() in ["coords_lon", "lon", "x", "longitude"]:
        lon_col = c
    if c.lower() in ["coords_lat", "lat", "y", "latitude"]:
        lat_col = c

def try_parse_latlon_text(s):
    """解析 'latlon' 形式的字符串，如 '13.745053, 100.523320' -> (lat, lon)"""
    if not isinstance(s, str):
        return None, None
    m = re.match(r"\s*([+-]?\d+(\.\d+)?)\s*,\s*([+-]?\d+(\.\d+)?)\s*$", s)
    if not m:
        return None, None
    lat = float(m.group(1))
    lon = float(m.group(3))
    return lat, lon

if lon_col is not None and lat_col is not None:
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
else:
    if "latlon" not in df.columns:
        raise ValueError("coords can not be found")
    parsed = df["latlon"].apply(try_parse_latlon_text)
    df["_lat"] = parsed.apply(lambda t: t[0])
    df["_lon"] = parsed.apply(lambda t: t[1])

# coords clean
df = df.dropna(subset=["_lon", "_lat"]).copy()

if df.empty:
    raise ValueError("coords column empty")

# === build GeoDataFrame（WGS84）===
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["_lon"], df["_lat"]),
    crs="EPSG:4326"   # WGS84
)

# === 5) 导出为 Shapefile（WGS84）===
out_wgs84_path = os.path.join(out_folder, out_name_wgs84)
gdf.to_file(out_wgs84_path, driver="ESRI Shapefile", encoding="utf-8")
print(f"out put：{out_wgs84_path}")


