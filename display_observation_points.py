import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import re
import os

# === 1) 输入与输出路径 ===
excel_path = "matched_roads_distance_based.xlsx"   # 你的Excel文件名
sheet_name = "Sheet1"                                      # 工作表名（本文件为Sheet1）
out_folder = "output_shp"
out_name_wgs84 = "matched_points_wgs84.shp"                # WGS84经纬度
out_name_utm47n = "matched_points_utm47n.shp"              # 可选：投影后

os.makedirs(out_folder, exist_ok=True)

# === 2) 读取Excel ===
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# === 3) 提取坐标 ===
# 优先使用 coords_lon / coords_lat
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
    # 直接用数值列
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
else:
    # 回退：从 'latlon' 文本列解析
    if "latlon" not in df.columns:
        raise ValueError("找不到坐标列（如 coords_lon/coords_lat 或 latlon）。请检查Excel列名。")
    parsed = df["latlon"].apply(try_parse_latlon_text)
    df["_lat"] = parsed.apply(lambda t: t[0])
    df["_lon"] = parsed.apply(lambda t: t[1])

# 清理非法坐标
df = df.dropna(subset=["_lon", "_lat"]).copy()

# （可选）过滤经纬度范围，避免脏数据
df = df[(df["_lon"] >= -180) & (df["_lon"] <= 180) & (df["_lat"] >= -90) & (df["_lat"] <= 90)].copy()

if df.empty:
    raise ValueError("坐标为空或无有效行，请检查坐标列内容。")

# === 4) 构建GeoDataFrame（WGS84）===
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["_lon"], df["_lat"]),
    crs="EPSG:4326"   # WGS84 经纬度
)

# === 5) 导出为 Shapefile（WGS84）===
out_wgs84_path = os.path.join(out_folder, out_name_wgs84)
gdf.to_file(out_wgs84_path, driver="ESRI Shapefile", encoding="utf-8")
print(f"[OK] 已导出WGS84点图层：{out_wgs84_path}")

# # === 6) （可选）投影到 UTM 47N（EPSG:32647），更适合曼谷地区度量 ===
# gdf_utm = gdf.to_crs(epsg=32647)
# out_utm_path = os.path.join(out_folder, out_name_utm47n)
# gdf_utm.to_file(out_utm_path, driver="ESRI Shapefile", encoding="utf-8")
# print(f"[OK] 已导出UTM 47N点图层：{out_utm_path}")

# === 7) 小贴士 ===
# - QGIS 中：图层 -> 添加图层 -> 添加矢量图层，选择上述 .shp 文件即可。
# - 字段名长度：Shapefile字段名最多10字符，超出会被截断；如需完整字段名，建议导出为 GeoPackage (.gpkg)：
#   gdf.to_file('matched_points.gpkg', layer='points', driver='GPKG')

