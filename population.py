# roads_pop_buffer_mean.py
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask

# Khwaeng(subdistricts).geojson
# ========== 参数 ==========
ROADS_PATH = "64b74998-fbdc-40e6-989b-fe2e4901add6.geojson"   # 道路 LineString
RASTER_PATH = "dff2dc6b-ac1d-417e-8f67-2db6273d7fca.tif"      # 人口密度 TIFF
OUT_PATH   = "roads_with_pop.geojson"                         # 结果输出
BUFFER_M   = 30                                               # 缓冲半径（米）
METRIC_EPSG = 32647                                           # 曼谷 UTM 47N

# ========== 读取道路并准备唯一ID ==========
roads = gpd.read_file(ROADS_PATH)
if "road_id" not in roads.columns:
    roads["road_id"] = np.arange(len(roads))

# ========== 在米坐标系里缓冲 ==========
roads_metric = roads.to_crs(epsg=METRIC_EPSG)
roads_buffer = roads_metric.copy()
roads_buffer["geometry"] = roads_metric.geometry.buffer(BUFFER_M)  # 生成面

# ========== 计算栅格统计 ==========
def zonal_mean_with_rasterio(poly_gdf, tif_path):
    """不依赖 rasterstats，直接用 rasterio.mask 计算均值"""
    means = []
    with rasterio.open(tif_path) as r:
        # 将缓冲面投影到栅格 CRS
        poly_in_raster_crs = poly_gdf.to_crs(r.crs)
        nodata = r.nodata
        for geom in poly_in_raster_crs.geometry:
            if geom is None or geom.is_empty:
                means.append(np.nan); continue
            try:
                out, _ = mask(r, [mapping(geom)], crop=True, nodata=nodata)
                arr = out[0].astype("float64")
                if nodata is not None:
                    arr = arr[arr != nodata]
                else:
                    arr = arr[~np.isnan(arr)]
                means.append(float(np.nanmean(arr)) if arr.size else np.nan)
            except Exception:
                means.append(np.nan)
    return means

try:
    # 优先用 rasterstats（快且简洁）
    from rasterstats import zonal_stats
    stats = zonal_stats(
        roads_buffer, RASTER_PATH, stats=["mean"], nodata=None, geojson_out=False
    )
    pop_mean = [s["mean"] if s and "mean" in s else np.nan for s in stats]
except Exception:
    # 回退到纯 rasterio 方案
    pop_mean = zonal_mean_with_rasterio(roads_buffer, RASTER_PATH)

# ========== 将均值写回“原始道路（线）” ==========
roads[f"pop_mean_buf{BUFFER_M}m"] = pop_mean

# ========== 保存 ==========
roads.to_file(OUT_PATH, driver="GeoJSON")
print(f"✅ Done! 输出: {OUT_PATH}")
