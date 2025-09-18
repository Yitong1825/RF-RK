# -*- coding: utf-8 -*-
"""
按分区将道路划分为 center / edge，两类分别计算三种克里金结果的 RMSE 与 R²
需求字段：
- 观测 CSV:   osm_id, aadt
- 道路 GeoJSON: osm_id, aadt_pred_rk, type, geometry (LineString/MultiLineString)
- 分区 GeoJSON: 任意分区标识列（可选），geometry (Polygon/MultiPolygon)
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

# ========= 路径配置（按需修改） =========
obs_csv_path = "osm_id_with_aadt.csv"                # 观测值：含 osm_id, aadt
khwaeng_path = "Khwaeng(subdistricts).geojson"    # 180 个分区边界
geojson_files = {
    "RK(linear)" : "roads_rk_pred.geojson",
    "RK(RF)"     : "roads_rk_pred2.geojson",
    "RK(SVR)"    : "roads_rk_pred4.geojson",
}
# ========= 划分中心区的配置 =========
# 方式 A：手动列出中心区标识（如代码或名称），并指定其列名。若留空，将使用“自动法”。
central_ids = set()           # 例如：{"103401","103402","103403", ...}
central_id_field = None       # 例如："Khwaeng_code" 或 "kh_name". 若用自动法，保持 None

# 方式 B：自动法参数（当 central_ids 为空或 central_id_field 不存在时启用）
center_ratio = 0.30  # 按与全市质心的距离，将最近的前 30% 分区视作“中心区”
work_epsg = 32647    # 曼谷常用 UTM 47N；若数据不是投影坐标系，会临时转为该投影计算距离
# ====================================

def load_obs(csv_path):
    df = pd.read_csv(csv_path)
    need = {"osm_id", "aadt"}
    if not need.issubset(df.columns):
        raise ValueError(f"观测 CSV 必须包含列 {need}，实际列：{df.columns.tolist()}")
    df = df[list(need)].dropna()
    df["osm_id"] = df["osm_id"].astype(str)
    return df

def ensure_proj(gdf, epsg):
    if gdf.crs is None:
        # 默认假设 WGS84
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    if gdf.crs.to_epsg() != epsg:
        return gdf.to_crs(epsg=epsg)
    return gdf

def make_zone_from_khwaeng(khwaeng_gdf):
    gdf = khwaeng_gdf.copy()
    # 优先使用手动清单
    if central_ids and (central_id_field is not None) and (central_id_field in gdf.columns):
        gdf["zone"] = gdf[central_id_field].astype(str).apply(lambda x: "center" if x in central_ids else "edge")
        return gdf[["zone", "geometry"]]

    # 自动法：根据与全体质心的距离划分
    gdf_m = ensure_proj(gdf, work_epsg)
    city_centroid = gdf_m.unary_union.centroid
    gdf_m["centroid"] = gdf_m.geometry.centroid
    gdf_m["dist_to_city_center_m"] = gdf_m["centroid"].distance(city_centroid)
    # 取距离分位数阈值
    thresh = gdf_m["dist_to_city_center_m"].quantile(center_ratio)
    gdf_m["zone"] = np.where(gdf_m["dist_to_city_center_m"] <= thresh, "center", "edge")
    return gdf_m[["zone", "geometry"]].to_crs(khwaeng_gdf.crs)

def load_roads(gj_path):
    gdf = gpd.read_file(gj_path)
    need = {"osm_id", "aadt_pred_rk", "geometry"}
    missing = need - set(gdf.columns)
    if missing:
        raise ValueError(f"{gj_path} 缺少列：{missing}")
    gdf = gdf[list(need)].dropna(subset=["osm_id", "aadt_pred_rk"])
    gdf["osm_id"] = gdf["osm_id"].astype(str)
    return gdf

def assign_zone_to_roads(roads_gdf, zones_gdf):
    # 空间连接：给每条路分配 zone（如跨多区，取出现次数最多的 zone）
    r = roads_gdf
    z = zones_gdf
    # 坐标系统一（用于空间相交）
    if r.crs is None and z.crs is None:
        r = r.set_crs(4326); z = z.set_crs(4326)
    elif r.crs != z.crs:
        r = r.to_crs(z.crs)

    sjoined = gpd.sjoin(r, z, how="inner", predicate="intersects")
    if sjoined.empty:
        # 若没有任何相交（极少见），全部标为 unknown
        r["zone"] = "unknown"
        return r

    # 对同一 osm_id 若落入多个 zone，取众数
    sjoined["cnt"] = 1
    pick = (
        sjoined.groupby(["osm_id", "zone"])["cnt"].sum()
        .reset_index()
        .sort_values(["osm_id", "cnt"], ascending=[True, False])
        .drop_duplicates(subset=["osm_id"])
        .rename(columns={"zone": "zone_pick"})[["osm_id", "zone_pick"]]
    )
    merged = r.drop(columns=["zone"], errors="ignore").merge(pick, on="osm_id", how="left")
    merged["zone"] = merged["zone_pick"].fillna("unknown")
    return merged.drop(columns=["zone_pick"])

def metrics_by_zone(df_merged):
    out = []
    for zone, sub in df_merged.groupby("zone"):
        if len(sub) >= 2:
            rmse = np.sqrt(mean_squared_error(sub["aadt"], sub["aadt_pred_rk"]))
            r2 = r2_score(sub["aadt"], sub["aadt_pred_rk"])
        elif len(sub) == 1:
            rmse = float(abs(sub["aadt"].iloc[0] - sub["aadt_pred_rk"].iloc[0]))
            r2 = np.nan
        else:
            rmse, r2 = np.nan, np.nan
        out.append({"zone": zone, "n": len(sub), "RMSE": rmse, "R2": r2})
    return pd.DataFrame(out).sort_values("zone")

def main():
    # 读分区并生成 zone
    kh = gpd.read_file(khwaeng_path)
    zones = make_zone_from_khwaeng(kh)[["zone", "geometry"]]

    # 读观测
    obs = load_obs(obs_csv_path)

    all_rows = []
    for model_name, gj in geojson_files.items():
        roads = load_roads(gj)
        roads_z = assign_zone_to_roads(roads, zones)

        # 与观测匹配
        merged = roads_z.merge(obs, on="osm_id", how="inner")
        merged = merged[(merged["aadt"].notna()) & (merged["aadt_pred_rk"].notna())]

        m_tbl = metrics_by_zone(merged)
        m_tbl["Model"] = model_name
        all_rows.append(m_tbl)

    res = pd.concat(all_rows, ignore_index=True)
    res = res[["zone", "Model", "n", "RMSE", "R2"]].sort_values(["zone", "Model"])

    # 输出
    out_dir = Path("zone_metrics")
    out_dir.mkdir(exist_ok=True)
    res.to_csv(out_dir / "zone_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n=== 按中心/边界（zone）分组的 RMSE / R² ===")
    print(res.round({"RMSE": 2, "R2": 3}))

if __name__ == "__main__":
    main()
