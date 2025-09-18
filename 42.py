# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# ========= 文件路径（请改成你的实际路径） =========
csv_path = "osm_id_with_aadt.csv"
geojson_files = {
    "RK(linear)" : "roads_rk_pred.geojson",
    "RK(RF)"     : "roads_rk_pred2.geojson",
    "RK(SVR)"    : "roads_rk_pred4.geojson",
}
# ==============================================

# 类型到等级映射表
type_to_level = {
    # Level 1
    "motorway": 1, "motorway_link": 1, "trunk": 1, "trunk_link": 1,
    "primary": 1, "primary_link": 1,
    # Level 2
    "secondary": 2, "secondary_link": 2, "tertiary": 2, "tertiary_link": 2,
    # Level 3
    "residential": 3, "living_street": 3,
    # Level 4
    "busway": 4, "road": 4, "unclassified": 4,
}

def load_obs(csv_path):
    df_obs = pd.read_csv(csv_path)
    if not {"osm_id", "aadt"}.issubset(df_obs.columns):
        raise ValueError("CSV必须包含 osm_id 和 aadt 列")
    df_obs = df_obs[["osm_id", "aadt"]].dropna()
    df_obs["osm_id"] = df_obs["osm_id"].astype(str)
    return df_obs

def load_pred(geojson_path):
    gdf = gpd.read_file(geojson_path)
    if not {"osm_id", "type", "aadt_pred_rk"}.issubset(gdf.columns):
        raise ValueError(f"{geojson_path} 必须包含 osm_id, type, aadt_pred_rk 列")
    df = gdf[["osm_id", "type", "aadt_pred_rk"]].dropna()
    df["osm_id"] = df["osm_id"].astype(str)
    # 映射到 Level
    df["class_level"] = df["type"].map(type_to_level)
    df = df.dropna(subset=["class_level"])
    # 同一条路可能有多个分段，按 (osm_id, class_level) 聚合
    df = df.groupby(["osm_id", "class_level"], as_index=False)["aadt_pred_rk"].mean()
    return df

def metrics_by_level(df_merged):
    rows = []
    for lvl, sub in df_merged.groupby("class_level"):
        if len(sub) < 2:
            rmse = np.sqrt(mean_squared_error(sub["aadt"], sub["aadt_pred_rk"])) if len(sub) == 1 else np.nan
            r2 = np.nan
        else:
            rmse = np.sqrt(mean_squared_error(sub["aadt"], sub["aadt_pred_rk"]))
            r2 = r2_score(sub["aadt"], sub["aadt_pred_rk"])
        rows.append({"Level": int(lvl), "n": len(sub), "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows).sort_values("Level")

# ============== 主流程 ==============
df_obs = load_obs(csv_path)
all_results = {}

for model_name, gj_path in geojson_files.items():
    df_pred = load_pred(gj_path)
    merged = df_pred.merge(df_obs, on="osm_id", how="inner")
    metrics = metrics_by_level(merged)
    metrics["Model"] = model_name
    all_results[model_name] = metrics

# 合并成一个长表
df_all = pd.concat(all_results.values(), ignore_index=True)

print("\n=== 按道路等级的 RMSE/R² ===")
print(df_all)

# 保存结果
df_all.to_csv("metrics_by_level.csv", index=False, encoding="utf-8-sig")

# ============== 绘图 ==============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RMSE 柱状图
for model_name, sub in df_all.groupby("Model"):
    axes[0].bar(sub["Level"] + {"RK(linear)": -0.25, "RK(RF)": 0, "RK(SVR)": 0.25}[model_name],
                sub["RMSE"], width=0.25, label=model_name)
axes[0].set_xticks([1, 2, 3, 4])
axes[0].set_xticklabels(["Level 1", "Level 2", "Level 3", "Level 4"])
axes[0].set_ylabel("RMSE")
axes[0].set_title("RMSE by Road Level")
axes[0].legend()

# R² 柱状图
for model_name, sub in df_all.groupby("Model"):
    axes[1].bar(sub["Level"] + {"RK(linear)": -0.25, "RK(RF)": 0, "RK(SVR)": 0.25}[model_name],
                sub["R2"], width=0.25, label=model_name)
axes[1].set_xticks([1, 2, 3, 4])
axes[1].set_xticklabels(["Level 1", "Level 2", "Level 3", "Level 4"])
axes[1].set_ylabel("R²")
axes[1].set_title("R² by Road Level")
axes[1].legend()

plt.tight_layout()
plt.savefig("metrics_by_level_barplots.png", dpi=300)
plt.show()
