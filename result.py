import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# === 1. 加载预测结果（GeoJSON）和观测值（CSV） ===
gdf_pred = gpd.read_file("roads_rk_pred4.geojson")  # 或其他插值结果
df_obs = pd.read_csv("osm_id_with_aadt.csv")

# === 2. 合并数据（通过 osm_id） ===
df_obs["osm_id"] = df_obs["osm_id"].astype("Int64")
gdf_pred["osm_id"] = gdf_pred["osm_id"].astype("Int64")

df_merged = pd.merge(df_obs, gdf_pred[["osm_id", "aadt_pred_rk"]], on="osm_id", how="inner")
df_merged = df_merged.dropna()

# === 3. 计算 R² ===
r2 = r2_score(df_merged["aadt"], df_merged["aadt_pred_rk"])

# === 4. 绘制散点图 ===
plt.figure(figsize=(6,6))
plt.scatter(df_merged["aadt"], df_merged["aadt_pred_rk"], alpha=0.6)
plt.plot([df_merged["aadt"].min(), df_merged["aadt"].max()],
         [df_merged["aadt"].min(), df_merged["aadt"].max()],
         color='red', linestyle='--', label="1:1 line")
plt.xlabel("Observed AADT")
plt.ylabel("Predicted AADT (RF→OK)")
plt.title(f"Prediction vs Observation (R² = {r2:.3f})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. 真值与预测值
y_true = df_merged["aadt"]
y_pred = df_merged["aadt_pred_rk"]

# 2. 手动计算 RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# 3. 打印结果
print(f"📊 Evaluation Metrics (RF→OK):")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE : {mae:.2f}")
print(f"  R²  : {r2:.4f}")

