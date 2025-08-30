
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pykrige.uk import UniversalKriging
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm
# gdf = gpd.read_file("roads_split.geojson").to_crs(epsg=32647)
# df_obs = pd.read_csv("osm_id_with_aadt.csv")
# === 1. 加载数据 ===
gdf = gpd.read_file("roads_split.geojson").to_crs(epsg=32647)
df_obs = pd.read_csv("osm_id_with_aadt.csv")
df_pop = pd.read_csv("bangkok_khwaeng_population.csv")  # 人口数据

gdf["osm_id"] = gdf["osm_id"].astype("Int64")
df_obs["osm_id"] = df_obs["osm_id"].astype("Int64")
gdf = gdf.merge(df_obs, on="osm_id", how="left")

# === 2. maxspeed 处理 ===
gdf["maxspeed_clean"] = gdf["maxspeed"].astype(str).str.extract(r"(\d+)").astype(float)
gdf["maxspeed_filled"] = gdf["maxspeed_clean"]
gdf["maxspeed_filled"] = gdf.groupby("type")["maxspeed_filled"].transform(lambda x: x.fillna(x.median()))
gdf["maxspeed_filled"] = gdf["maxspeed_filled"].fillna(gdf["maxspeed_clean"].median())

# === 3. 道路长度和密度 ===
gdf["length"] = gdf.geometry.length
centroids = gdf.geometry.centroid
tree = cKDTree(np.array([(p.x, p.y) for p in centroids]))
gdf["density"] = [len(tree.query_ball_point([pt.x, pt.y], r=300)) for pt in centroids]

# === 4. 合并人口信息（需要行政区编码字段 Khwaeng_code）===
# 你需要确保 gdf 中有 Khwaeng_code，可以通过 spatial join 匹配得到
# 这里假设已经有该字段：
if "Khwaeng_code" in gdf.columns:
    df_pop["Khwaeng_code"] = df_pop["Khwaeng_code"].astype(str)
    gdf["Khwaeng_code"] = gdf["Khwaeng_code"].astype(str)
    gdf = gdf.merge(df_pop[["Khwaeng_code", "Population_total"]], on="Khwaeng_code", how="left")
else:
    gdf["Population_total"] = 0  # 如果没有行政区编码就默认填 0

# === 5. 特征工程 ===
cat = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
type_encoded = cat.fit_transform(gdf[["type"]].fillna("unknown"))
type_feature_names = cat.get_feature_names_out(["type"])

numerical = gdf[["maxspeed_filled", "density", "length", "Population_total"]].fillna(0).values
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical)
X = np.hstack([type_encoded, numerical_scaled])
y = gdf["aadt"].values

# === 6. 划分训练测试集（仅限观测值）===
mask_known = gdf["aadt"].notna()
X_known = X[mask_known]
y_known = y[mask_known]
coords_known = np.array(gdf[mask_known].geometry.centroid.apply(lambda p: (p.x, p.y)).tolist())

X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
    X_known, y_known, coords_known, test_size=0.2, random_state=42
)

# === 7. 定义模型列表 ===
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
    "SVR": SVR(C=1.0, gamma="scale")
}

rk_scores = {}
reg_scores = {}
feature_weights = {}

# === 8. 回归 + 克里金残差插值 ===
for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_reg = model.score(X_test, y_test)
    reg_scores[name] = r2_reg

    residuals = y_train - y_pred_train
    x_krig = p_train[:, 0]
    y_krig = p_train[:, 1]

    uk = UniversalKriging(
        x_krig, y_krig, residuals,
        variogram_model="spherical", verbose=False, enable_plotting=False
    )

    rk_residuals, _ = uk.execute("points", p_test[:, 0], p_test[:, 1])
    rk_pred = y_pred_test + rk_residuals
    ss_res = np.sum((y_test - rk_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2_rk = 1 - ss_res / ss_tot
    rk_scores[name] = r2_rk

    print(f"Regression Score: {r2_reg:.4f}")
    print(f"RK Score:         {r2_rk:.4f}")

    if hasattr(model, "coef_"):
        weights = np.abs(model.coef_)
        weights /= weights.sum()
        feature_weights[name] = weights
    elif hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
        weights /= weights.sum()
        feature_weights[name] = weights

    coords_all = np.array(gdf.geometry.centroid.apply(lambda p: (p.x, p.y)).tolist())
    y_reg_all = model.predict(X)
    rk_full_residuals, _ = uk.execute("points", coords_all[:, 0], coords_all[:, 1])
    gdf[f"rk_aadt_{name.lower()}"] = np.round(y_reg_all + rk_full_residuals, 0)
    gdf[f"rk_aadt_{name.lower()}"] = gdf[f"rk_aadt_{name.lower()}"].clip(lower=0)
    gdf[["osm_id", "geometry", f"rk_aadt_{name.lower()}"]].to_file(f"rk_result_{name.lower()}.geojson", driver="GeoJSON")
    print(f"插值输出：rk_result_{name.lower()}.geojson")

# === 9. 模型得分可视化 ===
plt.figure(figsize=(6, 4))
plt.bar(reg_scores.keys(), reg_scores.values(), label="Regression", alpha=0.6)
plt.bar(rk_scores.keys(), rk_scores.values(), label="RK", alpha=0.8)
plt.ylabel("R² Score")
plt.title("Regression vs Regression Kriging")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# === 10. 特征影响展示 ===
if "LinearRegression" in feature_weights and "RandomForest" in feature_weights:
    feature_names = list(type_feature_names) + ["maxspeed", "density", "length", "Population_total"]
    x = np.arange(len(feature_names))
    width = 0.35
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, feature_weights["LinearRegression"], width, label="LinearRegression")
    plt.bar(x + width/2, feature_weights["RandomForest"], width, label="RandomForest")
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.ylabel("Normalized Feature Influence")
    plt.title("Feature Importance Comparison")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()
