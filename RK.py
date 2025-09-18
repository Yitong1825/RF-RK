# regression_kriging_from_roads_and_csv.py
# ------------------------------------------------------------
# 输入：
#   1) roads_with_poi_feats.geojson   —— 全部道路（含 population/maxspeed/roadtype/road_density/POI 特征），无 AADT
#   2) osm_id_with_aadt.csv           —— 部分道路的 ID + AADT 值（优先列名：osm_id, aadt）
#
# 输出：
#   roads_rk_pred.geojson —— 包含 aadt_pred_reg（回归）、rk_resid（残差克里金）、aadt_pred_rk（最终预测）
#                            以及 aadt_obs（若该道路在CSV中有观测）
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from shapely.geometry import LineString, MultiLineString, Point
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# ============== 路径配置（按需修改） ==============
ROADS_PATH = "roads_with_poi_feats.geojson"
CSV_PATH   = "osm_id_with_aadt.csv"
OUT_PATH   = "roads_rk_pred.geojson"

# ============== 模型/流程参数 ==============
RANDOM_SEED = 42
TEST_SIZE   = 0.2
CRS_METRIC  = 32647   # UTM 47N（曼谷）

# 半变异函数（指数）默认参数（当样本不足以拟合时使用）
DEFAULT_NUGGET = 0.05
DEFAULT_C      = 1.0    # 会乘以残差方差
DEFAULT_A      = 1500.0 # 相关程 a（米量级，城市路网常见 1-3km）
RIDGE          = 1e-6   # 线性代数稳定项

# ============== 小工具函数 ==============
def find_id_key(roads_df, csv_df):
    """自动寻找两边共同ID列（优先 osm_id）"""
    candidates = ["osm_id", "osmid", "id", "road_id", "roadid"]
    roads_cols = {c.lower(): c for c in roads_df.columns}
    csv_cols   = {c.lower(): c for c in csv_df.columns}
    for lc in candidates:
        if lc in roads_cols and lc in csv_cols:
            return roads_cols[lc], csv_cols[lc]
    # 若没命中，尝试任意交集名相同的列
    inter = set(roads_cols.keys()) & set(csv_cols.keys())
    if inter:
        lc = list(inter)[0]
        return roads_cols[lc], csv_cols[lc]
    raise ValueError("找不到共同ID列；请确保 roads 与 csv 都包含相同的 ID 字段（如 osm_id）。")

def find_aadt_col(csv_df):
    candidates = ["aadt", "AADT", "aadt_value", "AADT_value", "value"]
    for c in candidates:
        if c in csv_df.columns:
            return c
    # 若完全找不到，尝试最后一列
    return csv_df.columns[-1]

def numeric_maxspeed(series):
    s = series.astype(str).str.extract(r"(\d+\.?\d*)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def choose_poi_columns(df):
    prefixes = ("front20_", "dens100_", "dens300_", "dens800_", "dist_", "entropy_")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    return cols

# 构建路网图
def build_graph_from_lines(gdf_metric):
    G = nx.Graph()
    def add_line(line: LineString):
        coords = list(line.coords)
        for u, v in zip(coords[:-1], coords[1:]):
            pu, pv = Point(u), Point(v)
            w = pu.distance(pv)
            if w <= 0:
                continue
            if not G.has_node(u): G.add_node(u, pos=u)
            if not G.has_node(v): G.add_node(v, pos=v)
            if G.has_edge(u, v):
                G[u][v]["weight"] = min(G[u][v]["weight"], w)
            else:
                G.add_edge(u, v, weight=w)

    for geom in gdf_metric.geometry:
        if geom is None or geom.is_empty: continue
        if isinstance(geom, LineString):
            add_line(geom)
        elif isinstance(geom, MultiLineString):
            for sub in geom.geoms:
                add_line(sub)
    return G

def kdtree_from_nodes(G):
    from scipy.spatial import cKDTree
    nodes = np.array([G.nodes[n]["pos"] for n in G.nodes])
    tree = cKDTree(nodes)
    node_list = list(G.nodes)
    return tree, nodes, node_list

def snap_point_to_graph_node(point: Point, tree, nodes, node_list):
    d, idx = tree.query([point.x, point.y], k=1)
    return node_list[idx]

def network_distances_from_sources(G, source_nodes, target_nodes):
    """从多源点到一组目标点的网络距离（字典列表）"""
    target_set = set(target_nodes)
    dist_maps = []
    for s in source_nodes:
        lengths = nx.single_source_dijkstra_path_length(G, s, weight="weight")
        dist_maps.append({t: lengths.get(t, np.inf) for t in target_set})
    return dist_maps

# 半变异拟合（简化指数模型）：gamma(h)=nugget + c*(1-exp(-h/a))
def fit_simple_variogram(h, gam, resid_var):
    try:
        from scipy.optimize import curve_fit
        ok = np.isfinite(h) & np.isfinite(gam) & (h < np.inf)
        if ok.sum() < 20:
            return DEFAULT_NUGGET, max(1e-6, min(resid_var, DEFAULT_C*resid_var)), DEFAULT_A
        def model(x, nugget, c, a):
            return nugget + c * (1.0 - np.exp(-x / a))
        p0 = [np.nanpercentile(gam[ok], 10), np.nanmax(gam[ok]) - np.nanpercentile(gam[ok], 10), np.nanpercentile(h[ok], 75)]
        popt, _ = curve_fit(model, h[ok], gam[ok], p0=p0, bounds=(0, [np.inf, np.inf, np.inf]), maxfev=10000)
        nugget, c, a = popt
        c = max(1e-6, min(resid_var, c))
        a = float(a) if np.isfinite(a) and a > 0 else DEFAULT_A
        return float(nugget), float(c), a
    except Exception:
        return DEFAULT_NUGGET, max(1e-6, min(resid_var, DEFAULT_C*resid_var)), DEFAULT_A

def cov_exp(h, nugget, c, a):
    return c * np.exp(-h / a)

def ordinary_kriging_weights(C, c_vec):
    n = C.shape[0]
    A = np.zeros((n+1, n+1), dtype=float)
    A[:n, :n] = C
    A[:n,  n] = 1.0
    A[ n, :n] = 1.0
    b = np.zeros(n+1, dtype=float)
    b[:n] = c_vec
    b[ n] = 1.0
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return sol[:n]

# ============== 主流程 ==============
# 1) 读取数据
roads = gpd.read_file(ROADS_PATH)
csv   = pd.read_csv(CSV_PATH)

# 2) 规范 ID & 合并 AADT
id_roads, id_csv = find_id_key(roads, csv)
aadt_col = find_aadt_col(csv)

# 将两边ID都转为字符串（稳健合并）
roads["_join_id"] = roads[id_roads].astype(str).str.strip()
csv["_join_id"]   = csv[id_csv].astype(str).str.strip()

# 若 CSV 有重复ID，则对 AADT 求均值（你也可改成 first/median）
csv_agg = (csv
           .dropna(subset=[aadt_col])
           .groupby("_join_id", as_index=False)[aadt_col].mean())

# 合并：保留所有道路，附上观测 AADT
roads = roads.merge(csv_agg, on="_join_id", how="left", suffixes=("",""))
roads = roads.rename(columns={aadt_col: "aadt_obs"})

# 3) 特征工程：抽取你指定的特征
# population
if "population" not in roads.columns:
    roads["population"] = 0.0

# maxspeed 数值化
roads["maxspeed_num"] = numeric_maxspeed(roads["maxspeed"]) if "maxspeed" in roads.columns else np.nan

# roadtype（类别 one-hot）
roadtype_col = None
for cand in ["roadtype", "type", "highway", "type_level"]:
    if cand in roads.columns:
        roadtype_col = cand
        break
if roadtype_col is None:
    roadtype_col = "roadtype_fallback"
    roads[roadtype_col] = "unknown"

# road_density
roaddens_col = None
for cand in ["road_density", "road_dens", "density"]:
    if cand in roads.columns:
        roaddens_col = cand
        break
if roaddens_col is None:
    roaddens_col = "road_density"
    roads[roaddens_col] = 0.0

# POI 特征自动识别
poi_cols = choose_poi_columns(roads)

# 组装特征列
num_cols = ["population", "maxspeed_num", roaddens_col] + poi_cols
num_cols = [c for c in dict.fromkeys(num_cols) if c in roads.columns]  # 去重并确认存在
cat_cols = [roadtype_col]

# 4) 训练集（仅有 aadt_obs 的道路）
labeled = roads[~roads["aadt_obs"].isna()].copy()
if len(labeled) < 20:
    raise ValueError(f"可用的 AADT 训练样本过少（{len(labeled)} 条）。请检查 CSV 合并是否成功。")

X = labeled[num_cols + cat_cols].copy()
y = labeled["aadt_obs"].astype(float).values

# 缺失处理
for c in num_cols: X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
X[roadtype_col] = X[roadtype_col].astype(str).fillna("unknown")

Xtr, Xte, ytr, yte, idxtr, idxte = train_test_split(
    X, y, labeled.index.values, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# 5) 回归器（可替换为线性/岭/SVR）
reg = Pipeline(steps=[
    ("prep", ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )),
    ("rf", RandomForestRegressor(
        n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1
    ))
])
reg.fit(Xtr, ytr)

yhat_tr = reg.predict(Xtr)
yhat_te = reg.predict(Xte)
res_tr  = ytr - yhat_tr
res_te  = yte - yhat_te

print(f"[回归] R2(train)={r2_score(ytr, yhat_tr):.3f}  RMSE(train)={math.sqrt(mean_squared_error(ytr, yhat_tr)):.1f}")
print(f"[回归] R2(test )={r2_score(yte, yhat_te):.3f}  RMSE(test )={math.sqrt(mean_squared_error(yte, yhat_te)):.1f}")

# 6) 建立网络图并计算网络距离
roads_m = roads.to_crs(epsg=CRS_METRIC)
labeled_m = roads_m.loc[labeled.index]

print("[Graph] 构建路网图…")
G = build_graph_from_lines(roads_m)
print(f"[Graph] 节点={G.number_of_nodes()} 边={G.number_of_edges()}")

# 用几何质心作为代表点，snap 到最近节点
train_pts = labeled_m.geometry.centroid
all_pts   = roads_m.geometry.centroid

from scipy.spatial import cKDTree
tree, nodes_arr, node_list = kdtree_from_nodes(G)
train_nodes = [snap_point_to_graph_node(pt, tree, nodes_arr, node_list) for pt in train_pts]
all_nodes   = [snap_point_to_graph_node(pt, tree, nodes_arr, node_list) for pt in all_pts]

print("[Graph] 计算网络距离（训练→训练 / 训练→全部）…")
# 训练→训练
dist_maps_tt = network_distances_from_sources(G, train_nodes, train_nodes)
D_tt = np.array([[dist_maps_tt[i][n_j] for n_j in train_nodes] for i in range(len(train_nodes))], dtype=float)

# 训练→全部
dist_maps_tp = network_distances_from_sources(G, train_nodes, all_nodes)
D_tp = np.array([[dist_maps_tp[i][n_j] for n_j in all_nodes] for i in range(len(train_nodes))], dtype=float)

# 7) 简化半变异拟合
r = res_tr.astype(float)
# 抽样对
pairs = []
N = len(r)
max_pairs = min(5000, N*(N-1)//2)
step = max(1, (N*(N-1)//2)//max_pairs)
for i in range(N):
    for j in range(i+1, N, step):
        pairs.append((i, j))
pairs = pairs[:max_pairs]
h   = np.array([D_tt[i,j] for (i,j) in pairs], dtype=float)
gam = 0.5 * (r[[i for i,_ in pairs]] - r[[j for _,j in pairs]])**2

nugget, c, a = fit_simple_variogram(h, gam, resid_var=np.var(r))
print(f"[Variogram] nugget={nugget:.4f}  c={c:.4f}  a={a:.1f} m")

# 8) 训练协方差矩阵 & 目标协方差向量
C = c * np.exp(-D_tt / a)
np.fill_diagonal(C, c + nugget)
C = C + np.eye(C.shape[0]) * RIDGE

C_targets = c * np.exp(-D_tp / a)  # (n_train, n_all)

def krige_one(c_vec, C_train, resid_train):
    # 普通克里金权重
    w = ordinary_kriging_weights(C_train, c_vec)
    return float(np.dot(w, resid_train))

print("[Kriging] 对全部道路做残差 OK 预测…")
rk_resid_all = np.zeros(C_targets.shape[1], dtype=float)
for j in range(C_targets.shape[1]):
    rk_resid_all[j] = krige_one(C_targets[:, j], C, r)

# 9) 组合为回归克里金预测，并评估
# 对测试集评估
mask_all_is_test = np.isin(roads.index.values, idxte)
rk_pred_test = yhat_te + rk_resid_all[mask_all_is_test]

print(f"[RK]  R2(test)  = {r2_score(yte, rk_pred_test):.3f}")
print(f"[RK]  RMSE(test)= {math.sqrt(mean_squared_error(yte, rk_pred_test)):.1f}")

# 全网预测
X_all = roads[num_cols + [roadtype_col]].copy()
for c in num_cols: X_all[c] = pd.to_numeric(X_all[c], errors="coerce").fillna(0.0)
X_all[roadtype_col] = X_all[roadtype_col].astype(str).fillna("unknown")
yhat_all = reg.predict(X_all)

roads_out = roads.copy()
roads_out["aadt_obs"]     = roads_out["aadt_obs"].astype(float)
roads_out["aadt_pred_reg"]= yhat_all.astype(float)
roads_out["rk_resid"]     = rk_resid_all.astype(float)
roads_out["aadt_pred_rk"] = (roads_out["aadt_pred_reg"] + roads_out["rk_resid"]).astype(float)

# 10) 导出
roads_out.to_file(OUT_PATH, driver="GeoJSON")
print(f"✅ 已输出：{OUT_PATH}")
# ================== 结尾：变量重要性 & R^2 汇总 ==================

print("\n================ 变量重要性 & R² 汇总 ================")

# 1) 回归 R²（已在前面打印过，这里汇总一下）
reg_r2_train = r2_score(ytr, yhat_tr)
reg_r2_test  = r2_score(yte, yhat_te)
rk_r2_test   = r2_score(yte, rk_pred_test)

print(f"[回归]   R²(train) = {reg_r2_train:.4f}")
print(f"[回归]   R²(test ) = {reg_r2_test :.4f}")
print(f"[回归克里金] R²(test ) = {rk_r2_test  :.4f}")

# 2) 提取随机森林的变量重要性（映射回可读的特征名）
rf  = reg.named_steps["rf"]
prep = reg.named_steps["prep"]

# 尝试自动拿到 ColumnTransformer 展开的特征名
try:
    feat_names = prep.get_feature_names_out()
    # 清理一下前缀 'num__' / 'cat__'
    feat_names = [fn.replace("num__", "").replace("cat__", "") for fn in feat_names]
except Exception:
    # 兼容老版本 sklearn：手动拼
    feat_names = []
    # 数值列原样
    feat_names.extend(num_cols)
    # 类别列（one-hot）展开
    try:
        ohe = None
        for name, trans, cols in prep.transformers_:
            if name == "cat":
                ohe = trans
                break
        if hasattr(ohe, "get_feature_names_out"):
            cat_names = list(ohe.get_feature_names_out([roadtype_col]))
        else:
            # 老版本：退化成用类别列名占位
            cat_names = [f"{roadtype_col}_<onehot_{i}>" for i in range(rf.n_features_in_ - len(num_cols))]
        feat_names.extend(cat_names)
    except Exception:
        pass

importances = rf.feature_importances_
feat_import_df = pd.DataFrame({"feature": feat_names, "importance": importances})
feat_import_df = feat_import_df.sort_values("importance", ascending=False).reset_index(drop=True)

# 打印前 30 个
print("\n[Top 30 Feature Importance]")
print(feat_import_df.head(30).to_string(index=False))

# 3) 可选：把 one-hot 的 roadtype 汇总成一个大类的重要性，POI 家族做粗汇总
def family_of(f):
    if f.startswith("dens100_") or f.startswith("dens300_") or f.startswith("dens800_"):
        return "poi_density"
    if f.startswith("front20_"):
        return "poi_frontage"
    if f.startswith("dist_"):
        return "poi_distance"
    if f.startswith("entropy_"):
        return "poi_entropy"
    if f.startswith(f"{roadtype_col}_") or f.startswith(roadtype_col):
        return "roadtype(onehot)"
    if f == "population":
        return "population"
    if f == "maxspeed_num":
        return "maxspeed"
    if f == roaddens_col:
        return "road_density"
    return "other"

feat_import_df["family"] = feat_import_df["feature"].map(family_of)
family_import = (feat_import_df
                 .groupby("family", as_index=False)["importance"].sum()
                 .sort_values("importance", ascending=False))

print("\n[Feature Importance by Family]")
print(family_import.to_string(index=False))

# 4) 保存结果
feat_import_df.to_csv("feature_importance_detailed.csv", index=False, encoding="utf-8-sig")
family_import.to_csv("feature_importance_family.csv", index=False, encoding="utf-8-sig")

print("\n📄 已保存：feature_importance_detailed.csv, feature_importance_family.csv")
print("=========================================================\n")
