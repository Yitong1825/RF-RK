import json, math
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from shapely.geometry import LineString, MultiLineString, Point
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.inspection import permutation_importance
from scipy.spatial import cKDTree

# =================== 手动参数（按需修改） ===================
ROADS_PATH = "roads_with_density.geojson"
CSV_PATH   = "osm_id_with_aadt.csv"
OUT_R_SVR  = "roads_svr_pred4.geojson"   # 纯回归预测
OUT_RK     = "roads_rk_pred4.geojson"    # 回归克里金预测

CRS_METRIC  = 32647     # 曼谷 UTM 47N（米）
TEST_RATIO  = 0.20
RANDOM_SEED = 42

# Kriging 指数变差函数默认参数（样本不足以拟合时使用）
DEFAULT_NUGGET = 0.05
DEFAULT_C      = 1.0
DEFAULT_A      = 1500.0  # 相关程（米）
KRIGING_RIDGE  = 1e-6    # 线性解稳定项
# ==========================================================


# ------------------ 工具函数 ------------------
def find_id_key(roads_df, csv_df):
    candidates = ["osm_id","osmid","id","road_id","roadid"]
    roads_cols = {c.lower(): c for c in roads_df.columns}
    csv_cols   = {c.lower(): c for c in csv_df.columns}
    for lc in candidates:
        if lc in roads_cols and lc in csv_cols:
            return roads_cols[lc], csv_cols[lc]
    inter = set(roads_cols.keys()) & set(csv_cols.keys())
    if inter:
        lc = list(inter)[0]
        return roads_cols[lc], csv_cols[lc]
    raise ValueError("找不到共同ID列；请确保两份数据都有相同的 ID 字段（如 osm_id）。")

def find_aadt_col(csv_df):
    for c in ["aadt","AADT","aadt_value","AADT_value","value"]:
        if c in csv_df.columns: return c
    return csv_df.columns[-1]

def numeric_maxspeed(series):
    s = series.astype(str).str.extract(r"(\d+\.?\d*)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def choose_poi_columns(df):
    prefixes = ("front20_", "dens100_", "dens300_", "dens800_", "dist_", "entropy_")
    return [c for c in df.columns if c.startswith(prefixes)]

def build_graph_from_lines(gdf_metric):
    G = nx.Graph()
    def add_line(line: LineString):
        coords = list(line.coords)
        for u, v in zip(coords[:-1], coords[1:]):
            pu, pv = Point(u), Point(v)
            w = pu.distance(pv)
            if w <= 0: continue
            if not G.has_node(u): G.add_node(u, pos=u)
            if not G.has_node(v): G.add_node(v, pos=v)
            if G.has_edge(u, v): G[u][v]["weight"] = min(G[u][v]["weight"], w)
            else: G.add_edge(u, v, weight=w)
    for geom in gdf_metric.geometry:
        if geom is None or geom.is_empty: continue
        if isinstance(geom, LineString): add_line(geom)
        elif isinstance(geom, MultiLineString):
            for sub in geom.geoms: add_line(sub)
    return G

def kdtree_from_nodes(G):
    nodes = np.array([G.nodes[n]["pos"] for n in G.nodes])
    tree  = cKDTree(nodes)
    node_list = list(G.nodes)
    return tree, nodes, node_list

def snap_point_to_graph_node(point: Point, tree, nodes, node_list):
    d, idx = tree.query([point.x, point.y], k=1)
    return node_list[idx]

def network_distances_from_sources(G, source_nodes, target_nodes):
    target_set = set(target_nodes)
    dist_maps = []
    for s in source_nodes:
        lengths = nx.single_source_dijkstra_path_length(G, s, weight="weight")
        dist_maps.append({t: lengths.get(t, np.inf) for t in target_set})
    return dist_maps

def fit_simple_variogram(h, gam, resid_var):
    # 指数模型 gamma(h) = nugget + c*(1-exp(-h/a))
    try:
        from scipy.optimize import curve_fit
        ok = np.isfinite(h) & np.isfinite(gam) & (h < np.inf)
        if ok.sum() < 20:
            return DEFAULT_NUGGET, max(1e-6, min(resid_var, DEFAULT_C*resid_var)), DEFAULT_A
        def model(x, nugget, c, a): return nugget + c*(1.0 - np.exp(-x/a))
        p0 = [np.nanpercentile(gam[ok], 10),
              np.nanmax(gam[ok]) - np.nanpercentile(gam[ok], 10),
              np.nanpercentile(h[ok], 75)]
        popt, _ = curve_fit(model, h[ok], gam[ok], p0=p0, bounds=(0, [np.inf,np.inf,np.inf]), maxfev=20000)
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
# -----------------------------------------------------------


# 1) 读入与合并 AADT
roads = gpd.read_file(ROADS_PATH)
csv   = pd.read_csv(CSV_PATH)

id_roads, id_csv = find_id_key(roads, csv)
aadt_col         = find_aadt_col(csv)

roads["_join_id"] = roads[id_roads].astype(str).str.strip()
csv["_join_id"]   = csv[id_csv].astype(str).str.strip()

csv_agg = (csv.dropna(subset=[aadt_col])
              .groupby("_join_id", as_index=False)[aadt_col].mean())
roads = roads.merge(csv_agg, on="_join_id", how="left")
roads = roads.rename(columns={aadt_col: "aadt_obs"})

# 2) 特征准备
if "population" not in roads.columns:
    roads["population"] = 0.0
roads["maxspeed_num"] = numeric_maxspeed(roads["maxspeed"]) if "maxspeed" in roads.columns else np.nan

roadtype_col = next((c for c in ["roadtype","type","highway","type_level"] if c in roads.columns), None)
if roadtype_col is None:
    roadtype_col = "roadtype_fallback"
    roads[roadtype_col] = "unknown"

roaddens_col = next((c for c in ["road_density","road_dens","density"] if c in roads.columns), None)
if roaddens_col is None:
    roaddens_col = "road_density"
    roads[roaddens_col] = 0.0

poi_cols = choose_poi_columns(roads)
num_cols = ["population", "maxspeed_num", roaddens_col] + poi_cols
num_cols = [c for c in dict.fromkeys(num_cols) if c in roads.columns]
cat_cols = [roadtype_col]

# 3) 仅用“有 AADT”的道路做带标注集；空间均匀 80/20 切分
labeled = roads[~roads["aadt_obs"].isna()].copy()
if len(labeled) < 30:
    raise ValueError(f"带 AADT 的样本太少（{len(labeled)}），不足以做稳定的空间拆分。")

labeled_m = labeled.to_crs(epsg=CRS_METRIC).copy()
coords = np.vstack([labeled_m.geometry.centroid.x.values,
                    labeled_m.geometry.centroid.y.values]).T

N = len(labeled_m)
n_clusters = int(np.clip(np.sqrt(N), 5, min(80, N)))
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
cluster_id = kmeans.fit_predict(coords)
labeled["cluster_id"] = cluster_id

# 分簇抽样：每簇约 20% 为测试
rng = np.random.RandomState(RANDOM_SEED)
test_index = []
for cid, idxs in labeled.groupby("cluster_id").groups.items():
    idxs = np.array(list(idxs))
    m = len(idxs)
    if m <= 4:  # 小簇全进训练
        continue
    k = max(1, int(round(m * TEST_RATIO)))
    chosen = rng.choice(idxs, size=k, replace=False)
    test_index.extend(chosen.tolist())
test_index = np.array(sorted(set(test_index)))
train_index = np.array([i for i in labeled.index.values if i not in test_index])

# 训练/测试表
Xtr = roads.loc[train_index, num_cols + cat_cols].copy()
ytr = roads.loc[train_index, "aadt_obs"].astype(float).values
Xte = roads.loc[test_index,  num_cols + cat_cols].copy()
yte = roads.loc[test_index,  "aadt_obs"].astype(float).values
for df in (Xtr, Xte):
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[roadtype_col] = df[roadtype_col].astype(str).fillna("unknown")

# 4) SVR 回归（RBF）+ 空间分组网格搜索（GroupKFold）
prep = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop"
)
pipe = Pipeline([
    ("prep",  prep),
    ("scale", StandardScaler(with_mean=True, with_std=True)),
    ("svr",   SVR(kernel="rbf"))
])

param_grid = {
    "svr__C": [3, 10, 30, 100],
    "svr__epsilon": [0.05, 0.1, 0.3],
    "svr__gamma": ["scale", 0.1, 0.03, 0.01],
}
groups = labeled.loc[train_index, "cluster_id"].values  # 只用训练集簇做 CV 分组
gkf = GroupKFold(n_splits=5)
gs = GridSearchCV(pipe, param_grid=param_grid, cv=gkf.split(Xtr, ytr, groups),
                  scoring="r2", n_jobs=-1, verbose=1)
gs.fit(Xtr, ytr)

best = gs.best_estimator_
print("Best params:", gs.best_params_, "CV R2:", gs.best_score_)

# 训练/测试集性能（纯回归）
yhat_tr = best.predict(Xtr)
yhat_te = best.predict(Xte)
res_tr  = ytr - yhat_tr

print(f"[回归] R2(train)={r2_score(ytr, yhat_tr):.3f}  RMSE(train)={math.sqrt(mean_squared_error(ytr, yhat_tr)):.1f}")
print(f"[回归] R2(test )={r2_score(yte, yhat_te):.3f}  RMSE(test )={math.sqrt(mean_squared_error(yte, yhat_te)):.1f}")

# 5) 网络距离 OK：仅用训练集构建克里金系统
roads_m = roads.to_crs(epsg=CRS_METRIC)
print("[Graph] 构建路网图…")
G = build_graph_from_lines(roads_m)
print(f"[Graph] 节点={G.number_of_nodes()} 边={G.number_of_edges()}")

train_pts = roads_m.loc[train_index, "geometry"].centroid
all_pts   = roads_m.geometry.centroid

tree, nodes_arr, node_list = kdtree_from_nodes(G)
train_nodes = [snap_point_to_graph_node(pt, tree, nodes_arr, node_list) for pt in train_pts]
all_nodes   = [snap_point_to_graph_node(pt, tree, nodes_arr, node_list) for pt in all_pts]

# 合并同节点训练样本（残差取均值）
from collections import defaultdict
node_to_resids = defaultdict(list)
for n, rval in zip(train_nodes, res_tr.astype(float)):
    node_to_resids[n].append(rval)

uniq_train_nodes = list(node_to_resids.keys())
r = np.array([np.mean(node_to_resids[n]) for n in uniq_train_nodes], dtype=float)

# 距离矩阵
dist_maps_tt = network_distances_from_sources(G, uniq_train_nodes, uniq_train_nodes)
D_tt = np.array([[dist_maps_tt[i][n_j] for n_j in uniq_train_nodes] for i in range(len(uniq_train_nodes))], dtype=float)

dist_maps_tp = network_distances_from_sources(G, uniq_train_nodes, all_nodes)
D_tp = np.array([[dist_maps_tp[i][n_j] for n_j in all_nodes] for i in range(len(uniq_train_nodes))], dtype=float)

# 6) 半变异拟合（简化抽样）
pairs = []
Nw = len(r)
max_pairs = min(5000, Nw*(Nw-1)//2)
step = max(1, (Nw*(Nw-1)//2)//max_pairs)
for i in range(Nw):
    for j in range(i+1, Nw, step):
        pairs.append((i, j))
pairs = pairs[:max_pairs]
h   = np.array([D_tt[i,j] for (i,j) in pairs], dtype=float)
gam = 0.5 * (r[[i for i,_ in pairs]] - r[[j for _,j in pairs]])**2

nugget, c, a = fit_simple_variogram(h, gam, resid_var=np.var(r))
print(f"[Variogram] nugget={nugget:.4f}  c={c:.4f}  a={a:.1f} m")

# 7) 残差 OK 预测（对全网）
C = cov_exp(D_tt, nugget, c, a)
np.fill_diagonal(C, c + nugget)
C = C + np.eye(C.shape[0]) * KRIGING_RIDGE
C_targets = cov_exp(D_tp, nugget, c, a)

def krige_one(c_vec, C_train, resid_train):
    w = ordinary_kriging_weights(C_train, c_vec)
    return float(np.dot(w, resid_train))

print("[Kriging] 对全部道路做残差 OK 预测…")
rk_resid_all = np.zeros(C_targets.shape[1], dtype=float)
for j in range(C_targets.shape[1]):
    rk_resid_all[j] = krige_one(C_targets[:, j], C, r)

# 8) 合成回归克里金预测（全网），并评估（用空间均匀测试集）
# 纯回归全网预测
X_all = roads[num_cols + [roadtype_col]].copy()
for c in num_cols: X_all[c] = pd.to_numeric(X_all[c], errors="coerce").fillna(0.0)
X_all[roadtype_col] = X_all[roadtype_col].astype(str).fillna("unknown")
yhat_all = best.predict(X_all)

# 输出表
roads_out_reg = roads.copy()
roads_out_reg["aadt_pred_reg_svr"] = yhat_all.astype(float)
roads_out_reg.to_file(OUT_R_SVR, driver="GeoJSON")
print(f"已输出：{OUT_R_SVR}（纯回归预测）")

roads_out_rk = roads.copy()
roads_out_rk["aadt_obs"]      = roads_out_rk["aadt_obs"].astype(float)
roads_out_rk["aadt_pred_reg"] = yhat_all.astype(float)
roads_out_rk["rk_resid"]      = rk_resid_all.astype(float)
roads_out_rk["aadt_pred_rk"]  = (roads_out_rk["aadt_pred_reg"] + roads_out_rk["rk_resid"]).astype(float)
roads_out_rk.to_file(OUT_RK, driver="GeoJSON")
print(f"已输出：{OUT_RK}（回归克里金预测）")

# 评估（仅测试集）
mask_all_is_test = roads.index.isin(test_index)
rk_pred_test = roads_out_rk.loc[mask_all_is_test, "aadt_pred_rk"].values
print(f"[RK]  R2(test)  = {r2_score(yte, rk_pred_test):.3f}")
print(f"[RK]  RMSE(test)= {math.sqrt(mean_squared_error(yte, rk_pred_test)):.1f}")

# 9) 保存网格搜索与训练/测试预测对比
pd.DataFrame(gs.cv_results_).to_csv("svr_gridcv_results.csv", index=False, encoding="utf-8-sig")
with open("svr_best_params.json", "w", encoding="utf-8") as f:
    json.dump(gs.best_params_, f, ensure_ascii=False, indent=2)

pd.DataFrame({
    "set": (["train"]*len(ytr)) + (["test"]*len(yte)),
    "_join_id": list(roads.loc[train_index, "_join_id"].values) + list(roads.loc[test_index, "_join_id"].values),
    "y_true": np.r_[ytr, yte],
    "y_pred_reg": np.r_[yhat_tr, yhat_te],
    "y_pred_rk":  np.r_[yhat_tr + rk_resid_all[roads.index.isin(train_index)],
                        rk_pred_test]
}).to_csv("svr_pred_train_test.csv", index=False, encoding="utf-8-sig")
print("已保存：svr_gridcv_results.csv, svr_best_params.json, svr_pred_train_test.csv")

# 10) Permutation Importance（基于测试集，粒度到展开特征）
prep_best  = best.named_steps["prep"]
scale_best = best.named_steps["scale"]
svr_best   = best.named_steps["svr"]

feat_names = prep_best.get_feature_names_out()
feat_names = [fn.replace("num__", "").replace("cat__", "") for fn in feat_names]

# 在“scale+svr”上做 permutation importance（输入为 prep 输出）
Xte_design = prep_best.transform(Xte)
Xte_df = pd.DataFrame(Xte_design, columns=feat_names)

from sklearn.base import clone
svr_est = Pipeline([("scale", scale_best), ("svr", svr_best)])
pi = permutation_importance(svr_est, Xte_df, yte, scoring="r2",
                            n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)

feat_import_df = (pd.DataFrame({
    "feature": feat_names,
    "importance_mean": pi.importances_mean,
    "importance_std":  pi.importances_std
}).sort_values("importance_mean", ascending=False).reset_index(drop=True))

# 家族汇总
def family_of(f):
    if f.startswith(("dens100_","dens300_","dens800_")): return "poi_density"
    if f.startswith("front20_"): return "poi_frontage"
    if f.startswith("dist_"):    return "poi_distance"
    if f.startswith("entropy_"): return "poi_entropy"
    if f.startswith(("roadtype","highway","type")): return "roadtype(onehot)"
    if f == "population":      return "population"
    if f == "maxspeed_num":    return "maxspeed"
    if f == roaddens_col:      return "road_density"
    return "other"

feat_import_df["family"] = feat_import_df["feature"].map(family_of)
family_import = (feat_import_df.groupby("family", as_index=False)["importance_mean"].sum()
                                .sort_values("importance_mean", ascending=False))

feat_import_df.to_csv("feature_importance_detailed4.csv", index=False, encoding="utf-8-sig")
family_import.to_csv("feature_importance_family4.csv", index=False, encoding="utf-8-sig")
print("out put：feature_importance_detailed.csv, feature_importance_family.csv")

print("\n================ 总结 ================")
print(f"[回归]   R²(train)={r2_score(ytr,yhat_tr):.4f}  R²(test)={r2_score(yte,yhat_te):.4f}")
print(f"[RK]     R²(test) ={r2_score(yte,rk_pred_test):.4f}")
print("输出：")
print(f" - 纯回归预测：{OUT_R_SVR}")
print(f" - 回归克里金：{OUT_RK}")
print(" - 网格搜索结果：svr_gridcv_results.csv / svr_best_params.json")
print(" - 训练/测试对比：svr_pred_train_test.csv")
print(" - 特征重要性：feature_importance_detailed.csv / feature_importance_family.csv")
