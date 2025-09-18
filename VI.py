# import warnings
# warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, MultiLineString, Point
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform

# =================== 配置 ===================
ROADS_PATH = "roads_with_density.geojson"
CSV_PATH   = "osm_id_with_aadt.csv"

CRS_METRIC  = 32647     # 曼谷 UTM 47N（米）
TEST_RATIO  = 0.20
RANDOM_SEED = 42

# 迭代与调参
MAX_ROUNDS          = 5      # 最多迭代轮数
KEEP_CUM_IMPORTANCE = 0.95   # 累计重要性覆盖阈值
MIN_IMPORTANCE      = 1e-4   # 极小重要性剔除阈值
EARLY_STOP_ROUNDS   = 2      # 连续几轮无提升则停止
N_RANDOM_SEARCH     = 30     # 每轮超参随机搜索次数
N_PERM_REPEATS      = 10     # 置换重要性重复次数

# =================== 工具函数 ===================
def find_id_key(roads_df, csv_df):
    candidates = ["osm_id", "osmid", "id", "road_id", "roadid"]
    roads_cols = {c.lower(): c for c in roads_df.columns}
    csv_cols   = {c.lower(): c for c in csv_df.columns}
    for lc in candidates:
        if lc in roads_cols and lc in csv_cols:
            return roads_cols[lc], csv_cols[lc]
    inter = set(roads_cols.keys()) & set(csv_cols.keys())
    if inter:
        lc = list(inter)[0]
        return roads_cols[lc], csv_cols[lc]
    raise ValueError("找不到共同ID列；请确保两表都包含相同的 ID 字段（如 osm_id）。")

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

def make_ohe():
    # 兼容不同 sklearn 版本的 OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_reg_pipeline(num_cols, cat_cols, random_state=RANDOM_SEED):
    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", make_ohe(), cat_cols),
        ],
        remainder="drop"
    )
    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1
    )
    return Pipeline(steps=[("prep", prep), ("rf", rf)])

def get_feature_names_from_pipeline(pipeline, num_cols, cat_cols):
    prep = pipeline.named_steps["prep"]
    try:
        feat_names = list(prep.get_feature_names_out())
        feat_names = [fn.replace("num__", "").replace("cat__", "") for fn in feat_names]
        return feat_names
    except Exception:
        names = []
        names.extend(num_cols)

        return names

# =================== 数据读取与特征准备 ===================
roads = gpd.read_file(ROADS_PATH)
csv   = pd.read_csv(CSV_PATH)

# 合并 AADT
id_roads, id_csv = find_id_key(roads, csv)
aadt_col         = find_aadt_col(csv)
roads["_join_id"] = roads[id_roads].astype(str).str.strip()
csv["_join_id"]   = csv[id_csv].astype(str).str.strip()
csv_agg = (csv.dropna(subset=[aadt_col])
              .groupby("_join_id", as_index=False)[aadt_col].mean())
roads = roads.merge(csv_agg, on="_join_id", how="left")
roads = roads.rename(columns={aadt_col: "aadt_obs"})

# 数值/类别特征列
if "population" not in roads.columns:
    roads["population"] = 0.0
roads["maxspeed_num"] = numeric_maxspeed(roads["maxspeed"]) if "maxspeed" in roads.columns else np.nan

roadtype_col = None
for cand in ["roadtype", "type", "highway", "type_level"]:
    if cand in roads.columns:
        roadtype_col = cand
        break
if roadtype_col is None:
    roadtype_col = "roadtype_fallback"
    roads[roadtype_col] = "unknown"

roaddens_col = None
for cand in ["road_density", "road_dens", "density"]:
    if cand in roads.columns:
        roaddens_col = cand
        break
if roaddens_col is None:
    roaddens_col = "road_density"
    roads[roaddens_col] = 0.0

poi_cols = choose_poi_columns(roads)
num_cols = ["population", "maxspeed_num", roaddens_col] + poi_cols
# 去重同时保持顺序
num_cols = [c for c in dict.fromkeys(num_cols) if c in roads.columns]
cat_cols = [roadtype_col]

# 仅用带 AADT 的样本
labeled = roads[~roads["aadt_obs"].isna()].copy()

# =================== 空间均匀 80/20 拆分 ===================
labeled_m = labeled.to_crs(epsg=CRS_METRIC).copy()
coords = np.vstack([labeled_m.geometry.centroid.x.values,
                    labeled_m.geometry.centroid.y.values]).T

N = len(labeled_m)
n_clusters = int(np.clip(np.sqrt(N), 10, 80))  # ~sqrt(N)；[10,80] 约束
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
cluster_id = kmeans.fit_predict(coords)
labeled["cluster_id"] = cluster_id

rng = np.random.RandomState(RANDOM_SEED)
test_index = []
for cid, idxs in labeled.groupby("cluster_id").groups.items():
    idxs = np.array(list(idxs))
    n_in_cluster = len(idxs)
    if n_in_cluster <= 4:
        continue
    k = max(1, int(round(n_in_cluster * TEST_RATIO)))
    chosen = rng.choice(idxs, size=k, replace=False)
    test_index.extend(chosen.tolist())

test_index = np.array(sorted(set(test_index)))
train_index = np.array([i for i in labeled.index.values if i not in test_index])

# 组装训练/测试特征
Xtr = roads.loc[train_index, num_cols + cat_cols].copy()
ytr = roads.loc[train_index, "aadt_obs"].astype(float).values
Xte = roads.loc[test_index,  num_cols + cat_cols].copy()
yte = roads.loc[test_index,  "aadt_obs"].astype(float).values

# 缺失处理
for df in (Xtr, Xte):
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in cat_cols: df[c] = df[c].astype(str).fillna("unknown")

# =================== 迭代：随机搜索 + 重要性裁剪 ===================
def run_iterative_rf(
    Xtr, ytr, Xte, yte, groups,
    init_num_cols, init_cat_cols,
    max_rounds=MAX_ROUNDS,
    keep_cum=KEEP_CUM_IMPORTANCE,
    min_imp=MIN_IMPORTANCE,
    early_stop=EARLY_STOP_ROUNDS,
    n_rs=N_RANDOM_SEARCH
):
    sel_num = list(init_num_cols)
    sel_cat = list(init_cat_cols)

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    best_reg = None
    best_score = -np.inf
    best_cols = (sel_num[:], sel_cat[:])
    no_improve = 0
    history = []

    for rnd in range(1, max_rounds + 1):
        pipe = make_reg_pipeline(sel_num, sel_cat, RANDOM_SEED)
        param_distributions = {
            "rf__n_estimators": randint(300, 900),
            "rf__max_depth": randint(6, 32),
            "rf__min_samples_split": randint(2, 10),
            "rf__min_samples_leaf": randint(1, 8),
            "rf__max_features": uniform(0.3, 0.7),  # 0.3~1.0
            "rf__bootstrap": [True, False],
        }
        rs = RandomizedSearchCV(
            pipe, param_distributions=param_distributions,
            n_iter=n_rs, cv=gkf.split(Xtr[sel_num + sel_cat], ytr, groups),
            scoring="r2", n_jobs=-1, random_state=RANDOM_SEED, verbose=0
        )
        rs.fit(Xtr[sel_num + sel_cat], ytr)

        cand_reg = rs.best_estimator_
        cand_reg.fit(Xtr[sel_num + sel_cat], ytr)

        yhat_tr = cand_reg.predict(Xtr[sel_num + sel_cat])
        yhat_te = cand_reg.predict(Xte[sel_num + sel_cat])

        tr_r2  = r2_score(ytr, yhat_tr)
        tr_rmse= math.sqrt(mean_squared_error(ytr, yhat_tr))
        te_r2  = r2_score(yte, yhat_te)
        te_rmse= math.sqrt(mean_squared_error(yte, yhat_te))

        # 记录历史
        history.append({
            "round": rnd,
            "n_num": len(sel_num), "n_cat": len(sel_cat),
            "cv_r2": rs.best_score_,
            "train_r2": tr_r2, "train_rmse": tr_rmse,
            "test_r2": te_r2,  "test_rmse": te_rmse,
            "best_params": rs.best_params_
        })

        # 保存本轮内置重要性
        rf = cand_reg.named_steps["rf"]
        feat_names = get_feature_names_from_pipeline(cand_reg, sel_num, sel_cat)
        importances = rf.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})\
                    .sort_values("importance", ascending=False).reset_index(drop=True)
        imp_df.to_csv(f"feature_importance_round{rnd}.csv", index=False, encoding="utf-8-sig")

        # 是否刷新最优
        improved = te_r2 > best_score + 1e-4
        if improved:
            best_reg = cand_reg
            best_score = te_r2
            best_cols = (sel_num[:], sel_cat[:])
            no_improve = 0
        else:
            no_improve += 1

        # 基于累计重要性裁剪，得到下一轮特征集合
        imp_df["cum"] = imp_df["importance"].cumsum()
        keep_mask = (imp_df["cum"] <= keep_cum) | (imp_df["importance"] >= min_imp)
        kept_feats = imp_df.loc[keep_mask, "feature"].tolist()

        # 数值列直接匹配；类别列策略：只要出现任意一个 one-hot 列，就保留原始类别列
        new_sel_num = [c in kept_feats and c for c in sel_num]
        new_sel_num = [c for c in new_sel_num if c]
        keep_cat = any([(c == f) or f.startswith(sel_cat[0]) for f in kept_feats]) if sel_cat else False
        new_sel_cat = sel_cat if keep_cat else []

        # 兜底：避免全删空
        if len(new_sel_num) + len(new_sel_cat) == 0:
            fallback = sel_num[:1] if sel_num else []
            new_sel_num = fallback

        sel_num, sel_cat = new_sel_num, new_sel_cat

        # 早停
        if no_improve >= early_stop:
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv("rf_iter_history.csv", index=False, encoding="utf-8-sig")
    return best_reg, best_cols, hist_df

# 以空间簇为 GroupKFold 的分组
# groups = roads.loc[train_index, "cluster_id"].values
groups = labeled.loc[train_index, "cluster_id"].values


best_reg, (best_num_cols, best_cat_cols), hist_df = run_iterative_rf(
    Xtr, ytr, Xte, yte, groups,
    init_num_cols=num_cols, init_cat_cols=cat_cols,
    max_rounds=MAX_ROUNDS, keep_cum=KEEP_CUM_IMPORTANCE,
    min_imp=MIN_IMPORTANCE, early_stop=EARLY_STOP_ROUNDS, n_rs=N_RANDOM_SEARCH
)

# =================== 最优模型评估与重要性输出 ===================
print("\n================ 迭代结束：最优模型评估（仅RF） ================")
yhat_tr_best = best_reg.predict(Xtr[best_num_cols + best_cat_cols])
yhat_te_best = best_reg.predict(Xte[best_num_cols + best_cat_cols])

print(f"[RF]  R²(train) = {r2_score(ytr, yhat_tr_best):.4f}  "
      f"RMSE(train) = {math.sqrt(mean_squared_error(ytr, yhat_tr_best)):.1f}")
print(f"[RF]  R²(test ) = {r2_score(yte, yhat_te_best):.4f}  "
      f"RMSE(test ) = {math.sqrt(mean_squared_error(yte, yhat_te_best)):.1f}")

rf_final = best_reg.named_steps["rf"]
feat_names_final = get_feature_names_from_pipeline(best_reg, best_num_cols, best_cat_cols)
imp_final = pd.DataFrame({"feature": feat_names_final, "importance": rf_final.feature_importances_})\
              .sort_values("importance", ascending=False).reset_index(drop=True)
imp_final.to_csv("feature_importance_best.csv", index=False, encoding="utf-8-sig")
print("out put：rf_iter_history.csv, feature_importance_best.csv and feature_importance_round*.csv")

# 尝试：测试集置换重要性（更稳健）
try:
    perm = permutation_importance(
        best_reg,
        Xte[best_num_cols + best_cat_cols],
        yte,
        n_repeats=N_PERM_REPEATS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scoring="r2"
    )
    perm_df = pd.DataFrame({
        "feature": feat_names_final,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std":  perm.importances_std
    }).sort_values("perm_importance_mean", ascending=False)
    perm_df.to_csv("feature_importance_permutation.csv", index=False, encoding="utf-8-sig")
    print("out put：feature_importance_permutation.csv")
except Exception as e:
    print("error")
