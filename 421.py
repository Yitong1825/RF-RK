# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====== 数据写死 ======
data = [
    {"Level": 1, "n": 163, "RMSE": 23586.157859, "R2": 0.085054, "Model": "OK"},
    {"Level": 2, "n": 339, "RMSE": 10311.246130, "R2": 0.245529, "Model": "OK"},
    {"Level": 3, "n":  59, "RMSE": 14640.406526, "R2": -0.079669, "Model": "OK"},
    {"Level": 4, "n":  16, "RMSE":  7685.819338, "R2": -3.634123, "Model": "OK"},
    {"Level": 1, "n": 163, "RMSE": 21420.891845, "R2": 0.398693, "Model": "RK(linear)"},
    {"Level": 2, "n": 339, "RMSE":  7831.508840, "R2": 0.564778, "Model": "RK(linear)"},
    {"Level": 3, "n":  59, "RMSE": 11952.194369, "R2": 0.280419, "Model": "RK(linear)"},
    {"Level": 4, "n":  16, "RMSE":  2994.514987, "R2": 0.296539, "Model": "RK(linear)"},
    {"Level": 1, "n": 163, "RMSE": 19120.861561, "R2": 0.245332, "Model": "RK(RF)"},
    {"Level": 2, "n": 339, "RMSE":  9760.580531, "R2": 0.323961, "Model": "RK(RF)"},
    {"Level": 3, "n":  59, "RMSE": 14048.066199, "R2": 0.005929, "Model": "RK(RF)"},
    {"Level": 4, "n":  16, "RMSE":  4214.354333, "R2": -0.393315,"Model": "RK(RF)"},
]
df = pd.DataFrame(data)

# ====== 配色（Seaborn Set2） ======
models = df["Model"].unique()
palette = sns.color_palette("Set2", n_colors=len(models))
colors = dict(zip(models, palette))

# ====== 绘制 RMSE 和 R² 柱状图 ======
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

levels = sorted(df["Level"].unique())
bar_width = 0.25
x = range(len(levels))
offsets = {m: i*bar_width - bar_width for i, m in enumerate(models)}

# RMSE 柱状图
for model in models:
    sub = df[df["Model"] == model]
    axes[0].bar([p + offsets[model] for p in x], sub["RMSE"],
                width=bar_width, label=model, color=colors[model])
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Level {l}" for l in levels])
axes[0].set_ylabel("RMSE")
axes[0].set_title("RMSE by Road Level")
axes[0].legend()

# R² 柱状图
for model in models:
    sub = df[df["Model"] == model]
    axes[1].bar([p + offsets[model] for p in x], sub["R2"],
                width=bar_width, label=model, color=colors[model])
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"Level {l}" for l in levels])
axes[1].set_ylabel("R²")
axes[1].set_title("R² by Road Level")
axes[1].legend()

plt.tight_layout()
plt.savefig("metrics_by_level_seaborn.png", dpi=300)
plt.show()
