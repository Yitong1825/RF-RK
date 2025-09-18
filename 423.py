# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====== 数据写死 ======
data = [

    {"zone": "center", "Model": "OK",    "n": 279, "RMSE": 17233.34, "R2": 0.252},
    {"zone": "center", "Model": "RK(linear)", "n": 279, "RMSE": 15550.60, "R2": 0.491},
    {"zone": "center", "Model": "RK(RF)",     "n": 279, "RMSE": 13062.44, "R2": 0.771},
    {"zone": "edge",   "Model": "OK",    "n": 298, "RMSE": 13885.04, "R2": 0.265},
    {"zone": "edge",   "Model": "RK(linear)", "n": 298, "RMSE": 13152.32, "R2": 0.341},
    {"zone": "edge",   "Model": "RK(RF)",     "n": 298, "RMSE": 11779.95, "R2": 0.471},
]
df = pd.DataFrame(data)

# ====== 配色：使用 Seaborn Set2 调色板 ======
palette = sns.color_palette("Set2", n_colors=df["Model"].nunique())
colors = dict(zip(df["Model"].unique(), palette))

# ====== 绘图 ======
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RMSE
sns.barplot(data=df, x="zone", y="RMSE", hue="Model", palette=colors, ax=axes[0])
axes[0].set_title("RMSE by Zone")
axes[0].set_ylabel("RMSE")

# R²
sns.barplot(data=df, x="zone", y="R2", hue="Model", palette=colors, ax=axes[1])
axes[1].set_title("R² by Zone")
axes[1].set_ylabel("R²")

# 图例只显示一次
axes[1].legend(title="Model")
axes[0].legend_.remove()

plt.tight_layout()
plt.savefig("zone_metrics_seaborn.png", dpi=300)
plt.show()
