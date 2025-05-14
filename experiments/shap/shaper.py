import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Patch

# Load SHAP values
xgb = pd.read_csv(
    "shap_XgBoost.txt", sep=":", header=None, names=["feature", "xgb_importance"]
)
lgbm = pd.read_csv(
    "shap_Lgbm.txt", sep=":", header=None, names=["feature", "lgbm_importance"]
)
nn = pd.read_csv(
    "shap_feedforward.txt", sep=":", header=None, names=["feature", "nn_importance"]
)
svm = pd.read_csv(
    "shap_svm.txt", sep=":", header=None, names=["feature", "svm_importance"]
)

# Merge all on feature
merged = pd.merge(xgb, lgbm, on="feature", how="outer")
merged = pd.merge(merged, nn, on="feature", how="outer")
merged = pd.merge(merged, svm, on="feature", how="outer").fillna(0)

# Compute mean importance
merged["mean_importance"] = merged[
    ["xgb_importance", "lgbm_importance", "nn_importance", "svm_importance"]
].mean(axis=1)

# Sort by mean importance
merged_sorted = merged.sort_values("mean_importance", ascending=False)

# 1. Top 10% features
top_n = int(len(merged_sorted) * 0.10)
top_features = merged_sorted.head(top_n).reset_index(drop=True)

# === PLOT 1 ===
plt.figure(figsize=(12, 10))
bar_width = 0.2
indices = np.arange(len(top_features))

plt.barh(
    indices + 1.5 * bar_width,
    top_features["xgb_importance"],
    height=bar_width,
    label="XGBoost",
)
plt.barh(
    indices + 0.5 * bar_width,
    top_features["lgbm_importance"],
    height=bar_width,
    label="LightGBM",
)
plt.barh(
    indices - 0.5 * bar_width,
    top_features["nn_importance"],
    height=bar_width,
    label="Feedforward NN",
)
plt.barh(
    indices - 1.5 * bar_width,
    top_features["svm_importance"],
    height=bar_width,
    label="SVM",
)

plt.yticks(indices, top_features["feature"], fontsize=18)
plt.xlabel("Přínos příznaků", fontsize=18)
plt.title("Nejpřínosnější příznaky dle SHAP", fontsize=20)
plt.gca().invert_yaxis()
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# === PLOT 2 ===
plt.figure(figsize=(10, 10))
plt.barh(merged_sorted["feature"], merged_sorted["mean_importance"], color="#7777aa")
plt.xlabel("Mean SHAP Importance", fontsize=16)
plt.title("Mean SHAP Importance Across All Models", fontsize=18)
plt.yticks([])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# === PLOT 3 ===
def extract_prefix(f):
    return f.split("_")[0] + "_" if "_" in f else f


merged["prefix"] = merged["feature"].apply(extract_prefix)
prefix_means = (
    merged.groupby("prefix")["mean_importance"].mean().sort_values(ascending=False)
)

unique_prefixes = prefix_means.index.tolist()
cmap = cm.get_cmap("tab20", len(unique_prefixes))
prefix_colors = {prefix: cmap(i) for i, prefix in enumerate(unique_prefixes)}

merged_shuffled = merged.sort_values("mean_importance", ascending=False).reset_index(
    drop=True
)
colors = merged_shuffled["prefix"].map(prefix_colors)

plt.figure(figsize=(10, 10))
bar_height = 0.8

plt.barh(
    y=np.arange(len(merged_shuffled)),
    width=merged_shuffled["mean_importance"],
    color=colors,
    height=bar_height,
)

plt.xlabel("Přínos příznaků", fontsize=16)
plt.title("Přínos SHAP (příznaky značeny barevně dle kategorie)", fontsize=18)
plt.yticks([])

plt.gca().invert_yaxis()

legend_patches = [Patch(color=prefix_colors[p], label=p) for p in unique_prefixes]
plt.legend(
    handles=legend_patches,
    title="Kategorie",
    title_fontsize=16,
    fontsize=14,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

plt.tight_layout()
plt.show()

# === PLOT 4 ===
plt.figure(figsize=(8, 6))
plt.barh(
    prefix_means.index,
    prefix_means.values,
    color=[prefix_colors[p] for p in prefix_means.index],
)
plt.xlabel("Přínos kategorií", fontsize=14)
plt.title("Průměrný přínos dle SHAP", fontsize=18)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
