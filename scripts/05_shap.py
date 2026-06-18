# -*- coding: utf-8 -*-
"""
05_shap.py
==========
TreeSHAP analysis for XGB+F3 (nonlinear comparator model).
SHAP is used as a physical plausibility diagnostic — it confirms that
the top-ranked features are physically consistent with known tropical
rainfall mechanisms, rather than spurious correlations.

Inputs  (outputs/ from 03_train.py)
------
    dataset_final.csv
    feature_sets.json

Outputs (outputs/)
------------------
    shap_feature_rank.csv       mean |SHAP| per feature per station

Outputs (outputs/figures/)
---------------------------
    fig_shap_barplot.png        global feature importance (all stations)
    fig_shap_beeswarm.png       beeswarm plot (Phuket only)
"""

import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"
FIG  = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATA    = OUT / "dataset_final.csv"
FS_JSON = OUT / "feature_sets.json"

TRAIN_YEARS = list(range(2011, 2020))
TEST_YEARS  = [2020, 2021]
RANDOM_SEED = 42
RAIN_THRESH = 0.1

STATIONS = {
    "phuket":   {"col": "RF_phuket",   "label": "Phuket"},
    "krabi":    {"col": "RF_krabi",    "label": "Krabi"},
    "phangnga": {"col": "RF_phangnga", "label": "Phang-nga"},
}

df = pd.read_csv(DATA, parse_dates=["date"])
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

with open(FS_JSON) as f:
    FS = json.load(f)
F3_FEATS = FS["F3"]

# ── Main ───────────────────────────────────────────────────────────────────────
rank_rows   = []
phuket_shap = None
phuket_test = None
phuket_feat = None

for stn, info in STATIONS.items():
    col = info["col"]
    if col not in df.columns:
        print(f"  SKIP {stn}: column {col} not found")
        continue

    sdf   = df[df[col].notna()].copy()
    avail = [c for c in F3_FEATS if c in sdf.columns]
    tr    = sdf[sdf["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    te    = sdf[sdf["year"].isin(TEST_YEARS)].reset_index(drop=True)

    y_tr  = (tr[col] >= RAIN_THRESH).astype(int).values
    X_tr  = tr[avail].fillna(tr[avail].mean()).values
    X_te  = te[avail].fillna(tr[avail].mean()).values

    n_neg, n_pos = (y_tr==0).sum(), (y_tr==1).sum()
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=RANDOM_SEED,
                             verbosity=0,
                             scale_pos_weight=n_neg/n_pos if n_pos else 1)
    clf.fit(X_tr, y_tr)

    explainer   = shap.TreeExplainer(clf)
    shap_vals   = explainer.shap_values(X_te)
    mean_abs    = np.abs(shap_vals).mean(axis=0)

    for feat, val in zip(avail, mean_abs):
        rank_rows.append(dict(station=stn, feature=feat,
                              mean_abs_shap=round(float(val), 6)))

    if stn == "phuket":
        phuket_shap = shap_vals
        phuket_test = X_te
        phuket_feat = avail
    print(f"  {stn}: top feature = {avail[np.argmax(mean_abs)]} "
          f"({mean_abs.max():.4f})")

shap_df = pd.DataFrame(rank_rows)
shap_df.to_csv(OUT / "shap_feature_rank.csv", index=False)
print(f"Saved shap_feature_rank.csv ({len(shap_df)} rows)")

# ── Bar plot (all stations) ────────────────────────────────────────────────────
colors = ["#1565C0", "#2E7D32", "#C62828"]
fig, ax = plt.subplots(figsize=(10, 6))
n_feats = len(phuket_feat) if phuket_feat else 0
x       = np.arange(n_feats)
w       = 0.25

for i, (stn, info) in enumerate(STATIONS.items()):
    sub  = shap_df[shap_df.station==stn].set_index("feature")
    vals = [sub.loc[f,"mean_abs_shap"] if f in sub.index else 0
            for f in (phuket_feat or [])]
    ax.bar(x + i*w, vals, w, label=info["label"], color=colors[i], alpha=0.85)

if phuket_feat:
    ax.set_xticks(x + w)
    ax.set_xticklabels(phuket_feat, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean |SHAP value|")
ax.set_title("Global Feature Importance — XGB+F3 (holdout 2020-2021)\n"
             "SHAP used as physical plausibility diagnostic", fontsize=10)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIG / "fig_shap_barplot.png"), dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()

# ── Beeswarm (Phuket) ─────────────────────────────────────────────────────────
if phuket_shap is not None:
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(phuket_shap, phuket_test, feature_names=phuket_feat,
                      show=False, plot_type="dot")
    plt.title("SHAP Beeswarm — XGB+F3, Phuket holdout 2020-2021", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(FIG / "fig_shap_beeswarm.png"), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()

print("05_shap.py complete.")
