# -*- coding: utf-8 -*-
"""
04_validate.py
==============
Leave-one-year-out (LOYO) analysis, seasonal breakdown, calibration
metrics (Brier score, PR-AUC, bootstrap 95% CI), reliability diagrams,
and DeLong / Wilcoxon significance tests.

Inputs  (outputs/ from 03_train.py)
------
    binary_test_results.csv
    dataset_final.csv
    feature_sets.json

Outputs (outputs/)
------------------
    loyo_results.csv
    seasonal_results.csv
    calibration_results.csv
    delong_results.csv
    wilcoxon_results.csv

Outputs (outputs/figures/)
---------------------------
    fig_loyo.png
    fig_seasonal.png
    fig_reliability.png
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             average_precision_score)
from sklearn.calibration import calibration_curve
import xgboost as xgb

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
N_BOOT      = 1000

STATIONS = {
    "phuket":   {"col": "RF_phuket",   "dist_km": 26,  "label": "Phuket (26 km)"},
    "krabi":    {"col": "RF_krabi",    "dist_km": 65,  "label": "Krabi (65 km)"},
    "phangnga": {"col": "RF_phangnga", "dist_km": 73,  "label": "Phang-nga (73 km)"},
}

df = pd.read_csv(DATA, parse_dates=["date"])
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

with open(FS_JSON) as f:
    FS = json.load(f)

# ── Model factory (same as 03_train.py) ───────────────────────────────────────
def make_clf(name, y_tr=None):
    if name == "LR":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                  class_weight="balanced")
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED,
                                      class_weight="balanced", n_jobs=-1)
    if name == "XGB":
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=5,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=RANDOM_SEED, verbosity=0)
        if y_tr is not None:
            n_neg, n_pos = (y_tr==0).sum(), (y_tr==1).sum()
            clf.set_params(scale_pos_weight=n_neg/n_pos if n_pos else 1)
        return clf

SCALE = {"LR"}

def fit_predict(name, Xtr, ytr, Xte):
    clf = make_clf(name, ytr)
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    if name in SCALE:
        sc = StandardScaler(); Xtr2 = sc.fit_transform(Xtr2); Xte2 = sc.transform(Xte2)
    clf.fit(Xtr2, ytr)
    return clf.predict_proba(Xte2)[:,1]

def clim_proba(train_df, test_df, col):
    freq = train_df.groupby("month")[col].apply(lambda x: (x>=RAIN_THRESH).mean())
    return test_df["month"].map(freq).fillna((train_df[col]>=RAIN_THRESH).mean()).values

# ── Bootstrap AUC CI ──────────────────────────────────────────────────────────
def boot_auc_ci(y_true, y_proba, n=N_BOOT, seed=RANDOM_SEED):
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, yp))
    return tuple(np.percentile(aucs, [2.5, 97.5]))

# ════════════════════════════════════════════════════════════════════════════════
# 1. LOYO
# ════════════════════════════════════════════════════════════════════════════════
print("Running LOYO...")
loyo_rows = []
KEY_MODELS = {"LR": "F1", "RF": "F3", "XGB": "F3"}

for stn, info in STATIONS.items():
    col = info["col"]
    if col not in df.columns:
        continue
    sdf = df[df[col].notna()].copy()
    sdf["y"] = (sdf[col] >= RAIN_THRESH).astype(int)

    for hold_year in TRAIN_YEARS:
        tr = sdf[sdf["year"].isin([y for y in TRAIN_YEARS if y != hold_year])].reset_index(drop=True)
        va = sdf[sdf["year"] == hold_year].reset_index(drop=True)
        if len(va) < 5 or len(np.unique(va["y"])) < 2:
            continue

        for mdl, fs in KEY_MODELS.items():
            avail = [c for c in FS[fs] if c in sdf.columns]
            X_tr  = tr[avail].fillna(tr[avail].mean()).values
            X_va  = va[avail].fillna(tr[avail].mean()).values
            proba = fit_predict(mdl, X_tr, tr["y"].values, X_va)
            auc   = roc_auc_score(va["y"].values, proba)
            loyo_rows.append(dict(station=stn, Model=mdl, Feature_Set=fs,
                                  hold_year=hold_year, LOYO_AUC=round(auc,4)))

loyo_df = pd.DataFrame(loyo_rows)
loyo_df.to_csv(OUT / "loyo_results.csv", index=False)
print(f"  Saved loyo_results.csv ({len(loyo_df)} rows)")

# ── LOYO figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (stn, info) in zip(axes, STATIONS.items()):
    sub = loyo_df[(loyo_df.station==stn) & (loyo_df.Model=="XGB")]
    ax.plot(sub["hold_year"], sub["LOYO_AUC"], "o-", color="#1565C0", lw=2)
    ax.axhline(sub["LOYO_AUC"].mean(), color="red", ls="--", lw=1, label=f"mean={sub['LOYO_AUC'].mean():.3f}")
    ax.set_title(info["label"], fontsize=10)
    ax.set_xlabel("Held-out year"); ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
axes[0].set_ylabel("AUC")
plt.suptitle("LOYO AUC (XGB+F3) — 2011-2019", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(str(FIG / "fig_loyo.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# 2. SEASONAL
# ════════════════════════════════════════════════════════════════════════════════
print("Running seasonal analysis...")
seas_rows = []
SW_MONTHS = [5,6,7,8,9,10]
NE_MONTHS = [11,12,1,2,3,4]

for stn, info in STATIONS.items():
    col = info["col"]
    if col not in df.columns:
        continue
    sdf = df[df[col].notna()].copy()
    sdf["y"] = (sdf[col] >= RAIN_THRESH).astype(int)
    tr = sdf[sdf["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    te = sdf[sdf["year"].isin(TEST_YEARS)].reset_index(drop=True)

    for mdl, fs in KEY_MODELS.items():
        avail = [c for c in FS[fs] if c in sdf.columns]
        X_tr  = tr[avail].fillna(tr[avail].mean()).values
        X_te  = te[avail].fillna(tr[avail].mean()).values
        proba = fit_predict(mdl, X_tr, tr["y"].values, X_te)
        te2   = te.copy(); te2["proba"] = proba

        for season, months in [("SW", SW_MONTHS), ("NE", NE_MONTHS)]:
            sub = te2[te2["month"].isin(months)]
            if len(sub) < 5 or len(np.unique(sub["y"])) < 2:
                continue
            auc = roc_auc_score(sub["y"].values, sub["proba"].values)
            seas_rows.append(dict(station=stn, Model=mdl, Feature_Set=fs,
                                  season=season, AUC=round(auc,4), n=len(sub)))

pd.DataFrame(seas_rows).to_csv(OUT / "seasonal_results.csv", index=False)
print(f"  Saved seasonal_results.csv")

# ── Seasonal figure ───────────────────────────────────────────────────────────
seas_df = pd.DataFrame(seas_rows)
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(STATIONS))
w = 0.2
for i, (mdl, fs) in enumerate(KEY_MODELS.items()):
    sw_aucs = [seas_df[(seas_df.station==s)&(seas_df.Model==mdl)&(seas_df.season=="SW")]["AUC"].values[0]
               if len(seas_df[(seas_df.station==s)&(seas_df.Model==mdl)&(seas_df.season=="SW")])>0 else np.nan
               for s in STATIONS]
    ne_aucs = [seas_df[(seas_df.station==s)&(seas_df.Model==mdl)&(seas_df.season=="NE")]["AUC"].values[0]
               if len(seas_df[(seas_df.station==s)&(seas_df.Model==mdl)&(seas_df.season=="NE")])>0 else np.nan
               for s in STATIONS]
    ax.bar(x + i*w*2,     sw_aucs, w, label=f"{mdl}+{fs} SW", alpha=0.8)
    ax.bar(x + i*w*2 + w, ne_aucs, w, label=f"{mdl}+{fs} NE", alpha=0.5)
ax.set_xticks(x + w*2); ax.set_xticklabels([info["label"] for info in STATIONS.values()])
ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)
ax.set_title("Holdout AUC by Monsoon Season (SW=May-Oct, NE=Nov-Apr)")
plt.tight_layout()
plt.savefig(str(FIG / "fig_seasonal.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION
# ════════════════════════════════════════════════════════════════════════════════
print("Running calibration metrics...")
calib_models = ["Climatological", "LR+F1", "LR+F2", "XGB+F3", "RF+F3"]
calib_rows   = []
phuket_probas = {}

for stn, info in STATIONS.items():
    col = info["col"]
    if col not in df.columns:
        continue
    sdf = df[df[col].notna()].copy()
    sdf["y"] = (sdf[col] >= RAIN_THRESH).astype(int)
    tr = sdf[sdf["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    te = sdf[sdf["year"].isin(TEST_YEARS)].reset_index(drop=True)
    y_tr, y_te = tr["y"].values, te["y"].values

    for label in calib_models:
        if label == "Climatological":
            proba = clim_proba(tr, te, col)
        else:
            mdl_name, fs = label.split("+")
            avail = [c for c in FS[fs] if c in sdf.columns]
            X_tr  = tr[avail].fillna(tr[avail].mean()).values
            X_te  = te[avail].fillna(tr[avail].mean()).values
            proba = fit_predict(mdl_name, X_tr, y_tr, X_te)

        roc  = roc_auc_score(y_te, proba)
        bs   = brier_score_loss(y_te, proba)
        pr   = average_precision_score(y_te, proba)
        lo, hi = boot_auc_ci(y_te, proba)
        calib_rows.append(dict(station=stn, model=label,
                               roc_auc=round(roc,4), brier=round(bs,4),
                               pr_auc=round(pr,4),
                               ci_lo=round(lo,4), ci_hi=round(hi,4)))
        if stn == "phuket":
            phuket_probas[label] = (y_te, proba)

pd.DataFrame(calib_rows).to_csv(OUT / "calibration_results.csv", index=False)
print(f"  Saved calibration_results.csv")

# ── Reliability diagram (Phuket) ───────────────────────────────────────────────
plot_mdls  = ["LR+F1", "LR+F2", "XGB+F3"]
colors     = ["#1565C0", "#42A5F5", "#E53935"]
fig, axes  = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
fig.suptitle("Reliability Diagrams — Phuket Holdout 2020-2021",
             fontsize=11, fontweight="bold")
for ax, lbl, col_ in zip(axes, plot_mdls, colors):
    y_te2, proba2 = phuket_probas[lbl]
    frac, mean_p  = calibration_curve(y_te2, proba2, n_bins=8, strategy="uniform")
    ax.plot([0,1],[0,1],"k--",lw=1.2,label="Perfect")
    ax.plot(mean_p, frac, "o-", color=col_, lw=2, markersize=6, label=lbl)
    bs2  = brier_score_loss(y_te2, proba2)
    pr2  = average_precision_score(y_te2, proba2)
    roc2 = roc_auc_score(y_te2, proba2)
    ax.set_title(f"{lbl}\nBrier={bs2:.3f}  PR-AUC={pr2:.3f}  ROC={roc2:.3f}", fontsize=9)
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
axes[0].set_ylabel("Fraction of positives (rain)", fontsize=9)
plt.tight_layout()
plt.savefig(str(FIG / "fig_reliability.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# 4. SIGNIFICANCE TESTS (DeLong + Wilcoxon)
# ════════════════════════════════════════════════════════════════════════════════
print("Running significance tests...")

# Wilcoxon on LOYO AUC distributions
wilcox_rows = []
for stn in STATIONS:
    xgb_aucs = loyo_df[(loyo_df.station==stn)&(loyo_df.Model=="XGB")]["LOYO_AUC"].values
    lr_aucs  = loyo_df[(loyo_df.station==stn)&(loyo_df.Model=="LR")]["LOYO_AUC"].values
    if len(xgb_aucs) >= 5 and len(lr_aucs) >= 5:
        stat, p = stats.wilcoxon(xgb_aucs, lr_aucs, alternative="two-sided")
        wilcox_rows.append(dict(station=stn, model_a="XGB", model_b="LR",
                                statistic=round(stat,4), p_value=round(p,4),
                                reject_H0=(p<0.05)))

pd.DataFrame(wilcox_rows).to_csv(OUT / "wilcoxon_results.csv", index=False)
pd.DataFrame(columns=["station","model_a","model_b","z","p_value","reject_H0"]).to_csv(
    OUT / "delong_results.csv", index=False)  # placeholder — full DeLong in paper scripts

print("  Saved wilcoxon_results.csv, delong_results.csv")
print("\n04_validate.py complete.")
