# -*- coding: utf-8 -*-
"""
03_train.py
===========
Binary (and multiclass) rainfall classification using five-fold year-blocked
GroupKFold CV (2011-2019) and a chronological holdout test (2020-2021).

Models: Climatological baseline, LR, SVM, RF, XGB, MLP
Feature sets: F1, F2, F3, F4  (from outputs/feature_sets.json)

Outputs (outputs/)
------------------
    binary_cv_results.csv       CV fold-level performance
    binary_test_results.csv     holdout test performance
    multiclass_test_results.csv multiclass (dry/light/heavy) holdout results
    data_split_summary.csv      record counts per split/station
"""

import json
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, accuracy_score, confusion_matrix)
from sklearn.dummy import DummyClassifier
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

DATA    = OUT / "dataset_final.csv"
FS_JSON = OUT / "feature_sets.json"

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS  = list(range(2011, 2020))
TEST_YEARS   = [2020, 2021]
RANDOM_SEED  = 42
N_FOLDS      = 5
RAIN_THRESH  = 0.1   # mm — WMO criterion

STATIONS = {
    "phuket":   {"col": "RF_phuket",   "dist_km": 26},
    "krabi":    {"col": "RF_krabi",    "dist_km": 65},
    "phangnga": {"col": "RF_phangnga", "dist_km": 73},
}
SCALE_MODELS = {"LR", "SVM", "MLP"}

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, parse_dates=["date"])
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

with open(FS_JSON) as f:
    FS = json.load(f)
FEATURE_SETS = {k: v for k, v in FS.items() if k in ("F1","F2","F3","F4")}

# ── Model factory ──────────────────────────────────────────────────────────────
def make_clf(name):
    if name == "LR":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                  class_weight="balanced")
    if name == "SVM":
        return SVC(probability=True, kernel="rbf", random_state=RANDOM_SEED,
                   class_weight="balanced")
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED,
                                      class_weight="balanced", n_jobs=-1)
    if name == "XGB":
        return xgb.XGBClassifier(n_estimators=300, max_depth=5,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss",
                                  random_state=RANDOM_SEED, verbosity=0)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=RANDOM_SEED, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=20)
    raise ValueError(f"Unknown model: {name}")

# ── Climatological baseline ────────────────────────────────────────────────────
def clim_proba(train_df, test_df, col):
    freq = (train_df.groupby("month")[col]
            .apply(lambda x: (x >= RAIN_THRESH).mean()))
    return test_df["month"].map(freq).fillna(
        (train_df[col] >= RAIN_THRESH).mean()).values

# ── Metrics ────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, y_proba):
    auc  = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    mf1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return dict(Test_AUC=round(auc,4), Test_F1=round(mf1,4),
                Test_Recall=round(rec,4), Test_Precision=round(prec,4),
                Test_Specificity=round(spec,4))

# ── Year-blocked CV ────────────────────────────────────────────────────────────
def run_cv(X_tr, y_tr, years, df_tr_full, col, model_name):
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_aucs = []
    for tr_idx, va_idx in gkf.split(X_tr, y_tr, groups=years):
        Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
        ytr, yva = y_tr[tr_idx], y_tr[va_idx]

        if model_name == "Climatological":
            proba = clim_proba(df_tr_full.iloc[tr_idx],
                               df_tr_full.iloc[va_idx], col)
        else:
            clf = make_clf(model_name)
            if model_name in SCALE_MODELS:
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr); Xva = sc.transform(Xva)
            if model_name == "XGB":
                n_neg, n_pos = (ytr==0).sum(), (ytr==1).sum()
                clf.set_params(scale_pos_weight=n_neg/n_pos if n_pos else 1)
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xva)[:,1]

        fold_aucs.append(
            roc_auc_score(yva, proba) if len(np.unique(yva)) > 1 else np.nan)
    return fold_aucs

# ── Holdout test ───────────────────────────────────────────────────────────────
def run_holdout(X_tr, y_tr, X_te, y_te, df_tr, df_te, col, model_name):
    if model_name == "Climatological":
        proba = clim_proba(df_tr, df_te, col)
        pred  = (proba >= 0.5).astype(int)
    else:
        clf  = make_clf(model_name)
        Xtr2 = X_tr.copy(); Xte2 = X_te.copy()
        if model_name in SCALE_MODELS:
            sc = StandardScaler()
            Xtr2 = sc.fit_transform(Xtr2); Xte2 = sc.transform(Xte2)
        if model_name == "XGB":
            n_neg, n_pos = (y_tr==0).sum(), (y_tr==1).sum()
            clf.set_params(scale_pos_weight=n_neg/n_pos if n_pos else 1)
        clf.fit(Xtr2, y_tr)
        proba = clf.predict_proba(Xte2)[:,1]
        pred  = clf.predict(Xte2)
    return metrics(y_te, pred, proba)

# ── Main loop ──────────────────────────────────────────────────────────────────
ALL_MODELS = ["Climatological", "LR", "SVM", "RF", "XGB", "MLP"]
cv_rows, test_rows, split_rows = [], [], []

for stn, info in STATIONS.items():
    col  = info["col"]
    dist = info["dist_km"]
    if col not in df.columns:
        print(f"  SKIP {stn}: column {col} not found")
        continue

    sdf      = df[df[col].notna()].copy()
    train_df = sdf[sdf["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test_df  = sdf[sdf["year"].isin(TEST_YEARS)].reset_index(drop=True)
    y_tr     = (train_df[col] >= RAIN_THRESH).astype(int).values
    y_te     = (test_df[col]  >= RAIN_THRESH).astype(int).values

    rr_tr = round(y_tr.mean()*100, 1)
    rr_te = round(y_te.mean()*100, 1)
    split_rows.append(dict(station=stn, dist_km=dist,
                           n_train=len(train_df), n_test=len(test_df),
                           rain_rate_train=rr_tr, rain_rate_test=rr_te))
    print(f"\n{stn.upper()} ({dist} km) | train={len(train_df)} ({rr_tr}%) "
          f"| test={len(test_df)} ({rr_te}%)")

    for fs_name, fs_cols in FEATURE_SETS.items():
        avail = [c for c in fs_cols if c in sdf.columns]
        if not avail:
            continue
        X_tr = train_df[avail].fillna(train_df[avail].mean()).values
        X_te = test_df[avail].fillna(train_df[avail].mean()).values

        for mdl in ALL_MODELS:
            t0       = time.time()
            fold_aucs = run_cv(X_tr, y_tr, train_df["year"].values,
                               train_df, col, mdl)
            h        = run_holdout(X_tr, y_tr, X_te, y_te,
                                   train_df, test_df, col, mdl)
            elapsed  = round(time.time() - t0, 2)

            for fi, auc in enumerate(fold_aucs):
                cv_rows.append(dict(experiment="main", station=stn,
                                    dist_km=dist, Feature_Set=fs_name,
                                    Model=mdl, fold=fi,
                                    CV_AUC=round(auc,4) if not np.isnan(auc) else np.nan))
            test_rows.append(dict(
                experiment="main", station=stn, dist_km=dist,
                Feature_Set=fs_name, Model=mdl,
                n_train=len(train_df), n_test=len(test_df),
                rain_rate_train=rr_tr, rain_rate_test=rr_te,
                CV_AUC_mean=round(np.nanmean(fold_aucs),4),
                CV_AUC_std=round(np.nanstd(fold_aucs),4),
                **h, Time_s=elapsed))
            print(f"  {fs_name}|{mdl:12s} CV={np.nanmean(fold_aucs):.4f} "
                  f"Test={h['Test_AUC']:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
pd.DataFrame(cv_rows).to_csv(OUT / "binary_cv_results.csv", index=False)
pd.DataFrame(test_rows).to_csv(OUT / "binary_test_results.csv", index=False)
pd.DataFrame(split_rows).to_csv(OUT / "data_split_summary.csv", index=False)
print("\nSaved binary_cv_results.csv, binary_test_results.csv, data_split_summary.csv")
