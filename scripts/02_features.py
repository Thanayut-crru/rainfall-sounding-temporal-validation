# -*- coding: utf-8 -*-
"""
02_features.py
==============
Aligns sounding indices with daily rainfall observations, computes
seasonal harmonics and wind-profile features, applies an unsupervised
correlation filter (|r| >= 0.90 on training data only), and assembles
the four feature sets (F1-F4) used in model training.

Inputs
------
    outputs/sounding_indices_raw.csv   (from 01_preprocess.py)
    data/raw/rainfall_tmd.csv          (TMD rain-gauge data — see data/README.md)
      OR
    data/sample/sample_dataset.csv     (demo only — set USE_SAMPLE=True below)

Outputs
-------
    outputs/dataset_final.csv          merged dataset with all features + labels
    outputs/feature_sets.json          feature-set membership lists F1-F4
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
USE_SAMPLE    = True          # set False when using real TMD data
CORR_THRESH   = 0.90          # |r| >= this on training data → drop from F1
TRAIN_YEARS   = list(range(2011, 2020))
RAIN_MM_THRESH = 0.1          # WMO criterion for a rain day

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ── F1 candidate pool (all thermodynamic indices from Wyoming) ─────────────────
F1_CANDIDATES = [
    "SHOW","LIFT","LIFV","SWET","KINX","CTOT","VTOT","TTOT",
    "CAPE","CAPV","CINS","CINV","EQLV","EQTV","LFCT","LFCV",
    "BRCH","BRCV","LCLT","LCLP","Equivalent","MLTH","MLMR","THTK","PWAT",
]
WIND_FEATURES = ["WS850","WDsin850","WDcos850","WS700","WS200","VWS","VWS_lo"]

# ── Load sounding indices ──────────────────────────────────────────────────────
if USE_SAMPLE:
    src = BASE / "data" / "sample" / "sample_dataset.csv"
    print(f"[DEMO] Loading sample dataset: {src}")
    df = pd.read_csv(src, parse_dates=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Sample dataset already has labels and seasonal features; just write outputs
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Build feature sets from available columns
    f1_avail = [c for c in F1_CANDIDATES if c in df.columns]
    train_df = df[df["year"].isin(TRAIN_YEARS)]
    corr_mat = train_df[f1_avail].corr().abs()
    upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop  = [c for c in upper.columns if any(upper[c] >= CORR_THRESH)]
    f1_final = [c for c in f1_avail if c not in to_drop]
    print(f"  F1 candidates: {len(f1_avail)}  dropped (|r|>={CORR_THRESH}): {len(to_drop)}  F1 final: {len(f1_final)}")

    wind_avail = [c for c in WIND_FEATURES if c in df.columns]
    feat_sets  = {
        "F1": f1_final,
        "F2": f1_final + ["sin_month", "cos_month"],
        "F3": f1_final + wind_avail,
        "F4": f1_final + ["sin_month", "cos_month"] + wind_avail,
        "DROPPED": to_drop,
        "_meta": {
            "F1_description": f"{len(f1_final)} stability indices after |r|>={CORR_THRESH} filter",
            "F2_description": "F1 + cyclical season encoding (sin/cos month)",
            "F3_description": f"F1 + {len(wind_avail)} wind features",
            "F4_description": "F1 + season + wind = all features",
        },
    }
    df.to_csv(OUT / "dataset_final.csv", index=False)
    with open(OUT / "feature_sets.json", "w") as f:
        json.dump(feat_sets, f, indent=2)
    print(f"  Saved {len(df)} rows -> outputs/dataset_final.csv")
    print(f"  Saved feature_sets.json  (F1: {len(f1_final)} features)")
    raise SystemExit(0)

# ── REAL DATA MODE ─────────────────────────────────────────────────────────────
snd_path  = OUT / "sounding_indices_raw.csv"
rain_path = BASE / "data" / "raw" / "rainfall_tmd.csv"

print(f"Loading sounding indices: {snd_path}")
snd = pd.read_csv(snd_path, parse_dates=["date"])

print(f"Loading TMD rainfall: {rain_path}")
rain = pd.read_csv(rain_path, parse_dates=["date"])
# Expected columns: date, RF_phuket, RF_krabi, RF_phangnga (daily totals, mm)

df = pd.merge(snd, rain, on="date", how="inner")
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

# Seasonal harmonics
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

# Binary labels (WMO 0.1 mm criterion)
for stn in ["phuket", "krabi", "phangnga"]:
    col = f"RF_{stn}"
    if col in df.columns:
        df[f"y_{stn}"] = (df[col] >= RAIN_MM_THRESH).astype(int)

# Correlation filter on F1 candidates (training data only)
f1_avail = [c for c in F1_CANDIDATES if c in df.columns]
train_df = df[df["year"].isin(TRAIN_YEARS)]
corr_mat = train_df[f1_avail].corr().abs()
upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
to_drop  = [c for c in upper.columns if any(upper[c] >= CORR_THRESH)]
f1_final = [c for c in f1_avail if c not in to_drop]
print(f"F1 candidates: {len(f1_avail)}  dropped: {len(to_drop)}  F1: {len(f1_final)}")

wind_avail = [c for c in WIND_FEATURES if c in df.columns]
feat_sets  = {
    "F1": f1_final,
    "F2": f1_final + ["sin_month", "cos_month"],
    "F3": f1_final + wind_avail,
    "F4": f1_final + ["sin_month", "cos_month"] + wind_avail,
    "DROPPED": to_drop,
    "_meta": {
        "F1_description": f"{len(f1_final)} stability indices after |r|>={CORR_THRESH} filter",
        "F2_description": "F1 + cyclical season encoding (sin/cos month)",
        "F3_description": f"F1 + {len(wind_avail)} wind features",
        "F4_description": "F1 + season + wind = all features",
    },
}

df.to_csv(OUT / "dataset_final.csv", index=False)
with open(OUT / "feature_sets.json", "w") as f:
    json.dump(feat_sets, f, indent=2)

print(f"Saved {len(df)} rows -> outputs/dataset_final.csv")
print(f"Saved feature_sets.json  (F1: {len(f1_final)} features)")
