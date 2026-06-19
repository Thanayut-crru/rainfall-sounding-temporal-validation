# Pipeline — Run Order

Run scripts in numbered order. Each script saves outputs to `outputs/` and
reads inputs from the previous step.

## Prerequisites

```bash
conda env create -f environment.yml
conda activate rainfall-env
```

Place raw data in:
- Sounding data: downloaded automatically by `01_preprocess.py`
- Rainfall data: `data/raw/rainfall_tmd.csv` (TMD, not included — see data/README.md)

---

## Step 1 — Preprocess (`01_preprocess.py`)

Downloads WMO 48565 soundings from University of Wyoming archive,
parses sounding indices, applies quality control.

**Output:** `outputs/sounding_indices_raw.csv`

---

## Step 2 — Feature Engineering (`02_features.py`)

Aligns sounding indices with daily rainfall observations,
computes wind-profile features, assembles F1–F4 feature sets,
applies correlation screening (|r| >= 0.90) on training data only.

**Input:** `outputs/sounding_indices_raw.csv` + rainfall data  
**Output:** `outputs/dataset_final.csv`, `outputs/feature_sets.json`

---

## Step 3 — Model Training & Validation (`03_train.py`)

Runs year-blocked 5-fold GroupKFold CV (2011–2019) and
chronological holdout evaluation (2020–2021) for all
model × feature-set combinations.

**Input:** `outputs/dataset_final.csv`, `outputs/feature_sets.json`  
**Output:** `outputs/binary_cv_results.csv`, `outputs/binary_test_results.csv`,
`outputs/multiclass_cv_results.csv`, `outputs/multiclass_test_results.csv`,
`outputs/data_split_summary.csv`

---

## Step 4 — Validation & Calibration (`04_validate.py`)

Runs LOYO analysis, seasonal breakdown, calibration metrics
(Brier score, reliability diagram, PR-AUC), bootstrap CIs,
distance analysis, and Wilcoxon/DeLong significance tests.

**Input:** `outputs/dataset_final.csv`, `outputs/binary_test_results.csv`  
**Output:** `outputs/loyo_results.csv`, `outputs/seasonal_results.csv`,
`outputs/calibration_results.csv`, `outputs/delong_results.csv`,
`outputs/wilcoxon_results.csv`, figures in `outputs/figures/`

---

## Step 5 — SHAP Diagnostics (`05_shap.py`)

Computes TreeSHAP values for XGB+F3 on the holdout test set,
generates beeswarm and bar plots.

**Input:** `outputs/dataset_final.csv`, `outputs/binary_test_results.csv`  
**Output:** `outputs/shap_feature_rank.csv`, SHAP figures

---

## Expected Runtime

| Script | Approx. time |
|---|---|
| 01_preprocess.py | 30–90 min (network dependent) |
| 02_features.py | < 1 min |
| 03_train.py | 5–15 min |
| 04_validate.py | 5–10 min |
| 05_shap.py | 2–5 min |
