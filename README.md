# rainfall-sounding-temporal-validation

A reproducible, temporally validated modelling framework for binary rainfall
occurrence prediction from radiosonde-derived atmospheric instability indices.



---

## Overview

This repository provides the complete pipeline to reproduce the results of the
associated paper, including:

- Data ingestion from the University of Wyoming sounding archive
- Feature extraction (34 atmospheric instability indices → 4 nested feature sets)
- Temporal validation: year-blocked CV, chronological holdout, LOYO
- Model comparison: LR, SVM, RF, XGB, MLP vs climatological baseline
- Calibration assessment: Brier score, reliability diagram, PR-AUC
- SHAP-based physical plausibility diagnostics

---

## Quick Start

```bash
git clone https://github.com/Thanayut-crru/rainfall-sounding-temporal-validation.git
cd rainfall-sounding-temporal-validation
conda env create -f environment.yml
conda activate rainfall-env
```

Then run scripts in order (see PIPELINE.md):

```bash
python scripts/01_preprocess.py
python scripts/02_features.py
python scripts/03_train.py
python scripts/04_validate.py
python scripts/05_shap.py
```

Demo notebook (no raw TMD data required):

```bash
jupyter notebook notebooks/demo_workflow.ipynb
```


## Repository Structure

```
rainfall-sounding-temporal-validation/
├── README.md
├── LICENSE                        MIT licence
├── requirements.txt               pip-compatible dependencies
├── environment.yml                conda environment spec
├── CITATION.cff                   citation metadata
├── PIPELINE.md                    step-by-step run order
│
├── data/
│   ├── README.md                  data access instructions
│   └── sample/
│       ├──sample_dataset.csv     sounding predictors + synthetic labels (demo) 
│
├── scripts/
│   ├── 01_preprocess.py           sounding download + QC
│   ├── 02_features.py             index extraction + temporal alignment
│   ├── 03_train.py                year-blocked CV + holdout
│   ├── 04_validate.py             LOYO + calibration + baseline
│   └── 05_shap.py                 SHAP TreeExplainer + figures
│
├── notebooks/
│   └── demo_workflow.ipynb        end-to-end demo with sample data
│
├── outputs/
    └── README.md                  expected output files
---

## Data Access

**Sounding data:** Publicly available from the University of Wyoming Upper Air
Sounding Archive (weather.uwyo.edu/upperair/). Script `01_preprocess.py`
downloads data automatically.

**Rainfall data:** Obtained from the Thai Meteorological Department (TMD) under
a data-sharing agreement. Raw data cannot be redistributed. Contact TMD
(www.tmd.go.th) for access. A sample processed dataset (no raw TMD values)
is provided in `data/sample/`.

---

## Station

| Parameter | Value |
|---|---|
| Sounding station | Phuket International Airport (WMO 48565) |
| Observation time | 00 UTC (07:00 LT) |
| Primary period | 2011–2021 |
| Rain-gauge stations | Phuket (26 km), Krabi (65 km), Phang-nga (73 km) |

---

## Requirements

Python 3.11. See `environment.yml` or `requirements.txt`.

Key packages: scikit-learn 1.4, xgboost 2.0, shap 0.44, pandas 2.1,
numpy 1.26, matplotlib 3.8, python-docx 1.1, requests, tqdm.


