# Data

## Sounding Data

Publicly available from the University of Wyoming Upper Air Sounding Archive:
https://weather.uwyo.edu/upperair/

Script `01_preprocess.py` downloads data automatically for station WMO 48565.

## Rainfall Data

Obtained from the Thai Meteorological Department (TMD) under a data-sharing
agreement. Raw data **cannot be redistributed**.

To replicate the full pipeline from raw inputs, contact TMD:
https://www.tmd.go.th

## Sample Data (this folder)

`sample/sample_sounding_indices.csv` — processed sounding indices (no raw TMD values).
`sample/sample_labels.csv` — anonymised binary rain/no-rain labels (aggregated monthly).

These files allow running `demo_workflow.ipynb` without TMD access.

### sample_sounding_indices.csv columns

| Column | Description |
|---|---|
| date | YYYY-MM-DD |
| SHOW | Showalter index |
| LIFT | Lifted index |
| SWET | SWEAT index |
| KINX | K-index |
| CAPE | CAPE (J/kg) |
| PWAT | Precipitable water (mm) |
| WDcos850 | cos(850 hPa wind direction) |
| WDsin850 | sin(850 hPa wind direction) |
| WS850 | 850 hPa wind speed (kt) |
| ... | (see feature_sets.json for full list) |
