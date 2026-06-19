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

`sample/sample_dataset.csv` — sample processed dataset containing sounding-derived
predictors (real WMO 48565 indices) and synthetic rainfall labels. These data allow
the demo workflow (`notebooks/demo_workflow.ipynb`) to run end-to-end without raw
TMD rainfall observations.

### sample_dataset.csv columns

| Column | Description |
|---|---|
| date | YYYY-MM-DD |
| SHOW | Showalter index |
| LIFT | Lifted index |
| SWET | SWEAT index |
| KINX | K-index |
| VTOT | Vertical totals index |
| CAPE | CAPE (J/kg) |
| CINS | Convective inhibition (J/kg) |
| PWAT | Precipitable water (mm) |
| WDcos850 | cos(850 hPa wind direction) |
| WDsin850 | sin(850 hPa wind direction) |
| WS850 | 850 hPa wind speed (kt) |
| RF_phuket | Synthetic daily rainfall label — Phuket (mm) |
| RF_krabi | Synthetic daily rainfall label — Krabi (mm) |
| RF_phangnga | Synthetic daily rainfall label — Phang-nga (mm) |
| ... | (see scripts/02_features.py for full column list) |

> **Note:** RF_* columns contain **synthetic** rainfall amounts derived from PWAT
> and seasonal signals. They replicate the statistical structure of the real TMD
> data but are not real observations. Use for demonstration only.
