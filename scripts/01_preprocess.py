# -*- coding: utf-8 -*-
"""
01_preprocess.py
================
Downloads 00 UTC radiosonde soundings for WMO station 48565 (Phuket)
from the University of Wyoming Upper Air Sounding Archive, parses all
station-information and sounding-index fields, applies basic quality
filters, and writes a tidy CSV of daily sounding indices.

Usage
-----
    python scripts/01_preprocess.py                     # 1-year backward from today
    python scripts/01_preprocess.py 2011-01-01 2021-12-31  # custom date range

Output
------
    outputs/sounding_indices_raw.csv
    outputs/failed_downloads.csv

Note
----
The Wyoming archive rate-limits repeated requests.  The script adds a
random 2-4 s sleep between calls; expect ~30-90 min for a full year.
"""

import argparse
import os
import re
import time
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
STATION   = "48565"
REGION    = "seasia"
HOUR_UTC  = "00"

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)
RAW_DIR = OUT / "sounding_raw_text"
RAW_DIR.mkdir(exist_ok=True)

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    today = datetime.utcnow().date()
    parser = argparse.ArgumentParser(description="Download Wyoming sounding indices.")
    parser.add_argument("start", nargs="?",
                        default=(today - timedelta(days=365)).strftime("%Y-%m-%d"),
                        help="Start date YYYY-MM-DD (default: 1 year ago)")
    parser.add_argument("end", nargs="?",
                        default=today.strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today)")
    return parser.parse_args()

# ── Wyoming URL builder ────────────────────────────────────────────────────────
def build_url(date_obj):
    ddhh = f"{date_obj.day:02d}{HOUR_UTC}"
    return (
        "https://weather.uwyo.edu/cgi-bin/sounding"
        f"?region={REGION}&TYPE=TEXT%3ALIST"
        f"&YEAR={date_obj.year}&MONTH={date_obj.month:02d}"
        f"&FROM={ddhh}&TO={ddhh}&STNM={STATION}"
    )

# ── Downloader with retry ──────────────────────────────────────────────────────
BAD_MARKERS = ["Can't get", "No data", "Invalid", "ERROR",
               "Server Error", "503 Service", "Service Unavailable"]

def download(date_obj, retries=4):
    tag      = date_obj.strftime("%Y%m%d") + HOUR_UTC
    raw_path = RAW_DIR / f"{tag}_{STATION}.txt"
    url      = build_url(date_obj)

    if raw_path.exists() and raw_path.stat().st_size > 1000:
        return raw_path.read_text(encoding="utf-8", errors="ignore"), "cached"

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=45)
            text = r.text
            ok = (r.status_code == 200
                  and len(text) > 1000
                  and "Station information and sounding indices" in text
                  and not any(m in text for m in BAD_MARKERS))
            if ok:
                raw_path.write_text(text, encoding="utf-8", errors="ignore")
                return text, "downloaded"
        except Exception:
            pass
        time.sleep(random.uniform(2, 5))
    return None, "failed"

# ── Parser ─────────────────────────────────────────────────────────────────────
PATTERNS = {
    "SHOW": r"Showalter index:\s*([-+]?\d+\.?\d*)",
    "LIFT": r"Lifted index:\s*([-+]?\d+\.?\d*)",
    "LIFV": r"LIFT computed using virtual temperature:\s*([-+]?\d+\.?\d*)",
    "SWET": r"SWEAT index:\s*([-+]?\d+\.?\d*)",
    "KINX": r"K index:\s*([-+]?\d+\.?\d*)",
    "CTOT": r"Cross totals index:\s*([-+]?\d+\.?\d*)",
    "VTOT": r"Vertical totals index:\s*([-+]?\d+\.?\d*)",
    "TTOT": r"Totals totals index:\s*([-+]?\d+\.?\d*)",
    "CAPE": r"Convective Available Potential Energy:\s*([-+]?\d+\.?\d*)",
    "CAPV": r"CAPE using virtual temperature:\s*([-+]?\d+\.?\d*)",
    "CINS": r"Convective Inhibition:\s*([-+]?\d+\.?\d*)",
    "CINV": r"CINS using virtual temperature:\s*([-+]?\d+\.?\d*)",
    "EQLV": r"Equilibrum Level:\s*([-+]?\d+\.?\d*)",
    "EQTV": r"Equilibrum Level using virtual temperature:\s*([-+]?\d+\.?\d*)",
    "LFCT": r"Level of Free Convection:\s*([-+]?\d+\.?\d*)",
    "LFCV": r"LFCT using virtual temperature:\s*([-+]?\d+\.?\d*)",
    "BRCH": r"Bulk Richardson Number:\s*([-+]?\d+\.?\d*)",
    "BRCV": r"Bulk Richardson Number using CAPV:\s*([-+]?\d+\.?\d*)",
    "LCLT": r"Temp \[K\] of the Lifted Condensation Level:\s*([-+]?\d+\.?\d*)",
    "LCLP": r"Pres \[hPa\] of the Lifted Condensation Level:\s*([-+]?\d+\.?\d*)",
    "Equivalent": r"Equivalent potential temp \[K\] of the LCL:\s*([-+]?\d+\.?\d*)",
    "MLTH": r"Mean mixed layer potential temperature:\s*([-+]?\d+\.?\d*)",
    "MLMR": r"Mean mixed layer mixing ratio:\s*([-+]?\d+\.?\d*)",
    "THTK": r"1000 hPa to 500 hPa thickness:\s*([-+]?\d+\.?\d*)",
    "PWAT": r"Precipitable water \[mm\] for entire sounding:\s*([-+]?\d+\.?\d*)",
}

def parse_indices(text):
    row = {}
    for col, pat in PATTERNS.items():
        m = re.search(pat, text, re.IGNORECASE)
        row[col] = float(m.group(1)) if m else np.nan
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    print(f"Station {STATION} | {args.start} to {args.end} | {len(dates)} dates")

    rows, failed = [], []
    for date_obj in dates:
        iso  = date_obj.strftime("%Y-%m-%d")
        text, status = download(date_obj)
        if text is None:
            failed.append({"date": iso, "status": status})
            continue
        row = {"date": iso}
        row.update(parse_indices(text))
        row["status"] = status
        rows.append(row)
        if status == "downloaded":
            time.sleep(random.uniform(1.5, 3.0))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "sounding_indices_raw.csv", index=False)
    pd.DataFrame(failed).to_csv(OUT / "failed_downloads.csv", index=False)

    print(f"Saved {len(df_out)} rows -> outputs/sounding_indices_raw.csv")
    if failed:
        print(f"  {len(failed)} dates failed -> outputs/failed_downloads.csv")

if __name__ == "__main__":
    main()
