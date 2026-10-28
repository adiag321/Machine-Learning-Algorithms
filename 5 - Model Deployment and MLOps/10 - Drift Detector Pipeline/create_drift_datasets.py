# -*- coding: utf-8 -*-
"""
create_drift_datasets.py — One-time script to generate reference and current datasets.

Loads Telco_Customer_Churn.csv, splits 80/20, injects synthetic drift into
the current slice, and saves both CSVs to Datasets/Drift_Detection/.

Run this once before using the drift detector pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
_BASE        = os.path.dirname(os.path.abspath(__file__))
_ML_ROOT     = os.path.abspath(os.path.join(_BASE, "../../.."))
SOURCE_CSV   = os.path.join(_ML_ROOT, "Datasets", "Telecom", "Telco_Customer_Churn.csv")
OUTPUT_DIR   = os.path.join(_ML_ROOT, "Datasets", "Drift_Detection")
REF_CSV      = os.path.join(OUTPUT_DIR, "reference_data.csv")
CURRENT_CSV  = os.path.join(OUTPUT_DIR, "current_data.csv")

RANDOM_STATE = 42
TARGET_COL   = "Churn"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load Telco CSV and fix known data quality issues."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # TotalCharges has blank strings that must be coerced to NaN then filled
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # Encode target to 0/1 so both datasets are model-ready
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def inject_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject synthetic drift to simulate a real production shift.

    Changes applied to the current dataset:
    - tenure: +2σ shift (simulates an older customer base arriving in production)
    - MonthlyCharges: +15% increase (simulates price hike in production period)
    - PaymentMethod: proportions shuffled (simulates channel mix change)
    - Churn: 15% of labels flipped (simulates concept drift — P(Y|X) has changed)
    """
    rng = np.random.default_rng(RANDOM_STATE)
    d = df.copy()

    # Feature drift: numeric magnitude shift
    tenure_std = d["tenure"].std()
    d["tenure"] = d["tenure"] + 2.0 * tenure_std
    d["MonthlyCharges"] = d["MonthlyCharges"] * 1.15

    # Feature drift: categorical proportion shuffle
    vals = d["PaymentMethod"].values.copy()
    rng.shuffle(vals)
    d["PaymentMethod"] = vals

    # Concept drift: flip 15% of labels
    n_flip = int(0.15 * len(d))
    flip_idx = rng.choice(len(d), size=n_flip, replace=False)
    d.iloc[flip_idx, d.columns.get_loc(TARGET_COL)] = (
        1 - d.iloc[flip_idx][TARGET_COL].astype(int)
    )

    return d


def main():
    if not os.path.exists(SOURCE_CSV):
        print(f"ERROR: Source CSV not found at:\n  {SOURCE_CSV}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_clean(SOURCE_CSV)
    print(f"Loaded {len(df):,} rows from Telco_Customer_Churn.csv")

    # Stratified 80/20 split — reference is the larger baseline
    ref_df, cur_raw = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )
    ref_df = ref_df.reset_index(drop=True)
    cur_raw = cur_raw.reset_index(drop=True)

    cur_df = inject_drift(cur_raw)

    ref_df.to_csv(REF_CSV, index=False)
    cur_df.to_csv(CURRENT_CSV, index=False)

    print(f"Reference dataset saved  ({len(ref_df):,} rows): {REF_CSV}")
    print(f"Current dataset saved    ({len(cur_df):,} rows): {CURRENT_CSV}")
    print("\nDrift injected into current dataset:")
    print("  • tenure          : +2σ shift")
    print("  • MonthlyCharges  : +15% increase")
    print("  • PaymentMethod   : proportions shuffled")
    print("  • Churn (labels)  : 15% flipped (concept drift)")
    print("\nDatasets ready. You can now run: python3 run_drift_detection.py")


if __name__ == "__main__":
    main()
