# -*- coding: utf-8 -*-
"""utils.py — Data loading, column-type detection, and batch-splitting helpers."""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from drift_detector.config import get_logger

log = get_logger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV, normalise column names, and replace blank strings with NaN."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols — {path}")
    return df


def encode_binary_target(series: pd.Series) -> pd.Series:
    """
    Encode a binary target column to int 0/1.

    Handles Yes/No, True/False, and already-numeric 0/1.
    Raises ValueError if the column contains unrecognised values.
    """
    unique_vals = series.dropna().unique()

    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        return series.astype(int)

    yes_words = {"yes", "true", "1", "positive", "churn", "fraud", "default"}
    no_words  = {"no", "false", "0", "negative"}
    encoded   = series.copy()

    for v in unique_vals:
        key = str(v).strip().lower()
        if key in yes_words:
            encoded = encoded.replace(v, 1)
        elif key in no_words:
            encoded = encoded.replace(v, 0)
        else:
            raise ValueError(
                f"Cannot auto-encode target value '{v}'. Pre-encode to 0/1 before running."
            )

    return encoded.astype(int)


def auto_detect_column_types(
    df: pd.DataFrame,
    target_col: str = None,
    max_categories: int = 50,
) -> Dict[str, List[str]]:
    """
    Classify all columns as numeric or categorical without manual labelling.

    Rules:
    - Object/bool/category dtypes → categorical
    - Numeric with ≤ max_categories unique values → categorical (low-cardinality ordinal)
    - Numeric with > max_categories unique values → numeric
    - Target column and all-unique columns (ID-like) → excluded

    Returns dict with keys: 'numeric', 'categorical', 'excluded'.
    """
    excluded = []
    if target_col and target_col in df.columns:
        excluded.append(target_col)

    # Exclude ID-like columns — every value is unique, making drift tests meaningless
    for col in df.columns:
        if col not in excluded and df[col].nunique() == len(df):
            excluded.append(col)

    numeric_cols, categorical_cols = [], []

    for col in df.columns:
        if col in excluded:
            continue
        dtype = df[col].dtype
        if dtype in [object, bool] or str(dtype) == "category":
            categorical_cols.append(col)
        elif np.issubdtype(dtype, np.number):
            categorical_cols.append(col) if df[col].nunique() <= max_categories else numeric_cols.append(col)
        else:
            excluded.append(col)  # datetime and other unsupported types

    log.info(f"Column types — numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}, excluded: {len(excluded)}")
    return {"numeric": numeric_cols, "categorical": categorical_cols, "excluded": excluded}


def align_columns(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reduce both DataFrames to their shared columns.

    Warns on schema mismatches so misaligned deployments are surfaced immediately.
    """
    ref_cols, cur_cols = set(ref_df.columns), set(cur_df.columns)

    if only_ref := ref_cols - cur_cols:
        log.warning(f"Columns in reference only (missing in current): {sorted(only_ref)}")
    if only_cur := cur_cols - ref_cols:
        log.warning(f"Columns in current only (missing in reference): {sorted(only_cur)}")

    common = list(ref_cols & cur_cols)
    return ref_df[common], cur_df[common]


def split_into_batches(
    df: pd.DataFrame,
    timestamp_col: str = None,
    n_batches: int = 5,
) -> List[pd.DataFrame]:
    """
    Split current data into time-ordered batches for drift trend analysis.

    Sorts by timestamp_col first if provided; otherwise uses row order.
    Used to track how drift evolves over time rather than as a single snapshot.
    """
    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    batches = np.array_split(df, n_batches)
    log.info(f"Split into {n_batches} batches (~{len(batches[0]):,} rows each)")
    return batches
