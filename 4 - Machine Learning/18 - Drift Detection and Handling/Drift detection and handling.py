# -*- coding: utf-8 -*-
"""Telco churn — drift simulation, detection and handling workflow."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

project_dir = "D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms"
os.chdir(project_dir)

##############################################
#                 Config 
##############################################
TELCO_CSV_PATH = "./Datasets/Telecom/Telco_Customer_Churn.csv"
RANDOM_STATE = 42

# Detection thresholds (tweak for your use case)
KS_P_THRESHOLD = 0.05
PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25
AUC_DROP_THRESHOLD = 0.05  # 5% drop considered notable
FEATURES_TO_MONITOR = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]  # sample numeric features
##############################################


df = pd.read_csv(TELCO_CSV_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"], errors="ignore")
y = df["Churn"]

# baseline split
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
print(f"Train size: {len(X_train)}, Holdout size: {len(X_holdout)}")

# ---------- Preprocessing + model helpers ----------
def build_pipeline(X: pd.DataFrame, random_state: int = RANDOM_STATE) -> Pipeline:
    # simple preprocessing: numeric scaling + one-hot for categorical
    num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Avoid extremely high cardinality encoding in this simple demo
    preproc = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ], remainder='drop')

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    pipeline = Pipeline([
        ("preproc", preproc),
        ("clf", model)
    ])
    return pipeline

def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    return pipeline

# Train baseline
model = train_baseline(X_train, y_train)
y_holdout_pred = model.predict_proba(X_holdout)[:, 1]
base_auc = roc_auc_score(y_holdout, y_holdout_pred)
print(f"Baseline AUC on holdout: {base_auc:.4f}")
print("----")

##############################################
#               DATA DRIFT 
##############################################
## Simulate Data Drift -> Changing the data
def simulate_data_drift_age_shift(X: pd.DataFrame, years_shift: int = 10) -> pd.DataFrame:
    Xd = X.copy()
    # here tenure is months of service; let's artificially increase tenure or shift numeric features
    Xd["tenure"] = Xd["tenure"] + years_shift  # illustrative shift
    
    if "SeniorCitizen" in Xd.columns and years_shift != 0:
        # small change in senior citizen distribution
        # flip a small fraction
        prob_flip = min(0.05, years_shift / 100)
        mask = np.random.rand(len(Xd)) < prob_flip
        Xd.loc[mask, "SeniorCitizen"] = 1

    # shift monthly charges a bit
    Xd["MonthlyCharges"] = Xd["MonthlyCharges"] * (1.0 + 0.05 * (years_shift / 10.0))
    return Xd

# ---------- PSI implementation ----------
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two numeric arrays. Uses percentile binning of expected distribution for stable bins.
    """
    eps = 1e-8
    expected = np.array(expected).ravel()
    actual = np.array(actual).ravel()

    # create bins using expected percentiles
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, quantiles)
    # ensure monotonic increasing breakpoints
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_perc = exp_counts / (len(expected) + eps)
    act_perc = act_counts / (len(actual) + eps)

    # add small value to avoid division by zero or log of zero
    psi_values = (exp_perc - act_perc) * np.log((exp_perc + eps) / (act_perc + eps))
    psi = np.sum(psi_values)
    return float(psi)

# ---------- Data drift detection ----------
def detect_data_drift(train_X: pd.DataFrame, current_X: pd.DataFrame, features: List[str] = None) -> Dict[str, Dict]:
    """
    Runs KS test and PSI for each feature in features (or numeric/categorical default set).
    Returns a dict with feature -> {ks_p, psi, drift_flag}
    """
    results = {}
    if features is None:
        # choose numeric features by default
        features = train_X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for f in features:
        if f not in train_X.columns or f not in current_X.columns:
            continue
        train_vals = train_X[f].dropna()
        cur_vals = current_X[f].dropna()

        try:
            ks_stat, ks_p = ks_2samp(train_vals, cur_vals)
        except Exception:
            ks_p = 1.0

        psi = compute_psi(train_vals.values, cur_vals.values, bins=10)
        drift_flag = (ks_p < KS_P_THRESHOLD) or (psi > PSI_WARN_THRESHOLD)

        results[f] = {"ks_p": float(ks_p), "psi": float(psi), "drift_flag": bool(drift_flag)}
    return results


def handle_data_drift_retrain(X_reference: pd.DataFrame, y_reference: pd.Series,
                              X_current: pd.DataFrame, y_current: pd.Series) -> Pipeline:
    """
    Simple handling: retrain the model on (reference + recent) or only recent data.
    Return new fitted pipeline.
    """
    # Here we retrain on combined, but you might prefer only recent (sliding window)
    X_comb = pd.concat([X_reference, X_current], ignore_index=True)
    y_comb = pd.concat([y_reference, y_current], ignore_index=True)
    pipeline = build_pipeline(X_comb)
    pipeline.fit(X_comb, y_comb)
    return pipeline

print("Simulating Data Drift (feature shift)...")
X_data_drift = simulate_data_drift_age_shift(X_holdout, years_shift=10)
# Detect data drift
data_drift_results = detect_data_drift(X_train, X_data_drift, features=FEATURES_TO_MONITOR)
print("Data drift detection (KS p / PSI / drift_flag):")
for f, v in data_drift_results.items():
    print(f" {f}: ks_p={v['ks_p']:.4f}, psi={v['psi']:.4f}, drift={v['drift_flag']}")

# Evaluate model performance on data-drifted dataset
auc_drift_data = roc_auc_score(y_holdout, model.predict_proba(X_data_drift)[:, 1])
print(f"AUC on data-drifted set: {auc_drift_data:.4f}, drop={base_auc - auc_drift_data:.4f}")

# Handle data drift: retrain on combined (simple strategy)
print("Handling data drift by retraining on combined reference + recent...")
model_after_data = handle_data_drift_retrain(X_train, y_train, X_data_drift, y_holdout)
auc_after_retrain = roc_auc_score(y_holdout, model_after_data.predict_proba(X_data_drift)[:, 1])
print(f"AUC after retrain: {auc_after_retrain:.4f} (should improve)")



# ---------- Concept drift detection ----------
def detect_concept_drift_by_performance(model: Pipeline, X_ref: pd.DataFrame, y_ref: pd.Series,
                                        X_curr: pd.DataFrame, y_curr: pd.Series, auc_drop_thresh: float = AUC_DROP_THRESHOLD) -> Dict:
    """
    Simple and practical: compare AUC on reference (validation) vs current.
    """
    y_pred_ref = model.predict_proba(X_ref)[:, 1]
    y_pred_curr = model.predict_proba(X_curr)[:, 1]
    auc_ref = roc_auc_score(y_ref, y_pred_ref)
    auc_curr = roc_auc_score(y_curr, y_pred_curr)
    drop = auc_ref - auc_curr
    drift_flag = drop > auc_drop_thresh
    return {"auc_ref": float(auc_ref), "auc_curr": float(auc_curr), "auc_drop": float(drop), "drift_flag": bool(drift_flag)}

def detect_concept_drift_by_label_feature_classifier(X_ref: pd.DataFrame, y_ref: pd.Series,
                                                     X_curr: pd.DataFrame, y_curr: pd.Series, threshold_auc: float = 0.7) -> Dict:
    """
    Train a classifier to distinguish (X,y) from ref vs curr. If classifier can separate them well, concept drift
    (joint distribution P(X,y) changed) is likely.
    """
    # combine features and label as additional column(s)
    def add_label_column(X, y):
        df = X.copy()
        df["_label_target"] = y.values
        return df

    df_ref = add_label_column(X_ref, y_ref)
    df_curr = add_label_column(X_curr, y_curr)
    df_ref["_is_new"] = 0
    df_curr["_is_new"] = 1

    combined = pd.concat([df_ref, df_curr], ignore_index=True)
    y_meta = combined["_is_new"]
    combined = combined.drop(columns=["_is_new"])

    # Simple preprocessing: fill na, convert objects to categories encoded by pandas
    combined = combined.fillna(-999)
    for col in combined.select_dtypes(include=["object"]).columns:
        combined[col] = combined[col].astype("category").cat.codes

    # Split and train a fast classifier
    X_tr, X_te, y_tr, y_te = train_test_split(combined, y_meta, test_size=0.3, random_state=RANDOM_STATE, stratify=y_meta)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict_proba(X_te)[:, 1]
    meta_auc = roc_auc_score(y_te, y_pred)

    drift_flag = meta_auc > threshold_auc
    return {"meta_auc": float(meta_auc), "drift_flag": bool(drift_flag)}

# ---------- Label drift detection ----------
def detect_label_drift(y_train: pd.Series, y_curr: pd.Series, threshold: float = 0.05) -> Dict:
    train_rate = float(np.mean(y_train))
    curr_rate = float(np.mean(y_curr))
    delta = abs(curr_rate - train_rate)
    flag = delta > threshold
    return {"train_rate": train_rate, "curr_rate": curr_rate, "delta": delta, "drift_flag": bool(flag)}

# ---------- Handlers ----------


def handle_concept_drift_retrain_recent(X_recent: pd.DataFrame, y_recent: pd.Series) -> Pipeline:
    """
    Retrain model using only recent data (sliding window strategy).
    """
    pipeline = build_pipeline(X_recent)
    pipeline.fit(X_recent, y_recent)
    return pipeline

def recalibrate_probs_bayes(pred_probs: np.ndarray, old_rate: float, new_rate: float) -> np.ndarray:
    """
    Bayes update to adjust predicted probabilities to new base rate.
    Works when model P(y|X) is stable but P(y) changed.
    """
    eps = 1e-8
    pred_probs = np.clip(pred_probs, eps, 1 - eps)
    odds_old = pred_probs / (1 - pred_probs)
    calibrated_odds = odds_old * (new_rate / (1 - new_rate)) / (old_rate / (1 - old_rate))
    calibrated = calibrated_odds / (1 + calibrated_odds)
    return calibrated

def adjust_threshold_by_metric(y_true: pd.Series, pred_probs: np.ndarray, metric: str = "f1") -> float:
    """
    Simple threshold selector using precision-recall curve (maximize F1-like product).
    Returns chosen threshold.
    """
    prec, rec, thr = precision_recall_curve(y_true, pred_probs)
    # compute F1-like score
    f1_scores = (2 * prec * rec) / (prec + rec + 1e-8)
    idx = np.nanargmax(f1_scores)
    if idx >= len(thr):
        # fallback
        return 0.5
    return float(thr[idx])

# ---------- Simulation functions ----------

def simulate_concept_drift_flip_balance_labels(X: pd.DataFrame, y: pd.Series, fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.Series]:
    Xc = X.copy()
    yc = y.copy().reset_index(drop=True)
    Xc = Xc.reset_index(drop=True)
    if "MonthlyCharges" in Xc.columns:
        high_mask = Xc["MonthlyCharges"] > Xc["MonthlyCharges"].quantile(0.8)
        high_idx = Xc[high_mask].sample(frac=fraction, random_state=RANDOM_STATE).index
        yc.loc[high_idx] = 1 - yc.loc[high_idx]  # flip labels for subset
    else:
        idx = yc.sample(frac=fraction, random_state=RANDOM_STATE).index
        yc.loc[idx] = 1 - yc.loc[idx]
    return Xc, yc

def simulate_label_drift_increase_churn(y: pd.Series, increase_frac: float = 0.1) -> pd.Series:
    y_new = y.copy().reset_index(drop=True)
    # take random subset of non-churners and mark churn
    non_idx = y_new[y_new == 0].sample(frac=increase_frac, random_state=RANDOM_STATE).index
    y_new.loc[non_idx] = 1
    return y_new
