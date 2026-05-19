"""Calibration in Machine Learning
This script shows what probability calibration means and why it matters.
We use the breast cancer dataset and compare 3 classifiers with 2 calibration methods (Platt Scaling and Isotonic Regression).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# ==========================================
# Utility Functions
# ==========================================
# --- Helper Function for ECE ---
def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates the Expected Calibration Error (ECE)"""
    bin_edges = np.linspace(0., 1., n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    ece = 0
    for b in range(1, n_bins + 1):
        mask = (bin_indices == b)
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(y_true)
    return ece

def apply_platt_scaling(base_model, X_calib, y_calib, X_test):
    """Applies Platt Scaling (sigmoid) calibration and returns calibrated probabilities."""
    calibrator = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)
    calibrator.fit(X_calib, y_calib)
    return calibrator, calibrator.predict_proba(X_test)[:, 1]

def apply_isotonic_regression(base_model, X_calib, y_calib, X_test):
    """Applies Isotonic Regression calibration and returns calibrated probabilities."""
    calibrator = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrator.fit(X_calib, y_calib)
    return calibrator, calibrator.predict_proba(X_test)[:, 1]

def plot_reliability_curves(y_true, prob_dict, ax, model_name):
    """Plots reliability diagram for multiple calibration states on one axis.
    
    Args:
        y_true: True labels
        prob_dict: dict of {label: probabilities}, e.g. {'Uncalibrated': prob_uncal, ...}
        ax: matplotlib axis to plot on
        model_name: title for the subplot
    """
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for label, probs in prob_dict.items():
        brier = brier_score_loss(y_true, probs)
        ece = calculate_ece(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac_pos, marker='s', label=f'{label} (AUC:{auc:.3f}, Brier:{brier:.3f}, ECE:{ece:.3f})')
    
    ax.set_title(f"{model_name}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="lower right", fontsize=7)

# ==========================================
# 1. Load Data & 3-Way Split
# ==========================================
data = load_breast_cancer()
X = data.data
y = data.target

# 3-way split: Train (60%) / Calibration (20%) / Test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

data_sizes = {'Train_Samples': len(y_train),
                'Calib_Samples': len(y_calib),
                'Test_Samples': len(y_test)
}
print(f"Split: Train={data_sizes['Train_Samples']}, Calib={data_sizes['Calib_Samples']}, Test={data_sizes['Test_Samples']}")

# ==========================================
# 2. Define Models
# ==========================================
models = {"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
          "Naive Bayes": GaussianNB(),
          "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42)
}

# ==========================================
# 3. Train, Calibrate & Collect Results
# ==========================================
all_summaries = []
all_prediction_logs = []

# Reliability diagrams: 1 subplot per model
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Reliability Diagrams (Breast Cancer Dataset)", fontsize=16)

for i, (model_name, model) in enumerate(models.items()):
    ax = axes[i]

    # --- Train base model ---
    model.fit(X_train, y_train)
    prob_uncal = model.predict_proba(X_test)[:, 1]

    # --- Apply calibration methods ---
    platt_model, prob_platt = apply_platt_scaling(model, X_calib, y_calib, X_test)
    iso_model, prob_iso = apply_isotonic_regression(model, X_calib, y_calib, X_test)

    # --- Plot reliability curves ---
    prob_dict = {'Uncalibrated': prob_uncal, 'Platt Scaling': prob_platt, 'Isotonic': prob_iso}
    plot_reliability_curves(y_test, prob_dict, ax, model_name)

    # --- DataFrame 1: Aggregate Summary (one row per calibration state) ---
    prob_dict = {'Uncalibrated': prob_uncal, 'Platt Scaling': prob_platt, 'Isotonic': prob_iso}

    for calib_state, probs in prob_dict.items():
        preds = (probs >= 0.50).astype(int)
        all_summaries.append({'Algorithm': model_name,
                              'Calibration_State': calib_state,
                              'Brier_Score': round(brier_score_loss(y_test, probs), 4),
                              'ECE': round(calculate_ece(y_test, probs), 4),
                              'ROC_AUC': round(roc_auc_score(y_test, probs), 4),
                              'Accuracy': round(accuracy_score(y_test, preds), 4),
                              'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
                              'Recall': round(recall_score(y_test, preds, zero_division=0), 4),
                              'Positive_Preds': int(np.sum(preds)),
                              **data_sizes
                              })

    # --- DataFrame 2: Row-Level Prediction Log (one row per observation) ---
    df_preds = pd.DataFrame({'Algorithm': model_name,
                              'y_actual': y_test,
                              'Prob_Uncalibrated': np.round(prob_uncal, 4),
                              'Prob_Platt': np.round(prob_platt, 4),
                              'Prob_Isotonic': np.round(prob_iso, 4),
                              'Pred_Uncalibrated': (prob_uncal >= 0.50).astype(int),
                              'Pred_Platt': (prob_platt >= 0.50).astype(int),
                              'Pred_Isotonic': (prob_iso >= 0.50).astype(int)
                              })
    all_prediction_logs.append(df_preds)

plt.tight_layout()
plt.show()

# ==========================================
# 4. Build Final DataFrames
# ==========================================
final_summary_df = pd.DataFrame(all_summaries)
final_predictions_df = pd.concat(all_prediction_logs, ignore_index=True)

# ==========================================
# 5. Combined Comparison DataFrame
# ==========================================
# For each Algorithm, show accuracy of each method + how many predictions calibration flipped and whether flips helped.

def calibration_comparison(predictions, summary):
    """
    Calculate flip and correctness metrics for each algorithm and calibration method.
    Args:
        predictions (pd.DataFrame): DataFrame with prediction columns.
        summary (pd.DataFrame): DataFrame with summary statistics.
    Returns:
        pd.DataFrame: Combined comparison with flip and correctness metrics.
    """
    records = []
    for algo in predictions['Algorithm'].unique():
        
        print("Performing Calibration Comparison on Algorithm:", algo)
        preds = predictions[predictions['Algorithm'] == algo]
        summary_df = summary[summary['Algorithm'] == algo]
        acc = summary_df.set_index('Calibration_State')['Accuracy'].to_dict()

        uncal_correct = (preds['Pred_Uncalibrated'] == preds['y_actual'])
        platt_correct = (preds['Pred_Platt'] == preds['y_actual'])
        iso_correct   = (preds['Pred_Isotonic'] == preds['y_actual'])

        platt_flips = (preds['Pred_Uncalibrated'] != preds['Pred_Platt']).sum()
        platt_fixed = ((~uncal_correct) & platt_correct).sum()
        platt_broke = (uncal_correct & (~platt_correct)).sum()

        iso_flips = (preds['Pred_Uncalibrated'] != preds['Pred_Isotonic']).sum()
        iso_fixed = ((~uncal_correct) & iso_correct).sum()
        iso_broke = (uncal_correct & (~iso_correct)).sum()

        records.append({
            'Algorithm':    algo,
            'Acc_Uncal':    acc.get('Uncalibrated', None),
            'Acc_Platt':    acc.get('Platt Scaling', None),
            'Acc_Iso':      acc.get('Isotonic', None),
            'Platt_Flips':  platt_flips,
            'Platt_Fixed':  platt_fixed,
            'Platt_Broke':  platt_broke,
            'Platt_Net':    platt_fixed - platt_broke,
            'Iso_Flips':    iso_flips,
            'Iso_Fixed':    iso_fixed,
            'Iso_Broke':    iso_broke,
            'Iso_Net':      iso_fixed - iso_broke,
        })
    return pd.DataFrame(records)

calibration_comparison_df = calibration_comparison(predictions = final_predictions_df, summary = final_summary_df)

print("--- Calibration Comparison (all models) ---")
print("Flips = predictions changed, Fixed = errors corrected,")
print("Broke = correct predictions ruined, Net = Fixed - Broke (+ve = helped)\n")
calibration_comparison_df

# ==========================================
# 6. Probability Distribution Histograms
# ==========================================
def plot_probability_distributions(models, X_train, y_train, X_calib, y_calib, X_test, y_test):
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 5))
    fig2.suptitle("Probability Distribution: Before vs After Calibration (Test Set)", fontsize=14)

    # Re-train to get probabilities for histograms
    for i, (model_name, model) in enumerate(models.items()):
        ax = axes2[i]
        model.fit(X_train, y_train)
        prob_uncal = model.predict_proba(X_test)[:, 1]

        _, prob_platt = apply_platt_scaling(model, X_calib, y_calib, X_test)

        ax.hist(prob_uncal[y_test == 1], bins=15, alpha=0.5, label='Actual 1 (Before)', color='blue')
        ax.hist(prob_uncal[y_test == 0], bins=15, alpha=0.5, label='Actual 0 (Before)', color='red')
        ax.hist(prob_platt[y_test == 1], bins=15, alpha=0.8, label='Actual 1 (After Platt)', color='cyan', histtype='step', linewidth=2)
        ax.hist(prob_platt[y_test == 0], bins=15, alpha=0.8, label='Actual 0 (After Platt)', color='orange', histtype='step', linewidth=2)

        ax.set_title(f'{model_name}')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=7)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return fig2, axes2

# Plot probability distributions
plot_probability_distributions(models, X_train, y_train, X_calib, y_calib, X_test, y_test)

# ==========================================
# 7. Simple Explanation
# ==========================================
print("\nExplanation:")
print("A calibrated model means its predicted probabilities match the true chance of the event.")
print("The Expected Calibration Error (ECE) measures the average difference between accuracy and confidence (predicted probability). Lower is better.")
print("The Brier score measures the overall mean squared error of the probability predictions: lower is better.")
print("The calibration curve compares your model against the 'Perfectly Calibrated' gray diagonal line.")