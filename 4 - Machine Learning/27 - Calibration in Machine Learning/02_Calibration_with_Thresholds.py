import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, precision_score, recall_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# ==========================================
# 1. Data Loading & 3-Way Split
# ==========================================
X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, random_state=42)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

data_sizes = {'Train_Samples': len(y_train),
              'Calib_Samples': len(y_calib),
              'Test_Samples': len(y_test)
}

# ==========================================
# 2. Core Functions
# ==========================================
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0., 1., n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        if np.any(bin_mask):
            bin_size = np.sum(bin_mask)
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
    return ece

def evaluate_and_plot(y_true, y_prob, model_name, ax):
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    ax.plot(prob_pred, prob_true, marker='s', label=f'{model_name} (AUC: {roc_auc:.3f}, Brier: {brier:.3f}, ECE: {ece:.3f})')
    return brier, ece, roc_auc

def apply_platt_scaling(base_model, X_c, y_c):
    calibrator = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)
    calibrator.fit(X_c, y_c)
    return calibrator

def apply_isotonic_regression(base_model, X_c, y_c):
    calibrator = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrator.fit(X_c, y_c)
    return calibrator

# ==========================================
# 3. Summary & Logging Functions
# ==========================================
def create_threshold_summary(y_true, prob_uncal, prob_platt, prob_iso, model_name, sizes):
    """Generates the aggregated Threshold metrics table"""
    thresholds = np.arange(0.00, 1.00, 0.05)
    records = []
    prob_dict = {'Uncalibrated': prob_uncal, 'Platt Scaling': prob_platt, 'Isotonic': prob_iso}
    
    calib_metrics = {}
    for state, probs in prob_dict.items():
        calib_metrics[state] = {
            'Brier': brier_score_loss(y_true, probs),
            'ECE': calculate_ece(y_true, probs),
            'ROC_AUC': roc_auc_score(y_true, probs)
        }
    
    for t in thresholds:
        for calib_state, probs in prob_dict.items():
            preds = (probs >= t).astype(int)
            records.append({
                'Algorithm': model_name,
                'Calibration_State': calib_state,
                'Threshold': round(t, 2),
                'Brier_Score': round(calib_metrics[calib_state]['Brier'], 4),
                'ECE': round(calib_metrics[calib_state]['ECE'], 4),
                'ROC_AUC': round(calib_metrics[calib_state]['ROC_AUC'], 4),
                'Accuracy': round(accuracy_score(y_true, preds), 4),
                'Precision': round(precision_score(y_true, preds, zero_division=0), 4),
                'Recall': round(recall_score(y_true, preds, zero_division=0), 4),
                'Positive_Preds': np.sum(preds),
                **sizes
            })
    return pd.DataFrame(records)

def create_prediction_log(y_true, prob_uncal, prob_platt, prob_iso, model_name):
    """Generates the observation-level log in a LONG format (Thresholds as rows)"""
    thresholds = np.arange(0.00, 1.00, 0.05)
    all_threshold_frames = []
    
    for t in thresholds:
        # Create a dataframe for this specific threshold
        df_t = pd.DataFrame({
            'Algorithm': model_name,
            'Threshold': round(t, 2),
            'y_actual': y_true,
            'Prob_Uncalibrated': np.round(prob_uncal, 4),
            'Prob_Platt': np.round(prob_platt, 4),
            'Prob_Isotonic': np.round(prob_iso, 4),
            'Pred_Uncalibrated': (prob_uncal >= t).astype(int),
            'Pred_Platt': (prob_platt >= t).astype(int),
            'Pred_Isotonic': (prob_iso >= t).astype(int)
        })
        all_threshold_frames.append(df_t)
        
    # Concatenate all thresholds vertically
    return pd.concat(all_threshold_frames, ignore_index=True)

# ==========================================
# 4. Execution Pipeline
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB()
}

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Reliability Diagrams (Quantile Binning)", fontsize=16)

all_summaries = []
all_prediction_logs = []

for i, (name, model) in enumerate(models.items()):
    ax = axes[i]
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    model.fit(X_train, y_train)
    prob_uncal = model.predict_proba(X_test)[:, 1]
    
    platt_model = apply_platt_scaling(model, X_calib, y_calib)
    prob_platt = platt_model.predict_proba(X_test)[:, 1]
    
    iso_model = apply_isotonic_regression(model, X_calib, y_calib)
    prob_iso = iso_model.predict_proba(X_test)[:, 1]
    
    evaluate_and_plot(y_test, prob_uncal, "Uncalibrated", ax)
    evaluate_and_plot(y_test, prob_platt, "Platt Scaling", ax)
    evaluate_and_plot(y_test, prob_iso, "Isotonic", ax)
    
    ax.set_title(f"{name} Calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="lower right")
    
    # 1. Store Aggregate Data
    df_summary = create_threshold_summary(y_test, prob_uncal, prob_platt, prob_iso, name, data_sizes)
    all_summaries.append(df_summary)
    
    # 2. Store Observation-Level Data (Now in Long format)
    df_preds = create_prediction_log(y_test, prob_uncal, prob_platt, prob_iso, name)
    all_prediction_logs.append(df_preds)

plt.tight_layout()
plt.show()

# --- Display the Results ---
final_summary_df = pd.concat(all_summaries, ignore_index=True)
final_predictions_df = pd.concat(all_prediction_logs, ignore_index=True)

print("\n--- 1. Snippet of Threshold Summary (Aggregated) ---")
print(final_summary_df[(final_summary_df['Algorithm'] == 'Random Forest') & (np.isclose(final_summary_df['Threshold'], 0.60))].to_string(index=False))

print("\n--- 2. Snippet of ROW-WISE Prediction Log (Showing Random Forest at t=0.50) ---")
# Filtering the long dataframe to show just a few rows at a specific threshold
mask = (final_predictions_df['Algorithm'] == 'Random Forest') & (np.isclose(final_predictions_df['Threshold'], 0.50))
print(final_predictions_df[mask].head(8).to_string(index=False))

# ==========================================
# 5. Combined Comparison DataFrame
# ==========================================
# Goal: For each Algorithm & Threshold, show:
#   - Accuracy of each calibration method (from summary table)
#   - How many predictions changed due to calibration (from predictions table)
#   - Whether those changes were fixes or new errors

records = []

for algo in final_predictions_df['Algorithm'].unique():
    for t in final_predictions_df['Threshold'].unique():

        # --- Filter to this Algorithm + Threshold ---
        preds = final_predictions_df[
            (final_predictions_df['Algorithm'] == algo) &
            (np.isclose(final_predictions_df['Threshold'], t))
        ]
        summary = final_summary_df[
            (final_summary_df['Algorithm'] == algo) &
            (np.isclose(final_summary_df['Threshold'], t))
        ]

        # --- Pull accuracy for each calibration state from summary ---
        acc = summary.set_index('Calibration_State')['Accuracy'].to_dict()

        # --- Count how many predictions flipped and whether the flip helped ---
        uncal_correct = (preds['Pred_Uncalibrated'] == preds['y_actual'])
        platt_correct = (preds['Pred_Platt'] == preds['y_actual'])
        iso_correct   = (preds['Pred_Isotonic'] == preds['y_actual'])

        platt_flips      = (preds['Pred_Uncalibrated'] != preds['Pred_Platt']).sum()
        platt_fixed      = ((~uncal_correct) & platt_correct).sum()     # was wrong, now right
        platt_broke       = (uncal_correct & (~platt_correct)).sum()     # was right, now wrong

        iso_flips        = (preds['Pred_Uncalibrated'] != preds['Pred_Isotonic']).sum()
        iso_fixed        = ((~uncal_correct) & iso_correct).sum()
        iso_broke         = (uncal_correct & (~iso_correct)).sum()

        records.append({
            'Algorithm':        algo,
            'Threshold':        round(t, 2),
            # Accuracy comparison
            'Acc_Uncal':        acc.get('Uncalibrated', None),
            'Acc_Platt':        acc.get('Platt Scaling', None),
            'Acc_Iso':          acc.get('Isotonic', None),
            # Platt impact
            'Platt_Flips':      platt_flips,
            'Platt_Fixed':      platt_fixed,
            'Platt_Broke':      platt_broke,
            'Platt_Net':        platt_fixed - platt_broke,     # +ve = calibration helped
            # Isotonic impact
            'Iso_Flips':        iso_flips,
            'Iso_Fixed':        iso_fixed,
            'Iso_Broke':        iso_broke,
            'Iso_Net':          iso_fixed - iso_broke,         # +ve = calibration helped
        })

combined_df = pd.DataFrame(records)

# --- Display ---
print("\n--- 3. Combined Comparison (Random Forest) ---")
print("  Columns: Flips = predictions that changed, Fixed = errors corrected,")
print("           Broke = correct predictions ruined, Net = Fixed - Broke (+ve = helped)\n")
print(combined_df[combined_df['Algorithm'] == 'Random Forest'].to_string(index=False))

print("\n--- 4. Combined Comparison (Naive Bayes) ---")
print(combined_df[combined_df['Algorithm'] == 'Naive Bayes'].to_string(index=False))