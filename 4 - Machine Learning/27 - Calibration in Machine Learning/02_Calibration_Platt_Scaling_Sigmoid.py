"""Calibration in Machine Learning
This script shows what probability calibration means and why it matters.
We use a built-in scikit-learn dataset and a simple classifier.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score
import matplotlib.pyplot as plt

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

# 1. Load a sample dataset from scikit-learn.
data = load_breast_cancer()
X = data.data
y = data.target

# Create the original DataFrame
df = pd.DataFrame(X, columns=data.feature_names)
df['Actual'] = y

# 2. Split the data, keeping track of original indices to map back to the DataFrame
indices = np.arange(len(y))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.3, random_state=42, stratify=y
)

# Tag Train/Test data in the DataFrame
df['Dataset'] = 'Train'
df.loc[idx_test, 'Dataset'] = 'Test'

# 3. Train a simple classifier.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Calibrate the classifier.
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)

# 5. Populate original dataframe with all predictions and probabilities
df['Predicted_Before'] = model.predict(X)
df['Prob_Before'] = model.predict_proba(X)[:, 1]

df['Predicted_After'] = calibrated.predict(X)
df['Prob_After'] = calibrated.predict_proba(X)[:, 1]

# Extract test data metrics for fair evaluation
probs_test = df.loc[idx_test, 'Prob_Before']
preds_test = df.loc[idx_test, 'Predicted_Before']
cal_probs_test = df.loc[idx_test, 'Prob_After']
cal_preds_test = df.loc[idx_test, 'Predicted_After']

# 6. Print metrics including ECE
print("--- Before Calibration (Test Data) ---")
print("Accuracy:", round(accuracy_score(y_test, preds_test), 4))
print("Brier score:", round(brier_score_loss(y_test, probs_test), 4))
print("ECE:", round(calculate_ece(y_test, probs_test), 4))

print("\n--- After Calibration (Test Data) ---")
print("Accuracy:", round(accuracy_score(y_test, cal_preds_test), 4))
print("Brier score:", round(brier_score_loss(y_test, cal_probs_test), 4))
print("ECE:", round(calculate_ece(y_test, cal_probs_test), 4))

# 7. Compare calibration curves and probability distributions.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Reliability Diagram (Miscalibrated vs Calibrated)
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs_test, n_bins=10, cv=5)
cal_fraction, cal_mean = calibration_curve(y_test, cal_probs_test, n_bins=10)

axes[0].plot(mean_predicted_value, fraction_of_positives, marker='o', label='Before calibration (Miscalibrated)')
axes[0].plot(cal_mean, cal_fraction, marker='s', label='After calibration')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
axes[0].set_title('Calibration Curve (Reliability Diagram)')
axes[0].set_xlabel('Mean predicted probability')
axes[0].set_ylabel('Fraction of positives (Expected)')
axes[0].legend()
axes[0].grid(True)

# Plot B: Actuals vs Predicted Probabilities (Distribution)
axes[1].hist(df.loc[(df['Dataset'] == 'Test') & (df['Actual'] == 1), 'Prob_Before'], bins=15, alpha=0.5, label='Actual 1 (Before)', color='blue')
axes[1].hist(df.loc[(df['Dataset'] == 'Test') & (df['Actual'] == 0), 'Prob_Before'], bins=15, alpha=0.5, label='Actual 0 (Before)', color='red')
axes[1].hist(df.loc[(df['Dataset'] == 'Test') & (df['Actual'] == 1), 'Prob_After'], bins=15, alpha=0.8, label='Actual 1 (After)', color='cyan', histtype='step', linewidth=2)
axes[1].hist(df.loc[(df['Dataset'] == 'Test') & (df['Actual'] == 0), 'Prob_After'], bins=15, alpha=0.8, label='Actual 0 (After)', color='orange', histtype='step', linewidth=2)

axes[1].set_title('Distribution of Predicted Probabilities by Actual Class')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 8. Displaying a sample of the tracking DataFrame
print("\n--- Original DataFrame with Tracking (Sample) ---")
tracking_cols = ['Dataset', 'Actual', 'Predicted_Before', 'Prob_Before', 'Predicted_After', 'Prob_After']
print(df[tracking_cols].sample(10, random_state=42))

## Instances where Probability changed
df['Predicted_Changed'] = np.where(df['Predicted_Before'] == df['Predicted_After'], 1, 0)
df[df['Predicted_Changed']==0]

# 9. Simple explanation printed to user
print("\nExplanation:")
print("A calibrated model means its predicted probabilities match the true chance of the event.")
print("The Expected Calibration Error (ECE) measures the average difference between accuracy and confidence (predicted probability). Lower is better.")
print("The Brier score measures the overall mean squared error of the probability predictions: lower is better.")
print("The calibration curve compares your model against the 'Perfectly Calibrated' gray diagonal line.")