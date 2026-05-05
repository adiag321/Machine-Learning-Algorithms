"""
Simple calibration example without custom functions.
This script:
1. Trains 2 models
2. Collects raw probability scores
3. Compares probabilities with actual labels
4. Creates reliability diagrams and calculates ECE
5. Applies Platt scaling and isotonic regression
6. Builds a detailed dataframe with all important columns
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BINS = 10
CONFIDENCE_THRESHOLD = 0.70

# 1. Load sample dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# 2. Create 2 algorithms
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    ),
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    ),
}

summary_rows = []
all_dataframes = []
plot_results = []

# 3. Train each model and apply calibration
for model_name, model in models.items():
    # Raw model
    model.fit(X_train, y_train)
    raw_train_prob = model.predict_proba(X_train)[:, 1]
    raw_test_prob = model.predict_proba(X_test)[:, 1]

    # Platt scaling
    platt_model = CalibratedClassifierCV(model, method="sigmoid", cv=5)
    platt_model.fit(X_train, y_train)
    platt_train_prob = platt_model.predict_proba(X_train)[:, 1]
    platt_test_prob = platt_model.predict_proba(X_test)[:, 1]

    # Isotonic regression
    isotonic_model = CalibratedClassifierCV(model, method="isotonic", cv=5)
    isotonic_model.fit(X_train, y_train)
    isotonic_train_prob = isotonic_model.predict_proba(X_train)[:, 1]
    isotonic_test_prob = isotonic_model.predict_proba(X_test)[:, 1]

    # Convert probabilities to labels
    raw_test_pred = (raw_test_prob >= 0.5).astype(int)
    platt_test_pred = (platt_test_prob >= 0.5).astype(int)
    isotonic_test_pred = (isotonic_test_prob >= 0.5).astype(int)

    # Accuracy
    raw_accuracy = accuracy_score(y_test, raw_test_pred)
    platt_accuracy = accuracy_score(y_test, platt_test_pred)
    isotonic_accuracy = accuracy_score(y_test, isotonic_test_pred)

    # Brier score
    raw_brier = brier_score_loss(y_test, raw_test_prob)
    platt_brier = brier_score_loss(y_test, platt_test_prob)
    isotonic_brier = brier_score_loss(y_test, isotonic_test_prob)

    # ECE for raw probabilities
    raw_ece = 0.0
    raw_bin_edges = np.linspace(0, 1, N_BINS + 1)
    raw_y_true_array = y_test.to_numpy()
    for i in range(N_BINS):
        left_edge = raw_bin_edges[i]
        right_edge = raw_bin_edges[i + 1]
        raw_in_bin = (raw_test_prob >= left_edge) & (raw_test_prob < right_edge)
        if i == N_BINS - 1:
            raw_in_bin = (raw_test_prob >= left_edge) & (raw_test_prob <= right_edge)
        if raw_in_bin.sum() > 0:
            raw_bin_accuracy = raw_y_true_array[raw_in_bin].mean()
            raw_bin_confidence = raw_test_prob[raw_in_bin].mean()
            raw_bin_weight = raw_in_bin.mean()
            raw_ece += abs(raw_bin_accuracy - raw_bin_confidence) * raw_bin_weight

    # ECE for Platt scaling
    platt_ece = 0.0
    platt_bin_edges = np.linspace(0, 1, N_BINS + 1)
    for i in range(N_BINS):
        left_edge = platt_bin_edges[i]
        right_edge = platt_bin_edges[i + 1]
        platt_in_bin = (platt_test_prob >= left_edge) & (platt_test_prob < right_edge)
        if i == N_BINS - 1:
            platt_in_bin = (platt_test_prob >= left_edge) & (platt_test_prob <= right_edge)
        if platt_in_bin.sum() > 0:
            platt_bin_accuracy = raw_y_true_array[platt_in_bin].mean()
            platt_bin_confidence = platt_test_prob[platt_in_bin].mean()
            platt_bin_weight = platt_in_bin.mean()
            platt_ece += abs(platt_bin_accuracy - platt_bin_confidence) * platt_bin_weight

    # ECE for Isotonic regression
    isotonic_ece = 0.0
    isotonic_bin_edges = np.linspace(0, 1, N_BINS + 1)
    for i in range(N_BINS):
        left_edge = isotonic_bin_edges[i]
        right_edge = isotonic_bin_edges[i + 1]
        isotonic_in_bin = (isotonic_test_prob >= left_edge) & (isotonic_test_prob < right_edge)
        if i == N_BINS - 1:
            isotonic_in_bin = (isotonic_test_prob >= left_edge) & (isotonic_test_prob <= right_edge)
        if isotonic_in_bin.sum() > 0:
            isotonic_bin_accuracy = raw_y_true_array[isotonic_in_bin].mean()
            isotonic_bin_confidence = isotonic_test_prob[isotonic_in_bin].mean()
            isotonic_bin_weight = isotonic_in_bin.mean()
            isotonic_ece += abs(isotonic_bin_accuracy - isotonic_bin_confidence) * isotonic_bin_weight

    # Summary table rows
    summary_rows.append(
        {
            "model_name": model_name,
            "version": "raw",
            "accuracy": raw_accuracy,
            "brier_score": raw_brier,
            "ece": raw_ece,
        }
    )
    summary_rows.append(
        {
            "model_name": model_name,
            "version": "platt_scaling",
            "accuracy": platt_accuracy,
            "brier_score": platt_brier,
            "ece": platt_ece,
        }
    )
    summary_rows.append(
        {
            "model_name": model_name,
            "version": "isotonic_regression",
            "accuracy": isotonic_accuracy,
            "brier_score": isotonic_brier,
            "ece": isotonic_ece,
        }
    )

    # Train dataframe
    train_df = X_train.copy().reset_index(drop=True)
    train_y = y_train.reset_index(drop=True)
    train_raw_pred = (raw_train_prob >= 0.5).astype(int)
    train_confidence = np.maximum(raw_train_prob, 1 - raw_train_prob)

    train_df["actual_label"] = train_y
    train_df["train_test"] = "train"
    train_df["model_name"] = model_name
    train_df["predicted_label_raw"] = train_raw_pred
    train_df["probability_score_raw"] = raw_train_prob
    train_df["confidence_score_raw"] = train_confidence
    train_df["model_is_confident"] = train_confidence >= CONFIDENCE_THRESHOLD
    train_df["is_prediction_correct_raw"] = train_raw_pred == train_y
    train_df["probability_vs_actual_gap_raw"] = np.abs(train_y - raw_train_prob)
    train_df["probability_score_platt"] = platt_train_prob
    train_df["probability_score_isotonic"] = isotonic_train_prob
    train_df["predicted_label_platt"] = (platt_train_prob >= 0.5).astype(int)
    train_df["predicted_label_isotonic"] = (isotonic_train_prob >= 0.5).astype(int)
    train_df["brier_score_raw"] = np.nan
    train_df["brier_score_platt"] = np.nan
    train_df["brier_score_isotonic"] = np.nan
    train_df["ece_raw"] = np.nan
    train_df["ece_platt"] = np.nan
    train_df["ece_isotonic"] = np.nan

    # Test dataframe
    test_df = X_test.copy().reset_index(drop=True)
    test_y = y_test.reset_index(drop=True)
    test_raw_pred = (raw_test_prob >= 0.5).astype(int)
    test_confidence = np.maximum(raw_test_prob, 1 - raw_test_prob)

    test_df["actual_label"] = test_y
    test_df["train_test"] = "test"
    test_df["model_name"] = model_name
    test_df["predicted_label_raw"] = test_raw_pred
    test_df["probability_score_raw"] = raw_test_prob
    test_df["confidence_score_raw"] = test_confidence
    test_df["model_is_confident"] = test_confidence >= CONFIDENCE_THRESHOLD
    test_df["is_prediction_correct_raw"] = test_raw_pred == test_y
    test_df["probability_vs_actual_gap_raw"] = np.abs(test_y - raw_test_prob)
    test_df["probability_score_platt"] = platt_test_prob
    test_df["probability_score_isotonic"] = isotonic_test_prob
    test_df["predicted_label_platt"] = (platt_test_prob >= 0.5).astype(int)
    test_df["predicted_label_isotonic"] = (isotonic_test_prob >= 0.5).astype(int)
    test_df["brier_score_raw"] = raw_brier
    test_df["brier_score_platt"] = platt_brier
    test_df["brier_score_isotonic"] = isotonic_brier
    test_df["ece_raw"] = raw_ece
    test_df["ece_platt"] = platt_ece
    test_df["ece_isotonic"] = isotonic_ece

    all_dataframes.append(train_df)
    all_dataframes.append(test_df)

    plot_results.append(
        {
            "model_name": model_name,
            "raw_test_prob": raw_test_prob,
            "platt_test_prob": platt_test_prob,
            "isotonic_test_prob": isotonic_test_prob,
        }
    )


# 4. Summary dataframe
metrics_df = pd.DataFrame(summary_rows)
metrics_df = metrics_df.sort_values(["model_name", "brier_score"]).reset_index(drop=True)

print("\nCalibration summary on test data")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 5. Detailed dataframe
calibration_results_df = pd.concat(all_dataframes, ignore_index=True)

print("\nDetailed dataframe preview")
preview_columns = [
    "model_name",
    "train_test",
    "actual_label",
    "predicted_label_raw",
    "probability_score_raw",
    "model_is_confident",
    "is_prediction_correct_raw",
    "probability_score_platt",
    "probability_score_isotonic",
    "brier_score_raw",
    "brier_score_platt",
    "brier_score_isotonic",
]
print(
    calibration_results_df[preview_columns]
    .head(12)
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)


# 6. Confidence comparison
confidence_summary_df = (
    calibration_results_df[calibration_results_df["train_test"] == "test"]
    .groupby(["model_name", "model_is_confident"], as_index=False)
    .agg(
        sample_count=("actual_label", "size"),
        average_probability=("probability_score_raw", "mean"),
        accuracy=("is_prediction_correct_raw", "mean"),
        average_gap_from_actual=("probability_vs_actual_gap_raw", "mean"),
    )
)

print("\nConfidence analysis on test data")
print(confidence_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 7. Reliability diagrams
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, result in zip(axes, plot_results):
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    raw_fraction, raw_mean = calibration_curve(y_test, result["raw_test_prob"], n_bins=N_BINS)
    platt_fraction, platt_mean = calibration_curve(y_test, result["platt_test_prob"], n_bins=N_BINS)
    iso_fraction, iso_mean = calibration_curve(y_test, result["isotonic_test_prob"], n_bins=N_BINS)

    ax.plot(raw_mean, raw_fraction, "o-", color="steelblue", label="Raw")
    ax.plot(platt_mean, platt_fraction, "o-", color="seagreen", label="Platt")
    ax.plot(iso_mean, iso_fraction, "o-", color="darkorange", label="Isotonic")

    ax.set_title(result["model_name"])
    ax.set_xlabel("Mean predicted probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Fraction of positives")
axes[-1].legend(loc="lower right")
plt.tight_layout()
plt.show()
plt.close()


# 8. Probability histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, result in zip(axes, plot_results):
    ax.hist(result["raw_test_prob"], bins=15, alpha=0.55, color="steelblue", label="Raw")
    ax.hist(result["platt_test_prob"], bins=15, alpha=0.45, color="seagreen", label="Platt")
    ax.hist(result["isotonic_test_prob"], bins=15, alpha=0.35, color="darkorange", label="Isotonic")

    ax.set_title(result["model_name"])
    ax.set_xlabel("Predicted probability")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Count")
axes[-1].legend(loc="upper center")
plt.tight_layout()
plt.show()
plt.close()
