import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

# -----------------------------
# 1. Load and prepare dataset
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# Make dataset imbalanced (keep all class 0, reduce class 1)
X_majority = X[y == 0]
y_majority = y[y == 0]

X_minority = X[y == 1][:50]  # reduce minority
y_minority = y[y == 1][:50]

X_imbalanced = np.vstack((X_majority, X_minority))
y_imbalanced = np.hstack((y_majority, y_minority))

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, stratify=y_imbalanced, random_state=42)

# -----------------------------
# 2. Define models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier()
}

# -----------------------------
# 3. Cost Function
# -----------------------------
COST_FP = 10
COST_FN = 100

def calculate_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fp * COST_FP + fn * COST_FN
    avg_precision = average_precision_score(y_true, y_pred)
    return total_cost, fp, fn, avg_precision

# -----------------------------
# 4. Train + Evaluate
# -----------------------------
thresholds = np.linspace(0.1, 0.95, 10)

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        cost, fp, fn, avg_precision = calculate_cost(y_test, y_pred)

        results.append({
            "Model": name,
            "Threshold": t,
            "Cost": cost,
            "FP": fp,
            "FN": fn,
            "Average Precision": avg_precision
        })

# -----------------------------
# 5. Convert to DataFrame
# -----------------------------
results_df = pd.DataFrame(results)

# Show best threshold per model
best_results = results_df.loc[results_df.groupby("Model")["Cost"].idxmin()]

print("\n=== Best Threshold per Model ===")
print(best_results)

# Show full table
print("\n=== All Results ===")
print(results_df.sort_values(["Model", "Threshold"]))

## Plots
for model_name in results_df["Model"].unique():
    subset = results_df[results_df["Model"] == model_name]
    plt.plot(subset["Threshold"], subset["Cost"], label=model_name)

plt.xlabel("Threshold")
plt.ylabel("Total Cost")
plt.legend()
plt.title("Cost vs Threshold")
plt.show()