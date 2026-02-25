"""
STACKING ENSEMBLE MODEL (Proper Implementation)

ARCHITECTURE:
-------------
This implementation uses PROPER Cross-Validation (K-Fold) to generate meta-features.
This prevents data leakage and overfitting at the meta-learner level.

    Level 0 (Base Learners):
    -------------------------
    Diverse models trained on the data using K-Fold Cross-Validation.

    Procedure for Meta-Features:
    1. Split data into K folds (e.g., 5 folds).
    2. For each fold:
       - Train base models on the other (K-1) folds.
       - Predict on the current fold (hold-out set).
    3. Stack these predictions to form the full meta-feature matrix.

    Level 1 (Meta-Learner):
    ------------------------
    A final model (Logistic Regression) trained on the meta-features to make
    the final prediction.

       +------------------+
       |   Original Data  |
       +--------+---------+
                |
       (K-Fold Cross-Validation)
                |
       +--------v---------+
       |  Base Predictions|  <-- Meta-Features (Unbiased due to CV)
       +--------+---------+
                |
                v
       +--------+---------+
       |   Meta-Learner   |
       +--------+---------+
                |
                v
         Final Prediction

"""
import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
warnings.filterwarnings("ignore")

# Load data
X, y = load_breast_cancer(return_X_y=True)

# 1. Define Cross-Validation Strategy
# We initially used K-Fold CV. Standard KFold does not preserve class percentage 
# (StratifiedKFold is usually preferred for classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Define Base Models
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# 3. Define Stacking Model
# The StackingClassifier automatically handles the cross-validation logic
# to generate unbiased predictions for the meta-learner.
stacking_model = StackingClassifier(
    estimators = [(name, model) for name, model in base_models.items()],
    final_estimator = LogisticRegression(max_iter=1000),
    cv = cv,                                            # Using the StratifiedKFold strategy defined above
    n_jobs = -1
)

# 4. Evaluation Helper
def evaluate_model(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()

# -------------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------------
print("Performance Evaluation (5-stratified Fold CV)")

# Evaluate individual base models
base_scores = {}
for name, model in base_models.items():
    mean_acc, std_acc = evaluate_model(model, X, y, cv)
    base_scores[name] = mean_acc
    print(f"{name:20s} | Mean: {mean_acc:.4f} | Std: {std_acc:.4f}")

avg_base = np.mean(list(base_scores.values()))
print(f"\nAverage Base Accuracy : {avg_base:.4f}")

# Evaluate Stacking Model
stack_mean, stack_std = evaluate_model(stacking_model, X, y, cv)
print(f"Stacking Classifier   : {stack_mean:.4f} (+{stack_mean - avg_base:.4f})")
