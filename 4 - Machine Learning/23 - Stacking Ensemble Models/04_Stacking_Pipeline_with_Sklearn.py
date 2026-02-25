"""
STACKING PIPELINE - Using Sklearn's StackingClassifier

Usage:
------
Pass your data (X, y), a dictionary of base models, and a meta model.
The function uses sklearn's StackingClassifier which handles K-Fold CV
internally, evaluates all models, and returns a results comparison.

# 02 (Manual):  you write the fold loop, build meta_train/meta_test yourself
# 03 (Sklearn): StackingClassifier handles it in one line
stacking_model = StackingClassifier(
    estimators = [(name, model) for name, model in base_models.items()],
    final_estimator = meta_model,
    cv = kf
)
stacking_model.fit(X_train, y_train)       # does everything internally
stacking_model.predict(X_test)             # one call to predict
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")


def stacking_pipeline_sklearn(X, y, base_models, meta_model, test_size = 0.2, n_splits = 5, random_state = 42):
    """
    Stacking pipeline using sklearn's StackingClassifier.

    Parameters:
    -----------
    X           : Feature matrix (numpy array or DataFrame)
    y           : Target vector
    base_models : dict → {"Model Name": model_object, ...}
    meta_model  : sklearn estimator for the meta-learner
    test_size   : fraction of data for testing (default 0.2)
    n_splits    : number of K-Fold splits (default 5)
    random_state: random seed for reproducibility

    Returns:
    --------
    results_df  : DataFrame comparing all models on eval metrics
    """

    # Convert to numpy if DataFrame is passed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # ===========================================================
    # STEP 1: Split data into Train / Test
    # ===========================================================

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ===========================================================
    # STEP 2: Define the StackingClassifier
    # ===========================================================
    # StackingClassifier takes:
    #   - estimators: list of (name, model) tuples → base models
    #   - final_estimator: the meta-learner
    #   - cv: K-Fold strategy for generating out-of-fold predictions
    # It handles the K-Fold CV internally (same logic as our manual pipeline)

    kf = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)

    stacking_model = StackingClassifier(
        estimators = [(name, clone(model)) for name, model in base_models.items()],
        final_estimator = clone(meta_model),
        cv = kf,
        n_jobs = -1
    )

    # ===========================================================
    # STEP 3: Train individual base models (for comparison)
    # ===========================================================
    print("\nTraining individual base models for comparison ...")
    trained_base_models = {}
    for name, model in base_models.items():
        m = clone(model)
        m.fit(X_train, y_train)
        trained_base_models[name] = m
        print(f"{name} trained")

    # ===========================================================
    # STEP 4: Train the StackingClassifier
    # ===========================================================
    # Internally this does:
    #   1. K-Fold CV on X_train → out-of-fold predictions → meta_train
    #   2. Trains base models on full X_train
    #   3. Trains meta-learner on meta_train

    print("\nTraining StackingClassifier ...")
    stacking_model.fit(X_train, y_train)
    print("Stacking model trained")

    # ===========================================================
    # STEP 5: Evaluate all models
    # ===========================================================
    results = []

    # Evaluate each base model individually
    for name, model in trained_base_models.items():
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average = "weighted"),
            "Recall": recall_score(y_test, preds, average = "weighted"),
            "F1 Score": f1_score(y_test, preds, average = "weighted"),
        })

    # Evaluate stacking model
    stacking_preds = stacking_model.predict(X_test)
    results.append({
        "Model": "Stacking (Meta Model)",
        "Accuracy": accuracy_score(y_test, stacking_preds),
        "Precision": precision_score(y_test, stacking_preds, average = "weighted"),
        "Recall": recall_score(y_test, stacking_preds, average = "weighted"),
        "F1 Score": f1_score(y_test, stacking_preds, average = "weighted"),
    })

    results_df = pd.DataFrame(results).set_index("Model")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(results_df.to_string())
    print("=" * 70)

    # Print detailed classification report for stacking
    print("\nStacking Classification Report:")
    print(classification_report(y_test, stacking_preds))

    return results_df

# =====================================================================
# EXAMPLE USAGE
# =====================================================================
if __name__ == "__main__":

    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC

    # Load data
    X, y = load_breast_cancer(return_X_y=True)

    # Define base models
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "SVM": SVC(random_state=42),
    }

    # Define meta model
    meta_model = LogisticRegression(max_iter=1000, random_state=42)

    # Run the pipeline
    results = stacking_pipeline_sklearn(X, y, base_models, meta_model)