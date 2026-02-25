"""
STACKING PIPELINE - Reusable Function

Usage:
------
Pass your data (X, y), a dictionary of base models, and a meta model.
The function handles everything: train/test split, K-Fold CV stacking,
evaluation metrics, and results comparison.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")


def stacking_pipeline(X, y, base_models, meta_model, test_size = 0.2, n_splits = 5, random_state = 42):
    """
    Complete stacking pipeline using K-Fold CV (no data leakage).

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
    # STEP 2: Create fresh copies of models (so originals stay untouched)
    # ===========================================================

    base_models_fresh = {name: clone(model) for name, model in base_models.items()}
    meta_model_fresh = clone(meta_model)

    # ===========================================================
    # STEP 3: Generate META-TRAIN using K-Fold CV on training set
    # ===========================================================
    # Each base model predicts on folds it was NOT trained on,
    # so the meta model learns from unbiased predictions.

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    meta_train = np.zeros((X_train.shape[0], len(base_models_fresh)))

    print(f"\nGenerating meta-train with {n_splits}-Fold CV ...")
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]

        for i, (name, model) in enumerate(base_models_fresh.items()):
            model.fit(X_fold_train, y_fold_train)
            meta_train[val_idx, i] = model.predict(X_fold_val)

        print(f"Fold {fold_num} complete")

    print(f"Meta_train shape: {meta_train.shape}")

    # ===========================================================
    # STEP 4: Generate META-TEST
    # ===========================================================
    # Train base models on FULL training set → predict on test set

    meta_test = np.zeros((X_test.shape[0], len(base_models_fresh)))

    print("\nGenerating meta-test (training on full train set) ...")
    for i, (name, model) in enumerate(base_models_fresh.items()):
        model.fit(X_train, y_train)
        meta_test[:, i] = model.predict(X_test)
        print(f"{name} done")

    print(f"Meta_test shape: {meta_test.shape}")

    # ===========================================================
    # STEP 5: Train Meta Model
    # ===========================================================

    meta_model_fresh.fit(meta_train, y_train)
    stacking_predictions = meta_model_fresh.predict(meta_test)

    # ===========================================================
    # STEP 6: Evaluate all models + stacking
    # ===========================================================

    results = []

    # Evaluate each base model
    for i, name in enumerate(base_models_fresh.keys()):
        preds = meta_test[:, i]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted"),
            "Recall": recall_score(y_test, preds, average="weighted"),
            "F1 Score": f1_score(y_test, preds, average="weighted"),
        })

    # Evaluate stacking
    results.append({
        "Model": "Stacking (Meta Model)",
        "Accuracy": accuracy_score(y_test, stacking_predictions),
        "Precision": precision_score(y_test, stacking_predictions, average="weighted"),
        "Recall": recall_score(y_test, stacking_predictions, average="weighted"),
        "F1 Score": f1_score(y_test, stacking_predictions, average="weighted"),
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
    print(classification_report(y_test, stacking_predictions))

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
    results = stacking_pipeline(X, y, base_models, meta_model)