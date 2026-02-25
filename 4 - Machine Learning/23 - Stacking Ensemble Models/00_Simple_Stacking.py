"""
SIMPLE STACKING ENSEMBLE MODEL

Architecture:
-------------
Level 0 (Base Learners): Multiple diverse models trained on original features.
Level 1 (Meta-Learner): A final model that combines base model predictions.

    Original Data (X)
          |
    +-----+-----+-----+-----+
    |     |     |     |     |
   LR    KNN   DT   SVM        <-- Base Models (Level 0)
    |     |     |     |
    v     v     v     v
  Pred1 Pred2 Pred3 Pred4
    |     |     |     |
    +-----+-----+-----+
          |
    Meta-Learner (LR)          <-- Level 1
          |
    Final Prediction

Procedure:
----------
1. Split data into Train and Test sets
2. Use K-Fold CV on the TRAIN set:
   - For each fold, train base models on (K-1) folds, predict on the held-out fold
   - Stack these out-of-fold predictions → becomes the TRAIN DATA FOR META MODEL
3. Train base models on the FULL training set, predict on the Test set
   - These predictions → become the TEST DATA FOR META MODEL
4. Train the meta model on the stacked train data, evaluate on the stacked test data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: Load Data & Split into Train / Test
# ============================================================

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ============================================================
# STEP 2: Define Base Models & K-Fold Strategy
# ============================================================

base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "SVM": SVC(random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# STEP 3: Generate META-TRAIN data using K-Fold on training set
# ============================================================
# For each base model we create a column of out-of-fold predictions.
# These predictions are "unseen" by the model that made them,
# so the meta model learns from unbiased predictions (NO data leakage).

# Placeholder array – one column per base model
meta_train = np.zeros((X_train.shape[0], len(base_models)))   # shape: (n_train, 4)

for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    # Split the training data into fold-train and fold-validation
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train = y_train[train_idx]

    print(f"\nFold {fold_num}  |  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    for i, (name, model) in enumerate(base_models.items()):
        # Train on fold-train
        model.fit(X_fold_train, y_fold_train)

        # Predict on fold-validation → store in the correct rows of meta_train
        meta_train[val_idx, i] = model.predict(X_fold_val)

        print(f"{name} trained on fold {fold_num}")

print("\nmeta_train shape:", meta_train.shape)
meta_train_df = pd.DataFrame(meta_train, columns = base_models.keys())
meta_train_df["target"] = y_train
print(meta_train_df.head())

# ============================================================
# STEP 4: Generate META-TEST data
# ============================================================
# Train each base model on the FULL training set,
# then predict on the test set. These predictions become
# the test data for the meta model.

meta_test = np.zeros((X_test.shape[0], len(base_models)))   # shape: (n_test, 4)

print("\nTraining base models on FULL training set for test predictions ...")
for i, (name, model) in enumerate(base_models.items()):
    model.fit(X_train, y_train)             # full training set
    meta_test[:, i] = model.predict(X_test) # predict on test set
    acc = accuracy_score(y_test, meta_test[:, i])
    print(f"  {name}  →  Test Accuracy: {acc:.4f}")

print("\nmeta_test shape:", meta_test.shape)
meta_test_df = pd.DataFrame(meta_test, columns=base_models.keys())
meta_test_df["target"] = y_test
print(meta_test_df.head())

# ============================================================
# STEP 5: Train Meta Model & Make Final Predictions
# ============================================================
# The meta model takes the base model predictions as features
# and learns how to best combine them.

meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(meta_train, y_train)        # train on stacked train predictions

# Final predictions on stacked test predictions
final_predictions = meta_model.predict(meta_test)
final_accuracy = accuracy_score(y_test, final_predictions)

# ============================================================
# STEP 6: Compare Results
# ============================================================

print("RESULTS COMPARISON")

for i, name in enumerate(base_models.keys()):
    acc = accuracy_score(y_test, meta_test[:, i])
    print(f"  {name:25s} →  {acc:.4f}")

print(f"{'Stacking (Meta Model)':25s} →  {final_accuracy:.4f}")
