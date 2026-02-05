"""
STACKING ENSEMBLE MODEL (Simple Implementation)

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

Note: This simple version trains the meta-learner on the same data used for base models. 
For production, use cross-validation (see 01_Stacking.py).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("X_train Shape", X_train.shape, "\nX_test Shape", X_test.shape, "\ny_train Shape", y_train.shape, "\ny_test Shape", y_test.shape)

# Step 2: Define base models (Level 0)
# Using diverse models to capture different patterns
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "SVM": SVC(random_state=42)
}

# Step 3: Train base models on training data
for name, model in base_models.items():
    print("Training", name)
    model.fit(X_train, y_train)

# Step 4: Generate meta-features (predictions from base models)
# Each column = one base model's predictions
meta_features_train = np.column_stack([model.predict(X_train) for model in base_models.values()])
print("Meta-features train shape:", meta_features_train.shape)

meta_features_test = np.column_stack([model.predict(X_test) for model in base_models.values()])
print("Meta-features test shape:", meta_features_test.shape)

# Step 5: Train meta-learner on meta-features
meta_learner = LogisticRegression(max_iter=1000, random_state=42)
meta_learner.fit(meta_features_train, y_train)
print("Meta-learner coefficients shape:", meta_learner.coef_.shape)

# Step 6: Evaluate all models
print("Base Model Performance:")
for name, model in base_models.items():
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name}: {acc:.4f}")

# Stacking prediction: base models predict -> meta-learner combines
stacking_pred = meta_learner.predict(meta_features_test)
stacking_acc = accuracy_score(y_test, stacking_pred)

print(f"\nStacking Ensemble: {stacking_acc:.4f}")
