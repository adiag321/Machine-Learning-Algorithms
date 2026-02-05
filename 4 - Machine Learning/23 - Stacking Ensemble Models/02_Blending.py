"""
BLENDING ENSEMBLE MODEL

ARCHITECTURE:
-------------
Blending distinguishes itself from Stacking by using a HOLD-OUT VALIDATION SET
instead of Cross-Validation to train the Meta-Learner.

    Data Split:
    -----------
    [ Train Set (60%) ]  [ Validation Set (20%) ]  [ Test Set (20%) ]
           |                      |                      |
           v                      v                      v
    Train Base Models      Predict -> Meta-Features   Final Eval
           |                      |
           |                      v
           +-----------------> Train Meta-Learner

    Step-by-Step Flow:
    1. Split data into TRAIN, VALIDATION, and TEST sets.
    2. Train Base Models on TRAIN set.
    3. Predict on VALIDATION set using Base Models -> These become Meta-Features.
    4. Train Meta-Learner on Meta-Features (using VALIDATION y as target).
    5. Evaluate everything on TEST set.

PROS: simpler, faster, no leakage.
CONS: uses less data for training base models (since validation is held out).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

############################################
# STEP 1: LOAD AND SPLIT DATA (3-WAY SPLIT)
############################################
print("STEP 1: Splitting Data (Train / Validation / Test)")

X, y = load_breast_cancer(return_X_y=True)
print(f"  Original Data  : {X.shape}")

# First split: Train (60% + 20%) vs Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  X_train (X_temp) : {X_temp.shape} (for Base Models)")
print(f"  X_test (X_test)       : {X_test.shape}   (for Final Evaluation)")
print(f"  Y_train (Y_temp)        : {y_temp.shape} (For Base Models)")
print(f"  Y_test (Y_test)         : {y_test.shape} (For Final Evaluation)")

# Second split: Train (60%) vs Validation (20%)
# The 0.25 here splits the remaining 80% into 60% and 20% (0.25 * 0.8 = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
print(f"  X_train (X_train)      : {X_train.shape} (for Base Models)")
print(f"  X_val (X_val) : {X_val.shape}   (for Meta-Learner)")
print(f"  Y_train (Y_train)        : {y_train.shape} (For Base Models)")
print(f"  Y_val (Y_val)          : {y_val.shape}   (For Meta-Learner)")

############################################
# STEP 2: DEFINE AND TRAIN BASE MODELS (LEVEL 0)
############################################
print("\nSTEP 2: Training Base Models on TRAIN Set...")

base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}
for name, model in base_models.items():
    model.fit(X_train, y_train)
    
print("The Base models have been trained.")

############################################
# STEP 3: GENERATE PREDICTIONS (META-FEATURES)
############################################
print("\nSTEP 3: Generating Meta-Features (Predictions on Val and Test sets)...")

# We need predictions on VALIDATION set to train the meta-learner
val_meta_features = np.column_stack([model.predict(X_val) for model in base_models.values()])

# We need predictions on TEST set to specificy evaluating the final ensemble
test_meta_features = np.column_stack([model.predict(X_test) for model in base_models.values()])

print(f"  Meta-Features Matrix (Val) : {val_meta_features.shape}")

############################################
# STEP 4: TRAIN META-LEARNER (LEVEL 1)
############################################
print("\nSTEP 4: Training Meta-Learner on Validation Predictions...")

meta_learner = LogisticRegression(max_iter=1000, random_state=42)

# Verify: We train on VALIDATION predictions, using VALIDATION targets
meta_learner.fit(val_meta_features, y_val)

print("The Meta-Learner has been trained.")

############################################
# STEP 5: FINAL EVALUATION ON TEST SET
############################################
print("\nSTEP 5: Final Evaluation on TEST Set")
print("-" * 40)

# 1. Base Models on Test Set
for name, model in base_models.items():
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:20s}: {acc:.4f}")

# 2. Blending Ensemble on Test Set
# We use the test_meta_features we generated in Step 3
blending_pred = meta_learner.predict(test_meta_features)
blending_acc = accuracy_score(y_test, blending_pred)

print("-" * 40)
print(f"{'Blending Ensemble':20s}: {blending_acc:.4f}")
