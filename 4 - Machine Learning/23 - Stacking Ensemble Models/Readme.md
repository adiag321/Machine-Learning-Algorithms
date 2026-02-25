# Stacking and Blending Ensemble Models

Stacking (Stacked Generalization) is an ensemble technique that combines multiple base models through a meta-learner to produce better predictions.

### Files in this Repository

| File | Description |
|------|-------------|
| `00_Simple_Stacking.py` | Manual stacking with K-Fold CV — step-by-step, easy to follow |
| `01_Stacking_with_sklearn.py` | Stacking using sklearn's `StackingClassifier` |
| `02_Blending.py` | Blending approach (single split, no CV) |
| `03_Stacking_Pipeline.py` | Reusable function — manual K-Fold CV stacking |
| `04_Stacking_Pipeline_with_Sklearn.py` | Reusable function — uses sklearn's `StackingClassifier` |

---

## 1. Stacking Ensemble Models
### Architecture
```
    Original Data (X)
          |
    +-----+-----+-----+-----+
    |     |     |     |     |
   LR    KNN   DT   SVM        ← Base Models (Level 0)
    |     |     |     |
    v     v     v     v
  Pred1 Pred2 Pred3 Pred4
    |     |     |     |
    +-----+-----+-----+
          |
    Meta-Learner (LR)           ← Level 1
          |
    Final Prediction
```

---

### Manual Stacking vs Sklearn StackingClassifier

#### A. Manual Stacking Workflow (`00_Simple_Stacking.py`, `03_Stacking_Pipeline.py`)

We write the K-Fold loop yourself, giving full visibility into how `meta_train` and `meta_test` are built.

```
Step 1: Split data → Train / Test
Step 2: K-Fold CV on Train set
        For each fold:
            Train base models on (K-1) folds
            Predict on held-out fold → meta_train
Step 3: Train base models on FULL Train set
        Predict on Test set → meta_test
Step 4: Train meta model on meta_train
        Predict on meta_test → Final Predictions
```

**Code:**
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((X_train.shape[0], len(base_models)))

# Build meta_train using K-Fold CV
for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train = y_train[train_idx]

    for i, (name, model) in enumerate(base_models.items()):
        model.fit(X_fold_train, y_fold_train)
        meta_train[val_idx, i] = model.predict(X_fold_val)

# Build meta_test — train on full train set, predict on test
meta_test = np.zeros((X_test.shape[0], len(base_models)))
for i, (name, model) in enumerate(base_models.items()):
    model.fit(X_train, y_train)
    meta_test[:, i] = model.predict(X_test)

# Train meta model
meta_model = LogisticRegression()
meta_model.fit(meta_train, y_train)
final_predictions = meta_model.predict(meta_test)
```

---

#### B. Sklearn StackingClassifier Workflow (`01_Stacking_with_sklearn.py`, `04_Stacking_Pipeline_with_Sklearn.py`)

Sklearn handles K-Fold CV, meta-feature generation, and model training internally.

```
Step 1: Split data → Train / Test
Step 2: Define StackingClassifier with base models + meta model
Step 3: Call .fit() → internally does K-Fold CV + meta-training
Step 4: Call .predict() → Final Predictions
```

**Code:**
```python
from sklearn.ensemble import StackingClassifier

stacking_model = StackingClassifier(
    estimators = [(name, model) for name, model in base_models.items()],
    final_estimator = LogisticRegression(max_iter=1000),
    cv = KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs = -1
)

stacking_model.fit(X_train, y_train)           # handles everything internally
final_predictions = stacking_model.predict(X_test)  # one call to predict
```

---

### Comparison

| Aspect | Manual Stacking | Sklearn StackingClassifier |
|--------|:-:|:-:|
| Data leakage prevention | K-Fold CV | K-Fold CV (internal) |
| Transparency | Full — can inspect `meta_train`, `meta_test` | Black box |
| Customizability | Different CV per model, custom meta-features | Limited |
| Code complexity | More code, more bookkeeping | Minimal code |
| Pipeline integration | Manual | Works with `GridSearchCV`, `Pipeline` |
| Model serialization | Save N+1 files separately | Save 1 file with `joblib` |
| Best for | Learning, debugging, Kaggle | Production, deployment |

---

## How Predictions Are Made

#### 1. New Prediction Phase
```
Input: New/unseen sample
  ↓
All 4 base models make independent predictions
  ↓
[LR_pred, KNN_pred, DT_pred, SVM_pred]  ← 4 predictions become features
  ↓
Meta-model (LogisticRegression) takes these 4 predictions as input
  ↓
Meta-model outputs FINAL prediction
```

#### 2. Why It Works
- Base models capture **different patterns** in data (diversity)
- Meta-model learns the **optimal way to combine** their strengths
- Reduces individual model weaknesses through ensemble learning

#### 3. Example with a Single Sample
```
Sample features: [f1, f2, ..., f30]

Base Model Predictions:
- Logistic Regression → 0.75
- KNN                 → 0.80
- Decision Tree       → 0.70
- SVM                 → 0.78

Meta-Model sees: [0.75, 0.80, 0.70, 0.78]
Meta-Model learns weights: [w1, w2, w3, w4]
Final Prediction = sigmoid(w1×0.75 + w2×0.80 + w3×0.70 + w4×0.78 + bias)
Final Prediction → 0.76 (class 1)
```

---

### When to Use Which

| Scenario | Recommended Approach |
|----------|---------------------|
| Learning how stacking works | `00_Simple_Stacking.py` (manual) |
| Quick prototyping | `01_Stacking_with_sklearn.py` (sklearn) |
| Understanding blending | `02_Blending.py` |
| Reusable pipeline with full control | `03_Stacking_Pipeline.py` (manual) |
| Production deployment | `04_Stacking_Pipeline_with_Sklearn.py` (sklearn) |
| Kaggle / custom meta-features | Manual approach |
| Hyperparameter tuning with GridSearchCV | Sklearn approach |

---

## 2. Blending
### Architecture



---

### Stacking vs Blending

| Aspect | Stacking (K-Fold CV) | Blending (Single Split) |
|--------|:--------------------:|:-----------------------:|
| Meta-train creation | Out-of-fold predictions via K-Fold CV | Predict on same train data or a hold-out set |
| Data leakage | No | Yes (if predicting on train data) |
| Data efficiency | Uses all training data | Wastes data on hold-out |
| Complexity | More involved | Very simple |
| Production-ready | Yes | Not recommended |

