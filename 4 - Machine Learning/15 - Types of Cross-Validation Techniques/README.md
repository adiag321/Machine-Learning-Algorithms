# Cross-Validation Techniques — Complete Guide

#### Table of Contents
1. [What is Cross-Validation?](#what-is-cross-validation)
2. [Why We Need Cross-Validation](#why-we-need-cross-validation)
3. [When to Use Cross-Validation](#when-to-use-cross-validation)
4. [Important Considerations](#important-considerations)
5. [Types of Cross-Validation](#types-of-cross-validation)

### What is Cross-Validation?

Cross-validation is a statistical technique used to evaluate the performance of a machine learning model on unseen data. It works by:

1. Dividing the dataset into multiple subsets (folds)
2. Training the model on some folds
3. Testing the model on the remaining fold(s)
4. Repeating this process multiple times
5. Averaging the results to get a more reliable estimate of model performance

**Key Concept:** Instead of using a single train-test split, CV uses multiple splits to provide a more robust evaluation metric that better generalizes to new data.

### Why We Need Cross-Validation

#### Problems with Single Train-Test Split:
- **High Variance**: A single split may give misleading results depending on which samples end up in train vs test
- **Data Wastage**: If we use 70% for training, we lose 30% of valuable training data
- **Lucky/Unlucky Splits**: Model performance can be artificially inflated or deflated based on the random split
- **Unstable Estimates**: Small differences in the split can cause large differences in performance metrics

#### Benefits of Cross-Validation:
- **Better Estimates**: Multiple evaluations reduce variance and provide more reliable performance metrics
- **Full Data Usage**: Every sample is used for both training and testing
- **Stability**: Results are less dependent on a single random train-test split
- **Statistical Confidence**: Can calculate variance/confidence intervals of performance
- **Better Model Selection**: More accurate comparison between different models

### When to Use Cross-Validation

✅ **Always use CV when:**
- Building production models for critical applications
- Comparing multiple algorithms or hyperparameters
- Dataset is moderately sized (100s to 10,000s of samples)
- Publishing research results
- Creating ensemble models

⚠️ **Consider CV when:**
- Dataset is small-to-medium sized
- Computing resources allow (CV is computationally more expensive)
- You need to tune hyperparameters

❌ **May not need extensive CV when:**
- Dataset is extremely large (millions of samples) — single split may suffice
- Computational budget is very limited
- You're doing exploratory data analysis

### Important Considerations

#### 1. Data Preprocessing and Splitting

⚠️ **Critical: Avoid Data Leakage!**

```python
# ❌ WRONG - Data leakage! Scaler fit on entire dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on entire dataset!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
```

```python
# ✅ CORRECT - No leakage! Scaler fit only on training data
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

cross_val_score(pipeline, X, y, cv=5)  # Scaler is fit within each fold
```

**Key Rules:**
1. **Never** fit preprocessing (scaling, encoding, feature selection) on entire dataset
2. Use `Pipeline` to ensure preprocessing is applied correctly within CV folds
3. Feature selection must happen inside the CV loop
4. Handle missing values inside the pipeline

### 2. **Data Requirements**

| Aspect | Guideline |
|--------|-----------|
| **Sample Size** | Minimum 20-30 samples; CV more valuable with 50-1000 samples |
| **Class Balance** | Use Stratified CV for imbalanced classification |
| **Feature Scale** | Preprocess features (scale/normalize) inside pipeline |
| **Missing Values** | Handle before CV or within pipeline |
| **Outliers** | Decide on removal/treatment before CV |

### 3. **Number of Folds (k)**

- **k=5**: Default, good balance between computation and reliability
- **k=10**: More thorough, computationally heavier
- **k=3**: Quick estimation, less reliable
- **k=n (LOOCV)**: Leave-One-Out, maximum reliability but computationally expensive
- **Rule of thumb**: Use k=5 or k=10; increase for small datasets, decrease for large datasets

### 4. **Random State and Reproducibility**

```python
# Always set random_state for reproducibility
cv = cross_val_score(model, X, y, cv=5, random_state=42)
```

### 5. **Stratification in Classification**

```python
# For imbalanced datasets, use StratifiedKFold
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)
```

---

## Types of Cross-Validation

### 1. **K-Fold Cross-Validation**

#### Definition
Splits data into k equal-sized folds. Model trains on k-1 folds and tests on 1 fold. Process repeats k times.

#### When to Use
- General purpose CV technique
- Balanced datasets
- When you need good bias-variance trade-off
- Default choice for most problems

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Fold scores: {scores}")
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_splits` | 3, 5, 10 (default: 5) | Number of folds |
| `shuffle` | True/False | Whether to shuffle before splitting |
| `random_state` | int or None | Random seed for reproducibility |

---

### 2. **Stratified K-Fold Cross-Validation**

#### Definition
K-Fold variant that maintains class distribution in each fold. Ensures each fold is representative of the overall class distribution.

#### When to Use
- **Classification tasks** with imbalanced classes
- When class balance matters
- **Always preferred over K-Fold for classification**
- Prevents biased folds with unequal class representation

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

print(f"Stratified fold scores: {scores}")
print(f"Mean F1: {scores.mean():.4f}")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_splits` | 3, 5, 10 | Number of folds with maintained class distribution |
| `shuffle` | True/False | Shuffle data before splitting |
| `random_state` | int or None | Random seed |

---

### 3. **Leave-One-Out Cross-Validation (LOOCV)**

#### Definition
Each sample is used as a test set exactly once, with remaining n-1 samples as training set. Repeats n times (where n = number of samples).

#### When to Use
- **Very small datasets** (< 100 samples)
- When you need maximum training data per fold
- When computational cost is not a concern
- Research/publication-grade evaluation needed

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.svm import SVC

model = SVC(kernel='rbf', random_state=42)
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

print(f"LOOCV accuracy: {scores.mean():.4f}")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| (No parameters) | — | Fixed behavior: n folds where n = number of samples |

#### Advantages & Disadvantages
- ✅ Advantages: Maximum use of data, no randomness
- ❌ Disadvantages: Very slow for large datasets, high variance, computationally expensive

---

### 4. **Repeated K-Fold Cross-Validation**

#### Definition
K-Fold CV repeated multiple times with different random seeds. Provides multiple estimates of model performance.

#### When to Use
- Need variance/confidence intervals of performance
- Model selection between similar algorithms
- When you want to estimate uncertainty
- More thorough evaluation needed

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=10, random_state=42)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf, scoring='accuracy')

print(f"Scores from {len(scores)} iterations:")
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
print(f"95% CI: [{scores.mean()-1.96*scores.std():.4f}, {scores.mean()+1.96*scores.std():.4f}]")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_splits` | 5, 10 | Folds per repetition |
| `n_repeats` | 5, 10, 20 | Number of times to repeat |
| `random_state` | int or None | Random seed |

---

### 5. **Shuffle Split Cross-Validation**

#### Definition
Randomly shuffles data and creates random train-test splits. Each split uses a random subset.

#### When to Use
- When you want custom train/test sizes
- Fixed number of iterations needed
- Efficient alternative to K-Fold for large datasets
- When data distribution is different from stratified approach

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ss = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7, random_state=42)
scores = cross_val_score(model, X, y, cv=ss, scoring='roc_auc')

print(f"ShuffleSplit scores: {scores}")
print(f"Mean ROC-AUC: {scores.mean():.4f}")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_splits` | 5, 10, 20 | Number of re-shuffling iterations |
| `test_size` | 0.2, 0.3, 0.5 | Proportion of data in test set |
| `train_size` | 0.7, 0.8, None | Proportion of data in train set |
| `random_state` | int or None | Random seed |

---

### 6. **Stratified Shuffle Split Cross-Validation**

#### Definition
Shuffle Split variant that maintains class distribution in each split.

#### When to Use
- **Classification with class imbalance**
- Need custom train/test split sizes with stratification
- Efficient alternative to StratifiedKFold for large datasets
- When you want fixed iteration count

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
scores = cross_val_score(model, X, y, cv=sss, scoring='f1_macro')

print(f"Stratified ShuffleSplit scores: {scores}")
print(f"Mean F1 (macro): {scores.mean():.4f}")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_splits` | 5, 10, 20 | Number of iterations |
| `test_size` | 0.2, 0.3 | Proportion in test with maintained class distribution |
| `train_size` | 0.7, 0.8 | Proportion in train |
| `random_state` | int or None | Random seed |

---

### 7. **Time Series Split**

#### Definition
Forward-chaining method where train set grows incrementally and test set is always in the future. Prevents data leakage in time series.

#### When to Use
- **Time series problems**
- Stock price prediction
- Sensor data analysis
- Any sequential/temporal data
- Prevents "peeking into the future"

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression

model = LinearRegression()
tscv = TimeSeriesSplit(n_splits=5, max_train_size=100, gap=0)
scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

print(f"Time Series CV scores: {scores}")
print(f"Mean R²: {scores.mean():.4f}")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_splits` | 3, 5, 10 | Number of splits |
| `max_train_size` | int or None | Maximum training set size |
| `gap` | 0, 1, 5 | Gap between train and test set (prevents leakage) |
| `test_size` | int or None | Fixed test set size |

---

### 8. **Group K-Fold Cross-Validation**

#### Definition
K-Fold variant where splits are made based on groups. All samples from a group go together (train or test).

#### When to Use
- Data has natural groups (e.g., patient IDs, store IDs, user IDs)
- Prevent samples from same group in both train and test
- Realistic evaluation when groups are meaningful
- Nested structure in data

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np

model = RandomForestRegressor(n_estimators=100, random_state=42)
gkf = GroupKFold(n_splits=5)
groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])  # Group labels

scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring='neg_mean_squared_error')

print(f"Group KFold scores: {-scores}")  # Negate because scoring is negative MSE
print(f"Mean MSE: {-scores.mean():.4f}")
```

#### Key Parameters
| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_splits` | 3, 5, 10 | Number of folds (must be <= number of groups) |
| `groups` | array | Group label for each sample |

#### Important Notes
- ⚠️ Must provide `groups` parameter to `cross_val_score()`
- Number of groups must be >= n_splits
- All samples from same group stay together (never split across train/test)

---

### 9. **Nested Cross-Validation**

#### Definition
Uses two levels of CV: outer CV for model evaluation, inner CV for hyperparameter tuning.

#### When to Use
- Hyperparameter tuning with cross-validation
- Unbiased performance estimation while tuning
- Comparing multiple hyperparameter sets
- When you need both tuning and evaluation

#### Sample Code
```python
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Inner CV for tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Outer CV for evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy')

# Nested CV: outer loop evaluates different CV folds
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='accuracy')

print(f"Nested CV scores: {nested_scores}")
print(f"Mean accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")
print(f"Best params from final model: {grid_search.best_params_}")
```

#### Key Parameters
- **Outer CV**: Use StratifiedKFold (k=5) for reliable evaluation
- **Inner CV**: Use StratifiedKFold (k=3) for faster tuning
- **Scoring**: Same as regular cross_val_score

#### Important Notes
- ⚠️ More computationally expensive than single CV
- Provides unbiased performance estimates
- Inner CV does hyperparameter selection
- Outer CV evaluates final model performance

---

## Quick Reference Table

| Technique | Use Case | Speed | Data Size | Key Parameter |
|-----------|----------|-------|-----------|----------------|
| **K-Fold** | General purpose | Fast | Any | n_splits (5-10) |
| **Stratified K-Fold** | Classification imbalance | Fast | Any | n_splits (5-10) |
| **Leave-One-Out** | Very small data | Slow | <100 | — |
| **Repeated K-Fold** | Variance estimation | Medium | Any | n_repeats (5-10) |
| **Shuffle Split** | Custom train/test ratio | Fast | Large | test_size |
| **Stratified Shuffle** | Classification + custom ratio | Fast | Large | test_size |
| **Time Series Split** | Sequential data | Fast | Any | gap (0-5) |
| **Group K-Fold** | Grouped data | Fast | Any | n_splits (must have enough groups) |
| **Nested CV** | Hyperparameter tuning | Very Slow | Any | inner + outer CV |

---

## Best Practices

1. ✅ **Always use a pipeline** to avoid data leakage
2. ✅ **Use stratification** for classification tasks
3. ✅ **Set random_state** for reproducibility
4. ✅ **Check class distribution** in each fold for classification
5. ✅ **Use appropriate metrics** for your problem (accuracy, F1, ROC-AUC, etc.)
6. ✅ **Report mean AND std** of CV scores
7. ✅ **Use nested CV** when tuning hyperparameters
8. ❌ **Never preprocess** entire dataset before CV
9. ❌ **Don't use default random_state=None** in production
10. ❌ **Don't compare CV scores** from different methods directly

---

## Folder Contents

This folder contains:
- `01_Cross_Validation_Techniques.py` — Implementation of all CV techniques with examples
- `README.md` — This comprehensive guide