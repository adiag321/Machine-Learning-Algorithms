# K-Nearest-Neighour

K-Nearest Neighbors (KNN) is a **supervised learning algorithm** used for **classification and regression** tasks. It is a **non-parametric**, **instance-based** (lazy learning) method that makes predictions based on similarity (distance) between data points.

## Key Concepts

- **Instance-based learning**: KNN memorizes the training dataset instead of learning a discriminative function.
- **No training phase**: KNN directly makes predictions using the entire training dataset.
- **Distance-based**: Uses distance metrics like Euclidean, Manhattan, or Minkowski to find the "nearest" neighbors.

## How KNN Works

1. Choose the number of neighbors **K**.
2. Calculate the distance between the test data and all training data.
3. Select the **K nearest neighbors** based on distance.
4. For classification:
   - Predict the **most common class** among neighbors.
5. For regression:
   - Predict the **average value** of neighbors.

## Distance Metrics

- **Euclidean Distance** (default):  
  \[
  d(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
  \]

- **Manhattan Distance**:  
  \[
  d(p, q) = \sum_{i=1}^n |p_i - q_i|
  \]

- **Minkowski Distance** (generalized form)

## Hyperparameters

| Parameter        | Description                                                   |
|------------------|---------------------------------------------------------------|
| `n_neighbors`    | Number of neighbors to use (K)                                |
| `metric`         | Distance metric to use (`'euclidean'`, `'manhattan'`, etc.)  |
| `weights`        | `'uniform'` (equal weight) or `'distance'` (weighted by dist) |

## Pros

- Simple to understand and implement
- No training time
- Works well with small datasets
- Naturally handles multi-class classification

## Cons

- Computationally expensive at prediction time (slow for large datasets)
- Sensitive to irrelevant or high-dimensional features
- Poor with imbalanced data
- Requires proper feature scaling (standardization or normalization)

## Preprocessing Tips

- Normalize or standardize features to ensure fair distance calculations.
- Remove noise and irrelevant features to improve accuracy.
- Choose optimal `K` using methods like cross-validation.

## KNN with Scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load sample data
data = load_iris()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN classifier
model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train, y_train)

# Prediction & Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))