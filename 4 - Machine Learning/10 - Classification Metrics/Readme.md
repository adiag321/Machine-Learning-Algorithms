# Classification Metrics: Binary vs Multi-Class

Understanding how evaluation metrics differ for **binary** vs **multi-class** classification problems is crucial for building and interpreting machine learning models effectively.


## Metric Comparison Overview

| Metric                  | Binary Classification                         | Multi-Class Classification                                             |
|------------------------|-----------------------------------------------|------------------------------------------------------------------------|
| **Accuracy**           | % of correctly predicted instances            | Same concept, but less informative with class imbalance                |
| **Precision / Recall / F1** | Defined for the positive class               | Computed per class and averaged (`macro`, `micro`, `weighted`)        |
| **Confusion Matrix**   | 2×2 matrix (TP, TN, FP, FN)                   | n×n matrix (for n classes)                                            |
| **AUC-ROC**            | Single ROC curve                              | One-vs-rest or macro-average of multiple ROC curves                   |
| **Log Loss**           | Penalizes incorrect probabilities             | Considers predicted probability for each class                        |
| **Specificity / Sensitivity** | Clear TP, TN, FP, FN definition          | Typically used per class, not common for multi-class tasks            |


## Detailed Differences

### 1. Confusion Matrix
- **Binary**: A simple 2×2 matrix.
- **Multi-Class**: An n×n matrix that visualizes misclassifications between multiple classes.

### 2. Precision / Recall / F1 Score
- **Binary**: Focused on the **positive class**.
- **Multi-Class**: Must be calculated **per class**, then averaged using:
  - `macro`: unweighted mean of all classes
  - `micro`: global average considering total TP, FP, FN
  - `weighted`: average weighted by support (number of true instances per class)

### 3. ROC-AUC
- **Binary**: One curve showing trade-off between TPR and FPR.
- **Multi-Class**:
  - One-vs-Rest (OvR): Separate ROC curve per class
  - Macro-Averaged AUC: Average of ROC-AUC scores for each class

### 4. Interpretation
- **Binary** metrics are sharper and focused.
- **Multi-Class** metrics require aggregation, and might obscure certain types of misclassifications.

## Pro Tips

- For **multi-class classification**:
  - Always report precision, recall, and F1 with `macro` and `weighted` averages.
  - Use a **confusion matrix** to understand misclassification patterns.
  - Consider **Top-k Accuracy** (e.g., Top-3 Accuracy) for high-class-count problems.

- For **binary classification**:
  - Precision-Recall and ROC curves are especially useful when the dataset is **imbalanced**.


## With Scikit-learn

Example usage in Python:

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# For multi-class classification
f1_score(y_true, y_pred, average='macro')     # Equal weight to all classes
f1_score(y_true, y_pred, average='weighted')  # Weighted by support