# Classification Metrics: Binary vs Multi-Class

A concise guide to evaluation metrics for binary and multi-class classification models.

---

## Quick Cheat Sheet: Decision Guide
- **Balanced Data:** Accuracy.
- **Imbalanced Data (False Positives are costly):** Precision.
- **Imbalanced Data (False Negatives are costly):** Recall.
- **Imbalanced Data (Require balance):** F1-Score (Binary) / Weighted-F1 (Multi-Class).
- **Evaluating Overall Class Separation:** ROC-AUC.
- **Focusing on Minority Class fairness in Multi-Class:** Macro-F1.

---

## 1. Binary Classification Metrics

Used when the target variable has exactly two classes (e.g., Spam vs. Not Spam). 

### Confusion Matrix
- **Interpretation:** Shows True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) matrix.
- **When to use:** Base for understanding specific error types made by the model.
- **Things to remember:** It is a 2x2 absolute count matrix, not a standalone comparative metric.
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

### Accuracy
- **Interpretation:** Percentage of correct predictions overall. `(TP + TN) / Total`
- **When to use:** When classes are evenly balanced and False Positives / False Negatives have similar costs.
- **Things to remember:** Highly misleading with imbalanced datasets (e.g., 99% fraud-free data).
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
```

### Precision
- **Interpretation:** Out of all positive *predictions*, how many were actually positive? `TP / (TP + FP)`
- **When to use:** When the cost of a False Positive is high (e.g., incorrectly marking good emails as Spam).
- **Things to remember:** High precision means a low false alarm rate.
```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
```

### Recall (Sensitivity / True Positive Rate)
- **Interpretation:** Out of all actual *positives*, how many were predicted correctly? `TP / (TP + FN)`
- **When to use:** When the cost of a False Negative is high (e.g., missing a cancerous tumor).
- **Things to remember:** High recall means few missed positive cases.
```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
```

### F1 Score
- **Interpretation:** Harmonic mean of Precision and Recall.
- **When to use:** When you need a balance between Precision and Recall on an imbalanced class distribution.
- **Things to remember:** Punishes extreme values; both Precision and Recall must be reasonably high to get a good F1.
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
```

### ROC-AUC
- **Interpretation:** Measures the model's ability to rank positive instances higher than negative ones (0.5 = random, 1.0 = perfect). Plot of TPR vs. FPR.
- **When to use:** To evaluate probabilistic models regardless of the hard classification threshold.
- **Things to remember:** Evaluates probabilities (`y_prob`), not hard predicted class labels.
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_prob) # Pass probabilities
```

### Log Loss (Cross-Entropy Loss)
- **Interpretation:** Penalizes false classifications based on the probability confidence.
- **When to use:** When you want well-calibrated probabilities, not just binary label outputs.
- **Things to remember:** Lower implies better models (0.0 is perfect).
```python
from sklearn.metrics import log_loss
loss = log_loss(y_true, y_prob)
```

---

## 2. Multi-Class Classification Metrics

Used when the target variable has more than two classes (e.g., Dog vs. Cat vs. Bird).

**Averaging Methods for Multi-Class:**
- **Macro Average:** Unweighted mean of the metric calculated per class. Uses this to treat all classes equally.
- **Weighted Average:** Mean of the metric calculated per class, weighted by the number of true instances. Best for imbalanced data.
- **Micro Average:** Global average counting total TPs, FPs, and FNs. (Same as overall multi-class Accuracy).

### Multi-Class Confusion Matrix
- **Interpretation:** An *N x N* matrix visualizing misclassifications across all *N* classes.
- **When to use:** To identify which specific classes the model is confusing with each other.
- **Things to remember:** A perfect model has values only on the main diagonal.
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred) # Yields N x N matrix
```

### Multi-Class Precision, Recall, and F1 Score
- **Interpretation:** Evaluates performance with respect to each class, requiring an aggregation strategy.
- **When to use:** Use `macro` to ensure minority classes are predicted well. Use `weighted` to prioritize overall system performance across imbalanced classes.
- **Things to remember:** Scikit-learn requires the `average` parameter to be set for multi-class inputs.
```python
from sklearn.metrics import precision_score, recall_score, f1_score

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

### Multi-Class ROC-AUC
- **Interpretation:** Generalizes AUC to multi-class using One-vs-Rest (OvR) or One-vs-One (OvO) strategies.
- **When to use:** To evaluate the general quality of class probabilities across multiple classes.
- **Things to remember:** Requires sending a probability matrix of shape `(n_samples, n_classes)`.
```python
from sklearn.metrics import roc_auc_score
# y_prob should be shape (n_samples, n_classes)
auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr') 
```

### Multi-Class Log Loss
- **Interpretation:** Evaluates the predicted probability distributions across all classes.
- **When to use:** Training metric for modern machine learning algorithms (Neural Networks, XGBoost).
- **Things to remember:** Heavily penalizes the model for being highly confident and completely wrong.
```python
from sklearn.metrics import log_loss
# y_prob should be shape (n_samples, n_classes)
loss = log_loss(y_true, y_prob) 
```