# -*- coding: utf-8 -*-
"""
Multi Class Classification: Precision, Recall, and F1 Score
Link: CampusX Youtube Channel - https://youtu.be/iK-kdhJ-7yI?si=q5OQ6w_p34JTQwDN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
                            classification_report, cohen_kappa_score, log_loss, roc_auc_score, roc_curve, auc)

import warnings
warnings.filterwarnings('ignore')

# Load iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = data.target

################################
## Split the data
################################
X = df.drop(columns=['Species'])
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_proba1 = clf1.predict_proba(X_test)
y_proba2 = clf2.predict_proba(X_test)

################################
# Evaluation function
################################
def evaluate_model(name, y_true, y_pred, y_proba=None):
    print(f"\n{name}\n{'-' * len(name)}")
    
    # Core metrics
    print(f"Accuracy             : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Cohen’s Kappa        : {cohen_kappa_score(y_true, y_pred):.4f}")
    
    # Per class metrics
    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        columns=[f"Pred {label}" for label in np.unique(y_true)],
        index=[f"Actual {label}" for label in np.unique(y_true)]
    ))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=data.target_names))
    
    print("Macro Avg Precision  : {:.4f}".format(precision_score(y_true, y_pred, average='macro')))
    print("Macro Avg Recall     : {:.4f}".format(recall_score(y_true, y_pred, average='macro')))
    print("Macro Avg F1 Score   : {:.4f}".format(f1_score(y_true, y_pred, average='macro')))
    
    print("Micro Avg Precision  : {:.4f}".format(precision_score(y_true, y_pred, average='micro')))
    print("Micro Avg Recall     : {:.4f}".format(recall_score(y_true, y_pred, average='micro')))
    print("Micro Avg F1 Score   : {:.4f}".format(f1_score(y_true, y_pred, average='micro')))

    if y_proba is not None:
        print(f"Log Loss: {log_loss(y_true, y_proba):.4f}")
        
        # Binarize labels for ROC AUC computation
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        if y_true_bin.shape[1] == 1:  # Binary case fallback
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        print(f"ROC AUC (OvR, macro) : {roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr'):.4f}")
        print(f"ROC AUC (OvR, weighted): {roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr'):.4f}")
        
        # Optional: Plot ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_true_bin.shape[1]

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"{data.target_names[i]} (AUC = {roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} - ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

# Evaluate both classifiers
evaluate_model("Logistic Regression", y_test, y_pred1, y_proba1)
evaluate_model("Decision Tree", y_test, y_pred2, y_proba2)

# Show a few predictions
result = pd.DataFrame({
    'Actual Label': y_test.values,
    'Logistic Regression Prediction': y_pred1,
    'Decision Tree Prediction': y_pred2
})
print("\nSample Predictions:\n", result.sample(10, random_state=1))
