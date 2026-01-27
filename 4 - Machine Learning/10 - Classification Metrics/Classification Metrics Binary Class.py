## Classification Metrics
'''
Binary Class Classification: Precision, Recall, and F1 Score
Link: CampusX Youtube Channel - https://youtu.be/iK-kdhJ-7yI?si=q5OQ6w_p34JTQwDN
'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report,
                             roc_curve, precision_recall_curve, auc, roc_auc_score, log_loss, balanced_accuracy_score, 
                             cohen_kappa_score)
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms')

# Load dataset
df = pd.read_csv('./Datasets/heart/heart.csv')

################################
## Split the data
################################
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

################################
# Initialize models
################################
log_reg = LogisticRegression(max_iter=1000)
dec_tree = DecisionTreeClassifier()
log_reg.fit(X_train, y_train)
dec_tree.fit(X_train, y_train)
log_reg_y_pred = log_reg.predict(X_test)
dec_tree_y_pred = dec_tree.predict(X_test)

################################
# Evaluation function
################################
def evaluate_model(name, y_true, y_pred, y_proba=None):
    print(f"\n{name}\n{'-' * len(name)}")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    # Confusion Matrix
    print("\nConfusion Matrix:\n", pd.DataFrame(cm,
                                              columns=["Pred 0", "Pred 1"],
                                              index=["Actual 0", "Actual 1"]))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity      : {specificity:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Cohen’s Kappa    : {cohen_kappa_score(y_true, y_pred):.4f}")
    
    if y_proba is not None:
        print(f"ROC AUC Score    : {roc_auc_score(y_true, y_proba):.4f}")
        print(f"Log Loss         : {log_loss(y_true, y_proba):.4f}")
    
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} - ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Precision-Recall Curve
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name} - Precision-Recall Curve")
        plt.grid(True)
        plt.show()
        
        # Precision-Recall vs Threshold with optimal threshold analysis
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-10)
        
        # Find optimal thresholds for different objectives
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds_pr[optimal_f1_idx]
        
        # Find threshold where precision and recall are most balanced (closest to each other)
        balance_idx = np.argmin(np.abs(precision_vals[:-1] - recall_vals[:-1]))
        balance_threshold = thresholds_pr[balance_idx]
        
        print(f"\nThreshold Analysis:")
        print(f"   Best F1 Score      : Threshold = {optimal_f1_threshold:.3f} (F1={f1_scores[optimal_f1_idx]:.3f}, P={precision_vals[optimal_f1_idx]:.3f}, R={recall_vals[optimal_f1_idx]:.3f})")
        print(f"   Balanced P-R       : Threshold = {balance_threshold:.3f} (P={precision_vals[balance_idx]:.3f}, R={recall_vals[balance_idx]:.3f})")
        print(f"   Max Precision      : Threshold = {thresholds_pr[-1]:.3f} (P={precision_vals[-2]:.3f}, R={recall_vals[-2]:.3f})")
        print(f"   Max Recall         : Threshold = {thresholds_pr[0]:.3f} (P={precision_vals[0]:.3f}, R={recall_vals[0]:.3f})")
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_pr, precision_vals[:-1], label="Precision", linewidth=2, color='blue')
        plt.plot(thresholds_pr, recall_vals[:-1], label="Recall", linewidth=2, color='orange')
        plt.plot(thresholds_pr, f1_scores, label="F1 Score", linewidth=2, color='green', linestyle='--')
        
        # Mark optimal thresholds
        plt.axvline(optimal_f1_threshold, color='green', linestyle=':', alpha=0.7, label=f'Best F1 (t={optimal_f1_threshold:.3f})')
        plt.axvline(balance_threshold, color='purple', linestyle=':', alpha=0.7, label=f'Balanced P-R (t={balance_threshold:.3f})')
        
        plt.xlabel("Threshold", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.title(f"{name} - Precision, Recall & F1 vs Threshold", fontsize=12, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.show()
    
################################
# Evaluate both models
################################
log_reg_y_proba = log_reg.predict_proba(X_test)[:, 1]
evaluate_model("Logistic Regression", y_test, log_reg_y_pred, y_proba=log_reg_y_proba)

dec_tree_y_proba = dec_tree.predict_proba(X_test)[:, 1]
evaluate_model("Decision Tree", y_test, dec_tree_y_pred, y_proba=dec_tree_y_proba)



