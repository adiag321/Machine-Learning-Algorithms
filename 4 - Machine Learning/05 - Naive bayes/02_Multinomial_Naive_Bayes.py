# Multinomial Naive Bayes Classifier
# Note: Multinomial NB expects non-negative integer counts; we scale Iris features to simulate counts
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score, precision_score, recall_score, 
                             f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef,
                             precision_recall_curve, average_precision_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
target_names = data.target_names

# Scale features to [0, 100] to simulate counts (Multinomial NB expects non-negative integers)
scaler = MinMaxScaler(feature_range=(0, 100))
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=data.feature_names)
X = X.astype(int)  # Convert to integers for Multinomial NB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

####################################
# Multinomial Naive Bayes Models
####################################
def multinomial_naivebayes(X_train, y_train, X_test, y_test, target_names):
    """
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    target_names : Class names
    
    Returns:
    --------
    model : Trained MultinomialNB model
    y_pred : Predictions on test data
    metrics : Dictionary containing all evaluation metrics
    """
    mnb = MultinomialNB()
    y_pred = mnb.fit(X_train, y_train).predict(X_test)
    
    # ===== Basic Metrics =====
    n_mislabeled = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)

    print("MULTINOMIAL NAIVE BAYES - COMPREHENSIVE EVALUATION METRICS")
    print("="*70)
    print(f"\nMislabeled Points: {n_mislabeled} out of {X_test.shape[0]} ({n_mislabeled/X_test.shape[0]*100:.2f}%)")
    print(f"Model Parameters: {mnb.get_params()}")
    print(f"\n{'Accuracy (%):'} {accuracy * 100:.4f}%")
    
    # ===== Model Parameters =====
    print(f"\n{'Class Labels:'} {mnb.classes_}")
    print(f"{'Class Priors (P(class)):'}")
    for i, cls in enumerate(mnb.classes_):
        print(f"  Class {target_names[cls]}: {mnb.class_log_prior_[i]:.4f} (log), {np.exp(mnb.class_log_prior_[i]):.4f}")
    
    # ===== Probability Predictions =====
    print(f"\n{'Predicted Probabilities (first 5 samples):'}")
    probs = mnb.predict_proba(X_test)
    for i in range(min(5, len(y_test))):
        print(f"  Sample {i}: {[f'{p:.4f}' for p in probs[i]]}")
    
    # ===== Precision, Recall, F1 (per class and overall) =====
    print(f"\n{'Precision (weighted):'} {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"{'Recall (weighted):'} {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"{'F1-Score (weighted):'} {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"{'Precision (macro):'} {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"{'Recall (macro):'} {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"{'F1-Score (macro):'} {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    # ===== Confusion Matrix & Heatmap =====
    print(f"\n{'Confusion Matrix:'}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # ===== Additional Classification Metrics =====
    print(f"\n{'Hamming Loss:'} {hamming_loss(y_test, y_pred):.4f}")
    print(f"{'Cohen Kappa Score:'} {cohen_kappa_score(y_test, y_pred):.4f}")
    print(f"{'Matthews Correlation Coeff:'} {matthews_corrcoef(y_test, y_pred):.4f}")
    
    # ===== Per-Class Metrics =====
    print(f"\nPer-Class Metrics:")
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    for i, cls in enumerate(mnb.classes_):
        print(f"  {target_names[cls]:>15} | Precision: {precision_per_class[i]:.4f} | Recall: {recall_per_class[i]:.4f} | F1: {f1_per_class[i]:.4f}")
    
    # ===== Detailed Classification Report =====
    print(f"\n{'Classification Report:'}")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # ===== ROC-AUC Scores (One-vs-Rest) =====
    print("\n" + "="*70)
    print("GENERATING ROC CURVES (One-vs-Rest)")

    y_bin = label_binarize(y_test, classes=mnb.classes_)
    y_proba = mnb.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(mnb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(len(mnb.classes_)):
        plt.plot(fpr[i], tpr[i], label=f"Class {target_names[mnb.classes_[i]]} (AUC = {roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title("ROC Curves (One-vs-Rest) - Multinomial Naive Bayes")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    ##############################
    # Confusion Matrix Heatmap
    ##############################
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Multinomial NB - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    # Store metrics in dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'hamming_loss': hamming_loss(y_test, y_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'matthews_corr': matthews_corrcoef(y_test, y_pred),
    }
    return mnb, y_pred, metrics

# Train model
mnb, y_pred, metrics = multinomial_naivebayes(X_train, y_train, X_test, y_test, target_names)
