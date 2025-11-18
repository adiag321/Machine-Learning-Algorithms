import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def xgboost_model(X_train, X_test, y_train, y_test, feature_names=None):
    """
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training target
    y_test : array-like
        Testing target
    feature_names : list, optional
        List of feature names for feature importance
    """
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,              # Number of boosting stages
        learning_rate=0.1,             # Shrinks contribution of each tree
        max_depth=5,                   # Maximum depth of individual trees
        min_child_weight=1,            # Minimum sum of weights in child node
        subsample=0.8,                 # Fraction of samples for fitting each tree
        colsample_bytree=0.8,          # Fraction of features for each tree
        reg_alpha=0.0,                 # L1 regularization
        reg_lambda=1.0,                # L2 regularization
        random_state=42,               # Reproducibility
        eval_metric='logloss',         # Evaluation metric
        use_label_encoder=False        # Suppress warning
    )
    
    # Train the model with early stopping
    xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False,
        # early_stopping_rounds=10
    )
    y_pred_train = xgb_clf.predict(X_train)
    y_pred_test = xgb_clf.predict(X_test)
    y_pred_proba_train = xgb_clf.predict_proba(X_train)[:, 1]
    y_pred_proba_test = xgb_clf.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    train_precision = precision_score(y_train, y_pred_train)
    test_precision = precision_score(y_test, y_pred_test)
    
    train_recall = recall_score(y_train, y_pred_train)
    test_recall = recall_score(y_test, y_pred_test)
    
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)
    
    train_roc_auc = roc_auc_score(y_train, y_pred_proba_train)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba_test)
    avg_precision = average_precision_score(y_test, y_pred_proba_test)
    
    cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='f1', error_score='raise')
    
    classification_rep = classification_report(y_test, y_pred_test, output_dict=True)
    
    # Print metrics
    print("\nACCURACY METRICS")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    
    print("\nPRECISION METRICS")
    print(f"  Train Precision: {train_precision:.4f}")
    print(f"  Test Precision:  {test_precision:.4f}")
    
    print("\nRECALL METRICS")
    print(f"  Train Recall: {train_recall:.4f}")
    print(f"  Test Recall:  {test_recall:.4f}")
    
    print("\nF1-SCORE METRICS")
    print(f"  Train F1-Score: {train_f1:.4f}")
    print(f"  Test F1-Score:  {test_f1:.4f}")
    print(f"  Cross-Val F1 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("\nROC-AUC METRICS")
    print(f"  Train ROC-AUC: {train_roc_auc:.4f}")
    print(f"  Test ROC-AUC:  {test_roc_auc:.4f}")
    
    print("\nAVERAGE PRECISION")
    print(f"  Average Precision (PR-AUC): {avg_precision:.4f}")
    
    print("\nCONFUSION MATRIX (Test Set)")
    print(f"  True Negatives:  {cm_test[0, 0]}")
    print(f"  False Positives: {cm_test[0, 1]}")
    print(f"  False Negatives: {cm_test[1, 0]}")
    print(f"  True Positives:  {cm_test[1, 1]}")
    
    print("\nCLASSIFICATION REPORT (Test Set)")
    report = classification_rep
    print(f"  Class 0 - Precision: {report['0']['precision']:.4f}, Recall: {report['0']['recall']:.4f}, F1: {report['0']['f1-score']:.4f}")
    print(f"  Class 1 - Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}, F1: {report['1']['f1-score']:.4f}")
    print(f"  Macro Avg - Precision: {report['macro avg']['precision']:.4f}, Recall: {report['macro avg']['recall']:.4f}, F1: {report['macro avg']['f1-score']:.4f}")
    
    print("\nFEATURE IMPORTANCE (Top 10)")
    feature_importance = xgb_clf.feature_importances_
    if feature_names is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        print(importance_df.head(10).to_string(index=False))
    else:
        importance_sorted = np.argsort(feature_importance)[::-1][:10]
        for idx, feat_idx in enumerate(importance_sorted, 1):
            print(f"  {idx}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")

    # Plotting results
    fig = plt.figure(figsize=(18, 12))
    
    # Confusion Matrix - Test Set
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
    ax1.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    
    # Precision-Recall Curve
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc="upper right")
    ax3.grid(alpha=0.3)
    
    # Metrics Comparison
    ax4 = plt.subplot(2, 3, 4)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    train_values = [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc]
    test_values = [test_accuracy, test_precision, test_recall, test_f1, test_roc_auc]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax4.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    ax4.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    ax4.set_ylabel('Score')
    ax4.set_title('Metrics Comparison (Train vs Test)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Feature Importance
    ax5 = plt.subplot(2, 3, 5)
    feature_imp = xgb_clf.feature_importances_
    top_n = 10
    top_indices = np.argsort(feature_imp)[-top_n:]
    
    if feature_names is not None:
        top_names = [feature_names[i] for i in top_indices]
    else:
        top_names = [f'Feature {i}' for i in top_indices]
    
    top_values = feature_imp[top_indices]
    ax5.barh(range(len(top_values)), top_values, color='steelblue')
    ax5.set_yticks(range(len(top_values)))
    ax5.set_yticklabels(top_names)
    ax5.set_xlabel('Importance')
    ax5.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # Confusion Matrix - Train Set
    ax6 = plt.subplot(2, 3, 6)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=ax6, cbar=False)
    ax6.set_title('Confusion Matrix (Train Set)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Predicted Label')
    ax6.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()

##############################################################
# Main Execution
##############################################################
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

# Data Splitting
print("\nSplitting data (80-20 stratified split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and Evaluate Model
xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names=feature_names)