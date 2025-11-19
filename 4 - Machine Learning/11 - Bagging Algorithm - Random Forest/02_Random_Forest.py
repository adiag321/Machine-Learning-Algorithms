import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')


def random_forest_comprehensive_example(X_train, X_test, y_train, y_test, feature_names, target_names):
    """
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training labels
    y_test : array-like
        Testing labels
    feature_names : list
        Names of features
    target_names : list
        Names of target classes
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing model, metrics, and data
    """
    
    # ==================== STEP 1: RANDOM FOREST WITH ALL PARAMETERS ====================    
    model = RandomForestClassifier(
        # Tree construction parameters
        n_estimators=100,              # Number of trees in forest
        max_depth=15,                  # Maximum depth of each tree
        min_samples_split=5,           # Minimum samples to split a node
        min_samples_leaf=2,            # Minimum samples at leaf node
        min_weight_fraction_leaf=0.0,  # Minimum fraction of weighted samples
        max_features='sqrt',           # Features to consider at split ('sqrt', 'log2', None, int, float)
        max_leaf_nodes=None,           # Maximum leaf nodes
        min_impurity_decrease=0.0,     # Minimum impurity decrease threshold
        
        # Randomness parameters
        bootstrap=True,                # Whether to use bootstrap samples
        oob_score=True,                # Whether to use out-of-bag samples
        random_state=42,
        
        # Parallelization
        n_jobs=-1,                     # Use all processors
        
        # Sampling parameters
        class_weight=None,             # Class weights ('balanced', dict, or None)
        criterion='gini',              # Split criterion ('gini' or 'entropy')
        
        # Behavior
        verbose=0                      # Verbosity level
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # ==================== STEP 2: PREDICTIONS ====================    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print(f"Predictions (first 10): {y_pred[:10]}")
    print(f"Prediction probabilities (first 5):\n{y_pred_proba[:5]}")
    
    # ==================== STEP 3: EVALUATION METRICS ====================
    print("\n Evaluation Metrics")

    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    metrics = {'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    print("Train Set Metrics:")
    print(f"Accuracy:  {model.score(X_train, y_train):.4f}")
    
    print("\nTest Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:12s}: {metric_value:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Out-of-bag score
    print(f"\nOut-of-Bag (OOB) Score: {model.oob_score_:.4f}")
    
    # ==================== STEP 4: FEATURE IMPORTANCE ====================
    print("Feature Importance Analysis")
    
    feature_importance = pd.DataFrame({'Feature': feature_names,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # ==================== STEP 5: CROSS-VALIDATION ====================
    print("Cross-Validation Analysis")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')
    print(f"5-Fold CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ==================== STEP 6: VISUALIZATIONS ====================
    print("Creating visualizations...")
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Feature Importance
    ax1 = plt.subplot(2, 3, 1)
    top_features = feature_importance.head(10)
    ax1.barh(range(len(top_features)), top_features['Importance'].values.astype(float), color='steelblue')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['Feature'].values)
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 10 Feature Importances')
    ax1.invert_yaxis()
    
    # Plot 2: Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    
    # Plot 3: ROC Curve
    ax3 = plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Cross-Validation Scores
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(range(1, len(cv_scores) + 1), cv_scores, color='seagreen', alpha=0.7)
    ax4.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Score')
    ax4.set_title('5-Fold Cross-Validation Scores')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Prediction Probability Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_pred_proba[:, 1], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax5.set_xlabel('Predicted Probability (Class 1)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Predicted Probabilities')
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Number of Trees vs OOB Score
    ax6 = plt.subplot(2, 3, 6)
    oob_scores = []
    n_trees_range = range(10, 101, 10)
    
    for n_trees in n_trees_range:
        temp_model = RandomForestClassifier(n_estimators=n_trees, oob_score=True, random_state=42, n_jobs=-1)
        temp_model.fit(X_train, y_train)
        oob_scores.append(temp_model.oob_score_)
    
    ax6.plot(n_trees_range, oob_scores, marker='o', linestyle='-', linewidth=2, color='purple')
    ax6.set_xlabel('Number of Trees')
    ax6.set_ylabel('OOB Score')
    ax6.set_title('OOB Score vs Number of Trees')
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ==================== STEP 7: MODEL SUMMARY ====================
    print("Model Summary")
    print(f"Problem Type: Binary Classification")
    print(f"Dataset: Breast Cancer")
    print(f"Model: Random Forest Classifier")
    print(f"Number of Trees: {model.n_estimators}")
    print(f"Max Depth: {model.max_depth}")
    print(f"Min Samples Split: {model.min_samples_split}")
    print(f"Min Samples Leaf: {model.min_samples_leaf}")
    print(f"Max Features: {model.max_features}")
    print(f"Bootstrap: {model.bootstrap}")
    print(f"Criterion: {model.criterion}")
    print(f"OOB Score: {model.oob_score_:.4f}")
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'cv_scores': cv_scores,
        'confusion_matrix': cm
    }


###########################
# Example Usage
###########################

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names
print(f"\nDataset: Breast Cancer")
print(f"Total Samples: {X.shape[0]}")
print(f"Total Features: {X.shape[1]}")
print(f"Target distribution:\n{pd.Series(y).value_counts()}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== RUN RANDOM FOREST EXAMPLE ====================
results = random_forest_comprehensive_example(X_train, X_test, y_train, y_test, feature_names, target_names)
