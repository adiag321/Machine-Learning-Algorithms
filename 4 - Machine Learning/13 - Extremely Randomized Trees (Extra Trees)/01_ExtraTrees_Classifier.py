"""
Extremely Randomized Trees (Extra Trees) Implementation
Key differences from Random Forest:
1. When splitting a node: Random Forests choose the best split among a random subset of features
2. Extra Trees draw random splits for each feature and pick the best among those
3. Extra Trees use the whole learning sample to grow the trees (no bootstrap)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def check_data_requirements(X, y):
    """
    Check if the data meets Extra Trees requirements and provide recommendations.
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    problem_type : str
        'classification' or 'regression'
    """
    print("Data Requirements Check:")
    print("-"*30)
    
    # Check sample size
    n_samples, n_features = X.shape
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    
    if n_samples < 100:
        print("Warning: Small sample size. Extra Trees work better with larger datasets.")
    
    # Check for missing values
    missing_vals = X.isnull().sum().sum()
    if missing_vals > 0:
        print(f"Warning: Found {missing_vals} missing values. Handle them before proceeding.")
    
    # Check feature variance
    zero_var_features = X.columns[X.var() == 0].tolist()
    if zero_var_features:
        print("Warning: Found features with zero variance:", zero_var_features)
    
    # Check target variable
    n_classes = len(np.unique(y))
    print(f"Number of classes: {n_classes}")
    class_counts = pd.Series(y).value_counts()
    print("\nClass distribution:")
    print(class_counts)
        
    # Check for class imbalance
    if (class_counts.max() / class_counts.min()) > 3:
        print("Warning: Significant class imbalance detected.")
    
    # Feature correlation
    corr_matrix = X.corr()
    high_corr_pairs = np.where(np.abs(corr_matrix) > 0.95)
    high_corr_pairs = [(X.columns[i], X.columns[j]) 
                       for i, j in zip(*high_corr_pairs) if i != j and i < j]
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (>0.95):")
        for feat1, feat2 in high_corr_pairs:
            print(f"{feat1} - {feat2}")
   
def train_extra_trees(X_train, X_test, y_train, y_test,
                     n_estimators = 100,
                     criterion = None,  # Will be set based on problem_type
                     min_samples_split = 2,
                     min_samples_leaf = 1,
                     max_features = 'sqrt',
                     max_depth = None,
                     class_weight = None,
                     n_jobs = -1):
    """
    Train and evaluate Extra Trees model with comprehensive metrics.
    
    Parameters:
    -----------
    problem_type : str
        'classification' or 'regression'
    n_estimators : int
        Number of trees in the forest
    criterion : str
        'gini' or 'entropy' for classification, 'squared_error' or 'absolute_error' for regression
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples required in a leaf node
    max_features : str or int
        Number of features to consider when looking for the best split
    max_depth : int
        Maximum depth of the trees
    class_weight : dict or 'balanced'
        Weights associated with classes (classification only)
    n_jobs : int
        Number of parallel jobs
    """
    # Set default criterion based on problem type
    if criterion is None:
        criterion = 'gini'
    
    # Choose model based on problem type
    model = ExtraTreesClassifier(
        n_estimators = n_estimators,
        criterion = criterion,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        class_weight = class_weight,
        n_jobs = n_jobs,
        random_state = 42
        )
    print("\nTraining Extra Trees Classifier...")
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and display metrics based on problem type
    y_pred_proba = model.predict_proba(X_test)
        
    print("\nClassification Metrics:")
    print("-----------------------")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve for binary classification
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()
    
    return model, feature_importance

def perform_grid_search(X, y, problem_type='classification', cv=5):
    """
    Perform grid search to find optimal hyperparameters.
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Choose model based on problem type
    if problem_type == 'classification':
        model = ExtraTreesClassifier(random_state=42)
        scoring = 'accuracy'
    else:
        model = ExtraTreesRegressor(random_state=42)
        scoring = 'r2'
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print("\nBest parameters:", grid_search.best_params_)
    print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
    
    return grid_search

# Example usage with a sample dataset
if __name__ == "__main__":

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Check data requirements
    check_data_requirements(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
    
    # Train and evaluate classifier

    etc_model, feature_imp = train_extra_trees(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Optional: Perform grid search
    print("\nPerforming Grid Search...")
    grid_search = perform_grid_search(X_train_scaled, y_train, 'classification')
    
    # Train with best parameters
    print("\nTraining with best parameters...")
    best_etc, best_feature_imp = train_extra_trees(
        X_train_scaled, X_test_scaled, y_train, y_test,
        problem_type='classification',
        **grid_search.best_params_
    )
    
    # Save the model if needed
    # joblib.dump(best_etc, 'extra_trees_model.pkl')

