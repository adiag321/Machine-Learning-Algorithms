"""
Extremely Randomized Trees (Extra Trees) Implementation for Regression
Key differences from Random Forest:
1. When splitting a node: Random Forests choose the best split among a random subset of features
2. Extra Trees draw random splits for each feature and pick the best among those
3. Extra Trees use the whole learning sample to grow the trees (no bootstrap)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from time import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error, mean_absolute_percentage_error)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def check_data_requirements(X, y):
    """
    Check if the data meets Extra Trees Regression requirements and provide recommendations.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable (continuous)
    """
    print("Data Requirements Check:")
    print("-----------------------")
    
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
    print("\nTarget Variable Statistics:")
    print(f"Mean: {np.mean(y):.2f}")
    print(f"Std: {np.std(y):.2f}")
    print(f"Range: {np.min(y):.2f} to {np.max(y):.2f}")
    
    # Check for outliers in target
    z_scores = np.abs((y - np.mean(y)) / np.std(y))
    outliers = np.sum(z_scores > 3)
    if outliers > 0:
        print(f"Warning: Found {outliers} potential outliers in target variable (|z-score| > 3)")
    
    # Feature correlation
    corr_matrix = X.corr()
    high_corr_pairs = np.where(np.abs(corr_matrix) > 0.95)
    high_corr_pairs = [(X.columns[i], X.columns[j]) 
                       for i, j in zip(*high_corr_pairs) if i != j and i < j]
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (>0.95):")
        for feat1, feat2 in high_corr_pairs:
            print(f"{feat1} - {feat2}")

def train_extra_trees_regressor(X_train, X_test, y_train, y_test,
                              n_estimators=100,
                              criterion='squared_error',
                              min_samples_split=2,
                              min_samples_leaf=1,
                              max_features='sqrt',
                              max_depth=None,
                              n_jobs=-1):
    """
    Train and evaluate Extra Trees Regressor with comprehensive metrics.
    
    Parameters:
    -----------
    n_estimators : int
        Number of trees in the forest
    criterion : {'squared_error', 'absolute_error', 'friedman_mse', 'poisson'}
        The function to measure the quality of a split
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples required in a leaf node
    max_features : int, float or {'auto', 'sqrt', 'log2'}
        Number of features to consider when looking for the best split
    max_depth : int
        Maximum depth of the trees
    n_jobs : int
        Number of parallel jobs
    """
    # Initialize and train model
    etr = ExtraTreesRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=42
    )
    
    # Fit the model and time it
    start_time = time()
    etr.fit(X_train, y_train)
    train_time = time() - start_time
    
    # Make predictions
    y_pred = etr.predict(X_test)
    
    # Calculate comprehensive metrics
    print("\nModel Performance Metrics:")
    print("--------------------------")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    ev_score = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.4f}")
    print(f"Explained Variance Score: {ev_score:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"Maximum Error: {max_err:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted Plot
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted Values')
    
    # 2. Residuals Plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted Values')
    
    # 3. Residuals Distribution
    sns.histplot(residuals, kde=True, ax=ax3)
    ax3.set_title('Residuals Distribution')
    ax3.set_xlabel('Residual Value')
    
    # 4. Feature Importance Plot
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': etr.feature_importances_
    }).sort_values('importance', ascending=True)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax4)
    ax4.set_title('Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Additional Analysis: Error Distribution
    plt.figure(figsize=(10, 6))
    error_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': np.abs(y_test - y_pred)
    }).sort_values('Error', ascending=False)
    
    plt.scatter(range(len(error_df)), error_df['Error'], alpha=0.5)
    plt.title('Error Distribution Across Samples')
    plt.xlabel('Sample Index (sorted by error)')
    plt.ylabel('Absolute Error')
    plt.show()
    
    return etr, feature_importance, error_df.head(10)

def perform_grid_search(X, y, cv=5):
    """
    Perform grid search to find optimal hyperparameters for regression.
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    # Initialize model
    model = ExtraTreesRegressor(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='r2',
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print("\nBest parameters:", grid_search.best_params_)
    print(f"Best R² score: {grid_search.best_score_:.4f}")
    
    # Create CV results dataframe
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_idx = cv_results['rank_test_score'] == 1
    print("\nBest Model Cross-validation Results:")
    print(f"Mean R² Score: {cv_results.loc[best_idx, 'mean_test_score'].values[0]:.4f}")
    print(f"Std R² Score: {cv_results.loc[best_idx, 'std_test_score'].values[0]:.4f}")
    
    return grid_search

# Example usage
if __name__ == "__main__":
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Check data requirements
    check_data_requirements(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # Train and evaluate regressor
    print("\nTraining Extra Trees Regressor...")
    etr_model, feature_imp, top_errors = train_extra_trees_regressor(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Perform grid search
    print("\nPerforming Grid Search...")
    grid_search = perform_grid_search(X_train_scaled, y_train)
    
    # Train with best parameters
    print("\nTraining with best parameters...")
    best_etr, best_feature_imp, best_errors = train_extra_trees_regressor(
        X_train_scaled, X_test_scaled, y_train, y_test,
        **grid_search.best_params_
    )
    
    # Print samples with highest prediction errors
    print("\nTop 10 Samples with Highest Prediction Errors:")
    print(best_errors)
    
    # Save the model if needed
    # joblib.dump(best_etr, 'extra_trees_regressor_model.pkl')
