#K-Nearest Neighbour (KNN) Regression
import os
import logging
from networkx import dfs_edges
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error)
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\Datasets')

# Set up logging to both file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler("knn_regressor_model.log", mode='w'),
        logging.StreamHandler()
        ])

##########################
#       FUNCTIONS
##########################
def data_manipulation(data = df, X = X, y = y):
    """
    This function handles:
    - Checking for missing values
    - Extracting features and target variable
    - Train-test split
    - Feature Scaling (Standardization)
    
    Returns:
        X_train_scaled (array): Scaled training features
        X_test_scaled (array): Scaled testing features
        y_train (array): Training target variable
        y_test (array): Testing target variable
        scaler (StandardScaler): Fitted scaler object for future transformations
    """
    logging.info(f"Dataset shape: {data.shape}")
    logging.info(f"Number of features: {X.shape[1]}")
    logging.info(f"Missing values:\n{data.isnull().sum()}")
    
    # Remove rows with missing values (if any)
    data = data.dropna()
    logging.info(f"Dataset shape after removing missing values: {data.shape}")
    
    # Extract features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Ensure numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")
    logging.info(f"Target variable statistics - Min: {float(np.min(y)):.2f}, Max: {float(np.max(y)):.2f}, Mean: {float(np.mean(y)):.2f}")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    
    # Feature Scaling (Standardization)
    logging.info("Applying Feature Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logging.info("Data manipulation completed successfully!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def knn_regression_pipeline(X_train, X_test, y_train, y_test):
    """
    This function handles:
    - Finding optimal K value using Elbow method
    - Hyperparameter tuning using GridSearchCV
    - Model training
    - Model evaluation with regression metrics (MSE, MAE, R², MAPE)
    - Cross-validation
    - Visualization of predictions vs actual values
    - Elbow curve for optimal K selection
    
    Args:
        X_train (array): Scaled training features
        X_test (array): Scaled testing features
        y_train (array): Training target variable
        y_test (array): Testing target variable
        
    Returns:
        best_model (KNeighborsRegressor): Trained KNN regressor with best parameters
        y_pred (array): Predictions on test set
        metrics (dict): Dictionary containing all evaluation metrics
    """
    # Step 1: Elbow Method to find optimal K
    logging.info("Step 1: Performing Elbow Method for optimal K selection...")
    mse_scores = []
    r2_scores = []
    k_range = range(1, 31)
    
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance', p=2, metric='minkowski')
        knn.fit(X_train, y_train)
        y_pred_k = knn.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred_k))
        r2_scores.append(r2_score(y_test, y_pred_k))
    
    # Plot Elbow Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, mse_scores, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', linewidth=2)
    plt.title('Elbow Method - MSE vs K')
    plt.xlabel('K Value')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, r2_scores, color='green', linestyle='dashed', marker='s', markerfacecolor='black', linewidth=2)
    plt.title('R² Score vs K')
    plt.xlabel('K Value')
    plt.ylabel('R² Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    logging.info(f"MSE scores range: {min(mse_scores):.4f} to {max(mse_scores):.4f}")
    logging.info(f"R² scores range: {min(r2_scores):.4f} to {max(r2_scores):.4f}")
    
    # Step 2: Hyperparameter Tuning using GridSearchCV
    logging.info("\nStep 2: Performing GridSearchCV for hyperparameter tuning...")
    param_grid = {
        'n_neighbors': list(range(3, 31, 2)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    }
    
    grid = GridSearchCV(
        KNeighborsRegressor(),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid.best_params_}")
    logging.info(f"Best CV MSE Score: {-grid.best_score_:.4f}")
    
    best_model = grid.best_estimator_
    
    # Step 3: Make Predictions
    logging.info("\nStep 3: Making predictions on test set...")
    y_pred = best_model.predict(X_test)
    
    # Step 4: Model Evaluation with Regression Metrics
    logging.info("\nStep 4: Evaluating the model...")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2,
        'MAPE': mape
    }
    
    logging.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logging.info(f"R² Score: {r2:.4f}")
    logging.info(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    
    # Step 5: Cross-Validation
    logging.info("\nStep 5: Performing cross-validation...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    logging.info(f"Cross-validation R² scores: {cv_scores}")
    logging.info(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Step 6: Visualization - Actual vs Predicted
    logging.info("\nStep 6: Visualizing predictions...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('KNN Regression: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('KNN Regression: Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    logging.info("\nKNN Regression pipeline completed successfully!")
    return best_model, y_pred, metrics

###############################################################################
#                                    MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    # Step 1: Data Manipulation
    logging.info("Loading Diabetes dataset from sklearn...")
    # getting column names
    column_names = load_diabetes().feature_names
    X, y = load_diabetes(return_X_y=True)    
    # Create a DataFrame for better visualization
    df = pd.DataFrame(X, columns=column_names)
    df['target'] = y
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = data_manipulation(data = df, X = X, y = y)
    
    # Step 2: KNN Regression Pipeline
    best_model, y_pred, metrics = knn_regression_pipeline(X_train_scaled, X_test_scaled, y_train, y_test)
    
    logging.info("\n" + "="*50)
    logging.info("FINAL MODEL SUMMARY")
    logging.info("="*50)
    logging.info(f"Model: {best_model}")
    logging.info(f"\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        logging.info(f"  {metric_name}: {metric_value:.4f}")
    logging.info("="*50)