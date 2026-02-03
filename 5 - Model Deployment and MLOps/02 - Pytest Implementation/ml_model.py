"""
This module demonstrates a simple ML workflow using the Iris dataset.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_data():
    """
    Load the Iris dataset.
    
    Returns:
    --------
    tuple : (X, y) where X is features and y is target labels
    """
    iris = load_iris()
    return iris.data, iris.target


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like
        Feature data
    y : array-like
        Target labels
    test_size : float
        Proportion of data to use for testing (default: 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """
    Scale features using standardization (mean=0, std=1).
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
        
    Returns:
    --------
    tuple : (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def create_model(model_type='logistic_regression', random_state=42):
    """
    Create a machine learning model.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('logistic_regression' or 'random_forest')
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : sklearn model object
    """
    if model_type == 'logistic_regression':
        return LogisticRegression(random_state=random_state, max_iter=200)
    elif model_type == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(model, X_train, y_train):
    """
    Train the machine learning model.
    
    Parameters:
    -----------
    model : sklearn model
        The model to train
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
        
    Returns:
    --------
    model : trained sklearn model
    """
    # Validate inputs
    if X_train is None or len(X_train) == 0:
        raise ValueError("Training data cannot be empty")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")
    
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    """
    Make predictions using the trained model.
    
    Parameters:
    -----------
    model : trained sklearn model
        The model to use for predictions
    X : array-like
        Features to predict on
        
    Returns:
    --------
    predictions : array
        Predicted class labels
    """
    if X is None or len(X) == 0:
        raise ValueError("Input data cannot be empty")
    
    return model.predict(X)


def predict_proba(model, X):
    """
    Get prediction probabilities for each class.
    
    Parameters:
    -----------
    model : trained sklearn model
        The model to use for predictions
    X : array-like
        Features to predict on
        
    Returns:
    --------
    probabilities : array
        Prediction probabilities for each class
    """
    return model.predict_proba(X)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return performance metrics.
    
    Parameters:
    -----------
    model : trained sklearn model
        The model to evaluate
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
        
    Returns:
    --------
    dict : Dictionary with accuracy and classification report
    """
    predictions = predict(model, X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': predictions
    }


def save_model(model, scaler, filepath):
    """
    Save the trained model and scaler to disk.
    
    Parameters:
    -----------
    model : trained sklearn model
        The model to save
    scaler : StandardScaler
        The fitted scaler
    filepath : str
        Path where to save the model
    """
    joblib.dump({
        'model': model,
        'scaler': scaler
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a trained model and scaler from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model file
        
    Returns:
    --------
    tuple : (model, scaler)
    """
    data = joblib.load(filepath)
    return data['model'], data['scaler']


def main():
    """
    Main function demonstrating the complete ML workflow.
    """
    # Step 1: Load data
    print("1. Loading Iris dataset...")
    X, y = load_data()
    print(f"   Dataset shape: {X.shape} (150 samples, 4 features)")
    print(f"   Classes: {np.unique(y)} (setosa, versicolor, virginica)")
    
    # Step 2: Split data
    print("2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Step 3: Scale features
    print("3. Scaling features (standardization)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"   Training data - Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
    
    # ===== LOGISTIC REGRESSION MODEL =====
    print("LOGISTIC REGRESSION MODEL")
    
    # Create model
    print("4a. Creating Logistic Regression model...")
    lr_model = create_model(model_type='logistic_regression', random_state=42)
    
    # Train model
    print("5a. Training the model...")
    lr_model = train_model(lr_model, X_train_scaled, y_train)
    
    # Make predictions
    print("6a. Making predictions on test set...")
    lr_predictions = predict(lr_model, X_test_scaled)
    
    # Evaluate model
    print("7a. Evaluating model performance...")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test)
    print(f"Logistic Regression Accuracy: {lr_results['accuracy']:.4f}")
    print("Classification Report:")
    print(lr_results['classification_report'])
    
    # ===== RANDOM FOREST MODEL =====
    print("RANDOM FOREST MODEL")
    
    # Create model
    print("4b. Creating Random Forest model...")
    rf_model = create_model(model_type='random_forest', random_state=42)
    
    # Train model
    print("5b. Training the model...")
    rf_model = train_model(rf_model, X_train_scaled, y_train)
    
    # Make predictions
    print("6b. Making predictions on test set...")
    rf_predictions = predict(rf_model, X_test_scaled)
    
    # Evaluate model
    print("7b. Evaluating model performance...")
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test)
    print(f"Random Forest Accuracy: {rf_results['accuracy']:.4f}")
    print("Classification Report:")
    print(rf_results['classification_report'])
    
    # ===== COMPARISON =====
    print("MODEL COMPARISON")
    
    print(f"Logistic Regression Accuracy: {lr_results['accuracy']:.4f}")
    print(f"Random Forest Accuracy:       {rf_results['accuracy']:.4f}")
    
    if lr_results['accuracy'] > rf_results['accuracy']:
        print("Logistic Regression performed better!")
    elif rf_results['accuracy'] > lr_results['accuracy']:
        print("Random Forest performed better!")
    else:
        print("Both models performed equally well!")    

if __name__ == "__main__":
    main()
