import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import joblib
import config

def generate_base_data(n_samples=1000, shift_mean=0.0):
    """Generates a toy binary classification dataset."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=5, 
        n_informative=3, 
        n_redundant=2, 
        random_state=42
    )
    # Inject synthetic mean shift to simulate data drift when requested
    X = X + shift_mean
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df

def run_training_pipeline(training_data: pd.DataFrame):
    """Trains a new challenger model and promotes it if it meets criteria."""
    print("\n--- Running Training Pipeline ---")
    
    X = training_data.drop(columns=["target"])
    y = training_data["target"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_test_split=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate Validation Metrics
    preds = model.predict(X_val)
    recall = recall_score(y_val, preds)
    print(f"Challenger Model Validation Recall: {recall:.4f}")
    
    if recall >= config.PERFORMANCE_THRESHOLD_RECALL:
        # Save model (Promote to Champion)
        joblib.dump(model, config.MODEL_PATH)
        # Save reference data for drift monitoring
        training_data.to_csv(config.REFERENCE_DATA_PATH, index=False)
        print("-> Challenger approved. Promoted to Champion Registry.")
    else:
        print("-> Training failed: Challenger performance below threshold.")