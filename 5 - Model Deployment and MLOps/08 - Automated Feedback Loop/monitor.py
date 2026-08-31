import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import recall_score
import joblib
import config

def check_data_drift(production_data: pd.DataFrame) -> bool:
    """Compares production features to baseline data using a KS Test."""
    if not np.path.exists(config.REFERENCE_DATA_PATH):
        return False
        
    reference_data = pd.read_csv(config.REFERENCE_DATA_PATH)
    features = production_data.drop(columns=["target"]).columns
    
    drift_detected = False
    for col in features:
        # Run Kolmogorov-Smirnov test between baseline and production distributions
        stat, p_value = ks_2samp(reference_data[col], production_data[col])
        if p_value < config.DRIFT_P_VALUE_THRESHOLD:
            print(f"  [Drift Alert] Feature '{col}' has drifted! (p-value: {p_value:.5f})")
            drift_detected = True
            
    return drift_detected

def evaluate_production_performance(production_data: pd.DataFrame) -> bool:
    """Evaluates live performance metrics. Returns True if retraining is triggered."""
    if not np.path.exists(config.MODEL_PATH):
        print("No active champion model found.")
        return True

    model = joblib.load(config.MODEL_PATH)
    X_prod = production_data.drop(columns=["target"])
    y_prod = production_data["target"]
    
    preds = model.predict(X_prod)
    current_recall = recall_score(y_prod, preds)
    print(f"Current Production Recall: {current_recall:.4f}")
    
    # Check 1: Metric Degradation
    if current_recall < config.PERFORMANCE_THRESHOLD_RECALL:
        print(f"  [Performance Alert] Recall dropped below threshold ({config.PERFORMANCE_THRESHOLD_RECALL})")
        return True
        
    # Check 2: Feature Drift
    if check_data_drift(production_data):
        print("  [Trigger] Feature drift detected above statistical boundaries.")
        return True
        
    return False