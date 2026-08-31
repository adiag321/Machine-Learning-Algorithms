import os

# Create artifact directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

MODEL_PATH = "artifacts/champion_model.joblib"
REFERENCE_DATA_PATH = "artifacts/reference_data.csv"

# MLOps Thresholds
PERFORMANCE_THRESHOLD_RECALL = 0.75  # Trigger retrain if recall falls below this
DRIFT_P_VALUE_THRESHOLD = 0.05       # KS test significance level