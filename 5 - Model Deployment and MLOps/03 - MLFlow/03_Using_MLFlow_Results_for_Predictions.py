#######################################################
# TRAINING THE MODELS AND LOGGING THEM TO MLFLOW
#######################################################
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# MLFlow libraries
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow import MlflowClient             # Added for model registry management

# 1. Setup Data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 2. Define Models and Hyperparameters to sweep
# This dictionary acts as your "Control Center"
model_configs = {
    "Random_Forest": {
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },
    "SVM": {
        "model_class": SVC,
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }
}

# 3. Set Experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000") # Using local sqlite for immediate testing
mlflow.set_experiment("Model_Comparison_Sweep1")

# 4. The Execution Loop
for pred_model_name, config in model_configs.items():
    
    # Start a PARENT run for the Model Type (e.g., "Random_Forest")
    # This groups all the hyperparam variations under one collapsible header in the UI
    with mlflow.start_run(run_name = pred_model_name) as parent_run:
        print(f"Running sweep for: {pred_model_name}")
        
        # Generate all combinations of parameters (Grid Search)
        param_combinations = list(ParameterGrid(config["params"]))
        
        for params in param_combinations:
            
            # Start a CHILD run for the specific hyperparameters (nested=True)
            with mlflow.start_run(run_name = f"combo_{param_combinations.index(params)}", nested=True):
                
                # A. Init and Train Model
                clf = config["model_class"](**params)
                clf.fit(X_train, y_train)
                
                # B. Predict and Evaluate
                predictions = clf.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                # C. Log everything to MLflow
                mlflow.log_params(params)                         # Log the specific hyperparameters
                mlflow.log_metric("accuracy", accuracy)             # Log the result
                mlflow.log_param("model_type", pred_model_name)        # Tag the model type
                
                # Signature details
                signature = infer_signature(X_train, clf.predict(X_train))

                # Log the actual model file (optional, can take space)
                model_info = mlflow.sklearn.log_model(sk_model = clf, model_type = pred_model_name, signature = signature, input_example = X_test[:5])
                
                print(f"Logged: {params} | Accuracy: {accuracy:.4f}")

print("Experiment Complete. Check MLflow UI.")


#######################################################
# LOADING THE BEST MODEL FROM TRACKED MODELS IN MLFLOW
########################################################
# 1. Connect to your tracking server
# mlflow.set_tracking_uri("http://127.0.0.1:5000") 
experiment_name = "Model_Comparison_Sweep1"

# 2. Get the Experiment ID
current_experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = current_experiment.experiment_id
print("Current Experiment Details:", current_experiment)
print(f"Experiment ID: {experiment_id}")

# 3. Search all runs in this experiment
# We order by 'metrics.accuracy' Descending to get the best one first
df_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.8",                         # Optional: Only consider decent models
    order_by=["metrics.accuracy DESC"]
)
#print(df_runs)

# 4. Extract the Best Run
if not df_runs.empty:
    best_run = df_runs.iloc[0]                                  # The first row is the best because we sorted it
    best_run_id = best_run.run_id
    best_accuracy = best_run["metrics.accuracy"]
    best_params = best_run["params.n_estimators"]             # Example of accessing params
    print(f"Best Run ID: {best_run_id}")
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Artifact URI: {best_run.artifact_uri}")
    
    # ---------------------------------------------------------
    # 5. Register the Model
    # ---------------------------------------------------------
    model_name = "Iris_Sweep_Champion"
    client = MlflowClient()
    
    best_model_uri = f"runs:/{best_run_id}/model"
    print(f"Registering model from: {best_model_uri}")
    
    registered_model = mlflow.register_model(model_uri = best_model_uri, name=model_name)
    
    # Assign the 'champion' alias
    client.set_registered_model_alias(
        name=model_name, 
        alias="champion", 
        version=registered_model.version
    )
    print(f"Registered '{model_name}' Version {registered_model.version} as @champion")

    # ---------------------------------------------------------
    # 6. HOW TO LOAD AND USE IT
    # ---------------------------------------------------------
    # Construct the model URI. 
    # Format: "runs:/<RUN_ID>/<ARTIFACT_PATH>"
    # 'model' is the name we used in log_model(clf, "model") previously
    # model_uri = f"runs:/{best_run_id}/model"
    # print(f"Loading model for Inference from: {model_uri}...")
    
    model_uri = f"models:/{model_name}@champion"
    print(f"\nLoading model for inference from: {model_uri}")

    # Load model as a generic PyFunc (works for sklearn, xgboost, etc.)
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Now we can use it just like a normal sklearn model
    # sample_data = ...
    prediction = loaded_model.predict(X_test)
    print(prediction)

else:
    print("No runs found.")