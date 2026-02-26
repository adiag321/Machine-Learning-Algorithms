# MLflow Documentation

## 1. What is MLflow?

**MLflow** is an open-source platform designed to manage the end-to-end machine learning lifecycle. It provides tools to track experiments, package code into reproducible runs, share and deploy models, and maintain a central model registry.

### Core Components:

- **MLflow Tracking**: Records and queries experiments (parameters, metrics, artifacts)
- **MLflow Projects**: Packages code in a reusable, reproducible format
- **MLflow Models**: Manages and deploys models from various ML libraries
- **MLflow Registry**: Central repository for managing model lifecycle (staging, production, archiving)

---

## 2. When to Use MLflow & Advantages

### When to Use MLflow:

**Use MLflow when you need to:**
- Track multiple experiments with different hyperparameters
- Compare model performance across different runs
- Collaborate with team members on ML projects
- Version and manage models in production
- Reproduce past experiments
- Deploy models across different platforms
- Maintain audit trails for compliance

### Advantages of Using MLflow

#### **Without MLflow:**
```python
# Manual experiment tracking - error-prone and unscalable
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Manual logging - prone to errors and inconsistencies
results = {
    'timestamp': str(datetime.now()),
    'n_estimators': 100,
    'max_depth': 5,
    'accuracy': accuracy
}

# Save to file manually
with open(f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(results, f)

print(f"Accuracy: {accuracy}")
# Problems:
# - No centralized tracking
# - Hard to compare experiments
# - Can't visualize metrics easily
# - Model not saved properly
# - No versioning
# - No reproducibility guarantees
```

#### **With MLflow:**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Set experiment name
mlflow.set_experiment("iris_classification")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="random_forest_v1"):
    # Define parameters
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42
    }
    
    # Log parameters
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {accuracy}")

# Benefits:
# ✓ Centralized tracking in MLflow UI
# ✓ Easy comparison across runs
# ✓ Interactive visualization
# ✓ Model versioning and registry
# ✓ One-click model deployment
# ✓ Complete reproducibility
```

### Key Advantages Summary:

| Aspect | Without MLflow | With MLflow |
|--------|---------------|-------------|
| **Experiment Tracking** | Manual, error-prone | Automatic, centralized |
| **Model Versioning** | File-based, messy | Systematic, queryable |
| **Collaboration** | Difficult to share | Easy sharing via UI |
| **Reproducibility** | Hard to reproduce | Fully reproducible |
| **Comparison** | Manual analysis | Visual comparison in UI |
| **Deployment** | Custom scripts needed | Built-in deployment tools |

---

## 3. Reproducible Code Snippets

### 3.1 Basic Experiment Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set experiment
mlflow.set_experiment("breast_cancer_classification")

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run experiment
with mlflow.start_run(run_name="logistic_regression_baseline"):
    # Parameters
    params = {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs'
    }
    
    # Log parameters
    mlflow.log_params(params)
    
    # Train
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    
    print(f"Metrics: {metrics}")
```

### 3.2 Hyperparameter Tuning with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Set experiment
mlflow.set_experiment("wine_classification_tuning")

# Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = [
    {'n_estimators': 50, 'max_depth': 3},
    {'n_estimators': 100, 'max_depth': 5},
    {'n_estimators': 200, 'max_depth': 7},
    {'n_estimators': 100, 'max_depth': 10}
]

# Run experiments for each parameter combination
for idx, params in enumerate(param_grid):
    with mlflow.start_run(run_name=f"rf_exp_{idx+1}"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Run {idx+1}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
```

### 3.3 Logging Artifacts (Plots, Files)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set experiment
mlflow.set_experiment("digits_classification")

# Load data
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="gradient_boosting"):
    # Parameters
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    }
    
    mlflow.log_params(params)
    
    # Train
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Log plot as artifact
    mlflow.log_artifact('confusion_matrix.png')
    
    # Create classification report
    report = classification_report(y_test, y_pred)
    
    # Save report to file and log as artifact
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')
    
    # Log model
    mlflow.sklearn.log_model(model, "gradient_boosting_model")
    
    print(f"Accuracy: {accuracy:.4f}")
```

### 3.4 Nested Runs (Parent-Child Experiments)

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set experiment
mlflow.set_experiment("model_comparison")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parent run
with mlflow.start_run(run_name="ensemble_comparison") as parent_run:
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    best_score = 0
    best_model_name = None
    
    for model_name, model in models.items():
        # Child run for each model
        with mlflow.start_run(run_name=model_name, nested=True):
            # Train
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mean_cv_score = cv_scores.mean()
            
            # Test score
            test_score = model.score(X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("cv_score_mean", mean_cv_score)
            mlflow.log_metric("cv_score_std", cv_scores.std())
            mlflow.log_metric("test_score", test_score)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            print(f"{model_name}: CV={mean_cv_score:.4f}, Test={test_score:.4f}")
            
            # Track best model
            if test_score > best_score:
                best_score = test_score
                best_model_name = model_name
    
    # Log best model info in parent run
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_test_score", best_score)
    
    print(f"\nBest Model: {best_model_name} with score {best_score:.4f}")
```

### 3.5 Loading and Using Logged Models

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris

# Load data for prediction
X, y = load_iris(return_X_y=True)

# Method 1: Load model by run_id
run_id = "your_run_id_here"  # Replace with actual run_id from MLflow UI
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Make predictions
predictions = loaded_model.predict(X[:5])
print(f"Predictions: {predictions}")

# Method 2: Load model from local path (after downloading from UI)
# loaded_model = mlflow.sklearn.load_model("path/to/model")

# Method 3: Load latest model from a specific experiment
client = mlflow.tracking.MlflowClient()
experiment_name = "iris_classification"

# Get experiment by name
experiment = client.get_experiment_by_name(experiment_name)

if experiment:
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        best_model_uri = f"runs:/{best_run.info.run_id}/model"
        best_model = mlflow.sklearn.load_model(best_model_uri)
        
        print(f"Best run ID: {best_run.info.run_id}")
        print(f"Best accuracy: {best_run.data.metrics['accuracy']}")
```

### 3.6 Autologging (Automatic Logging)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Set experiment
mlflow.set_experiment("wine_autolog")

# Enable autologging - automatically logs params, metrics, and model
mlflow.sklearn.autolog()

# Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start run
with mlflow.start_run(run_name="autolog_demo"):
    # Train model - MLflow will automatically log everything!
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # MLflow automatically logs:
    # - All model parameters
    # - Training metrics
    # - Model artifacts
    # - Feature importance plots
    
    score = model.score(X_test, y_test)
    print(f"Test Score: {score:.4f}")

# Note: You can disable autolog anytime
# mlflow.sklearn.autolog(disable=True)
```

### 3.7 Setting Tags and Notes

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_experiment("tagged_experiments")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="production_candidate"):
    # Set tags for organization
    mlflow.set_tag("model_type", "ensemble")
    mlflow.set_tag("dataset", "iris")
    mlflow.set_tag("environment", "development")
    mlflow.set_tag("team", "data-science")
    mlflow.set_tag("deployment_ready", "yes")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")
    
    # Add notes to the run
    mlflow.set_tag("mlflow.note.content", 
                   "This is a baseline model for iris classification. "
                   "Achieved good accuracy with default parameters. "
                   "Ready for production testing.")
```

---

## How to Start MLflow UI

To view your experiments in the MLflow UI, run:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your web browser.

To specify a different port:

```bash
mlflow ui --port 5001
```

---

## Best Practices

1. **Always Set Experiment Names**: Use `mlflow.set_experiment()` to organize related runs
2. **Use Descriptive Run Names**: Makes it easier to identify runs in the UI
3. **Log Everything**: Parameters, metrics, artifacts, tags - comprehensive logging helps reproducibility
4. **Use Nested Runs**: For comparing multiple models or configurations
5. **Enable Autologging**: For supported frameworks, it saves time and ensures consistency
6. **Tag Runs Appropriately**: Use tags for filtering and organization
7. **Version Your Models**: Use MLflow Model Registry for production models

---

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
