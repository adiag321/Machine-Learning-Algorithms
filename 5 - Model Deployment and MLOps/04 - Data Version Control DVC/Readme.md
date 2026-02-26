# Data Version Control (DVC)

## Table of Contents
- [What is DVC?](#what-is-dvc)
- [Why Use DVC?](#why-use-dvc)
- [Quick Start Guide](#quick-start-guide)
- [Core Concepts](#core-concepts)
- [Common DVC Workflows](#common-dvc-workflows)
- [Integration with Git](#integration-with-git)
- [Remote Storage Setup](#remote-storage-setup)
- [Best Practices](#best-practices)

## What is DVC?

**DVC (Data Version Control)** is an open-source version control system specifically designed for machine learning projects. It extends Git's capabilities to handle large files, datasets, models, and ML pipelines while maintaining reproducibility and collaboration.

### Key Features:
- **Data Versioning**: Track large datasets and model files without bloating Git repositories
- **Pipeline Management**: Define and version ML workflows with dependencies
- **Experiment Tracking**: Compare model metrics across experiments
- **Remote Storage**: Store data in S3, GCS, Azure Blob, or local storage
- **Reproducibility**: Recreate any experiment or model version



## Why Use DVC?

### Problems DVC Solves:

1. **Git Can't Handle Large Files**
   - Git becomes slow with large datasets
   - Repository size explodes with binary files
   - DVC stores file metadata in Git, actual data elsewhere

2. **Data Pipeline Complexity**
   - Hard to track data transformations
   - Difficult to reproduce results
   - DVC creates reproducible pipelines

3. **Collaboration Issues**
   - Team members can't easily share large datasets
   - Model versioning is manual and error-prone
   - DVC provides centralized data storage

4. **Experiment Management**
   - Hard to track which data/code produced which results
   - DVC links data versions with code versions

## Quick Start Guide

### 1. Initialize DVC in Your Project

```bash
# Initialize Git (if not already)
git init

# Initialize DVC
dvc init

# Commit DVC configuration
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

**What Happens:**
- Creates `.dvc/` directory with configuration
- Creates `.dvcignore` file (like `.gitignore` for DVC)
- Configures Git to ignore DVC cache

### 2. Track Your First Data File

```bash
# Add a large dataset to DVC
dvc add data/train.csv

# DVC creates train.csv.dvc (metadata file)
# Original file is moved to .dvc/cache/
git add data/train.csv.dvc data/.gitignore
git commit -m "Track training data with DVC"
```

**File Structure:**
```
project/
├── data/
│   ├── train.csv          # Large file (not in Git)
│   ├── train.csv.dvc      # Metadata file (in Git)
│   └── .gitignore         # Ignores train.csv
├── .dvc/
│   ├── cache/             # Cached data files
│   └── config             # DVC configuration
```

### 3. Share Data with Remote Storage

```bash
# Add remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Push data to remote
dvc push

# Commit remote configuration
git add .dvc/config
git commit -m "Configure remote storage"
git push
```

### 4. Retrieve Data (New Team Member)

```bash
# Clone repository
git clone <repo-url>
cd <repo>

# Pull data from remote
dvc pull
```



## Core Concepts

### 1. DVC Files (`.dvc`)

Metadata files that track large files:

```yaml
# data/train.csv.dvc
outs:
- md5: a304afb96060aad90176268345e10355
  size: 50000000
  path: train.csv
```

### 2. DVC Cache (`.dvc/cache/`)

Local storage for tracked files:
- Content-addressable storage (uses MD5 hashes)
- Shared across all branches
- Can be safely deleted and restored with `dvc pull`

### 3. DVC Remote Storage

External storage for collaboration:
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- SSH/SFTP servers
- Local network drives

### 4. DVC Pipelines (`dvc.yaml`)

Define reproducible ML workflows:

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess.py
    deps:
      - data/raw/train.csv
      - preprocess.py
    outs:
      - data/processed/train_processed.csv
  
  train:
    cmd: python train.py
    deps:
      - data/processed/train_processed.csv
      - train.py
    params:
      - model.learning_rate
      - model.n_estimators
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false
```



## Common DVC Workflows

### Workflow 1: Version a Dataset

```bash
# Add new dataset version
dvc add data/dataset_v2.csv
git add data/dataset_v2.csv.dvc
git commit -m "Update dataset to v2"
dvc push

# Switch to previous version
git checkout <commit-hash> data/dataset_v1.csv.dvc
dvc checkout data/dataset_v1.csv.dvc
```

### Workflow 2: Create an ML Pipeline

**Step 1: Create DVC Pipeline**

```bash
# Add preprocessing stage
dvc stage add -n preprocess \
  -d data/raw/data.csv \
  -d src/preprocess.py \
  -o data/processed/data.csv \
  python src/preprocess.py

# Add training stage
dvc stage add -n train \
  -d data/processed/data.csv \
  -d src/train.py \
  -o models/model.pkl \
  -M metrics/scores.json \
  python src/train.py
```

**Step 2: Run Pipeline**

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train
```

**Step 3: Track Changes**

```bash
git add dvc.yaml dvc.lock
git commit -m "Add ML pipeline"
dvc push
```

### Workflow 3: Experiment Tracking

```bash
# Run experiment with different parameters
dvc exp run --set-param model.lr=0.001

# List all experiments
dvc exp show

# Compare experiments
dvc exp diff <experiment-name>

# Apply best experiment
dvc exp apply <experiment-name>
```

### Workflow 4: Update Data & Retrain

```bash
# Update dataset
cp new_data.csv data/train.csv
dvc add data/train.csv

# Reproduce pipeline (automatically detects changes)
dvc repro

# Commit changes
git add data/train.csv.dvc dvc.lock
git commit -m "Update training data and retrain model"
dvc push
```



## Integration with Git

### Typical Git + DVC Workflow

```bash
# 1. Create feature branch
git checkout -b experiment/new-model

# 2. Modify code and data
vim train.py
dvc add data/new_features.csv

# 3. Run pipeline
dvc repro

# 4. Commit everything
git add train.py data/new_features.csv.dvc dvc.lock
git commit -m "Implement new model with additional features"

# 5. Push data and code
dvc push
git push origin experiment/new-model

# 6. Merge after review
git checkout main
git merge experiment/new-model
```

### Branch Switching with DVC

```bash
# Switch branch
git checkout feature-branch

# Sync data files to match branch
dvc checkout
```

## Remote Storage Setup

### AWS S3

```bash
# Configure S3 remote
dvc remote add -d s3remote s3://my-bucket/path

# Set credentials (optional, uses AWS CLI config by default)
dvc remote modify s3remote access_key_id <key>
dvc remote modify s3remote secret_access_key <secret>

# Push data
dvc push -r s3remote
```

### Google Cloud Storage

```bash
# Configure GCS remote
dvc remote add -d gcs gs://my-bucket/path

# Authenticate (use gcloud CLI)
gcloud auth application-default login

# Push data
dvc push
```

### Local Remote (For Testing)

```bash
# Create local remote
dvc remote add -d local_remote /tmp/dvc-storage

# Push data
dvc push
```

## Best Practices

### 1. **Directory Organization**

```
project/
├── data/
│   ├── raw/              # Original data (track with DVC)
│   ├── processed/        # Processed data (pipeline output)
│   └── external/         # External datasets
├── models/               # Trained models (track with DVC)
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
├── metrics/              # Metrics and plots
├── dvc.yaml             # Pipeline definition
├── dvc.lock             # Pipeline lock file
└── params.yaml          # Hyperparameters
```

### 2. **What to Track with DVC**

**Track:**
- Raw datasets (> 50 MB)
- Processed/transformed data
- Trained models
- Large feature files
- Pre-trained embeddings

**Don't Track:**
- Source code (use Git)
- Small config files (< 1 MB)
- Temporary files
- Log files

### 3. **Use Parameters File**

```yaml
# params.yaml
model:
  learning_rate: 0.001
  n_estimators: 100
  max_depth: 5

preprocessing:
  test_size: 0.2
  random_state: 42
```

**Reference in code:**

```python
import yaml

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

lr = params['model']['learning_rate']
```

### 4. **Commit DVC Files with Git**

Always commit `.dvc` files:

```bash
# Good
git add data/train.csv.dvc dvc.yaml dvc.lock
git commit -m "Update dataset and pipeline"

# Bad - missing DVC files
git add src/train.py
git commit -m "Update training script"  # Lost data tracking!
```

### 5. **Use `.dvcignore`**

```
# .dvcignore
*.tmp
*.log
__pycache__/
.DS_Store
```

### When to Use Each:

**Use DVC:**
- Need reproducible ML pipelines
- Version datasets and code together
- Want lightweight experiment tracking
- Focus on data lineage

**Use Git LFS:**
- Only need file versioning (no ML features)
- Simple projects with large binary files
- Already using Git workflows

**Use MLflow:**
- Primary focus is experiment tracking
- Need model registry and serving
- Want rich UI for comparing runs
- Can use together with DVC!

### Combined Approach:

```bash
# Use DVC for data versioning and pipelines
dvc add data/train.csv
dvc repro

# Use MLflow for experiment tracking
mlflow run . -P learning_rate=0.01
```

## Example: Complete ML Project with DVC

### Project Structure

```
ml-project/
├── data/
│   ├── raw/
│   │   └── dataset.csv.dvc
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/
├── metrics/
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

### 1. `params.yaml`

```yaml
preprocessing:
  test_size: 0.2
  random_state: 42

model:
  type: RandomForest
  n_estimators: 100
  max_depth: 10
  learning_rate: 0.01
```

### 2. `dvc.yaml`

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/dataset.csv
      - src/preprocess.py
    params:
      - preprocessing
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed/train.csv
      - src/train.py
    params:
      - model
    outs:
      - models/model.pkl

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/test.csv
      - src/evaluate.py
    metrics:
      - metrics/scores.json:
          cache: false
    plots:
      - metrics/confusion_matrix.json:
          cache: false
```

### 3. `src/preprocess.py`

```python
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load data
df = pd.read_csv('data/raw/dataset.csv')

# Split data
train, test = train_test_split(
    df,
    test_size=params['preprocessing']['test_size'],
    random_state=params['preprocessing']['random_state']
)

# Save processed data
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)

print(f"Train size: {len(train)}, Test size: {len(test)}")
```

### 4. `src/train.py`

```python
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load data
train = pd.read_csv('data/processed/train.csv')
X_train = train.drop('target', axis=1)
y_train = train['target']

# Train model
model = RandomForestClassifier(
    n_estimators=params['model']['n_estimators'],
    max_depth=params['model']['max_depth'],
    random_state=42
)
model.fit(X_train, y_train)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained successfully!")
```

### 5. `src/evaluate.py`

```python
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
test = pd.read_csv('data/processed/test.csv')
X_test = test.drop('target', axis=1)
y_test = test['target']

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1': f1_score(y_test, y_pred, average='weighted')
}

# Save metrics
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### 6. Running the Project

```bash
# Initialize DVC
dvc init

# Track raw data
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc data/raw/.gitignore

# Run entire pipeline
dvc repro

# View metrics
dvc metrics show

# Compare experiments
dvc exp run --set-param model.n_estimators=200
dvc exp show

# Push everything
git add .
git commit -m "Complete ML pipeline with DVC"
dvc push
git push
```

## Useful Commands Reference

```bash
# Initialize
dvc init                          # Initialize DVC in project

# Tracking
dvc add <file>                    # Track large file
dvc checkout                      # Update workspace to match .dvc files
dvc status                        # Show pipeline status

# Remote Storage
dvc remote add -d <name> <url>    # Add remote storage
dvc push                          # Push data to remote
dvc pull                          # Pull data from remote
dvc fetch                         # Download data to cache (don't update workspace)

# Pipelines
dvc stage add                     # Add pipeline stage
dvc repro                         # Reproduce pipeline
dvc dag                           # Show pipeline DAG

# Experiments
dvc exp run                       # Run experiment
dvc exp show                      # Show all experiments
dvc exp diff                      # Compare experiments
dvc exp apply                     # Apply experiment

# Metrics
dvc metrics show                  # Show metrics
dvc metrics diff                  # Compare metrics
dvc plots show                    # Show plots

# Misc
dvc cache dir                     # Show cache directory
dvc gc                            # Garbage collect cache
dvc diff                          # Show changes
```


## Additional Resources

- [Official DVC Documentation](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DVC with MLflow](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial)
- [DVC vs Git LFS](https://dvc.org/doc/user-guide/related-technologies)


## Summary

DVC is essential for:
* Versioning large datasets and models  
* Creating reproducible ML pipelines  
* Collaborating on ML projects  
* Tracking experiments systematically  

**Key Takeaway:** Think of DVC as "Git for data" - it extends Git's capabilities to handle the unique challenges of machine learning projects while maintaining the familiar Git workflow.
