# End-to-End MLOps Template

A lightweight, modular, and reproducible template for Machine Learning Operations. This structure is designed to separate concerns—from data ingestion to model serving and monitoring—ensuring your machine learning lifecycle is scalable, easy to debug, and production-ready.

# Getting Started
1. Run `python scaffold.py` to generate the file structure.

2. Initialize your environment using Poetry (`poetry install`) or Conda (`conda env create -f environment.yml`).

3. Set your database credentials as environment variables (e.g., `DB_USER`, `DB_PASSWORD`).

4. Fill out `config/config.py` with your dataset parameters.

5. Fill in the implementation of the pipeline scripts sequentially in `src/pipelines/`.

# Project Architecture

```text
├── .github/
│   └── workflows/
│       └── main.yml           # Automated CI/CD pipeline (runs tests on push to main)
├── config/
│   └── config.py              # Centralized configuration (hyperparameters, paths, thresholds)
├── data/                      # Local data storage (ignored by git)
│   ├── 01-raw/                # Unprocessed data splits
│   ├── 02-preprocessed/       # Cleaned, imputed, and encoded data
│   └── 03-features/           # Engineered features ready for training
├── models/                    # Saved artifacts (joblib, model outputs, predictions)
├── mlruns/                    # MLflow tracking artifacts
├── model.db                   # Local MLflow SQLite database
├── pyproject.toml             # Modern strict dependency management (Poetry)
├── environment.yml            # Strict dependency management (Conda alternative)
├── src/                       
│   ├── pipelines/             # Core pipeline modules
│   │   ├── data_ingestion.py  # Loads data and creates train/test/holdout splits
│   │   ├── data_process.py    # Builds and applies preprocessing pipelines
│   │   ├── FE.py              # Applies specific feature engineering
│   │   ├── train_and_eval.py  # Tunes, trains, tracks to MLflow, and saves best model
│   │   ├── inference.py       # FastAPI application for serving the model
│   │   ├── data_monitoring.py # Checks for feature drift
│   │   ├── label_monitoring.py# Checks for target drift
│   │   ├── model_monitoring.py# Checks for performance degradation
│   │   ├── monitor.py         # Synthesizes monitoring reports
│   │   └── retrain_model.py   # Automated trigger for retraining
│   └── utils/
│       └── db_utils.py        # MySQL database connection and analytical data fetching
└── tests/                     # Pytest suite
```

## Mandatory (The Core Lifecycle)
These files are necessary to build, train, and track a reproducible model.

* pyproject.toml / environment.yml: Locks in your library versions (Scikit-Learn, MLflow, FastAPI) ensuring your environment is perfectly reproducible.

* config.py: Never hardcode paths or parameters. Keep them here.

* src/utils/db_utils.py: Replaces hardcoded CSV loading. Use this to securely pull data from your analytical database.

* data_ingestion.py: Essential for securing isolated data splits early.

* data_process.py: Essential for preventing data leakage during cleaning and scaling.

* train_and_eval.py: The heart of the project. Required for tuning, logging to MLflow, and saving the .joblib artifacts and batch prediction CSVs.

## Recommended (For Better Performance & Safety)
* FE.py: Separating feature engineering from preprocessing makes your code cleaner, but for simple datasets, it can technically be merged into data_process.py.

* .github/workflows/main.yml: Highly recommended for public repositories. Automatically runs your test suite to ensure code health before merging.

* tests/: Unit testing your ingestion, preprocessing, and database queries ensures garbage data doesn't break your training loop.

## Optional (For Production & Day-2 Operations)
If you are just doing R&D or a quick proof of concept, you can skip these until you need to deploy:

* inference.py: Only necessary if you need to serve the model via a live REST API. If you only need batch predictions, the CSV outputs from train_and_eval.py are sufficient.

* *_monitoring.py & monitor.py: Only needed if the model is in production and receiving continuous real-world data.

* retrain_model.py: Advanced MLOps. Skip this until you have an active monitoring pipeline that requires automated triggers.