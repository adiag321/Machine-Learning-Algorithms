import os
from pathlib import Path

def create_project_structure(base_dir="."):
    """Scaffolds the enhanced MLOps project directory and file structure."""
    base_path = Path(base_dir)

    # Define the directory tree
    directories = [
        ".github/workflows",
        "config",
        "data/01-raw",
        "data/02-preprocessed",
        "data/03-features",
        "models",
        "mlruns",
        "src/pipelines",
        "src/utils",
        "tests",
    ]

    # Define files that just need to be created (empty)
    empty_files = [
        "config/config.py",
        "src/pipelines/__init__.py",
        "src/pipelines/data_ingestion.py",
        "src/pipelines/data_process.py",
        "src/pipelines/FE.py",
        "src/pipelines/train_and_eval.py",
        "src/pipelines/inference.py",
        "src/pipelines/data_monitoring.py",
        "src/pipelines/label_monitoring.py",
        "src/pipelines/model_monitoring.py",
        "src/pipelines/monitor.py",
        "src/pipelines/retrain_model.py",
        "tests/__init__.py",
        "tests/test_ingestion.py",
        "tests/test_preprocessing.py",
        "tests/test_training.py",
        "tests/test_inference.py",
        "requirements.txt",
        "Makefile"
    ]

# Define files with injected boilerplate code
file_contents = {
    ".gitignore": """
# Environments
.env
.venv
env/
venv/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class

# Data & Models
data/*/*.csv
models/*.joblib
models/*.json
model.db
mlruns/

# IDEs
.vscode/
.idea/
""",
        ".github/workflows/main.yml": """
name: MLOps Pipeline CI

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest scikit-learn mlflow fastapi httpx pandas numpy pymysql
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        
    - name: Run pytest
      run: |
        pytest tests/
""",
        "pyproject.toml": """
[tool.poetry]
name = "mlops-template"
version = "0.1.0"
description = "End-to-End MLOps Pipeline Template"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
scikit-learn = "^1.3.0"
mlflow = "^2.6.0"
fastapi = "^0.103.0"
uvicorn = "^0.23.2"
pandas = "^2.0.0"
numpy = "^1.24.0"
pymysql = "^1.1.0"
SQLAlchemy = "^2.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
httpx = "^0.24.1"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
""",
        "environment.yml": """
name: mlops-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - scikit-learn=1.3.0
  - mlflow=2.6.0
  - fastapi=0.103.0
  - uvicorn=0.23.2
  - pandas=2.0.0
  - numpy=1.24.0
  - pymysql=1.1.0
  - sqlalchemy=2.0.0
  - pytest=7.4.0
  - httpx=0.24.1
""",

"src/utils/db_utils.py": """
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

def get_mysql_engine():
    \"\"\"
    Creates a SQLAlchemy engine for a MySQL database.
    Expects environment variables for secure connection.
    \"\"\"
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    database = os.getenv("DB_NAME", "analytics_db")
    
    connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(connection_string)

def fetch_analytical_data(query: str) -> pd.DataFrame:
    \"\"\"
    Fetches raw data from the MySQL database for downstream cleaning and target encoding.
    \"\"\"
    engine = get_mysql_engine()
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
            return df
    except SQLAlchemyError as e:
        print(f"Database connection failed: {e}")
        raise
"""
    }

    print(f"Scaffolding enhanced MLOps project in: {base_path.absolute()}\n")

    # Create directories
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / ".gitkeep").touch()
        print(f"📁 Created directory: {directory}")

    # Create empty files
    for file in empty_files:
        file_path = base_path / file
        file_path.touch(exist_ok=True)
        print(f"📄 Created file: {file}")

    # Create files with injected content
    for file_name, content in file_contents.items():
        file_path = base_path / file_name
        with open(file_path, "w") as f:
            f.write(content.strip())
        print(f"Wrote boilerplate to: {file_name}")

    print("\nProject scaffolding complete!")

if __name__ == "__main__":
    create_project_structure()