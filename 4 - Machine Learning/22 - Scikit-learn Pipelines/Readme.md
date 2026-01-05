# Scikit-learn Pipelines

Pipelines in scikit-learn streamline machine learning workflows by chaining multiple preprocessing and modeling steps into a single object. This ensures consistent data transformations and prevents common mistakes like data leakage.

## When to Use Pipelines

Use pipelines when you need to:
- **Combine multiple preprocessing steps** (scaling, encoding, imputation) with a final estimator
- **Ensure consistency** between training and test data transformations
- **Prevent data leakage** by fitting transformers only on training data
- **Simplify model deployment** by saving a single pipeline object
- **Reduce code complexity** when working with multiple transformers
- **Automate parameter tuning** across the entire pipeline using GridSearchCV

## Key Things to Remember

### Data Requirements
- **Input data format**: Pipelines work with NumPy arrays or pandas DataFrames
- **Categorical vs. numerical**: Separate columns by type before passing to `ColumnTransformer`
- **Feature order**: Order matters! Pass categorical and numerical columns in consistent order

### Important Parameters & Concepts
- **remainder='passthrough'**: Keeps columns not specified in transformers (don't forget this in `ColumnTransformer`)
- **Steps are sequential**: Each step's output becomes the next step's input
- **Only final estimator can have .predict()**: Earlier steps must be transformers with `.fit()` and `.transform()`
- **Fit only on training data**: `pipeline.fit(X_train, y_train)` - never fit on test data
- **Use on both train and test**: `pipeline.transform(X_test)` applies the same transformations learned from training data

## Basic Syntax

### Simple Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create a pipeline with 2 steps
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Pipeline with Preprocessing for Mixed Data Types
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Define transformers for different column types
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

# Combine using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_columns),
    ('cat', categorical_transformer, categorical_columns)
], remainder='passthrough')

# Create final pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

### Full Example with Feature Selection
```python
from sklearn.feature_selection import SelectKBest, chi2

# Create preprocessing pipeline
preproc_pipe = ColumnTransformer([
    ('num', numerical_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='passthrough')

# Create training pipeline with feature selection
train_pipe = Pipeline([
    ('kbest', SelectKBest(chi2, k='all')),
    ('model', RandomForestClassifier(n_estimators=100))
])

# Combine both pipelines
complete_pipe = Pipeline([
    ('preprocessor', preproc_pipe),
    ('training', train_pipe)
])

# Fit the complete pipeline
complete_pipe.fit(X_train, y_train)
score = complete_pipe.score(X_test, y_test)
```

## Saving and Loading Pipelines

```python
import joblib

# Save
joblib.dump(pipeline, 'my_pipeline.joblib')

# Load
loaded_pipeline = joblib.load('my_pipeline.joblib')

# Use for predictions on new data
predictions = loaded_pipeline.predict(new_data)
```

## Key Benefits

* **Reproducibility**: Same transformations applied consistently  
* **Prevents leakage**: Transformers fit only on training data  
* **Clean code**: Single object instead of multiple steps  
* **Easy deployment**: Save and load the entire workflow  
* **Works with GridSearchCV**: Tune all parameters at once  

## Common Mistakes to Avoid

* Fitting transformers on entire dataset (use training data only)  
* Forgetting `remainder='passthrough'` in `ColumnTransformer`  
* Mixing data types without using `ColumnTransformer`  
* Applying different transformations to test data than training data  
* Using transformers that don't have `.fit()` and `.transform()` methods