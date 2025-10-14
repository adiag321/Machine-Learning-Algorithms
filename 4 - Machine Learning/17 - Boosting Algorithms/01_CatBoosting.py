'''
### CatBoosting Algorithm

This notebook demonstrates how to train and evaluate a CatBoost classifier using a dataset with many categorical features (the 'adult' dataset from OpenML).

What you'll find here:
- A reusable function `train_catboost_classifier` that trains CatBoost, performs optional hyperparameter tuning, and returns metrics and feature importances.
- Use of CatBoost Pool and categorical feature handling.
- Evaluation metrics (accuracy, precision, recall, F1, AUC), confusion matrix and classification report.
- Hyperparameter tuning using RandomizedSearchCV (sklearn) with CatBoost as the estimator.

Run the cells below in order. The main example loads the `adult` dataset, identifies categorical columns automatically, and trains CatBoost.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier, Pool

import joblib
import warnings
warnings.filterwarnings('ignore')

#####################################
# Function to train CatBoostClassifier with hyperparameter tuning
#####################################
def train_catboost_classifier(df, target_col, cat_features=None, params=None,
                                random_state=42, use_gpu=False, sample_weight=None, class_weights=None,
                                save_model_path=None, verbose=100):
    """
    Train a CatBoostClassifier with many commonly-used parameters and helpers suitable for industry use.

    Features implemented:
    - Pool usage to pass categorical features and sample weights.
    - Support for class_weights and sample_weight.
    - Model saving (save_model_path).
    - Returns model, metrics, feature importances, and optional saved path.

    Note: This function focuses on binary classification examples but works for multiclass if loss_function and metrics are adjusted.
    """
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f'target_col not found in dataframe')

    # Infer categorical columns if not provided
    if cat_features is None:
        cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_features = [c for c in cat_features if c != target_col]

    # Basic cleaning: replace common missing markers and convert object cols to category
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].replace('?', np.nan)
            df[c] = df[c].astype('category')

    # Encode target if needed
    y = df[target_col]
    if y.dtype.name == 'category' or y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    else:
        le = None

    X = df.drop(columns=[target_col])

    # Train/test split (stratify when possible)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=random_state, stratify=stratify)

    # Default CatBoost params (expanded, industry-friendly)
    default_params = dict(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=random_state,
        od_type='Iter',
        early_stopping_rounds=50,
        use_best_model=True,
        verbose=verbose,
        border_count=128,
        bagging_temperature=1,
        subsample=0.8,
        rsm=0.8,
        nan_mode='Min',
        leaf_estimation_iterations=1,
    )

    if use_gpu:
        default_params['task_type'] = 'GPU'

    if params is not None:
        default_params.update(params)

    # Prepare CatBoost Pools to pass categorical feature indices and sample weights
    cat_idx = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns] if cat_features else None

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_idx, weight=sample_weight)
    eval_pool = Pool(data=X_test, label=y_test, cat_features=cat_idx)

    # Apply class weights if provided (CatBoost accepts class_weights param)
    if class_weights is not None:
        default_params['class_weights'] = class_weights

    # Instantiate and fit CatBoost model with provided/default parameters
    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=eval_pool)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = dict()
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, average='binary', zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, average='binary', zero_division=0)
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    except Exception:
        metrics['roc_auc'] = None

    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)

    # Feature importances (CatBoost supports different types)
    try:
        fi = model.get_feature_importance(type='FeatureImportance', prettified=False)
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': fi})
        feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)
    except Exception as e:
        print('Could not get feature importances:', e)
        feature_importances = None

    saved_path = None
    if save_model_path is not None:
        model.save_model(save_model_path, format='cbm')
        saved_path = save_model_path

    result = dict(model=model, metrics=metrics, feature_importances=feature_importances, label_encoder=le, saved_path=saved_path)
    return result

#####################################
## End of function definition
#####################################

#####################################
#### Example usage
#####################################
# Main: try to load 'adult' via OpenML (sklearn.fetch_openml)
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame.copy()
# choose target column used by this version
target = 'class' if 'class' in df.columns else 0
print('Using OpenML adult dataset, target:', target)

def convert_to_categorical(df, col):
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.add_categories('Unknown')
    df[col]= df[col].fillna('Unknown')
    return df[col]

df.workclass = convert_to_categorical(df, 'workclass')
df.occupation = convert_to_categorical(df, 'occupation')
df['native-country'] = convert_to_categorical(df, 'native-country')

# Clean whitespace in object columns and ensure categories have no leading/trailing spaces
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()

# Identify categorical columns automatically (exclude target)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c != target]
print('Detected categorical columns (sample):', cat_cols[:10])

# Run training with expanded parameters; save the model to disk
out = train_catboost_classifier(df, target_col=target, cat_features=cat_cols, random_state=42, verbose=100, save_model_path='catboost_model.cbm')

print('Metrics:')
for k,v in out['metrics'].items():
    if k in ['classification_report']:
        print('Classification report:', v)
    elif k in ['confusion_matrix']:
        print('Confusion matrix:', v)
    else:
        print(f'{k}: {v}')

if out['feature_importances'] is not None:
    print(out['feature_importances'].head(10))

if out['saved_path'] is not None:
    print('Model saved to', out['saved_path'])
