"""
Cross-validation technique utilities

This module provides reusable functions for many cross-validation strategies commonly used in real-world data science workflows.
Each CV technique is implemented as a separate function that accepts an estimator, feature matrix X
and target y (and other technique-specific arguments). The functions focus on the cross-validation wiring (splitting, scoring, params)
rather than modeling.

Usage pattern (example):

from sklearn.linear_model import LogisticRegression
from cross_validation_utils import stratified_kfold_cv, get_classification_data

X, y = get_classification_data()
clf = LogisticRegression(max_iter=1000)
scores = stratified_kfold_cv(clf, X, y, n_splits=5, scoring='accuracy')

All functions return dicts with arrays of scores so you can integrate them into pipelines, model selection, or reporting.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    LeaveOneOut,
    LeavePOut,
    LeaveOneGroupOut,
    RepeatedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#################################################################################
## Data helpers
#################################################################################
def get_classification_data():
    """Load sample classification dataset (breast cancer)."""
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=[data.target.name])
    y = data.frame[data.target.name]
    return X, y


def get_regression_data():
    """Load sample regression dataset (diabetes)."""
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]
    return X, y


#################################################################################
## Cross-validation functions
#################################################################################
def _wrap_scores(scores):
    """Helper to return consistent dict format."""
    return {"test_score": np.array(scores)}

# Used when you have a classification target
def kfold_cv(estimator, X, y, n_splits=5, shuffle=False, random_state=None,
             scoring=None, n_jobs=1, verbose=0):
    """K-Fold cross-validation."""
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring,
                             n_jobs=n_jobs, verbose=verbose)
    return _wrap_scores(scores)

# Used when you have a regression target
def regression_kfold_cv(estimator, X, y, n_splits=5, shuffle=True, random_state=None,
                        scoring="r2", n_jobs=1):
    """K-Fold CV wrapper for regression tasks (defaults to r2)."""
    return kfold_cv(estimator, X, y, n_splits=n_splits,
                    shuffle=shuffle, random_state=random_state,
                    scoring=scoring, n_jobs=n_jobs)

# Used when you have class imbalance
def stratified_kfold_cv(estimator, X, y, n_splits=5, shuffle=False,
                        random_state=None, scoring=None, n_jobs=1, verbose=0):
    """Stratified K-Fold: preserves class balance in each fold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring,
                             n_jobs=n_jobs, verbose=verbose)
    return _wrap_scores(scores)

# Used when you have groups
def group_kfold_cv(estimator, X, y, groups, n_splits=5, scoring=None, n_jobs=1):
    """GroupKFold: keep samples from the same group together in a fold."""
    if groups is None or len(groups) != len(y):
        raise ValueError("`groups` must be provided and same length as y.")
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(estimator, X, y, cv=cv.split(X, y, groups=groups),
                             scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# Used when you have groups and want to leave one group out each fold
def leave_one_group_out_cv(estimator, X, y, groups, scoring=None, n_jobs=1):
    """Leave-One-Group-Out CV."""
    if groups is None or len(groups) != len(y):
        raise ValueError("`groups` must be provided and same length as y.")
    cv = LeaveOneGroupOut()
    scores = cross_val_score(estimator, X, y, cv=cv.split(X, y, groups=groups),
                             scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# Used for time series
def time_series_cv(estimator, X, y, n_splits=5, max_train_size=None, gap=0,
                   scoring=None, n_jobs=1):
    """TimeSeriesSplit: forward-chaining split for time-ordered data."""
    cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, gap=gap)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# Used for leave-one-out cross-validation
def leave_one_out_cv(estimator, X, y, scoring=None):
    """Leave-One-Out cross-validation (LOO)."""
    cv = LeaveOneOut()
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return _wrap_scores(scores)

# Used for leave-p-out
def leave_p_out_cv(estimator, X, y, p=2, scoring=None):
    """Leave-P-Out cross-validation."""
    cv = LeavePOut(p=p)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return _wrap_scores(scores)

# Used for more stable estimates by repeating K-Fold multiple times
def repeated_kfold_cv(estimator, X, y, n_splits=5, n_repeats=10, random_state=None,
                      scoring=None, n_jobs=1):
    """Repeated K-Fold: repeats K-Fold with different random seeds for stability."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# Used for random train/test splits
def shuffle_split_cv(estimator, X, y, n_splits=10, train_size=None, test_size=None,
                     random_state=None, scoring=None, n_jobs=1):
    """ShuffleSplit: random train/test splits."""
    cv = ShuffleSplit(n_splits=n_splits, train_size=train_size,
                      test_size=test_size, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# Used for classification tasks to maintain class balance in random splits
def stratified_shuffle_split_cv(estimator, X, y, n_splits=10, train_size=None, test_size=None,
                                random_state=None, scoring=None, n_jobs=1):
    """StratifiedShuffleSplit: stratified version of ShuffleSplit for classification."""
    cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size,
                                test_size=test_size, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# Used when you have groups and want random splits that respect group boundaries
def group_shuffle_split_cv(estimator, X, y, groups, n_splits=10, train_size=None, test_size=None,
                           random_state=None, scoring=None, n_jobs=1):
    """GroupShuffleSplit: random splits that respect group boundaries."""
    if groups is None or len(groups) != len(y):
        raise ValueError("`groups` must be provided and same length as y.")
    cv = GroupShuffleSplit(n_splits=n_splits, train_size=train_size,
                           test_size=test_size, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv.split(X, y, groups=groups),
                             scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# Used for evaluating multiple metrics at once
def cross_validate_with_scoring(estimator, X, y, cv, scoring=None,
                                return_train_score=True, n_jobs=1):
    """Wrapper around sklearn.model_selection.cross_validate supporting multiple metrics."""
    results = cross_validate(estimator, X, y, cv=cv, scoring=scoring,
                             return_train_score=return_train_score, n_jobs=n_jobs)
    return results

# Used for hyperparameter tuning with grid search
def nested_cv_grid_search(estimator, param_grid, X, y, inner_cv, outer_cv,
                          scoring=None, n_jobs=1, verbose=0):
    """Nested cross-validation using GridSearchCV as the inner loop."""
    grid = GridSearchCV(estimator, param_grid, cv=inner_cv,
                        scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    scores = cross_val_score(grid, X, y, cv=outer_cv, scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)

# used for hyperparameter tuning with random search
def nested_cv_random_search(estimator, param_distributions, X, y, inner_cv, outer_cv,
                            n_iter=20, scoring=None, n_jobs=1, verbose=0, random_state=None):
    """Nested cross-validation using RandomizedSearchCV as the inner loop."""
    random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter,
                                       cv=inner_cv, scoring=scoring,
                                       n_jobs=n_jobs, verbose=verbose,
                                       random_state=random_state)
    scores = cross_val_score(random_search, X, y, cv=outer_cv,
                             scoring=scoring, n_jobs=n_jobs)
    return _wrap_scores(scores)


#################################################################################
## End of Cross-validation functions
#################################################################################

if __name__ == '__main__':
    # Classification example
    Xc, yc = get_classification_data()
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # 1) Standard K-Fold
	kf_scores = kfold_cv(clf, Xc, yc, n_splits=5, shuffle=True, random_state=42, scoring='accuracy')
	print('K-Fold (accuracy) scores:', kf_scores)
	print('K-Fold mean accuracy:', np.mean(kf_scores["test_score"]))
    
    # 2) Stratified K-Fold (preserves class balance)
	skf_scores = stratified_kfold_cv(clf, Xc, yc, n_splits=5, shuffle=True, random_state=42, scoring='accuracy')
	print('Stratified K-Fold (accuracy) scores:', skf_scores)
	print('Stratified K-Fold mean accuracy:', np.mean(skf_scores["test_score"]))

    # 3) GroupKFold: create simple synthetic groups (e.g., by index modulo n_groups)
	n_groups = 5
	groups = (np.arange(len(yc)) % n_groups).tolist()
	gkf_scores = group_kfold_cv(clf, Xc, yc, groups=groups, n_splits=n_groups, scoring='accuracy')
	print('GroupKFold (accuracy) scores:', gkf_scores)
	print('GroupKFold mean accuracy:', np.mean(gkf_scores["test_score"]))

	# 4) Leave-One-Out (be aware: this will fit N models; can be slow)
	loo_scores = leave_one_out_cv(clf, Xc, yc, scoring='accuracy')
	print('Leave-One-Out (accuracy) scores (first 10 shown):', loo_scores[:10])
	print('Leave-One-Out mean accuracy:', np.mean(loo_scores["test_score"]))
    
    # Regression example
    Xr, yr = get_regression_data()
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    rkf_scores = regression_kfold_cv(reg, Xr, yr, n_splits=5, scoring="r2")
    print("Regression K-Fold mean R2:", np.mean(rkf_scores["test_score"]))

