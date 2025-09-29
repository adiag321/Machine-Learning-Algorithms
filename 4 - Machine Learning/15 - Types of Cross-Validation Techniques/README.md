# Cross-Validation Techniques — README

This folder contains `01_Cross_Validation_Techniques.py`, a compact utilities module implementing many common cross-validation splitters and demo code that runs a sample classifier and regressor through a selection of CV workflows.

#### Key points about the utilities
- Each CV function accepts an estimator, X, y and relevant splitter parameters. They call sklearn's `cross_val_score` or `cross_validate` and return a consistent dictionary format:
  - `{ "test_score": np.ndarray([...]) }`
  - The wrapper `_wrap_scores` is used across most functions so demo code can index `result["test_score"]` for summaries.

#### Available functions (high level)
- `kfold_cv`, `regression_kfold_cv` — standard K-Fold for classification/regression
- `stratified_kfold_cv` — class-stratified folds (classification)
- `group_kfold_cv`, `leave_one_group_out_cv` — group-aware splitting
- `time_series_cv` — forward-chaining splits for time series (no shuffle)
- `leave_one_out_cv`, `leave_p_out_cv` — exhaustive small-data CV options
- `repeated_kfold_cv` — repeated K-Fold for variance estimation
- `shuffle_split_cv`, `stratified_shuffle_split_cv`, `group_shuffle_split_cv` — randomized holdouts
- `cross_validate_with_scoring` — wrapper for multiple metrics via `cross_validate`
- `nested_cv_grid_search`, `nested_cv_random_search` — nested CV for hyperparameter selection (GridSearchCV and RandomizedSearchCV as inner loops)

#### Demo behavior
- The module's `__main__` demo performs the following:
  - Loads the breast cancer dataset (classification) and runs a `RandomForestClassifier` through K-Fold, Stratified K-Fold, GroupKFold (with synthetic groups), and Leave-One-Out. It prints per-fold scores and means.
  - Loads a diabetes regression dataset and runs a `RandomForestRegressor` through K-Fold for an example regression score (R² by default via `regression_kfold_cv`).

#### Recommended parameter choices
- K-Fold / Stratified K-Fold: `n_splits=5` (or 10), `shuffle=True` with `random_state` for reproducibility.
- GroupKFold: set `n_splits` based on number of groups (e.g., 4–10) and ensure groups are balanced enough for folds.
- TimeSeriesSplit: `n_splits=5`, set `max_train_size` for a rolling window, set `gap`>0 to prevent leakage if needed.
- Leave-One-Out / Leave-P-Out: only for small datasets because they perform many fits.
- ShuffleSplit / StratifiedShuffleSplit: `n_splits=10`, `train_size=0.7`, `test_size=0.3` for stability checks.
- RepeatedKFold: `n_splits=5`, `n_repeats=5` to estimate variance.
- Nested CV: outer K-Fold (5) + inner GridSearch with StratifiedKFold (3–5) when tuning hyperparameters.

#### Practical tips
- Always set `random_state` when using shuffling so results are reproducible.
- For classification prefer stratified splits when class imbalance exists.
- When using group-aware splitters, verify `groups` alignment (same length as `y`) and that there are enough groups for the chosen `n_splits`.
- Use `n_jobs` to speed up `cross_val_score` and grid/random search where supported.