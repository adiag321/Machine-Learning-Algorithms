## Utility Function to validate Train Test Splits

import pandas as pd
import numpy as np

def validate_train_test_split(X_train, X_test, y_train, y_test, threshold=0.1):
    """
    Validates train-test split for classification problems.
    
    Parameters:
    - X_train, X_test: feature dataframes
    - y_train, y_test: target series
    - threshold: acceptable relative difference for numeric stats
    """

    print("\n" + "="*50)
    print("1. TARGET DISTRIBUTION CHECK")
    print("="*50)
    
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    target_df = pd.concat([train_dist, test_dist], axis=1)
    target_df.columns = ['Train %', 'Test %']
    print(target_df)

    print("\n" + "="*50)
    print("2. FEATURE DISTRIBUTION CHECK (Numerical)")
    print("="*50)

    num_cols = X_train.select_dtypes(include=np.number).columns
    drift_cols = []

    for col in num_cols:
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()

        if train_mean != 0:
            diff = abs(train_mean - test_mean) / abs(train_mean)
        else:
            diff = abs(train_mean - test_mean)

        if diff > threshold:
            drift_cols.append(col)

    print(f"Columns with drift > {threshold*100}%:", drift_cols)

    print("\n" + "="*50)
    print("3. MISSING VALUE CHECK")
    print("="*50)

    missing_train = X_train.isnull().mean()
    missing_test = X_test.isnull().mean()

    missing_df = pd.concat([missing_train, missing_test], axis=1)
    missing_df.columns = ['Train Missing %', 'Test Missing %']
    print(missing_df[missing_df.sum(axis=1) > 0])

    print("\n" + "="*50)
    print("4. DUPLICATE ROW CHECK (Leakage Risk)")
    print("="*50)

    common_rows = pd.merge(X_train, X_test, how='inner')
    print(f"Number of overlapping rows: {len(common_rows)}")

    print("\n" + "="*50)
    print("5. CATEGORICAL FEATURE CHECK (Unseen Categories)")
    print("="*50)

    cat_cols = X_train.select_dtypes(include='object').columns

    for col in cat_cols:
        train_cats = set(X_train[col].dropna().unique())
        test_cats = set(X_test[col].dropna().unique())

        unseen = test_cats - train_cats

        if unseen:
            print(f"{col}: Unseen categories in test -> {unseen}")

    print("\n" + "="*50)
    print("6. DATASET SIZE CHECK")
    print("="*50)

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    if len(X_test) < 0.1 * (len(X_train) + len(X_test)):
        print("Warning: Test set might be too small.")

    print("\n" + "="*50)
    print("7. BASIC LEAKAGE HEURISTIC CHECK")
    print("="*50)

    # Check for columns highly correlated with target
    if y_train.dtype in [np.int64, np.float64]:
        corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        high_corr = corr[corr > 0.9]
        if not high_corr.empty:
            print("Highly correlated features with target (possible leakage):")
            print(high_corr)
        else:
            print("No obvious leakage via high correlation.")
    else:
        print("Skipping correlation check (non-numeric target).")

    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)


validate_train_test_split(X_train, X_test, y_train, y_test)