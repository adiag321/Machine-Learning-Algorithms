"""Reusable SHAP utilities for tree models and a small demo runner.

Functions are intentionally small and focused so you can import them into
other projects and pass your cleaned data (X, y) and trained models.

Design goals / contract:
 - Inputs: pandas.DataFrame for X and pandas.Series for y (cleaned, preprocessed)
 - Outputs: trained model, importance DataFrames, shap values, and plots/CSVs
 - Error modes: functions raise exceptions on missing dependencies; callers
   should handle them.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        n_estimators: int, max_features, random_state: int, n_jobs: int):
    """Train a RandomForestRegressor and return the fitted model and train time.

    NOTE: This function does not perform any evaluation. It only trains and
    returns the fitted model. All hyperparameters must be provided by the caller.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed

# The module no longer computes or exports RandomForest impurity-based or
# permutation importances. The script focuses only on SHAP-based explanations.

###################################
## SHAP utilities
###################################
def generate_shap_report(model, X: pd.DataFrame, sample_size: int, random_state: int, outdir='.'):
    """Compute SHAP explanations and save summary plots, dependence plots and CSV.

    Returns a list of saved file paths.
    """
    os.makedirs(outdir, exist_ok=True)

    ## SHAP computations
    import shap  # local import so module can be used without shap installed
    explainer = shap.TreeExplainer(model)
    if X.shape[0] > sample_size:
        X_shap = X.sample(sample_size, random_state=random_state)
    else:
        X_shap = X.copy()

    shap_values = explainer.shap_values(X_shap)

    saved = []
    feature_names = X_shap.columns.tolist()

    ## SHAP Summary plots
    out_dot = os.path.join(outdir, 'shap_summary_dot.png')
    plt.figure()
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    plt.savefig(out_dot, dpi=150, bbox_inches='tight')
    plt.close()
    saved.append(out_dot)

    ## SHAP Bar plots
    out_bar = os.path.join(outdir, 'shap_summary_bar.png')
    plt.figure()
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, plot_type='bar', show=False)
    plt.savefig(out_bar, dpi=150, bbox_inches='tight')
    plt.close()
    saved.append(out_bar)

    ## SHAP Dependence plots for top 3 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = [feature_names[i] for i in top_idx]
    for feat in top_features:
        fname = os.path.join(outdir, f'shap_dependence_{feat}.png')
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_shap, feature_names=feature_names, show=False)
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        saved.append(fname)

    ## SHAP Force plots
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], X_shap.iloc[0, :], 
                                    feature_names=feature_names, matplotlib=False)
    html_file = os.path.join(outdir, 'shap_force_sample.html')
    shap.save_html(html_file, force_plot)
    saved.append(html_file)


    shap_vals_df = pd.DataFrame(shap_values, columns=X_shap.columns, index=X_shap.index)
    csv_path = os.path.join(outdir, 'shap_values_sample.csv')
    shap_vals_df.to_csv(csv_path)
    saved.append(csv_path)

    return saved
#end of support utilities
###################################


if __name__ == '__main__':
    # Minimal example: train a RandomForest only to produce a model to explain.
    # The module's primary purpose is SHAP explanations; training here is optional
    # and shown only as a convenience when running this file directly.
    project_dir = "D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms/4 - Machine Learning/08 - Feature Importance"
    outdir = os.path.join(project_dir, '/shap_outputs')
    os.makedirs(outdir, exist_ok=True)

   
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ###################################
    ## Train the model
    # Explicit hyperparameters must be provided (no defaults in module)
    ###################################
    model, train_time = train_random_forest(X_train, y_train,
                                            n_estimators=200, max_features='sqrt', random_state=42, n_jobs=-1)
    print(f"Trained model in {train_time:.1f}s")

    # Compute and save SHAP-based interpretations (sample_size and random_state are required)
    saved = generate_shap_report(model, X_test, sample_size=1000, random_state=42, outdir=outdir)
    print('Saved SHAP files:', saved)