"""
Model Agnostic Approach For Regression:
1. Permutation Feature Importance
2. SHAP (SHapley Additive exPlanations) for Tree Models
3. Lime (Local Interpretable Model-agnostic Explanations) for Black Box Models
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import shap
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

project_dir = "D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms/4 - Machine Learning/08 - Feature Importance"

###################################
## Load the data
###################################
data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##########################################
# 1. Permutation Feature Importance
##########################################
def perm_feature_importance(X, y, X_train, y_train, X_test, y_test):
    '''
    Trains a Random Forest model and evaluates its performance.
    '''
    # Model training
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Model performance
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Permutation Feature Importance
    perm_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({'Feature': X.columns, 'Permutation Importance': perm_result.importances_mean}).sort_values(by='Permutation Importance', ascending=False)

    # Plot permutation importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Permutation Importance', y='Feature', data=perm_df)
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.show()
    return rf_model, perm_df

model_perm, perm_feat_df  = perm_feature_importance(X, y, X_train, y_train, X_test, y_test)

###################################
##  2. SHAP Values
###################################
def generate_shap_report(X, sample_size, cur_dir):
    """
    Compute SHAP explanations and save summary plots, dependence plots and CSV.
    Returns a list of saved file paths.
    """
    outdir = os.path.join(cur_dir, '/shap_outputs')
    os.makedirs(outdir, exist_ok=True)
    
    model = RandomForestRegressor(n_estimators = 200, max_features = 'sqrt', random_state = 42, n_jobs = -1)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Model trained in {elapsed:.2f} seconds.")
    
    ## SHAP computations
    explainer = shap.TreeExplainer(model)
    if X.shape[0] > sample_size:
        X_shap = X.sample(sample_size, random_state=42)
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

# Example with regression (California housing dataset)
saved_regression = generate_shap_report(X = X_test, sample_size = 1000, cur_dir = project_dir)
print('Saved SHAP files (regression):', saved_regression)

##################################################################
# 3. Lime (Local Interpretable Model-agnostic Explanations)
##################################################################
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor

explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['target'], random_state=42)
explanation = explainer.explain_instance(X_test.iloc[0], model.predict, num_features=5)
print(explanation.as_list())
