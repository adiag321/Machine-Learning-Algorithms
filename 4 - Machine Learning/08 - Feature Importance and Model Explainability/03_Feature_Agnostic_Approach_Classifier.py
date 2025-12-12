"""
Model Agnostic Approach For Classification Problems:
1. Permutation Feature Importance
2. SHAP (SHapley Additive exPlanations) for Tree Models
3. Lime (Local Interpretable Model-agnostic Explanations) for Black Box Models
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import shap
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

project_dir = "D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms/4 - Machine Learning/08 - Feature Importance"

#######################################
## Load the data and train the model
#######################################
# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target
X = df.drop(columns = 'target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)     # Make predictions
print(classification_report(y_pred, y_test))        # Classification report

#############################################
## SHAP Explainer
#############################################
def shap_apply(model, X_train_data, X_test_data):
    '''
    Parameters
    ----------
    model : sklearn model. The model to explain
    X_train_data : pandas DataFrame
    X_test_data : pandas DataFrame

    Returns
    -------
    shap : shap object
    shap_values : numpy array. The SHAP values for the test set

    Notes
    -----
    This function provides a way to apply SHAP values to a given model and data. It can be used to explain the predictions of a model.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> rf_model.fit(X_train, y_train)
    >>> shap, shap_values = shap_apply(rf_model, X_train, X_test)
    '''
    # Create an explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set: Shap_values is in 3d array 
    shap_values = explainer(X_test_data)

    print(f'Shape of test data: {X_test_data.shape}')
    print(f'Type of Shap_values: {type(shap_values)}. Length of the list: {len(shap_values)}')
    print(f'Shape of Shap_values: {shap_values.shape}')

    # Summary Plot
    # Used for Global Interpretation
    print("Summary Plot")
    shap.summary_plot(shap_values[:,:,1], X_test_data, plot_type="bar")

    # Violin Plot
    print("Violin Plot")
    shap.plots.violin(shap_values[:,:,1])

    # Dependence Plot   
    # Its a type of scatter plot that displays how a model's predictions are affected by a specific feature
    print("Dependence Plot")
    shap.dependence_plot("mean radius", shap_values[:,:,1].values, X_test_data, interaction_index = "worst area")

    # Force Plot
    # Want to examine the first sample in the testing set to determine which features contributed to the "0" or "1" result.
    print("Force Plot")
    shap.plots.force(explainer.expected_value[1], shap_values[0, :, 1].values, X_test_data.iloc[0, :], matplotlib = True)  ## For class 1    
    shap.plots.force(explainer.expected_value[0], shap_values[0, :, 0].values, X_test_data.iloc[0, :], matplotlib = True)  ## For class 0

    # Decision Plot
    # It visually shows the model decisions by mapping the cumulative SHAP values for each prediction
    print("Decision Plot")
    shap.decision_plot(explainer.expected_value[1], shap_values[:, :, 1].values, X_test_data.columns)       # For class 1
    shap.decision_plot(explainer.expected_value[0], shap_values[:, :, 0].values, X_test_data.columns)       # For class 0
    ''' 
    Note for Decision Plot:
    Each plotted line on the decision plot shows how strongly the individual features contributed to a 
    single model prediction, thus explaining what feature values pushed the prediction.
    '''

    # Waterfall Plot:
    # Generate waterfall plot for a single instance (local interpretation)
    print("Waterfall Plot")
    shap_values_positive = shap_values[:, :, 1]  # All samples, all features, class 1
    shap_values_negative = shap_values[:, :, 0]  # All samples, all features, class 0

    # Now waterfall plot works with single index
    shap.waterfall_plot(shap_values_positive[0])
    shap.waterfall_plot(shap_values_negative[0])
    plt.show()

    return shap, shap_values

shap_explainer, shap_vals = shap_apply(model = rf_model, X_train_data = X_train, X_test_data = X_test)