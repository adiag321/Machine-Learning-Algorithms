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
from sklearn.datasets import load_breast_cancer
import shap
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

project_dir = "D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms/4 - Machine Learning/08 - Feature Importance"

###################################
## Load the data
###################################
# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

import shap

# Create an explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# Global Interpretation with SHAP Summary Plot
shap.summary_plot(shap_values, X_test, feature_names=data.feature_names)


#Local Interpretation with SHAP Waterfall Plot:
# Generate waterfall plot for a single instance
# Get only the positive class explanations
shap_values_positive = shap_values[:, :, 1]  # All samples, all features, class 1
shap_values_negative = shap_values[:, :, 0]  # All samples, all features, class 0

# Now waterfall plot works with single index
shap.waterfall_plot(shap_values_positive[0])
shap.waterfall_plot(shap_values_negative[0])
plt.show()
