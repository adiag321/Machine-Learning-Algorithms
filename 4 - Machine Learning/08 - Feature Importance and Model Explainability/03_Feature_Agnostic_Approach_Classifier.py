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
# Create an explainer
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for the test set: Shap_values is in 3d array 
shap_values = explainer(X_test)

# Summary Plot
# Used for Global Interpretation 
shap.summary_plot(shap_values[:,:,1], X_test, plot_type="bar")

# Violin Plot
shap.plots.violin(shap_values[:,:,1])

# Dependence Plot
# Its a type of scatter plot that displays how a model's predictions are affected by a specific feature
shap.dependence_plot("mean radius", shap_values[:,:,1].values, X_test, interaction_index = "worst area")

# Force Plot
# Want to examine the first sample in the testing set to determine which features contributed to the "0" or "1" result.
shap.plots.force(explainer.expected_value[1], shap_values[0, :, 1].values, X_test.iloc[0, :], matplotlib = True)  ## For class 1    
shap.plots.force(explainer.expected_value[0], shap_values[0, :, 0].values, X_test.iloc[0, :], matplotlib = True)  ## For class 0

# Decision Plot
# It visually shows the model decisions by mapping the cumulative SHAP values for each prediction
shap.decision_plot(explainer.expected_value[1], shap_values[:, :, 1].values, X_test.columns)       # For class 1
shap.decision_plot(explainer.expected_value[0], shap_values[:, :, 0].values, X_test.columns)       # For class 0
''' 
Note for Decision Plot:
Each plotted line on the decision plot shows how strongly the individual features contributed to a 
single model prediction, thus explaining what feature values pushed the prediction.
'''

# Waterfall Plot:
# Generate waterfall plot for a single instance (local interpretation)
shap_values_positive = shap_values[:, :, 1]  # All samples, all features, class 1
shap_values_negative = shap_values[:, :, 0]  # All samples, all features, class 0

# Now waterfall plot works with single index
shap.waterfall_plot(shap_values_positive[0])
shap.waterfall_plot(shap_values_negative[0])
plt.show()
