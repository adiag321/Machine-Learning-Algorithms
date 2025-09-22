# Import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\4 - Machine Learning\08 - Feature Importance')

data = load_diabetes()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

###################################
## Split the data
###################################
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

###################################
# Random Forest 
# for Feature Selection
###################################
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

###################################
# Feature Importance
###################################
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 
                             'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Gini-based Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

###################################
# Permutation importance
###################################
perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({'Feature': X.columns, 
        'Permutation Importance': perm_result.importances_mean}).sort_values(by='Permutation Importance', ascending=False)

# Plot permutation importance
plt.figure(figsize=(10, 5))
sns.barplot(x='Permutation Importance', y='Feature', data=perm_df)
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()


# Exporting feature importances
#importance_df.to_csv("feature_importance_gini.csv", index=False)
#perm_df.to_csv("feature_importance_permutation.csv", index=False)
