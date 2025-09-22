# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:39:10 2025

@author: adiag
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Change directory if needed
os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\Datasets')

# Load the dataset
dataset = pd.read_csv('Advertising_data.csv')
dataset['Gender'] = dataset['Gender'].replace({'Male': '1', 'Female': '0'})
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Base model for tuning
base_svc = SVC()

###############################
## Applying RandomizedSearchCV
###############################
param_dist = {
    'C': uniform(loc=0.1, scale=100),          # Uniformly distributed C values between 0.1 and 100
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'] + list(np.linspace(0.1, 1.0, 10)),
    'degree': [2, 3, 4, 5],                    # Relevant only for 'poly' kernel
}

random_search = RandomizedSearchCV(
    estimator = base_svc,
    param_distributions = param_dist,
    n_iter = 50,                # Try 50 different combinations
    scoring = 'accuracy',
    cv = 10,
    verbose = 1,
    random_state = 42,
    n_jobs = -1
)

random_search.fit(X_train, y_train)

# Best results from Random Search
best_accuracy = random_search.best_score_
best_params = random_search.best_params_

print("\nRandom Search CV Accuracy with best parameters:", best_accuracy)
print("Best parameters from Random Search:", best_params)

###############################
# Train best model and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set using best Random Search model:", accuracy)
