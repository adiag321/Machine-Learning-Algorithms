# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:50:06 2025
@author: adiag
"""
!pip install tpot

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Change directory if needed
os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\Datasets')

dataset = pd.read_csv('Advertising_data.csv')
dataset['Gender'] = dataset['Gender'].replace({'Male': '1', 'Female': '0'})
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##############################
## TPOT AutoML Classifier
##############################
tpot = TPOTClassifier(
    generations=5,          # Number of iterations to run pipeline optimization
    population_size=20,     # Number of models to keep per generation
    verbose=2,            # How much output to show
    scorers='accuracy',     # Optimization goal
    random_state=42,
    n_jobs=-1               # Use all processors
)

tpot.fit(X_train, y_train)

# Evaluate on test set
y_pred = tpot.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nTest Set Accuracy (TPOT):", accuracy)
print("Confusion Matrix:\n", cm)

# Export best pipeline code
os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\4 - Machine Learning\14 - Hyperparameter Tunning')
#tpot.export('best_pipeline.py')
