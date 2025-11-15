"""
Created on Wed Jul 16 11:23:43 2025
@author: adiag
"""
## Use GridSearchCV and select the best hyperparamter for Support Vector machine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\Datasets')

# Importing the dataset
dataset = pd.read_csv('Advertising_data.csv')
dataset['Gender'] = dataset['Gender'].replace({'Male': '1', 'Female': '0'})
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
SVC_accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of SVC without best parameters", SVC_accuracy)

###############################
## Applying GridSearchCV
###############################
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

grid_search_best_accuracy = grid_search.best_score_
print("Grid Search CV Accuracy with best parameters", grid_search_best_accuracy)

grid_srch_params = grid_search.best_params_
print("Parameters of Grid Search CV ", grid_srch_params)

###############################
## Training on best params
###############################
classifier = SVC(kernel = 'rbf', gamma=0.7)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
SVC_best_params_accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of SVC with best parameters", SVC_best_params_accuracy)
















