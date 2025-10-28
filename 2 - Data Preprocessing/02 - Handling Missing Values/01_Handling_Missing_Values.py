##################################
# Handling Missing values
# 1. Identifing missing values
# 2. Simple Imputer (Numeric + Cat)
# 3. KNN Imputer (Numeric + Cat)
# 4. Most frequent category (Cat)
##################################
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer                   # For Numeric
from sklearn.neighbors import KNeighborsClassifier      # For categorical

import warnings
warnings.filterwarnings("ignore")

project_dir = "D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms"
os.chdir(project_dir)

# loading the data 
df = pd.read_csv("./Datasets/Titanic/titanic.csv")

####################################################
#              Identify Missing Values
####################################################
def find_missing_values(data):
    missing_value_df = pd.DataFrame()
    print("Missing values in the data are: ")
    print(data.isnull().sum())
    
    ## Heatmap to plot missing Values
    print("HeatMap to show missing values: ")
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')

    # Inserting all the missing values information into a new dataframe for analysis
    missing_value_df['columns'] = data.columns
    missing_value_df['Missing Values'] = data.isnull().sum().values
    missing_value_df['Datatype'] = data.dtypes.values
    
    return missing_value_df
    
missing_value_info_df = find_missing_values(data = df)

####################################################
#                  Simple Imputer: 
# For numerical (mean/median/mode)
# For categorical (Unknown)
####################################################
def simple_imputer(data, col, strategy):
    if data[col].dtype == 'int64' or data[col].dtype == 'float64':
        imputer = SimpleImputer(strategy = strategy)  # or 'median', 'most_frequent'
        data[[col]] = imputer.fit_transform(data[[col]])
    
    elif data[col].dtype == 'object':
        imputer_cat = SimpleImputer(strategy = 'most_frequent')
        data[[col]] = imputer_cat.fit_transform(data[[col]])

    return data

num_simp_imputer = simple_imputer(data = df, col = 'Age', strategy = 'mean')

####################################################
#                  KNN Imputer
####################################################
def knn_imputer(data, col, neighbor):
    # Imputation for Numeric data
    if data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
        print("Performed KNN Imputation for", data[col].dtype, "input variable")
        knn_imputer = KNNImputer(n_neighbors = neighbor)
        data[[col]] = knn_imputer.fit_transform(data[[col]])
    
    elif data[col].dtype == 'object' and data[col].isnull().sum() > 0:
        print("Performed KNN Imputation for", data[col].dtype, "input variable")
        # Separate rows with and without missing values
        missing_mask = data[col].isnull()
        not_missing_mask = ~missing_mask
        numeric_df = data.select_dtypes(include=np.number)
        
        numeric_df = numeric_df.fillna(0)
        
        # Define training and missing sets
        X_train = numeric_df.loc[not_missing_mask]
        y_train = data.loc[not_missing_mask, col]
        X_missing = numeric_df.loc[missing_mask]
        
        # Fit KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=neighbor, weights='distance')
        knn.fit(X_train, y_train)
        # Predict missing categories
        y_pred = knn.predict(X_missing)
        
        # Replace missing values in the original dataframe
        data.loc[missing_mask, col] = y_pred
        print(f"Filled {missing_mask.sum()} missing values in '{col}' using KNN Classifier.")
                 
    return data

knn_imputer_data = knn_imputer(data = df, col = 'Age', neighbor = 5)
knn_imputer_cat_data_1 = knn_imputer(data = df, col = 'Cabin', neighbor = 5)

####################################################
##               Most frequent category
####################################################
def most_freq_cat(data, col):
    
    # Most frequent category Count
    if data[col].nunique() <= 10:
        print("Bar Plot for count for feature: ", col)
        data[col].value_counts().plot.bar()
    else:
        print(col, "input feature has more than 10 unique values, so not able to show bar plot!")
        
    # Value Counts
    print(data[col].value_counts())
    # Imputing by most frequent categry
    most_frequent_category = data[col].mode()[0]
    data[col].fillna(most_frequent_category,inplace=True)
    return data

freq_data_imputation = most_freq_cat(data = df, col = 'Cabin')

####################################################













