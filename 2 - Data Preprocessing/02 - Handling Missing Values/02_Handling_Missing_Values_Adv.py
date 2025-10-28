#####################################################
## Handling Missing Values: Advanced Techniques
# 1. 3M Imputation (Mean, Median, Mode)
# 2. Missing Case Analysis (missing at random)
# 3. Random Sample Imputation: Imputing at Random
#####################################################

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

project_dir = "D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms"
os.chdir(project_dir)

# loading the data 
data = pd.read_csv("https://raw.githubusercontent.com/adiag321/Data-Science-datasets/main/House%20Price%20train.csv")

############################################################
##      3M Imputation (Mean, Median, Mode)
############################################################
continous_var_na = []  
def continous_var_null(data, cont_var_null):
    '''
    Find all numerical predictor variables with null values
    '''
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content) and data[label].isnull().sum() > 0:
            cont_var_null.append(label)
    ## Show data distribution
    print("The data distribution for Continous variables with Null values:")
    print(data[cont_var_null].describe())
    #creating plots for continous variables
    data[cont_var_null].hist(bins=50, figsize=(10,5))
    
    return cont_var_null

# Mean, Median and mode imputation
def impute_missing_vals(data, var, strategy):
    """
    Fill the missing value using Mean/Median/Mode.
    It also plots the distribution of the original and imputed data
    """
    if strategy == "mode":
        impute_val = data[var].mode()[0] #finding value using mode
        data[var + "_" + strategy] = data[var].fillna(impute_val) #creating new column and storing the values
        
    elif strategy == "mean":
        impute_val = data[var].mean()
        data[var + "_" + strategy] = data[var].fillna(impute_val)
        
    elif strategy == "median":
        impute_val = data[var].median()
        data[var + "_" + strategy] = data[var].fillna(impute_val)
    
    # plotting data after and before imputation
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    data[var].plot(kind='kde', ax=ax)
    data[var + "_" + strategy].plot(kind='kde', ax=ax, color='orange')
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')
    
    return data

continous_var_na = continous_var_null(data, continous_var_na)
mean_imp_data = impute_missing_vals(data, "GarageYrBlt", "mean")

############################################################
##      Missing Case Analysis (missing at random)
############################################################
# storing variables those variables which has missing values 
var_na = []
def missing_data(data, missing_var_list):
    '''
    Find all the columns with missing values (Numeric + Categorical)
    '''
    for var in data.columns:
        if data[var].isnull().sum() > 0:
            missing_var_list.append(var)
    # printing % of missing values in the columns 
    print(data[missing_var_list].isnull().sum()*100/len(data))
    # data[var_na].isnull().sum()*100/data.shape[1]
    
def missing_at_random_imputer(data, threshold, var):
    '''
    We simply drop the columns if the missing value is less than threshold.
    And check if the distribution didnt change
    '''
    # Storing those variables which have missing values less than 5 %
    var_cca = [var for var in data.columns if data[var].isnull().mean() < threshold]

    # We simply drop all the null values
    data_imputed = data[var_cca].dropna()
    print("Shape of orignal data:", data.shape)
    print("Shape of Imputed data:", data_imputed.shape)
    
    # Plotting histogram for Integer variables
    if data[var].dtype == 'int64':
        print("Plotting for integer variables")
        fig = plt.figure(figsize = (16,4))
    
        ax = fig.add_subplot(131) # defining where we want our plot 
        data[var].hist(bins=50, ax=ax, color='orange', alpha=1, label = "Before_dropping") #plotting distribution
        plt.title(var) 
        plt.legend() # getting the legends 
        
        ax = fig.add_subplot(132)
        data_imputed[var].hist(bins=50, ax=ax, color='green', alpha=0.8, label = "After_dropping")
        plt.title(var)
        plt.legend()
    
        ax = fig.add_subplot(133)
        data[var].hist(bins=50, ax=ax, color='orange', alpha=1, label = "Befor_dropping")
        data_imputed[var].hist(bins=50, ax=ax, color='green',  alpha=0.8, label = "After_dropping")
        plt.title("Distribution change")
        plt.legend()
    
    ## Plotting for Categorical Variables
    elif data[var].dtype == 'object':
        # setting the figure size for our distrbution 
        fig = plt.figure(figsize = (16,4))
        
        ax = fig.add_subplot(121) #defining where we ant our plot 
        graph = sns.countplot(ax=ax,x=var, data=data)
        plt.title("Before Complete Case Analysis")
        
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x()+p.get_width()/2., height + 0.2,height ,ha="center",fontsize=15)
            
        ax = fig.add_subplot(122)
        graph = sns.countplot(ax=ax, x=var, data = data_imputed)
        plt.title("After Complete Case Analysis")
        
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x()+p.get_width()/2., height + 0.2,height, ha="center",fontsize=15)          
    else:
        print("Predictor is not of INTEGER or OBJECT data dtype")
    return data

imputed_data = missing_at_random_imputer(data, 0.05, "GrLivArea")

############################################################
##      Random Sample Imputation: Imputing at Random
############################################################
def random_sample_imputation(df, col):
    '''
    Fill the missing values at random
    '''
    random_sample = df[col].dropna().sample(df[col].isnull().sum(), random_state=0)
    random_sample.index = df[df[col].isnull()].index
    df.loc[df[col].isnull(), col] = random_sample
    return df

random_imputed_df = random_sample_imputation(df = data, col = 'GrLivArea')













































