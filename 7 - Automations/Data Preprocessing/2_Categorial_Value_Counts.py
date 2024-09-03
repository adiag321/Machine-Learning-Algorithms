# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:18:11 2024

@author: adiag
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\Datasets/Clean_data")
df.head()

# Creating a small subset of the data
df = df[["Alley","LotShape","GarageCond","MasVnrArea","SalePrice"]]
df.head(5)
df_cat_values = df.drop(["MasVnrArea","SalePrice"],axis=1)


def value_counts(data):
    cat_list = data.select_dtypes(include='object').columns.tolist()
    print("The categorical columns are:", cat_list)
    
    for i in cat_list:
        print(df[i].value_counts(),'\n')
        
value_counts(df)
    

