# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 14:58:02 2025

@author: adiag
"""
# importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher

import warnings
warnings.filterwarnings('ignore')

# Loading data
df = sns.load_dataset('tips')

################################################
##           Label Encoding
################################################
def label_encoder(data, col_name, encoder):
    le = encoder
    encoded_col_name = f"encoded_{col_name}"
    data[encoded_col_name] = le.fit_transform(data[col_name])
    return data

label_data = label_encoder(data = df, col_name = 'time', encoder = LabelEncoder())

################################################
#               Ordinal Encoding
################################################
def ordinal_encoder(data, col_name, encoder, categries):
    oe = encoder(categories=categries) 
    encoded_col_name = f"encoded_{col_name}"
    data[encoded_col_name] = oe.fit_transform(data[[col_name]])
    return data


ordinal_data = ordinal_encoder(data = df, col_name = 'day', encoder = OrdinalEncoder, 
                               categries = [['Fri' , 'Sat', 'Sun' , 'Thur']])

################################################
#               One Hot Encoding
################################################
def one_hot_encoding(data, col_name, encoder):
    ohe = encoder
    encoded_col = f"encoded_{col_name}"
    ## Convert sparse matrix to dense matrix
    encoded_col = ohe.fit_transform(data[[col_name]]).toarray()
    encoded_df = pd.DataFrame(encoded_col, columns=ohe.get_feature_names_out([col_name]))
    new_data = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return new_data

one_hot_encoded_data = one_hot_encoding(data = df, col_name = 'sex', encoder = OneHotEncoder())

################################################
#               Category Encoding
################################################
def category_encoding(data, col_name, encoder):
    # pip install category_encoders
    
    return data

cat_encoding = category_encoding(data = df, col_name = 'sex')

################################################
#                   Dummy Encoding
################################################
def get_dummies(data, col_name):
    # use pandas get dummies
    get_dummies = pd.get_dummies(data, columns=[col_name])
    
    return get_dummies

dummy_encoding = get_dummies(data = df, col_name = 'sex')

################################################
#            Frequency Encoding
################################################
def freq_encoding(data, col_name, encoder):
    # grouping by frequency
    fq = data.groupby(col_name).size()/len(df)
    encoded_col = f"freq_encoded_{col_name}"
    ## Convert sparse matrix to dense matrix
    data[encoded_col] = data[col_name].map(fq)
    return data
  
freq_encoding_df = freq_encoding(data = df, col_name = 'smoker', encoder = None)

################################################
#             Feature Hasher Encoding
################################################

def feat_hasher(data, col_name, num_features, input_type):
    # n_features contains the number of bits you want in your hash value.
    h = FeatureHasher(n_features = num_features, input_type = input_type) 
    # transforming the column after fitting
    hashed_Feature = h.fit_transform(data[col_name])
    hashed_Feature = hashed_Feature.toarray()
    data = pd.concat([data, pd.DataFrame(hashed_Feature)], axis = 1)
    
    return data

df['sex'] = df['sex'].astype(object)
feat_hasher_df = feat_hasher(data = df, col_name = 'sex', num_features = 3,
                             input_type = 'string')