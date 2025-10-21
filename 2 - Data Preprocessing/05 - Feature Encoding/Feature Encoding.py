# -*- coding: utf-8 -*-
# #######################################
# Label Encoding
# Ordinal Encoding
# Dummy Encoding
# One Hot Encoding
# Frequency Encoding
# Count Encoding
# Target Mean Encoding
# Frequency hasher Encoding
# Category Encoding
# Weight of Evidence Encoding
# #######################################

# importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from feature_engine.encoding import WoEEncoder
import warnings
warnings.filterwarnings('ignore')

# Loading data
df = sns.load_dataset('tips')

########### Ordinal Features ###################
################################################
##           Label Encoding
################################################
def label_encoder(data, col_name):
    le = LabelEncoder()
    encoded_col_name = f"encoded_{col_name}"
    data[encoded_col_name] = le.fit_transform(data[col_name])
    return data

label_data = label_encoder(data = df, col_name = 'time' )

################################################
#               Ordinal Encoding
################################################
def ordinal_encoder(data, col_name, categries):
    oe = OrdinalEncoder(categories=categries) 
    encoded_col_name = f"encoded_{col_name}"
    data[encoded_col_name] = oe.fit_transform(data[[col_name]])
    return data

ordinal_data = ordinal_encoder(data = df, col_name = 'day', categries = [['Fri' , 'Sat', 'Sun' , 'Thur']])

########### Nominal Features ###################
################################################
#                   Dummy Encoding
################################################
def get_dummies(data, col_name):
    # use pandas get dummies
    get_dummies = pd.get_dummies(data, columns=[col_name])
    return get_dummies

dummy_encoding = get_dummies(data = df, col_name = 'sex')

################################################
#               One Hot Encoding
################################################
def one_hot_encoding(data, col_name):
    ohe = OneHotEncoder(categories="auto", #learning the category automatically
                        drop="first", #creating k-1 category
                        #sparse=False, # this will return numpy array else it will return sparse matix
                        handle_unknown="error")
    encoded_col = f"encoded_{col_name}"
    ## Convert sparse matrix to dense matrix
    encoded_col = ohe.fit_transform(data[[col_name]]).toarray()
    # Add data to original dataframe
    encoded_df = pd.DataFrame(encoded_col, columns=ohe.get_feature_names_out([col_name]))
    new_data = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return new_data

one_hot_encoded_data = one_hot_encoding(data = df, col_name = 'sex')

################################################
#            Frequency Encoding
################################################
def freq_encoding(data, col_name):
    # grouping by frequency
    fq = data.groupby(col_name).size()/len(df)
    encoded_col = f"freq_encoded_{col_name}"
    ## Convert sparse matrix to dense matrix
    data[encoded_col] = data[col_name].map(fq)
    return data
  
freq_encoding_df = freq_encoding(data = df, col_name = 'smoker')

################################################
#            Count Encoding
################################################
def count_encoding(data, col_name):
    # grouping by frequency
    count = data[col_name].value_counts().to_dict()
    encoded_col = f"col_encoded_{col_name}"
    ## Convert sparse matrix to dense matrix
    data[encoded_col] = data[col_name].map(count)
    return data
  
count_encoding_df = count_encoding(data = df, col_name = 'smoker')


################################################
#            Target Mean Encoding
################################################
def target_mean_encoding(data, col_name, target):
    """
    Takes 4 input train,test, variable and target. It converts them into target mean encoding
    """
    ## plotting before encoding
    fig = plt.figure()
    fig = data.groupby([col_name])[target].mean().plot()
    fig.set_title(f'Relationship between {col_name} and {target}')
    plt.show()
    mapper = data.groupby([col_name])[target].mean().to_dict()
    data[col_name] = data[col_name].map(mapper)
    
    # Plotting after encoding
    fig = plt.figure()
    fig = data.groupby([col_name])[target].mean().plot()
    fig.set_title(f'Relationship between {col_name} and {target}')
    plt.show()
    return data

# col_name = categorical, target = numerical/binary
target_mean_encoding_df = target_mean_encoding(data = df, col_name = 'smoker', target = 'size')

################################################
#             Feature Hasher Encoding
################################################
def feat_hasher(data, col_name, num_features, input_type):
    # n_features contains the number of bits you want in your hash value.
    hasher = FeatureHasher(n_features = num_features, input_type = input_type) 
    feats = data[[col_name]].apply(tuple ,axis = 1)
    # Apply hash encoding
    
    hashed_Feature = hasher.fit_transform(feats)
    # Convert hashed_features to  array
    hashed_Feature = hashed_Feature.toarray()
    data = pd.concat([data, pd.DataFrame(hashed_Feature)], axis = 1)
    return data

df['sex'] = df['sex'].astype(str)
feat_hasher_df = feat_hasher(data = df, col_name = 'sex', num_features = 2, input_type = 'string')

################################################
#               Category Encoding
################################################
def category_encoding(data, col_name, encoder):
    # pip install category_encoders
    return data

cat_encoding = category_encoding(data = df, col_name = 'sex')

################################################
#       Weight of Evidence Encoding
################################################
# the target column should have binary values
df['smoker_binary'] = df['smoker'].map({'Yes': 1, 'No': 0}).astype(int)
# Convert pred variable to string (not object)
df['sex'] = df['sex'].astype(str)

def woe_encoding(data, col_name, target_name):
    # pip install feature-engine
    data[col_name] = data[col_name].astype(str)
    
    woe_encoder = WoEEncoder(variables=[col_name], ignore_format=True, fill_value=0)
    woe_encoder.fit(data[[col_name]], data[target_name])
    
    print("WOE mapping:", woe_encoder.encoder_dict_)
    data_woe = woe_encoder.transform(data[[col_name]])
    # Concat with the original data
    data[col_name + '_woe'] = data_woe[col_name]
    return data

woe_encoding_df = woe_encoding(data=df, col_name='sex', target_name='smoker_binary')
