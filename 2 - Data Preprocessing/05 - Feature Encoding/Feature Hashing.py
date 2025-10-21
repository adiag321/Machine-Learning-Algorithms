# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:49:33 2025

@author: adiag
"""
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Dummy dataset: Transaction, user, and product categories
raw_data = pd.DataFrame({
    'transaction_id': ['T1', 'T2', 'T3', 'T4'],
    'user_id': ['A12', 'B34', 'A12', 'C56'],
    'product': ['apple', 'banana', 'orange', 'apple']
})

# Prepare features as dicts for FeatureHasher
features = raw_data[['transaction_id', 'user_id', 'product']].to_dict(orient='records')

# Initialize FeatureHasher for 8 dimensions
hasher = FeatureHasher(n_features=8, input_type='dict')
hashed_features = hasher.transform(features)

# Convert to a DataFrame for easy display
hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"hash_{i}" for i in range(8)])
result = pd.concat([raw_data, hashed_df], axis=1)
print(result)


################################################

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import FeatureHasher

# Load text data
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball'])
texts = data.data[:6]  # Use a few for demonstration

# Prepare text as token count dicts
def text_to_token_dict(text):
    tokens = text.lower().split()
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    return freq

features = [text_to_token_dict(txt) for txt in texts]

# Feature hashing into 10 dimensions
hasher = FeatureHasher(n_features=10, input_type='dict')
hashed_matrix = hasher.transform(features)

print(hashed_matrix.toarray())

