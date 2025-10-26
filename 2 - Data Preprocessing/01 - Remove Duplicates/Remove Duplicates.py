# ## Code to Remove Duplicates

# string = input()
# result = string[0]
# def check(string, result):
#     string = string[1:]
#     try:
#         if(result[-1] != string[0]):
#             result = result + string[0]
#             return check(string, result)
#         else:
#             return check(string, result)
#     except IndexError:
#         print(result)
# check(string, result)

import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create base data
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
ages = [25, 30, 35, 40, 45]
cities = ['NY', 'LA', 'SF', 'Chicago', 'Houston']

# Generate 80 unique records
unique_data = {
    'Name': [random.choice(names) for _ in range(80)],
    'Age': [random.choice(ages) for _ in range(80)],
    'City': [random.choice(cities) for _ in range(80)]
}

df_unique = pd.DataFrame(unique_data)

# Introduce 20 duplicates by repeating random rows from the unique set
duplicates = df_unique.sample(20, random_state=1)
df = pd.concat([df_unique, duplicates], ignore_index=True)

# Shuffle rows to mix duplicates randomly
df = df.sample(frac=1, random_state=2).reset_index(drop=True)

print("Original DataFrame (with duplicates):")
print(df)

# Find duplicates
duplicates_found = df[df.duplicated(keep=False)]
print("\nDuplicates Found:")
print(duplicates_found)

# Remove duplicates
df_no_duplicates = df.drop_duplicates()

print("\nDataFrame after removing duplicates:")
print(df_no_duplicates)
