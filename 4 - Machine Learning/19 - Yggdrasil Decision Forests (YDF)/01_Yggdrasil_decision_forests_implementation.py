#################
# Yggdrasil Decision Forests
#################

# pip install yggdrasil-decision-forests
# Import the necessary libraries
#import yggdrasil_decision_forests as ydf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

#################
# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv', header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# Display the first few rows of the dataset
print(data.head())