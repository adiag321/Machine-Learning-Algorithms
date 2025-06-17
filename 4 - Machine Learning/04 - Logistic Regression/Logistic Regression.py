# -*- coding: utf-8 -*-
### Logistic Regression

# Importing required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Set working directory
os.chdir('D:\\OneDrive - Northeastern University\\Jupyter Notebook\\Machine Learning Algorithms')

# Load datasets
train_data = pd.read_csv("./Datasets/Titanic/train_titanic.csv")
test_data = pd.read_csv("./Datasets/Titanic/test_titanic.csv")

# Shape of the dataset
print('Shape of training data:', train_data.shape)
print('Shape of testing data:', test_data.shape)

# Separate the independent and target variables
train_x = train_data.drop(['Survived'], axis=1)
train_y = train_data['Survived']
test_x = test_data.drop(['Survived'], axis=1)
test_y = test_data['Survived']


######################################
## Logistic Regression
######################################

model = LogisticRegression(solver='liblinear')
model.fit(train_x, train_y)

# Model coefficients and intercept
print('Coefficients of the model:', model.coef_)
print('Intercept of the model:', model.intercept_)

# Probability predictions for training data
train_prob_df = pd.DataFrame(model.predict_proba(train_x), columns=["Probability of Not Survived", "Probability of Survived"])
train_prob_df["Actual"] = train_y.values

# Probability predictions for testing data
test_prob_df = pd.DataFrame(model.predict_proba(test_x), columns=["Probability of Not Survived", "Probability of Survived"])
test_prob_df["Actual"] = test_y.values

# Helper function to predict, compute accuracy, and plot confusion matrix
def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True, cmap="Blues", fmt=".2f")
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix')
    plt.show()
    return preds

# Run predictions and evaluation
predict_and_plot(train_x, train_y, name='Train')
predict_and_plot(test_x, test_y, name='Test')

# Save the trained model
# joblib.dump(model, 'logistic_regression_titanic_model.pkl')
