# Naive Bayes Classifier using Iris Dataset

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

##############################
# Load the Iris dataset
##############################
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

##############################
# Split the data
##############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##############################
# Naive Bayes Model
##############################
def gbnaivebayes(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print(f"Number of mislabeled points out of a total {X_test.shape[0]} points: {(y_test != y_pred).sum()}")
    print("Model parameters:", gnb.get_params())
    print("Accuracy (in %):", accuracy_score(y_test, y_pred) * 100)
    
    return gnb, y_pred

# Train and get predictions
gnb, y_pred = gbnaivebayes(X_train, y_train, X_test, y_test)

##############################
# Additional Accuracy Report
##############################
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Accuracy
print("Gaussian Naive Bayes model accuracy (in %):", accuracy_score(y_test, y_pred) * 100)
