## Classification Metrics
'''
Binary Class Classification: Precision, Recall, and F1 Score
Link: CampusX Youtube Channel - https://youtu.be/iK-kdhJ-7yI?si=q5OQ6w_p34JTQwDN
'''

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report


# Set working directory (adjust if needed)
os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\Datasets')

# Load dataset
df = pd.read_csv('./heart/heart.csv')

################################
## Split the data
################################
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

################################
# Initialize models
################################
clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

################################
# Evaluation function
################################
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}\n{'-' * len(name)}")
    
    print(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision    : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall       : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score     : {f1_score(y_true, y_pred):.4f}")
    
    print("Confusion Matrix:\n", pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        columns=["Pred 0", "Pred 1"],
        index=["Actual 0", "Actual 1"]
    ))


# Evaluate both models
evaluate_model("Logistic Regression", y_test, y_pred1)
evaluate_model("Decision Tree", y_test, y_pred2)
