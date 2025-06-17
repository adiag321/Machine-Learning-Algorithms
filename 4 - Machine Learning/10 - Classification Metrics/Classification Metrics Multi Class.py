# -*- coding: utf-8 -*-
"""
Multi Class Classification: Precision, Recall, and F1 Score
Link: CampusX Youtube Channel - https://youtu.be/iK-kdhJ-7yI?si=q5OQ6w_p34JTQwDN
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Load iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = data.target

################################
## Split the data
################################
X = df.drop(columns=['Species'])
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

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
    
    print("Confusion Matrix:\n", pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        columns=[f"Pred {label}" for label in np.unique(y_true)],
        index=[f"Actual {label}" for label in np.unique(y_true)]
    ))
    print("\nPer-Class Precision, Recall, F1:")
    print(classification_report(y_true, y_pred, target_names=data.target_names))
    
    # Optional: show macro/micro averages explicitly
    print(f"Macro Avg Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Macro Avg Recall   : {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Macro Avg F1 Score : {f1_score(y_true, y_pred, average='macro'):.4f}")


# Evaluate both classifiers
evaluate_model("Logistic Regression", y_test, y_pred1)
evaluate_model("Decision Tree", y_test, y_pred2)

# Show a few predictions
result = pd.DataFrame({
    'Actual Label': y_test.values,
    'Logistic Regression Prediction': y_pred1,
    'Decision Tree Prediction': y_pred2
})
print("\nSample Predictions:\n", result.sample(10, random_state=1))
