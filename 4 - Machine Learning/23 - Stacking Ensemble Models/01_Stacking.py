import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings("ignore")

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Define base models
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Helper function to evaluate models
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()

# Evaluate base models
print("Individual Model Performance")
base_results = {}

for name, model in base_models.items():
    mean_acc, std_acc = evaluate_model(model, X, y)
    base_results[name] = mean_acc
    print(f"{name:20s} | Mean: {mean_acc:.4f} | Std: {std_acc:.4f}")

avg_base_accuracy = np.mean(list(base_results.values()))
print(f"\nAverage Base Model Accuracy: {avg_base_accuracy:.4f}")

# Define stacking model
stacking_model = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

# Evaluate stacking model
stack_mean, stack_std = evaluate_model(stacking_model, X, y)

print("\nStacking Model Performance")
print(f"Stacking Classifier | Mean: {stack_mean:.4f} | Std: {stack_std:.4f}")

# Compare improvement
improvement = stack_mean - avg_base_accuracy
print("\nPerformance Comparison")
print(f"Average Base Accuracy : {avg_base_accuracy:.4f}")
print(f"Stacking Accuracy     : {stack_mean:.4f}")
print(f"Improvement           : {improvement:+.4f}")
