#K-Nearest Neighbour(K-NN)
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, classification_report, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set the path to the dataset
dataset_path = os.path.join("..", "..", "Datasets")
os.chdir(dataset_path)

# Set up logging to both file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler("knn_model.log", mode='w'),
        logging.StreamHandler()
        ])

##########################
#       FUNCTIONS
##########################
# Choosing value of K using Elbow method
def plot_elbow(X_train, y_train, X_test, y_test):
    logging.info("Plotting Elbow Method and Accuracy vs K...")
    error_rate, accuracy_list = [], []
    k_range = range(1, 21)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_k = knn.predict(X_test)
        error_rate.append(np.mean(pred_k != y_test))
        accuracy_list.append(accuracy_score(y_test, pred_k))

    
    # Look for the point where accuracy levels off or error stops decreasing sharply
    # Elbow plot
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, error_rate, color='red', linestyle='dashed', marker='o', markerfacecolor='blue')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('K Value')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, accuracy_list, color='green', linestyle='dashed', marker='s', markerfacecolor='black')
    plt.title('Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def find_best_k(X_train, y_train):
    logging.info("Performing Grid Search for best hyperparameters...")
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
        }
    
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"AUC Score: {auc:.4f}")
    logging.info("Classification Report:\n" + report)

    return y_pred, y_proba

def cross_validation_score(model, X, y):
    logging.info("Performing cross-validation...")
    scores = cross_val_score(model, X, y, cv=10)
    logging.info(f"Cross-validation Accuracy (mean ± std): {scores.mean():.4f} ± {scores.std():.4f}")

def plot_decision_boundary(classifier, X_set, y_set, title):
    x1, x2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1], c='red', label='0')
    plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1], c='green', label='1')
    plt.title(f'K-NN - {title}')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

#############################
# MAIN WORKFLOW
#############################
logging.info("Loading dataset...")
data = pd.read_csv(".\Social_Network_Ads.csv")
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
logging.info("Scaling dataset...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Find the best value of K
plot_elbow(X_train_scaled, y_train, X_test_scaled, y_test)

# Train best model
best_model = find_best_k(X_train_scaled, y_train)

# Evaluate model
y_pred, y_proba = evaluate_model(best_model, X_test_scaled, y_test)

# Cross-validation
cross_validation_score(best_model, X, y)

# Visualizations
plot_decision_boundary(best_model, X_train_scaled, y_train, 'Training Set')
plot_decision_boundary(best_model, X_test_scaled, y_test, 'Testing Set')
plot_roc_curve(y_test, y_proba)