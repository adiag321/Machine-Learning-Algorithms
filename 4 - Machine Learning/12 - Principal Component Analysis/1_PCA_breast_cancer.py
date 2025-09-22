"""
Created on Mon Jul 14 10:37:02 2025

@author: adiag
"""
## Implementing PCA on Breast Cancer Data
## https://www.youtube.com/watch?v=QdBy02ExhGI&ab_channel=KrishNaik

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Load and standardize the dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
y = pd.Series(cancer['target'], name='target')

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=cancer['feature_names'])

# Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

# Baseline Decision Tree (No PCA)
def train_decision_tree(X_train, X_test, y_train, y_test, title="Decision Tree Results"):
    """Train and evaluate Decision Tree classifier."""
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\n{title}")
    print(f"{'-'*len(title)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return clf

train_decision_tree(X_train, X_test, y_train, y_test, title="Before PCA")

# Apply PCA
def apply_pca(X_train, X_test, n_components=None):
    """Fit PCA and transform train and test data."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return pca, X_train_pca, X_test_pca

n_components = 3
pca, X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=n_components)

# PCA variance insights
print("\nExplained Variance Ratio by Component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")
print(f"Total Variance Explained by {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")


# Visualizations
def plot_explained_variance(pca):
    """Plot the explained variance ratio (scree plot)."""
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='blue')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot - Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_explained_variance(pca)

def plot_component_variance(pca):
    """Plot bar chart of individual component explained variance."""
    plt.figure(figsize=(8, 5))
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    plt.bar(components, pca.explained_variance_ratio_, color='skyblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Individual Component Variance Explained')
    plt.xticks(components)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_component_variance(pca)

def plot_2d_projection(X_pca, y, title='PCA - 2D Projection'):
    """Scatter plot of the first 2 PCA components."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_2d_projection(X_train_pca, y_train, title='2D PCA Projection - Training Set')

# Feature contribution
def inspect_component_loadings(pca, feature_names, top_n=5):
    """Show top contributing features for each component."""
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=feature_names)
    print("\nTop contributing features for each component:")
    for pc in loadings.columns:
        print(f"\n{pc}:")
        print(loadings[pc].abs().sort_values(ascending=False).head(top_n))

inspect_component_loadings(pca, feature_names=cancer.feature_names)

# Decision Tree after PCA
train_decision_tree(X_train_pca, X_test_pca, y_train, y_test, title="After PCA")
