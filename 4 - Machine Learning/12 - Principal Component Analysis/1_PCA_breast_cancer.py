## Implementing PCA on Breast Cancer Data
## https://www.youtube.com/watch?v=QdBy02ExhGI&ab_channel=KrishNaik

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Breast Cancer dataset
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

# 2. Standardize the feature set (PCA requires scaling)
def feature_scaling(df):
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    return scaled_data

df_scaled = pd.DataFrame(feature_scaling(df), columns=cancer['feature_names'])
df_scaled['target'] = cancer['target']

# 3. Prepare feature and target variables
X = df_scaled.drop(columns=['target'])
y = df_scaled['target']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Baseline model: Decision Tree without PCA
def decision_trees(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

print("Accuracy BEFORE PCA:")
decision_trees(X_train, X_test, y_train, y_test)

# 6. Apply PCA to reduce dimensionality
pca = PCA(n_components=3)  # Reduce to 3 principal components or using three columns
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 7. Variance explained by each principal component
print("\nExplained Variance Ratio by Component:")
print(pca.explained_variance_ratio_)
print(f"Total Variance Explained by 3 components: {np.sum(pca.explained_variance_ratio_):.4f}")

# 8. Scree Plot: Cumulative variance explained
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Visualize the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolor='k')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - Training Data Projection')
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Model training after PCA
print("Accuracy AFTER PCA:")
decision_trees(X_train_pca, X_test_pca, y_train, y_test)


