"""
PCA Analysis with Multiple Models on Digits Dataset
Tests which models benefit from PCA dimensionality reduction
"""
## Implementing PCA on Breast Cancer Data
## https://www.youtube.com/watch?v=QdBy02ExhGI&ab_channel=KrishNaik

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

###################################################################
## Model Training and Evaluation
###################################################################

def evaluate_models(X_train, X_test, y_train, y_test, label=""):
    """Train multiple models and return accuracy scores."""
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Accuracy': accuracy})
        print(f"{name:20} - Accuracy: {accuracy*100:.2f}%")
    
    return pd.DataFrame(results)

def apply_pca_and_visualize(X_train, X_test, n_components):
    """Apply PCA and show variance explained."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"\n--- PCA Analysis (n_components={n_components}) ---")
    print(f"Original shape: {X_train.shape}")
    print(f"PCA shape: {X_train_pca.shape}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Visualizations
    plt.figure(figsize=(12, 5))
    # Scree plot
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='blue', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.grid(True, alpha=0.3)
    
    # Individual variance
    plt.subplot(1, 2, 2)
    plt.bar(range(1, n_components+1), pca.explained_variance_ratio_, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Individual Component Variance')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    return pca, X_train_pca, X_test_pca


###################################################################
## Main Execution
###################################################################
# Load data
X, y = load_digits(return_X_y=True)
X = np.asarray(X)
y = np.asarray(y)
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# Step 1: Models WITHOUT PCA
print("STEP 1: Models WITHOUT PCA")
results_before = evaluate_models(X_train, X_test, y_train, y_test)

# Step 2: Apply PCA
print("STEP 2: Applying PCA (20 components)")
pca, X_train_pca, X_test_pca = apply_pca_and_visualize(X_train, X_test, n_components=20)

# Step 3: Models WITH PCA
print("STEP 3: Models WITH PCA")
results_after = evaluate_models(X_train_pca, X_test_pca, y_train, y_test)

# Step 4: Comparison
print("COMPARISON: PCA Impact on Model Performance")
comparison = pd.DataFrame({
    'Model': results_before['Model'],
    'Without PCA': results_before['Accuracy'],
    'With PCA': results_after['Accuracy'],
    'Difference': results_after['Accuracy'] - results_before['Accuracy']
})
print(comparison.to_string(index=False))

# Identify improved models
improved = comparison[comparison['Difference'] > 0.01]['Model'].tolist()
print(f"\nModels improved with PCA: {improved if improved else 'None'}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x = np.arange(len(comparison))
width = 0.35
plt.bar(x - width/2, comparison['Without PCA'], width, label='Without PCA', alpha=0.8, color='skyblue')
plt.bar(x + width/2, comparison['With PCA'], width, label='With PCA', alpha=0.8, color='orange')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, list(comparison['Model']), rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
colors = ['green' if d > 0 else 'red' for d in comparison['Difference']]
plt.bar(list(comparison['Model']), comparison['Difference'], color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Models')
plt.ylabel('Accuracy Difference')
plt.title('PCA Impact (With - Without)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


