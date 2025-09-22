# Naive Bayes Classifier using Iris Dataset
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##############################
# Naive Bayes Model Function
##############################
def gbnaivebayes(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print(f"\nNumber of mislabeled points out of a total {X_test.shape[0]}: {(y_test != y_pred).sum()}")
    print("Model parameters:", gnb.get_params())
    print("Accuracy (in %):", accuracy_score(y_test, y_pred) * 100)

    return gnb, y_pred

# Train model
gnb, y_pred = gbnaivebayes(X_train, y_train, X_test, y_test)

# Prior Probabilities & Feature Stats
print("\nClass Labels:", gnb.classes_)
print("Class Priors (P(class)):", gnb.class_prior_)
print("\nFeature Means (per class):\n", pd.DataFrame(gnb.theta_, columns=X.columns, index=data.target_names))

# Log Probabilities
log_probs = gnb.predict_log_proba(X_test)
print("\nLog probabilities of first 5 test samples:\n", log_probs[:5])

##############################
# Confusion Matrix & Report
##############################
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
print("Gaussian Naive Bayes model accuracy (in %):", accuracy_score(y_test, y_pred) * 100)

##############################
# Cross-Validation
##############################
cv_scores = cross_val_score(GaussianNB(), X, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Mean CV Accuracy (in %):", np.mean(cv_scores) * 100)

##############################
# PCA + Decision Boundary Plot
##############################
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

gnb_2d = GaussianNB()
gnb_2d.fit(X_train_red, y_train_red)

# Create mesh for decision boundary
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = gnb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data.target_names[y])
plt.title("GaussianNB Decision Boundary (PCA reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

##############################
# ROC Curve (OvR - One-vs-Rest)
##############################
# Binarize the output
y_bin = label_binarize(y, classes=[0, 1, 2])
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)

gnb_ovr = OneVsRestClassifier(GaussianNB())
y_score = gnb_ovr.fit(X_train_bin, y_train_bin).predict_proba(X_test_bin)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {data.target_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

