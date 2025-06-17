#K-Nearest Neighbour(K-NN)

'''
* "Always scale your features before applying KNN, as it’s a distance-based algorithm."
* "Fit the scaler only on the training data to avoid data leakage."
* "Split the data first, then apply standardization separately to train and test sets."
* "Use the Elbow Method to find the optimal value of K for your dataset."
* "KNN is a lazy learner — it stores training data and computes distances at prediction time."
* "Avoid using KNN on high-dimensional data unless you've applied dimensionality reduction."
* "KNN can perform poorly on imbalanced datasets — consider resampling techniques."
* "Test different distance metrics (e.g., Euclidean vs Manhattan) to see what works best."
'''
# Importing the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Machine Learning Algorithms\Datasets')

data = pd.read_csv(".\Social_Network_Ads.csv")

##########################
#    Split the data
##########################
X = data.iloc[:, [2, 3]].values  
y = data.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

##########################
#    Feature Scaling
##########################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##########################
# Choosing value of K
# Use Elbow method
##########################
error_rate = []
accuracy_list = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error_rate.append(np.mean(pred_k != y_test))
    accuracy_list.append(accuracy_score(y_test, pred_k))

# Plotting the Elbow Method
plt.figure(figsize=(10,5))
plt.plot(k_range, error_rate, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=8)
plt.title('Elbow Method for Optimal K')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

# Optional: Plot Accuracy vs K
plt.figure(figsize=(10,5))
plt.plot(k_range, accuracy_list, color='green', linestyle='dashed', marker='s',
         markerfacecolor='black', markersize=8)
plt.title('Accuracy vs K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

##########################
#         KNN
##########################
## We will take k = 5 (when error rate starts to level off)
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

##########################
# Evaluate model results
##########################
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix: \n", cm)
# F1 Score
print("F1 Score for the testing data and prediction dataset:", round(f1_score(y_test, y_pred)*100,3),"\n")
# Accuracy
print("Accuracy of the model is", round(accuracy_score(y_test, y_pred)*100, 2),"\n")
# Classification report
print("Classification Report", classification_report(y_pred, y_test))


##########################
# Visualize the results
##########################
def KNN_visulization(x_set, y_set, name):
    from matplotlib.colors import ListedColormap
    x_set, y_set = x_set, y_set
    x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1, stop=x_set[:,0].max()+1, step=0.01), np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
    plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75, cmap= ListedColormap(('red','green')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
        
    plt.title(f'K-NN {name}')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

# Visualize training set
KNN_visulization(X_train, y_train, 'Training')

# Visualize testing set
KNN_visulization(X_test, y_test, 'Testing')

