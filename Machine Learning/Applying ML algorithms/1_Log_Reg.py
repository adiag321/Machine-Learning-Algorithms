# importing required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn.metrics

train_data = pd.read_csv("/Users/adityaagarwal/Aditya Ag/Jupyter Notebook/Machine Learning Algorithms/17 - Applying ML algorithms/dataset/train_titanic.csv")

test_data = pd.read_csv("/Users/adityaagarwal/Aditya Ag/Jupyter Notebook/Machine Learning Algorithms/17 - Applying ML algorithms/dataset/test_titanic.csv")

print(train_data.sample(5))

# shape of the dataset
print('Shape of training data :',train_data.shape)
print('Shape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived

# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['Survived'],axis=1)
train_y = train_data['Survived']

# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['Survived'],axis=1)
test_y = test_data['Survived']

'''
Create the object of the Logistic Regression model
You can also add other parameters and test your code here
Some parameters are : fit_intercept and penalty
Documentation of sklearn LogisticRegression: 

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

 '''


### Model creation
model = LogisticRegression()

# fit the model with the training data
model.fit(train_x,train_y)

# coefficeints of the trained model
print('Coefficient of model :', model.coef_)

# intercept of the model
print('Intercept of model',model.intercept_)

# predict the target on the train dataset
predict_train = model.predict(train_x)
print('Target on train data',predict_train) 

# predict the target on the test dataset
predict_test = model.predict(test_x)
print('Target on test data',predict_test) 


## Accuracy Metric
# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('accuracy_score on train dataset : ', round((accuracy_train)*100,3), '%')

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('accuracy_score on test dataset : ', round((accuracy_test)*100,3), '%')