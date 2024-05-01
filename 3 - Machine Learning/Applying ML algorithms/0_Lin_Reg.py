# importing required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sklearn.metrics


train_data = pd.read_csv("/Users/adityaagarwal/Aditya Ag/Jupyter Notebook/Machine Learning Algorithms/17 - Applying ML algorithms/dataset/train.csv")

test_data = pd.read_csv("/Users/adityaagarwal/Aditya Ag/Jupyter Notebook/Machine Learning Algorithms/17 - Applying ML algorithms/dataset/test.csv")

print(train_data.head())

print('\nShape of training data :',train_data.shape)
print('\nShape of testing data :',test_data.shape)

# target variable - Item_outcome_sales

# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y = train_data['Item_Outlet_Sales']

# seperate the independent and target variable on training data
test_x = test_data.drop(columns=['Item_Outlet_Sales'],axis=1)
test_y = test_data['Item_Outlet_Sales']


## Creating model
# Creating model for Linear Regression
model = LinearRegression()

# fit the model with the training data
model.fit(train_x,train_y)

# coefficeints of the trained model
print('\nCoefficient of model :', model.coef_)

# intercept of the model
print('\nIntercept of model',model.intercept_)

## Predict
# predict the target on the test dataset
predict_train = model.predict(train_x)
print('\nItem_Outlet_Sales on training data',predict_train)

# predict the target on the testing dataset
predict_test = model.predict(test_x)
print('\nItem_Outlet_Sales on test data',predict_test) 


## RMSE 
# Root Mean Squared Error on training dataset
rmse_train = mean_squared_error(train_y,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

# Root Mean Squared Error on testing dataset
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
print('\nRMSE on test dataset : ', rmse_test)


## R_2 score  - (0,1)
# R_2 score on training dataset
r_score_train = sklearn.metrics.r2_score(train_y, predict_train)
print('\nR2 score on Train dataset', r_score_train)

# R_2 score on testing data
r_score_test = sklearn.metrics.r2_score(test_y, predict_test)
print('\nR2 score on Test dataset', r_score_test)


## Mean_absolute_error
# Mean absolute error
mae_train = sklearn.metrics.mean_absolute_error(train_y, predict_train)
print("\nMean absolute Error for training data", mae_train)

mae_test = sklearn.metrics.mean_absolute_error(test_y, predict_test)
print("\nMean absolute Error for Testing data", mae_test)