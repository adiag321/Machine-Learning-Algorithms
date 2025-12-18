import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

#####################################
## Load the data
#####################################
os.chdir(r'D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms')
bank_df = pd.read_csv('./Datasets/bank_churn/train.csv', index_col="id")
bank_df = bank_df.drop(['CustomerId', 'Surname'], axis=1)
bank_df = bank_df.sample(frac=1)

X = bank_df.drop(["Exited"],axis=1)
y = bank_df.Exited
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# Identify numerical and categorical columns
cat_col = ['Geography', 'Gender']
num_col = X.columns.difference(cat_col)

#####################################
# Buidlind Data pipelines
#####################################
# Transformers for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Transformers for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

# Combine pipelines using ColumnTransformer
preproc_pipe = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_col),
        ('cat', categorical_transformer, cat_col)
    ],
    remainder="passthrough")

# Selecting the best features
KBest = SelectKBest(chi2, k="all")

# Random Forest Classifier
model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# KBest and model pipeline
train_pipe = Pipeline(
    steps=[
        ("KBest", KBest),
        ("RFmodel", model),
    ])

# Combining the preprocessing and training pipelines
complete_pipe = Pipeline(
    steps=[  
        ("preprocessor", preproc_pipe),
        ("train", train_pipe),
    ])

# running the complete pipeline
complete_pipe.fit(X_train,y_train)

# model accuracy
complete_pipe.score(X_test, y_test)

#####################################
# Save and load the Pipeline
#####################################
loc = r'D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms/4 - Machine Learning/22 - Scikit-learn Pipelines'
os.chdir(loc)
joblib.dump(complete_pipe, "bank_churn_pipeline.joblib")

pipeline = joblib.load("bank_churn_pipeline.joblib")

# New customer data (raw, unprocessed)
new_data = pd.DataFrame({
    'CreditScore': [650],
    'Geography': ['France'],
    'Gender': ['Female'],
    'Age': [42],
    'Tenure': [3],
    'Balance': [120000],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [70000]
})

# Directly predict
prediction = pipeline.predict(new_data)
probability = pipeline.predict_proba(new_data)

print("The predicted class for the new customer is:", prediction[0])
print("The predicted probabilities for the new customer are:", probability[0])
