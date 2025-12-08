# Implementing Hyperparameter Tuning Using scikit-optimize
'''
!pip install scikit-optimize
'''
import numpy as np
import pandas as pd
import os
from regex import B
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import StandardScaler
from skopt.searchcv import BayesSearchCV
from skopt.space import Integer, Real, Categorical 
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import warnings
warnings.filterwarnings("ignore")

os.chdir('D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms/Datasets')

# load data 
data = pd.read_csv("mobile_dataset/train.csv") 
X = data.drop("price_range", axis=1).values 
y = data.price_range.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

###############################################
# (a) First Approach: Use BayesSearchCV to perform hyperparameter optimization for the Random Forest algorithm
# The benefit of BayesSearchCV is that the search procedure is performed automatically, requiring minimal configuration. 
# The class can be used in the same way as the Scikit-Learn (GridSearchCV and RandomizedSearchCV).
###############################################
# Create classifier 
rf_classifier = RandomForestClassifier(n_jobs=-1) 

# Define Search Space
params = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": (1, 9),
    "criterion": ["gini", "entropy"],
}

def BaseSearchcv_tuning(classifier, base_params, x_data, y_data):
    # Define the BayesSearchCV configuration
    search = BayesSearchCV(
        estimator = classifier,
        search_spaces = base_params,
        n_jobs=1,
        cv=5,
        n_iter=30,
        scoring="accuracy",
        verbose=4,
        random_state=42
    )
    # Fine Tune the Model
    search.fit(x_data, y_data)
    # report the best result
    print("The Best score using BaseSearchCV is: ", search.best_score_)
    print("The Best parameters using BaseSearchCV is: ", search.best_params_)
    
    return search, search.best_score_, search.best_params_

search, best_score, best_params = BaseSearchcv_tuning(classifier = rf_classifier, base_params = params, 
                                                      x_data = X_scaled, y_data = y)

##############################################
## (b) Second Approach
# We first define the search space by using the space methods provided by scikit-optimize 
# which are Categorical and Integer.
##############################################
# define the space of hyperparameters to search
search_space = list()
search_space.append(Categorical([100, 200, 300, 400], name='n_estimators'))
search_space.append(Categorical(['gini', 'entropy'], name='criterion'))
search_space.append(Integer(1, 9, name='max_depth'))

# Defining a Function to Minimize (Objective Function)
@use_named_args(search_space)
def evaluate_model(**params):
    # configure the model with specific hyperparameters
    clf = RandomForestClassifier(**params, n_jobs=-1)
    acc = cross_val_score(clf, X_scaled, y, scoring="accuracy").mean()
    return -acc

## Note: The use_named_args() decorator allows your objective function to receive the parameters as keyword arguments. 
# This is particularly convenient when you want to set scikit-learn estimator parameters.

# Fine Tune the Model
result = gp_minimize(
    func = evaluate_model,
    dimensions = search_space,
    n_calls = 30,
    random_state = 42,
    verbose = True,
    n_jobs = 1,
)

# summarizing finding:

print('Best Accuracy: %.3f' % (result.fun)) 
print('Best Parameters: %s' % (result.x))



plot_convergence(result) 