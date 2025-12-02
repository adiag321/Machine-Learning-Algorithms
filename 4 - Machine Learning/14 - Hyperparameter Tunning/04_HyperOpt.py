# Implementing Hyperparameter Tuning Using HyperOpt
'''
!pip install hyperopt
'''
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
import warnings
warnings.filterwarnings("ignore")

os.chdir('D:/OneDrive - Northeastern University/Jupyter Notebook/Machine Learning Algorithms/Datasets')

# load data 
data = pd.read_csv("mobile_dataset/train.csv") 
X = data.drop("price_range", axis=1).values 
y = data.price_range.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

################################################
# HyperOpt Function
################################################
# Define Parameter Space for Optimization
space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    #"max_depth": hp.quniform("max_depth", 1, 15, 1),
     "max_depth": hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

# Defining a Function to Minimize (Objective Function)
'''
Our function to minimize is called hyperparamter_tuning and the classification algorithm to optimize 
its hyperparameter is Random Forest. we use cross-validation to avoid overfitting and then the function 
will return a loss values and it’s status.
'''
def hyperparameter_tuning(params):
        clf = RandomForestClassifier(**params, n_jobs = -1)
        acc = cross_val_score(clf, X_scaled, y, cv=5, 
                              scoring="accuracy", error_score='raise').mean()
        return {"loss": -acc, "status": STATUS_OK}

## Note: hyperopt minimizes the function, that why we add negative sign in the accuracy

def hyperopt_space(objective_function, space_params):
    # Fine Tune the Model
    ## Finally first instantiate the Trial object, fine-tuning the model, and then print the best loss with 
    # its hyperparameters values.

    # Initialize trials object
    trials = Trials()
    best = fmin(
        fn = objective_function,
        space = space_params, 
        algo = tpe.suggest, 
        max_evals = 100, 
        trials = trials
    )
    print("Best Parameters are: {}".format(best))

    ## Analyze results by using trials object
    print("Results are", trials.results)

    ### shows a list of losses
    print("trail losses are", trials.losses())

    ### shows a list of status strings
    print("Status of trails are", trials.statuses())
    
    return best, trials

############################################################
best_params, trials = hyperopt_space(objective_function = hyperparameter_tuning, space_params = space)
