# Hyper Parameter Tuning
Hyperparameters are different parameter values that are used to control the learning process and have a significant effect on the performance of machine learning models. Example of hyperparameters in the Random Forest algorithm is the number of estimators (n_estimators), maximum depth(max_depth), and criterion. These parameters are tunable and can directly affect how well model trains.
Then hyperparameter optimization is a process of finding the right combination of hyperparameter values in order to achieve maximum performance on the data in a reasonable amount of time. It plays a vital role in the prediction accuracy of a machine learning algorithm.

## Common Hyperparameter Tuning Techniques

### 1. Grid Search
This is a widely used traditional method that performing hyperparameter tuning in order to determine the optimal values for a given model. The Grid search works by trying every possible combination of parameters you want to try in your model, this means it will take a lot of time to perform the entire search which can get very computationally expensive.

### 2. Random Search
This method works differently where random combinations of the values of the hyperparameters are used to find the best solution for the built model. The drawback of Random Search is sometimes could miss important points(values) in the search space.

## Alternative Hyperparameter Optimization techniques

### 1. Hyperopt
Hyperopt uses a form of Bayesian optimization for parameter tuning that allows you to get the best parameters for a given model. It can optimize a model with hundreds of parameters on a large scale.

#### Features of Hyperopt
##### (a) Search Space
The hyperopt have different functions to specify ranges for input parameters, these are stochastic search spaces. The most common options for a search space to choose are :

* hp.choice(label, options) — This can be used for categorical parameters, it returns one of the options, which should be a list or tuple.Example: hp.choice(“criterion”, [“gini”,”entropy”,])
* hp.randint(label, upper) — This can be used for Integer parameters, it returns a random integer in the range (0, upper).Example: hp.randint(“max_features”,50)
* hp.uniform(label, low, high) — It returns a value uniformly between low and highExample: hp.uniform("max_leaf_nodes",1,10)

Other option you can use are:

* hp.normal(label, mu, sigma) — This returns a real value that’s normally-distributed with mean mu and standard deviation sigma
* hp.qnormal(label, mu, sigma, q) — This returns a value like round(normal(mu, sigma) / q) * q
* hp.lognormal(label, mu, sigma) — This returns a value drawn according to exp(normal(mu, sigma))
* hp.qlognormal(label, mu, sigma, q) — This returns a value like round(exp(normal(mu, sigma)) / q) * q

You can learn more search space options <a href = 'https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions'>here</a>.

##### (b) Objective Function
This is a function to minimize that receives hyperparameters values as input from the search space and returns the loss. This means during the optimization process, we train the model with selected hyperparameters values and predict the target feature and then evaluate the prediction error and give it back to the optimizer. The optimizer will decide which values to check and iterate again.

##### (c) fmin
The fmin function is the optimization function that iterates on different sets of algorithms and their hyperparameters and then minimizes the objective function. the fmin takes 5 inputs which are:-

The objective function to minimize
The defined search space
The search algorithm to use such as Random search, TPE (Tree Parzen Estimators), and Adaptive TPE.
NB: hyperopt.rand.suggest and hyperopt.tpe.suggest provides logic for a sequential search of the hyperparameter space.
Maximum number of evaluations.
The trials object (optional).

##### (d) Trial Object
The Trials object is used to keep All hyperparameters, loss, and other information, this means you can access them after running optimization. Also, trials can help you to save important information and later load and then resume the optimization process. (you will learn more in the practical example).

### How to use Hyperopt?
1. Initialize the space over which to search.
2. Define the objective function.
3. Select the search algorithm to use.
4. Run hyperopt function.
5. Analyze the evaluation outputs stored in the trials object.

### 2. Scikit Optimize
It implements several methods for sequential model-based optimization. The library is very easy to use and provides a general toolkit for Bayesian optimization that can be used for hyperparameter tuning.

#### Features of scikit-optimize
##### (a) Space
The most common options for a search space to choose are :

* Real — This is a search space dimension that can take on any real value. You need to define the lower bound and upper bound and both are inclusive.
Example: Real(low=0.2, high=0.9, name="min_samples_leaf")
* Integer — This is a search space dimension that can take on integer values.
Example: Integer(low=3, high=25, name="max_features")
* Categorical — This is a search space dimension that can take on categorical values.
Example: Categorical(["gini","entropy"],name="criterion")

<b>Note: In each search space you have to define the hyperparameter name to optimize by using the name argument.</b>

##### (b) BayesSearchCV
BayesSearchCV class provides an interface similar to GridSearchCV or RandomizedSearchCV but it performs Bayesian optimization over hyperparameters. BayesSearchCV implements a “fit” and a “score” method and other common methods (predict(),predict_proba(), decision_function(), transform() and inverse_transform() ) if they are implemented in the estimator used.
<i><b>In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.</b></i>

##### (c) Objective Function
This is a function that will be called by the search procedure, it receives hyperparameters values as input from the search space and returns the loss (the lower the better). This means during the optimization process, we train the model with selected hyperparameters values and predict the target feature and then evaluate the prediction error and give it back to the optimizer. The optimizer will decide which values to check and iterate again.

##### (d) Optimizer
This is the function that performs the Bayesian Hyperparameter Optimization process. The optimization function iterates at each model and the search space to optimize and then minimizes the objective function.
There are different optimization functions provided by the scikit-optimize library such as:-

* dummy_minimize — Random search by uniform sampling within the given bounds.
* forest_minimize — Sequential optimization using decision trees.
* gbrt_minimize — Sequential optimization using gradient boosted trees.
* gp_minimize — Bayesian optimization using Gaussian Processes.


### 3. Optuna

