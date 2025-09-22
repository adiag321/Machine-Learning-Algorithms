### Logistic Regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('D:\\OneDrive - Northeastern University\\Jupyter Notebook\\Machine Learning Algorithms')

# Load datasets
train_data = pd.read_csv("./Datasets/Titanic/train_titanic.csv")
test_data = pd.read_csv("./Datasets/Titanic/test_titanic.csv")

print('Shape of training data:', train_data.shape)
print('Shape of testing data:', test_data.shape)

# Separate the independent and target variables
train_x = train_data.drop(['Survived'], axis=1)
train_y = train_data['Survived']
test_x = test_data.drop(['Survived'], axis=1)
test_y = test_data['Survived']

######################################
## Logistic Regression Pipeline
######################################
def logistic_regression(train_x, train_y, test_x, test_y, C, penalty):
    model = LogisticRegression(solver='liblinear',  C=C, penalty=penalty)
    model.fit(train_x, train_y)

    # Model coefficients and intercept
    print('Coefficients of the model:', model.coef_)
    print('\n')
    print('Intercept of the model:', model.intercept_)

    # Probability predictions for training data
    train_prob_df = pd.DataFrame(model.predict_proba(train_x), columns=["Probability of Not Survived", "Probability of Survived"])
    train_prob_df["Actual"] = train_y.values

    # Probability predictions for testing data
    test_prob_df = pd.DataFrame(model.predict_proba(test_x), columns=["Probability of Not Survived", "Probability of Survived"])
    test_prob_df["Actual"] = test_y.values
    return model


# Helper function to predict, compute accuracy, and plot confusion matrix
def evaluate_model(model, X, y, dataset_name=""):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    
    print(f"\n{dataset_name} Accuracy: {(acc)*100:.4f}%")
    print(f"{dataset_name} ROC AUC: {auc:.4f}")
    print(f"{dataset_name} Classification Report:\n", classification_report(y, preds))
    
    # Confusion matrix
    cf = confusion_matrix(y, preds, normalize='true')
    plt.figure(figsize=(5, 4))
    sns.heatmap(cf, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    return preds, probs

# Run predictions and evaluation
model = logistic_regression(train_x, train_y, test_x, test_y, C=1.0, penalty='l2')

# Evaluate
train_preds, train_probs = evaluate_model(model, train_x, train_y, dataset_name = "Train")
test_preds, test_probs = evaluate_model(model, test_x, test_y, dataset_name = "Test")

#print("Training predictions and probabilities:",train_preds, train_probs)
#print("Testing predictions and probabilities:",test_preds, test_probs)

### Save predictions and probabilities in original datasets
train_data["Predicted"] = train_preds
train_data["Probability_Survived"] = train_probs
train_data["Dataset"] = "Train"

test_data["Predicted"] = test_preds
test_data["Probability_Survived"] = test_probs
test_data["Dataset"] = "Test"

# Combine both for easy inspection or export
full_results = pd.concat([train_data, test_data], ignore_index=True)

# View sample
print(full_results.head())

# Optional: Save to CSV if needed
# full_results.to_csv("titanic_predictions_with_probabilities.csv", index=False)

# Save the trained model
# joblib.dump(model, 'logistic_regression_titanic_model.pkl')

