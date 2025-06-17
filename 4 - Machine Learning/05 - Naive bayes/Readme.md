# Naive Bayes Classifier - Iris Dataset

## What is Naive Bayes?

Naive Bayes is a family of probabilistic classifiers based on **Bayes’ Theorem**. The algorithm assumes that the presence of a particular feature in a class is **independent** of the presence of any other feature. It is particularly suited for **text classification**, **spam detection**, and problems where the **independence assumption** holds true or the dataset is high-dimensional.

## Algorithm Type

- **Type**: Supervised Learning
- **Category**: Classification
- **Variant Used**: Gaussian Naive Bayes (assumes normal distribution of features)


## Model Overview

The Gaussian Naive Bayes classifier was used to classify the iris species based on the four features. After splitting the data into training and testing sets, the model was trained and evaluated on unseen data.

## Evaluation Metrics

The model was evaluated using the following metrics:

- **Accuracy Score**  
- **Confusion Matrix**  
- **Precision, Recall, F1-Score** (via Classification Report)

## Advantages

- Fast and simple to implement
- Works well on small datasets
- Performs well with high-dimensional data
- Not sensitive to irrelevant features

## Limitations

- Assumes feature independence (rarely true in real-world data)
- Performs poorly when features are correlated
- Struggles with highly imbalanced datasets

