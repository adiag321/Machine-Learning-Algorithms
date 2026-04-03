"""
Pytest Test Suite for Simple ML Functions. This file contains comprehensive tests for the ML functions.

Test Categories:
1. Data loading tests
2. Data splitting tests
3. Feature scaling tests
4. Model creation tests
5. Model training tests
6. Prediction tests
7. Evaluation tests
8. Model persistence tests
9. Integration tests
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import tempfile

# Import all functions from our ML module
from ml_model import (
    load_data,
    split_data,
    scale_features,
    create_model,
    train_model,
    predict,
    predict_proba,
    evaluate_model,
    save_model,
    load_model
)


# ============================================================================
# FIXTURES
# Fixtures are reusable test data that can be shared across multiple tests.
# They run automatically before tests that use them.
# ============================================================================

@pytest.fixture
def iris_data():
    """
    Fixture that loads the Iris dataset.
    Any test can use this by adding 'iris_data' as a parameter.
    """
    X, y = load_data()
    return X, y


@pytest.fixture
def train_test_data(iris_data):
    """
    Fixture that provides pre-split train/test data.
    This fixture depends on iris_data fixture.
    """
    X, y = iris_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def scaled_data(train_test_data):
    """
    Fixture that provides scaled train/test data.
    """
    X_train, X_test, y_train, y_test = train_test_data
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


@pytest.fixture
def trained_lr_model(scaled_data):
    """
    Fixture that provides a trained Logistic Regression model.
    Useful for tests that need a pre-trained model.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    model = create_model('logistic_regression', random_state=42)
    trained_model = train_model(model, X_train_scaled, y_train)
    return trained_model


# ============================================================================
# DATA LOADING TESTS
# Test that data loads correctly with expected properties
# ============================================================================

def test_load_data_returns_correct_shape():
    """
    Test that load_data returns data with correct shape.
    Iris dataset has 150 samples and 4 features.
    """
    X, y = load_data()
    
    assert X.shape == (150, 4), "Features should have shape (150, 4)"
    assert y.shape == (150,), "Labels should have shape (150,)"


def test_load_data_returns_numpy_arrays():
    """
    Test that load_data returns numpy arrays (not lists or other types).
    """
    X, y = load_data()
    
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"


def test_load_data_has_valid_values():
    """
    Test that loaded data has reasonable values.
    """
    X, y = load_data()
    
    # Features should be positive (measurements in cm)
    assert X.min() >= 0, "Features should be non-negative"
    
    # Iris has 3 classes: 0, 1, 2
    assert set(y) == {0, 1, 2}, "Labels should be 0, 1, or 2"


# ============================================================================
# DATA SPLITTING TESTS
# Test that data is split correctly into train and test sets
# ============================================================================

def test_split_data_correct_sizes(iris_data):
    """
    Test that split_data creates correct train/test split sizes.
    With test_size=0.2, we expect 80% train and 20% test.
    """
    X, y = iris_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    total_samples = len(X)
    expected_test_size = int(total_samples * 0.2)
    expected_train_size = total_samples - expected_test_size
    
    assert len(X_train) == expected_train_size, f"Expected {expected_train_size} training samples"
    assert len(X_test) == expected_test_size, f"Expected {expected_test_size} test samples"


def test_split_data_maintains_total_count(iris_data):
    """
    Test that no data is lost during splitting.
    Train + test should equal original dataset size.
    """
    X, y = iris_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    assert len(X_train) + len(X_test) == len(X), "No samples should be lost"
    assert len(y_train) + len(y_test) == len(y), "No labels should be lost"


def test_split_data_is_reproducible(iris_data):
    """
    Test that splitting with same random_state gives same results.
    This is important for reproducibility in ML.
    """
    X, y = iris_data
    
    # Split twice with same random state
    X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)
    
    # Results should be identical
    assert np.array_equal(X_train1, X_train2), "Splits should be reproducible"
    assert np.array_equal(y_train1, y_train2), "Splits should be reproducible"


# ============================================================================
# FEATURE SCALING TESTS
# Test that feature scaling works correctly
# ============================================================================

def test_scale_features_standardizes_data(train_test_data):
    """
    Test that scaling produces standardized data.
    After scaling, mean should be ~0 and std should be ~1.
    """
    X_train, X_test, y_train, y_test = train_test_data
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Check standardization (mean ≈ 0, std ≈ 1)
    assert np.abs(X_train_scaled.mean()) < 0.1, "Mean should be close to 0"
    assert np.abs(X_train_scaled.std() - 1.0) < 0.1, "Std should be close to 1"


def test_scale_features_preserves_shape(train_test_data):
    """
    Test that scaling doesn't change data shape.
    """
    X_train, X_test, y_train, y_test = train_test_data
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    assert X_train_scaled.shape == X_train.shape, "Training shape should be preserved"
    assert X_test_scaled.shape == X_test.shape, "Test shape should be preserved"


def test_scale_features_returns_scaler(train_test_data):
    """
    Test that scale_features returns a fitted scaler object.
    """
    X_train, X_test, y_train, y_test = train_test_data
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Check scaler is fitted (has mean_ and scale_ attributes)
    assert hasattr(scaler, 'mean_'), "Scaler should be fitted"
    assert hasattr(scaler, 'scale_'), "Scaler should be fitted"


# ============================================================================
# MODEL CREATION TESTS
# Test that models are created correctly
# ============================================================================

def test_create_logistic_regression_model():
    """
    Test that create_model returns a LogisticRegression model.
    """
    model = create_model('logistic_regression', random_state=42)
    
    assert isinstance(model, LogisticRegression), "Should return LogisticRegression"
    assert model.random_state == 42, "Random state should be set correctly"


def test_create_random_forest_model():
    """
    Test that create_model returns a RandomForestClassifier model.
    """
    model = create_model('random_forest', random_state=42)
    
    assert isinstance(model, RandomForestClassifier), "Should return RandomForestClassifier"
    assert model.random_state == 42, "Random state should be set correctly"


def test_create_model_invalid_type():
    """
    Test that invalid model type raises ValueError.
    This tests error handling.
    """
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model('invalid_model_type')


# ============================================================================
# MODEL TRAINING TESTS
# Test that model training works correctly
# ============================================================================

def test_train_model_succeeds(scaled_data):
    """
    Test that train_model successfully trains a model.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    model = create_model('logistic_regression')
    
    # Train the model
    trained_model = train_model(model, X_train_scaled, y_train)
    
    # Check that model has been fitted (has coef_ attribute after training)
    assert hasattr(trained_model, 'coef_'), "Model should be trained"


def test_train_model_with_empty_data():
    """
    Test that training with empty data raises ValueError.
    This is an important edge case.
    """
    model = create_model('logistic_regression')
    
    with pytest.raises(ValueError, match="Training data cannot be empty"):
        train_model(model, np.array([]), np.array([]))


def test_train_model_with_mismatched_lengths():
    """
    Test that training with different X and y lengths raises error.
    """
    model = create_model('logistic_regression')
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = np.array([0])  # Wrong length!
    
    with pytest.raises(ValueError, match="must have the same length"):
        train_model(model, X, y)


# ============================================================================
# PREDICTION TESTS
# Test that predictions work correctly
# ============================================================================

def test_predict_returns_correct_shape(trained_lr_model, scaled_data):
    """
    Test that predict returns predictions with correct shape.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    predictions = predict(trained_lr_model, X_test_scaled)
    
    assert len(predictions) == len(y_test), "Should have one prediction per test sample"


def test_predict_returns_valid_classes(trained_lr_model, scaled_data):
    """
    Test that predictions are valid class labels (0, 1, or 2 for Iris).
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    predictions = predict(trained_lr_model, X_test_scaled)
    
    # All predictions should be one of the three classes
    assert set(predictions).issubset({0, 1, 2}), "Predictions should be 0, 1, or 2"


def test_predict_with_empty_data(trained_lr_model):
    """
    Test that predicting with empty data raises ValueError.
    """
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        predict(trained_lr_model, np.array([]))


def test_predict_proba_returns_probabilities(trained_lr_model, scaled_data):
    """
    Test that predict_proba returns valid probability distributions.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    probabilities = predict_proba(trained_lr_model, X_test_scaled)
    
    # Check shape: should be (n_samples, n_classes)
    assert probabilities.shape == (len(X_test_scaled), 3), "Should have 3 class probabilities"
    
    # Probabilities for each sample should sum to 1
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
    
    # All probabilities should be between 0 and 1
    assert (probabilities >= 0).all() and (probabilities <= 1).all(), "Probabilities in [0,1]"


# ============================================================================
# EVALUATION TESTS
# Test that model evaluation works correctly
# ============================================================================

def test_evaluate_model_returns_metrics(trained_lr_model, scaled_data):
    """
    Test that evaluate_model returns expected metrics.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    results = evaluate_model(trained_lr_model, X_test_scaled, y_test)
    
    # Check all expected keys are present
    assert 'accuracy' in results, "Results should contain accuracy"
    assert 'classification_report' in results, "Results should contain report"
    assert 'predictions' in results, "Results should contain predictions"


def test_evaluate_model_achieves_good_accuracy(trained_lr_model, scaled_data):
    """
    Test that model achieves reasonable accuracy on Iris dataset.
    Iris is an easy dataset, so we expect >80% accuracy.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    results = evaluate_model(trained_lr_model, X_test_scaled, y_test)
    
    assert results['accuracy'] > 0.8, "Should achieve >80% accuracy on Iris"
    assert results['accuracy'] <= 1.0, "Accuracy should not exceed 1.0"


# ============================================================================
# MODEL PERSISTENCE TESTS
# Test saving and loading models
# ============================================================================

def test_save_and_load_model(trained_lr_model, scaled_data):
    """
    Test that model can be saved and loaded correctly.
    Predictions should be identical before and after.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    
    # Get predictions before saving
    predictions_before = predict(trained_lr_model, X_test_scaled)
    
    # Save model to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Save the model
        save_model(trained_lr_model, scaler, tmp_path)
        
        # Load the model
        loaded_model, loaded_scaler = load_model(tmp_path)
        
        # Get predictions after loading
        predictions_after = predict(loaded_model, X_test_scaled)
        
        # Predictions should be identical
        assert np.array_equal(predictions_before, predictions_after), "Predictions should match"
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# PARAMETRIZED TESTS
# Test multiple scenarios with different parameters in one test
# ============================================================================

@pytest.mark.parametrize("model_type", ['logistic_regression', 'random_forest'])
def test_both_model_types_work(model_type, scaled_data):
    """
    Parametrized test that runs for both model types.
    This test runs twice: once for logistic regression, once for random forest.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scaled_data
    
    # Create model
    model = create_model(model_type, random_state=42)
    
    # Train model
    trained_model = train_model(model, X_train_scaled, y_train)
    
    # Make predictions
    predictions = predict(trained_model, X_test_scaled)
    
    # Evaluate
    results = evaluate_model(trained_model, X_test_scaled, y_test)
    
    # Both models should achieve good accuracy
    assert results['accuracy'] > 0.8, f"{model_type} should achieve >80% accuracy"


@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.4])
def test_different_test_sizes(iris_data, test_size):
    """
    Test that different test sizes work correctly.
    This demonstrates parametrized testing with different values.
    """
    X, y = iris_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    
    expected_test_samples = int(len(X) * test_size)
    
    # Allow ±1 sample difference due to rounding
    assert abs(len(X_test) - expected_test_samples) <= 1, "Test size should match"


@pytest.mark.parametrize("random_state", [0, 42, 99, 123])
def test_different_random_states(iris_data, random_state):
    """
    Test that different random states all work and produce valid models.
    """
    X, y = iris_data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    model = create_model('logistic_regression', random_state=random_state)
    trained_model = train_model(model, X_train_scaled, y_train)
    results = evaluate_model(trained_model, X_test_scaled, y_test)
    
    # Should still achieve reasonable accuracy
    assert results['accuracy'] > 0.7, "Should work with any random state"


# ============================================================================
# INTEGRATION TESTS
# Test the complete end-to-end workflow
# ============================================================================

def test_complete_ml_workflow():
    """
    Integration test for complete ML workflow from start to finish.
    Tests: load → split → scale → create → train → predict → evaluate
    """
    # 1. Load data
    X, y = load_data()
    assert X.shape[0] == 150, "Should load 150 samples"
    
    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == 150, "Should preserve all samples"
    
    # 3. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape, "Scaling preserves shape"
    
    # 4. Create model
    model = create_model('logistic_regression', random_state=42)
    assert model is not None, "Model should be created"
    
    # 5. Train model
    trained_model = train_model(model, X_train_scaled, y_train)
    assert hasattr(trained_model, 'coef_'), "Model should be trained"
    
    # 6. Make predictions
    predictions = predict(trained_model, X_test_scaled)
    assert len(predictions) == len(y_test), "Should predict for all test samples"
    
    # 7. Evaluate
    results = evaluate_model(trained_model, X_test_scaled, y_test)
    assert results['accuracy'] > 0.8, "Should achieve good accuracy"
    
    # 8. Get probabilities
    probabilities = predict_proba(trained_model, X_test_scaled)
    assert probabilities.shape == (len(X_test), 3), "Should have class probabilities"


def test_workflow_comparison_lr_vs_rf():
    """
    Integration test comparing Logistic Regression vs Random Forest.
    """
    # Load and prepare data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train Logistic Regression
    lr_model = create_model('logistic_regression', random_state=42)
    lr_trained = train_model(lr_model, X_train_scaled, y_train)
    lr_results = evaluate_model(lr_trained, X_test_scaled, y_test)
    
    # Train Random Forest
    rf_model = create_model('random_forest', random_state=42)
    rf_trained = train_model(rf_model, X_train_scaled, y_train)
    rf_results = evaluate_model(rf_trained, X_test_scaled, y_test)
    
    # Both should achieve good accuracy
    assert lr_results['accuracy'] > 0.8, "LR should achieve >80%"
    assert rf_results['accuracy'] > 0.8, "RF should achieve >80%"


# ============================================================================
# HOW TO RUN THESE TESTS
# ============================================================================
# Run all tests:              pytest test_ml_model.py -v
# Run specific test:          pytest test_ml_model.py::test_load_data_returns_correct_shape -v
# Run with coverage:          pytest test_ml_model.py --cov=ml_model --cov-report=html -v
# Run tests matching pattern: pytest test_ml_model.py -k "load" -v
# Show print statements:      pytest test_ml_model.py -v -s
# ============================================================================
