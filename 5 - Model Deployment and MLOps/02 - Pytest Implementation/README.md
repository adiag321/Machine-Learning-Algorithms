# Pytest Tutorial for Machine Learning

## Table of Contents
1. [What is Pytest?](#what-is-pytest)
2. [Why Use Pytest in Machine Learning?](#why-use-pytest-in-machine-learning)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Key Pytest Concepts](#key-pytest-concepts)
6. [Running the Tests](#running-the-tests)
7. [Understanding the Test Suite](#understanding-the-test-suite)
8. [Best Practices](#best-practices)
9. [Advanced Features](#advanced-features)



## What is Pytest?

**Pytest** is a powerful and popular testing framework for Python that makes it easy to write simple and scalable test cases. It is widely used in the software development community and is particularly valuable for testing machine learning projects.

### Key Features of Pytest:

- **Simple Syntax**: Write tests using plain `assert` statements (no need for complex assertion methods)
- **Auto-Discovery**: Automatically finds and runs test files that match the pattern `test_*.py` or `*_test.py`
- **Fixtures**: Reusable setup code that can be shared across multiple tests
- **Parametrization**: Run the same test with different input values
- **Rich Plugin Ecosystem**: Extend functionality with plugins (coverage, parallel execution, etc.)
- **Detailed Failure Reports**: Clear error messages showing exactly what went wrong
- **Integration Support**: Works well with CI/CD pipelines, code coverage tools, and IDEs



## Why Use Pytest in Machine Learning?

Machine learning projects involve complex data pipelines, model training, and evaluation processes that can be error-prone. Testing is crucial for ensuring:

### 1. **Data Integrity**
- Validate that data loading works correctly
- Check feature shapes and data types
- Ensure data preprocessing doesn't introduce errors
- Verify data transformations maintain expected properties

### 2. **Model Correctness**
- Ensure models train without errors
- Validate prediction outputs are in expected ranges
- Check that model performance meets minimum thresholds
- Test different model configurations

### 3. **Reproducibility**
- Verify that models produce consistent results with the same random seed
- Test model serialization (save/load) works correctly
- Ensure preprocessing steps are deterministic

### 4. **Error Handling**
- Test how your code handles invalid inputs
- Verify appropriate errors are raised for edge cases
- Ensure graceful degradation when something goes wrong

### 5. **Regression Prevention**
- Catch bugs early before they make it to production
- Ensure code changes don't break existing functionality
- Maintain code quality as your project grows

### 6. **Documentation**
- Tests serve as living documentation of how your code should work
- Examples of usage for different scenarios
- Clear expectations for inputs and outputs

### 7. **Refactoring Confidence**
- Safely refactor code knowing tests will catch breaking changes
- Improve code quality without fear
- Speed up development in the long run



## Installation

Install pytest and required dependencies:

```bash
# Install pytest
pip install pytest

# Install pytest with commonly used plugins
pip install pytest pytest-cov pytest-xdist

# Install project dependencies
pip install numpy scikit-learn joblib
```

**Package Explanation:**
- `pytest`: Core testing framework
- `pytest-cov`: Code coverage reports (shows which lines are tested)
- `pytest-xdist`: Run tests in parallel for faster execution



## Project Structure

```
15 - Pytest Tutorial/
│
├── ml_model.py              # Main ML code (model implementation)
├── test_ml_model.py         # Test suite for the ML code
└── README.md                # This file (documentation)
```



## Key Pytest Concepts

### 1. **Test Functions**

Test functions must start with `test_` to be auto-discovered:

```python
def test_model_initialization():
    """Test that model initializes correctly."""
    model = IrisClassifier()
    assert model.is_trained == False
```

### 2. **Fixtures**

Fixtures are reusable setup code. They help avoid duplication and ensure consistent test data:

```python
@pytest.fixture
def iris_data():
    """Load Iris dataset for testing."""
    iris = load_iris()
    return iris.data, iris.target

def test_data_loading(iris_data):
    """Test uses the iris_data fixture."""
    X, y = iris_data
    assert X.shape == (150, 4)
```

**Benefits:**
- Avoid code duplication
- Consistent test setup
- Automatic setup and teardown
- Can depend on other fixtures

### 3. **Assert Statements**

Pytest uses simple `assert` statements with clear failure messages:

```python
assert model.is_trained == True  # Simple boolean check
assert accuracy > 0.8            # Comparison
assert len(predictions) == 30    # Length check
assert X.shape == (150, 4)       # Tuple comparison
```

### 4. **Parametrized Tests**

Run the same test with different inputs:

```python
@pytest.mark.parametrize("model_type", ['logistic_regression', 'random_forest'])
def test_model_types(model_type):
    """This test runs twice, once for each model type."""
    classifier = IrisClassifier(model_type=model_type)
    assert classifier.model is not None
```

### 5. **Exception Testing**

Test that your code raises appropriate errors:

```python
def test_invalid_input():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="Training data cannot be empty"):
        classifier.train([], [])
```

### 6. **Test Organization**

Tests are organized into categories:
- **Unit Tests**: Test individual functions/methods in isolation
- **Integration Tests**: Test complete workflows
- **Edge Case Tests**: Test boundary conditions and error handling



## Running the Tests

### Basic Commands

```bash
# Run all tests in the file
pytest test_ml_model.py

# Run with verbose output (shows each test name)
pytest test_ml_model.py -v

# Run a specific test function
pytest test_ml_model.py::test_initialization_logistic_regression -v

# Run tests matching a pattern
pytest test_ml_model.py -k "initialization" -v

# Show print statements during test execution
pytest test_ml_model.py -v -s
```

### Advanced Commands

```bash
# Run with code coverage report
pytest test_ml_model.py --cov=ml_model --cov-report=html -v

# Run tests in parallel (faster)
pytest test_ml_model.py -n auto

# Stop at first failure
pytest test_ml_model.py -x

# Run only failed tests from last run
pytest test_ml_model.py --lf

# Show detailed output for failures
pytest test_ml_model.py -vv
```

### Expected Output

When you run the tests, you should see output like:

```
======================== test session starts ========================
platform win32 -- Python 3.x.x, pytest-x.x.x
collected 30 items

test_ml_model.py::test_initialization_logistic_regression PASSED  [ 3%]
test_ml_model.py::test_initialization_random_forest PASSED        [ 6%]
test_ml_model.py::test_initialization_invalid_model_type PASSED   [10%]
...
test_ml_model.py::test_complete_workflow PASSED                   [100%]

======================== 30 passed in 2.34s =========================
```



## Understanding the Test Suite

Our test suite (`test_ml_model.py`) is organized into the following sections:

### 1. **Fixtures** (Lines 20-70)
- `iris_data`: Loads the Iris dataset
- `split_data`: Provides train-test split
- `trained_lr_classifier`: Pre-trained Logistic Regression model
- `trained_rf_classifier`: Pre-trained Random Forest model

### 2. **Initialization Tests** (Lines 72-105)
Tests that models initialize correctly with different parameters.

### 3. **Data Loading Tests** (Lines 107-140)
Validates that data is loaded with correct shapes and value ranges.

### 4. **Preprocessing Tests** (Lines 142-175)
Tests data scaling and normalization.

### 5. **Training Tests** (Lines 177-210)
Ensures models can be trained and handles invalid inputs.

### 6. **Prediction Tests** (Lines 212-265)
Tests prediction functionality and probability outputs.

### 7. **Evaluation Tests** (Lines 267-310)
Validates model evaluation metrics.

### 8. **Model Persistence Tests** (Lines 312-365)
Tests saving and loading trained models.

### 9. **Parametrized Tests** (Lines 367-405)
Tests multiple scenarios with different parameters.

### 10. **Integration Tests** (Lines 407-450)
Tests complete end-to-end workflows.



## Best Practices

### For ML Testing

1. **Test Data Shapes**
   ```python
   assert X_train.shape == (120, 4)
   assert predictions.shape == (30,)
   ```

2. **Test Value Ranges**
   ```python
   assert (probabilities >= 0).all() and (probabilities <= 1).all()
   assert accuracy > 0.0 and accuracy <= 1.0
   ```

3. **Test Determinism** 
   ```python
   # Use fixed random seeds
   classifier = IrisClassifier(random_state=42)
   ```

4. **Test Edge Cases**
   ```python
   # Empty data, mismatched shapes, invalid inputs
   with pytest.raises(ValueError):
       classifier.train([], [])
   ```

5. **Use Fixtures for Expensive Operations**
   ```python
   @pytest.fixture(scope="module")  # Runs once per module
   def trained_model():
       # Expensive training here
       return model
   ```

### General Testing Principles

- **Keep tests independent**: Each test should be able to run in isolation
- **Use descriptive names**: `test_predict_with_empty_data` is better than `test_predict_1`
- **Test one thing per test**: Easier to debug when something fails
- **Use fixtures to avoid duplication**: DRY principle
- **Add docstrings**: Explain what the test is checking
- **Test error cases**: Don't just test the happy path
- **Keep tests fast**: Slow tests won't get run frequently



## Advanced Features

### 1. **Test Coverage**

Code coverage shows which lines of code are tested:

```bash
pytest test_ml_model.py --cov=ml_model --cov-report=html
```

This generates an HTML report in `htmlcov/index.html` showing:
- Which lines are covered by tests (green)
- Which lines are not tested (red)
- Overall coverage percentage

**Goal**: Aim for >80% coverage in ML projects.

### 2. **Markers**

Mark tests for selective execution:

```python
@pytest.mark.slow
def test_long_running_training():
    # Time-consuming test
    pass

@pytest.mark.gpu
def test_gpu_training():
    # Requires GPU
    pass
```

Run only marked tests:
```bash
pytest -m slow       # Run only slow tests
pytest -m "not slow" # Skip slow tests
```

### 3. **Fixtures with Scopes**

Control how often fixtures run:

```python
@pytest.fixture(scope="function")  # Default: run before each test
@pytest.fixture(scope="class")     # Run once per test class
@pytest.fixture(scope="module")    # Run once per module
@pytest.fixture(scope="session")   # Run once per test session
```

### 4. **Mocking**

Replace expensive operations with mock objects:

```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('ml_model.load_iris') as mock_load:
        mock_load.return_value = (Mock_X, Mock_y)
        # Test without actually loading data
```

### 5. **Approximate Comparisons**

For floating-point comparisons:

```python
assert accuracy == pytest.approx(0.95, abs=0.01)  # Within ±0.01
assert np.allclose(predictions, expected, rtol=1e-5)
```

### 6. **Temporary Files**

For testing model persistence:

```python
import tempfile

def test_save_model():
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
        model.save(tmp.name)
        # File automatically cleaned up
```



## Real-World ML Testing Scenarios

### Scenario 1: Testing Data Preprocessing Pipeline
```python
def test_preprocessing_pipeline():
    """Ensure preprocessing maintains data integrity."""
    raw_data = load_raw_data()
    processed_data = preprocess(raw_data)
    
    # Check no NaN values introduced
    assert not processed_data.isna().any().any()
    
    # Check number of samples preserved
    assert len(processed_data) == len(raw_data)
    
    # Check feature ranges
    assert processed_data['age'].between(0, 120).all()
```

### Scenario 2: Testing Model Performance Thresholds
```python
def test_model_meets_performance_threshold():
    """Ensure model meets minimum accuracy requirement."""
    model = train_model(X_train, y_train)
    accuracy = evaluate(model, X_test, y_test)
    
    # Regression test: model should be at least as good as baseline
    assert accuracy >= 0.85, f"Model accuracy {accuracy} below threshold"
```

### Scenario 3: Testing Model Fairness
```python
def test_model_fairness():
    """Check that model doesn't have significant bias."""
    predictions_group_a = model.predict(X_group_a)
    predictions_group_b = model.predict(X_group_b)
    
    accuracy_a = accuracy_score(y_group_a, predictions_group_a)
    accuracy_b = accuracy_score(y_group_b, predictions_group_b)
    
    # Accuracies should be similar (within 5%)
    assert abs(accuracy_a - accuracy_b) < 0.05
```



## Common Pytest Issues and Solutions

| Issue | Solution |
|-|-|
| Tests not discovered | Ensure files/functions start with `test_` |
| Import errors | Add `__init__.py` or adjust `PYTHONPATH` |
| Fixtures not found | Check fixture scope and availability |
| Slow tests | Use `pytest-xdist` for parallel execution |
| Flaky tests | Use fixed random seeds, avoid time-dependent code |



## Resources for Further Learning

- **Official Pytest Documentation**: [docs.pytest.org](https://docs.pytest.org/)
- **Pytest Best Practices**: [pytest.org/en/latest/goodpractices.html](https://docs.pytest.org/en/latest/goodpractices.html)
- **Testing ML Models**: [madewithml.com/courses/mlops/testing](https://madewithml.com/courses/mlops/testing/)
- **Python Testing with pytest (Book)** by Brian Okken
- **YouTube**: Search for "pytest tutorial" for video guides



## Conclusion

Testing is not just a best practice—it's essential for building reliable machine learning systems. Pytest makes testing:
- **Easy**: Simple syntax with powerful features
- **Fast**: Quick test execution and parallel support
- **Maintainable**: Clear structure and reusable fixtures
- **Comprehensive**: Full coverage of your ML pipeline

Start testing your ML code today to catch bugs early, ensure reproducibility, and build confidence in your models! 🚀



## Quick Reference Card

```bash
# Essential Commands
pytest test_ml_model.py              # Run all tests
pytest test_ml_model.py -v           # Verbose output
pytest test_ml_model.py -k "name"    # Run tests matching name
pytest test_ml_model.py --cov        # With coverage
pytest test_ml_model.py -x           # Stop at first failure
pytest test_ml_model.py --lf         # Run last failed tests

# In Python Tests
assert condition                      # Basic assertion
assert x == pytest.approx(y)         # Approximate comparison
pytest.raises(Exception)             # Test exceptions
@pytest.fixture                      # Reusable test data
@pytest.mark.parametrize(...)        # Multiple test cases
```
