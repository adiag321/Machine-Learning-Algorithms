"""
BentoML + scikit-learn Iris deployment example.

This file is intentionally self-contained:
1. It trains a small model on scikit-learn's built-in Iris dataset.
2. It saves the trained model to the local BentoML model store.
3. It defines a BentoML service that can serve predictions over HTTP.

Setup:
    pip install bentoml scikit-learn numpy

Train and save the model:
    python bentoml_sklearn_iris.py train

Run a quick local prediction without starting a server:
    python bentoml_sklearn_iris.py predict 5.1 3.5 1.4 0.2

Serve the model locally with BentoML:
    bentoml serve bentoml_sklearn_iris.py:IrisClassifier

Then open http://localhost:3000 or call the /predict endpoint.
"""

from __future__ import annotations

import argparse
from typing import Any

import bentoml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAME = "iris_logistic_regression"
MODEL_TAG = f"{MODEL_NAME}:latest"
RANDOM_STATE = 42


FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


SAMPLE_INPUT = [5.1, 3.5, 1.4, 0.2]


def train_and_save_model() -> None:
    """Train a simple classifier and save it to BentoML's model store."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    saved_model = bentoml.sklearn.save_model(
        MODEL_NAME,
        model,
        metadata={
            "dataset": "sklearn.datasets.load_iris",
            "test_accuracy": float(test_accuracy),
            "target_names": iris.target_names.tolist(),
            "feature_names": FEATURE_NAMES,
        },
    )

    print("Model training complete.")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Saved BentoML model: {saved_model.tag}")
    print(f"Serve with: bentoml serve {__file__}:IrisClassifier")


def load_latest_model() -> Any:
    """Load the latest saved model from the local BentoML model store."""
    try:
        return bentoml.sklearn.load_model(MODEL_TAG)
    except bentoml.exceptions.NotFound as error:
        raise SystemExit(
            f"Model '{MODEL_TAG}' was not found. Run this first:\n"
            "    python bentoml_sklearn_iris.py train"
        ) from error


def predict_species(features: list[float]) -> dict[str, Any]:
    """Return a human-readable Iris prediction for one row of four features."""
    if len(features) != 4:
        raise ValueError(
            "Expected exactly 4 values: sepal_length, sepal_width, "
            "petal_length, petal_width."
        )

    iris = load_iris()
    model = load_latest_model()
    input_array = np.array([features], dtype=float)

    predicted_class = int(model.predict(input_array)[0])
    probabilities = model.predict_proba(input_array)[0]

    return {
        "input": dict(zip(FEATURE_NAMES, features)),
        "predicted_class": predicted_class,
        "predicted_species": str(iris.target_names[predicted_class]),
        "class_probabilities": {
            str(species): round(float(probability), 4)
            for species, probability in zip(iris.target_names, probabilities)
        },
    }


@bentoml.service(resources={"cpu": "1"}, traffic={"timeout": 10})
class IrisClassifier:
    """BentoML service for serving the saved Iris classifier."""

    def __init__(self) -> None:
        self.model = load_latest_model()
        self.target_names = load_iris().target_names

    @bentoml.api
    def predict(self, features: list[float]) -> dict[str, Any]:
        """
        Predict one Iris species from four numeric measurements.

        Example request body:
            {"features": [5.1, 3.5, 1.4, 0.2]}
        """
        if len(features) != 4:
            raise ValueError(
                "Expected exactly 4 values: sepal_length, sepal_width, "
                "petal_length, petal_width."
            )

        input_array = np.array([features], dtype=float)
        predicted_class = int(self.model.predict(input_array)[0])
        probabilities = self.model.predict_proba(input_array)[0]

        return {
            "input": dict(zip(FEATURE_NAMES, features)),
            "predicted_class": predicted_class,
            "predicted_species": str(self.target_names[predicted_class]),
            "class_probabilities": {
                str(species): round(float(probability), 4)
                for species, probability in zip(self.target_names, probabilities)
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and serve a scikit-learn Iris model with BentoML."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train and save the Iris classifier.")

    predict_parser = subparsers.add_parser(
        "predict", help="Run one local prediction using the saved model."
    )
    predict_parser.add_argument(
        "features",
        nargs="*",
        type=float,
        default=SAMPLE_INPUT,
        help="Four Iris measurements. Defaults to a setosa sample.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        train_and_save_model()
        return

    if args.command == "predict":
        result = predict_species(args.features)
        print(result)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
