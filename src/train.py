"""
train.py

Purpose:
--------
Train a machine learning model using the Iris dataset
and save the trained model to disk.

Author: Manish
"""

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model():
    """
    Load dataset, train model and save model.

    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """

    # Load built-in Iris dataset
    iris = load_iris()

    # Features
    X = iris.data

    # Target labels
    y = iris.target

    # Create model object
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, "model.pkl")

    print("Model trained successfully.")
    print("Model saved as model.pkl")

    return model


# Execute only when file is run directly
if __name__ == "__main__":
    train_model()