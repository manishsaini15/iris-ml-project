"""
predict.py

Purpose:
--------
Load trained model and perform prediction
on a sample input.

Author: Manish
"""

# Import required libraries
import joblib


def predict_species(sample_data):
    """
    Predict Iris flower species.

    Parameters
    ----------
    sample_data : list
        Input feature values.

    Returns
    -------
    prediction : ndarray
        Predicted class label.
    """

    # Load saved model
    model = joblib.load("model.pkl")

    # Generate prediction
    prediction = model.predict(sample_data)

    return prediction


if __name__ == "__main__":

    # Example flower measurements
    sample = [[5.1, 3.5, 1.4, 0.2]]

    result = predict_species(sample)

    print(f"Prediction: {result}")