"""
test_train.py

Purpose:
--------
Unit test for model training.

Author: Manish
"""

from src.train import train_model


def test_model_training():
    """
    Verify that model training
    returns a valid model object.
    """

    model = train_model()

    assert model is not None