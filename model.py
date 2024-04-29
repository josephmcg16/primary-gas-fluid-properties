"""Polynomial regression model to predict fluid properties as a function of temperature and pressure."""
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression


class PolynomialRegression(Pipeline):
    """
    Polynomial regression model to predict fluid properties as a function of temperature and pressure.

    Inherits from sklearn's Pipeline class and tailored for fluid properties
    correlations with temperature and pressure as the input features.

    Parameters
    ----------
    degree: int
        The degree of the polynomial features.

    Attributes
    ----------
    degree: int
        The degree of the polynomial features.

    Methods
    -------
    fit(X, y)
        Fit the polynomial regression model.

    predict(X)
        Predict the fluid properties using the polynomial regression model."""

    def __init__(self, degree=2, **kwargs):
        self.degree = degree
        super().__init__(
            [
                ("poly", PolynomialFeatures(degree=degree)),
                ("linear", LinearRegression(**kwargs)),
            ]
        )

    def fit(self, X, y):
        """Fit the polynomial regression model.

        Parameters
        ----------
        X: np.ndarray
            The input data for the model.

        y: np.ndarray
            The output data for the model.

        Returns
        -------
        PolynomialRegression
            The fitted polynomial regression model."""
        return super().fit(X, y)

    def predict(self, X):
        """Predict the fluid properties using the polynomial regression model.

        Parameters
        ----------
        X: np.ndarray
            The input data for the model.

        Returns
        -------
        np.ndarray
            The predicted fluid properties."""
        return super().predict(X)

    def save(self) -> str:
        """Save the polynomial regression model to a text file.

        Returns
        -------
        str
            A string containing the functional form of the polynomial regression model.
        """
        functional_form_string = f'{self.named_steps["linear"].intercept_} + '
        for coeff, powers in zip(
            self.named_steps["linear"].coef_, self.named_steps["poly"].powers_
        ):
            functional_form_string = (
                f"{functional_form_string}{coeff}*T**{powers[0]}*P**{powers[1]} + "
            )
        functional_form_string = functional_form_string[:-3]
        return functional_form_string
