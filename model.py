import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

class PolynomialRegression(Pipeline):
    def __init__(self, degree=2, **kwargs):
        self.degree = degree
        super().__init__(
            [
                ("poly", PolynomialFeatures(degree=degree)),
                ("linear", LinearRegression(**kwargs)),
            ]
        )

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

    def save(self) -> str:
        functional_form_string = f'{self.named_steps["linear"].intercept_} + '
        for coeff, powers in zip(self.named_steps["linear"].coef_, self.named_steps["poly"].powers_):
            functional_form_string = f"{functional_form_string}{coeff}*T**{powers[0]}*P**{powers[1]} + "
        functional_form_string = functional_form_string[:-3]
        return functional_form_string
