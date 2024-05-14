"""Fit fluid property models for a real gas. Temperature and pressure are the input features."""

from matplotlib.pylab import f
import numpy as np
import pandas as pd

from model import PolynomialRegression
from loader import Loader
from scraper.fluid_constants import SpecificGasConstants, SutherlandsConstants


def fit_gas_density(config: dict) -> tuple[pd.DataFrame, str]:
    """
    Fit a model to the compressibility factor data and calculate the density of the gas.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Must contain the following keys:
            - "reference_data_path": Path to the reference data.
            - "reference_fluid": Name of the sheet in the reference data.
            - "model_poly_degree": Degree of the polynomial regression model.

    Returns
    -------
    tuple[pd.DataFrame, str]
        A tuple containing the DataFrame with the results and the functional form of the compressibility factor model.
    """

    config["reference_data_path"] = (
        config["reference_data_directory"] + "/" + config["reference_data_filename"]
    )
    loader = Loader(config["reference_data_path"])
    X, y = loader("Compressibility")

    temperature = X[:, 0]  # C
    pressure = X[:, 1]  # bar
    compressibility = y
    density = loader("Density (kg/m3)")[1]  # kg/m3

    # COMPRESSIBILITY FACTOR MODEL
    model = PolynomialRegression(degree=config["model_poly_degree"])
    model.fit(X, y)
    compressibility_functional_form = model.save()
    compressibility_hat = model.predict(X)

    # IDEAL GAS DENSITY MODEL
    density_ideal = (
        pressure
        * 10**5
        / (
            SpecificGasConstants.fluid_name_dict[config["reference_fluid"]]
            * (temperature + 273.15)
        )
    )

    # FULL DENSITY MODEL
    density_hat = density_ideal / compressibility_hat
    density_error = density_hat - density

    # SAVE DATA
    functional_form_text = f'P * 10**5 / {SpecificGasConstants.fluid_name_dict[config["reference_fluid"]]} / (T + 273.15) / ({compressibility_functional_form})'

    df = pd.DataFrame(
        {
            "Temperature (C)": temperature,
            "Pressure (bar)": pressure,
            "Compressibility": compressibility,
            "Compressibility_hat": compressibility_hat,
            "Density (kg/m3)": density,
            "Density_hat (kg/m3)": density_hat,
            "Density Error (kg/m3)": density_error,
            "Density Relative Error (%)": 100 * density_error / density,
        }
    )
    return df, functional_form_text


def fit_gas_viscosity(config: dict) -> tuple[pd.DataFrame, str]:
    """
    Fit a model to the gas viscosity data and calculate the viscosity of the gas.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Must contain the following keys:
            - "reference_data_path": Path to the reference data.
            - "reference_fluid": Name of the sheet in the reference data.
            - "model_poly_degree": Degree of the polynomial regression model.

    Returns
    -------
    tuple[pd.DataFrame, str]
        A tuple containing the DataFrame with the results and the functional form of the viscosity model.
    """

    config["reference_data_path"] = (
        config["reference_data_directory"] + "/" + config["reference_data_filename"]
    )
    loader = Loader(config["reference_data_path"])
    X, y = loader("Viscosity (Pa*s)")

    temperature = X[:, 0]  # C
    pressure = X[:, 1]  # bar
    dynamic_viscosity = y  # Pa.s

    mu_0 = SutherlandsConstants.fluid_name_dict[config["reference_fluid"]]["mu_0"]
    T_0 = SutherlandsConstants.fluid_name_dict[config["reference_fluid"]]["T_0"]
    S = SutherlandsConstants.fluid_name_dict[config["reference_fluid"]]["S"]
    mu_ideal = (
        mu_0
        * (T_0 + S)
        / (temperature + 273.15 + S)
        * ((temperature + 273.15) / T_0) ** (3 / 2)
    )

    # FULL VISCOSITY MODEL
    Y = dynamic_viscosity / mu_ideal
    model = PolynomialRegression(degree=config["model_poly_degree"])
    model.fit(X, Y)
    viscosity_model_functional_form = model.save()
    mu_hat = model.predict(X) * mu_ideal

    # SAVE DATA
    functional_form_text = f"{mu_0} * ((T + 273.15) / {T_0})**(3/2) * ({T_0} + {S}) / (T + 273.15 + {S}) * ({viscosity_model_functional_form})"

    mu_error = mu_hat - dynamic_viscosity
    df = pd.DataFrame(
        {
            "Temperature (C)": temperature,
            "Pressure (bar)": pressure,
            "Viscosity (Pa*s)": dynamic_viscosity,
            "Viscosity_hat (Pa*s)": mu_hat,
            "Viscosity Error (Pa*s)": mu_error,
            "Viscosity Relative Error (%)": 100 * mu_error / dynamic_viscosity,
        }
    )

    return df, functional_form_text


def fit_isentropic_exponent(config: dict) -> tuple[pd.DataFrame, str]:
    """
    Fit a model to the isentropic exponent data and calculate the isentropic exponent of the gas.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Must contain the following keys:
            - "reference_data_path": Path to the reference data.
            - "reference_fluid": Name of the sheet in the reference data.
            - "model_poly_degree": Degree of the polynomial regression model.

    Returns
    -------
    tuple[pd.DataFrame, str]
    """
    config["reference_data_path"] = (
        config["reference_data_directory"] + "/" + config["reference_data_filename"]
    )
    loader = Loader(config["reference_data_path"])
    X, y = loader.load_data("Isentropic Exponent")

    temperature = X[:, 0]  # C
    pressure = X[:, 1]  # bar
    isentropic_exponent = y

    # ISOTROPIC EXPONENT MODEL
    model = PolynomialRegression(degree=config["model_poly_degree"])
    model.fit(X, y)
    isentropic_exponent_functional_form = model.save()
    isentropic_exponent_hat = model.predict(X)

    # SAVE DATA
    functional_form_text = isentropic_exponent_functional_form

    df = pd.DataFrame(
        {
            "Temperature (C)": temperature,
            "Pressure (bar)": pressure,
            "Isentropic Exponent": isentropic_exponent,
            "Isentropic Exponent_hat": isentropic_exponent_hat,
            "Isentropic Exponent Error": isentropic_exponent_hat - isentropic_exponent,
            "Isentropic Exponent Relative Error (%)": 100
            * (isentropic_exponent_hat - isentropic_exponent)
            / isentropic_exponent,
        }
    )

    return df, functional_form_text


def fit_critical_flow_factor(config: dict) -> tuple[pd.DataFrame, str]:
    """
    Fit a model to the critical flow factor data and calculate the critical flow factor of the gas.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Must contain the following keys:
            - "reference_data_path": Path to the reference data.
            - "reference_fluid": Name of the sheet in the reference data.
            - "model_poly_degree": Degree of the polynomial regression model.

    Returns
    -------
    tuple[pd.DataFrame, str]
    """
    config["reference_data_path"] = (
        config["reference_data_directory"] + "/" + config["reference_data_filename"]
    )
    loader = Loader(config["reference_data_path"])
    X, y = loader("Critical Flow Factor")

    temperature = X[:, 0]  # C
    pressure = X[:, 1]  # bar
    critical_flow_factor = y

    # ISOTROPIC EXPONENT MODEL
    df_isentropic_exponent, isentropic_exponent_functional_form_test = (
        fit_isentropic_exponent(config)
    )
    isentropic_exponent_hat = df_isentropic_exponent["Isentropic Exponent_hat"].values

    # CRITICAL FLOW FACTOR MODEL
    C_ideal = np.sqrt(
        isentropic_exponent_hat
        * (2 / (isentropic_exponent_hat + 1))
        ** ((isentropic_exponent_hat + 1) / (isentropic_exponent_hat - 1))
    )

    Y = critical_flow_factor / C_ideal

    model = PolynomialRegression(degree=config["model_poly_degree"])
    model.fit(X, Y)
    critical_flow_factor_functional_form = model.save()
    critical_flow_factor_hat = model.predict(X) * C_ideal

    # SAVE DATA
    functional_form_text = f"{isentropic_exponent_functional_form_test} * (2 / ({isentropic_exponent_functional_form_test} + 1))**(({isentropic_exponent_functional_form_test} + 1) / ({isentropic_exponent_functional_form_test} - 1)) * ({critical_flow_factor_functional_form})"

    df = pd.DataFrame(
        {
            "Temperature (C)": temperature,
            "Pressure (bar)": pressure,
            "Isentropic Exponent_hat": isentropic_exponent_hat,
            "Critical Flow Factor": critical_flow_factor,
            "Critical Flow Factor_hat": critical_flow_factor_hat,
            "Critical Flow Factor Error": critical_flow_factor_hat
            - critical_flow_factor,
            "Critical Flow Factor Relative Error (%)": 100
            * (critical_flow_factor_hat - critical_flow_factor)
            / critical_flow_factor,
        }
    )

    return df, functional_form_text
