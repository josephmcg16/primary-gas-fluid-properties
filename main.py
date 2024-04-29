"""Script to fit gas viscosity and density models to reference data and save the best models as .txt files to disk."""

import re
import pandas as pd
import numpy as np

from fit_fluid_properties import (
    fit_gas_density,
    fit_gas_viscosity,
    fit_isentropic_exponent,
)


CONFIG_LIST = [
    {
        "reference_data_directory": "data",
        "reference_data_filename": "H2 Nist Properties 0.95-120 bar.xlsx",
        "reference_fluid": "H2",
    },
    {
        "reference_data_directory": "data",
        "reference_data_filename": "N2 Nist Properties 0.95-120 bar.xlsx",
        "reference_fluid": "N2",
    },
    {
        "reference_data_directory": "data",
        "reference_data_filename": "CH4 Nist Properties 0.95-120 bar.xlsx",
        "reference_fluid": "CH4",
    },
]


def fit_fluid_property_models(
    reference_data_directory: str,
    reference_data_filename: str,
    reference_fluid: str,
    model_poly_degree_range: np.ndarray = np.arange(1, 11, 1),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit the best models to the reference data and return the best models for viscosity, density, and isentropic exponent.

    Parameters
    ----------
    reference_data_directory: str
        The directory containing the reference data.
    reference_data_filename: str
        The filename of the reference data.
    reference_fluid: str
        The reference fluid to be used for the model fitting.
    model_poly_degree_range: np.ndarray
        The range of polynomial degrees to be searched for model selection.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the DataFrames with the results for viscosity, density, and isentropic exponent models.
    """
    config = {
        "reference_data_directory": reference_data_directory,
        "reference_data_filename": reference_data_filename,
        "reference_fluid": reference_fluid,
    }
    viscosity_fits = []
    density_fits = []
    isentropic_exponent_fits = []
    for model_poly_degree in model_poly_degree_range:
        config["model_poly_degree"] = model_poly_degree
        df_viscosity, viscosity_model_functional_form_text = fit_gas_viscosity(config)
        df_density, density_model_functional_form_text = fit_gas_density(config)
        df_isentropic_exponent, isentropic_exponent_model_functional_form_text = (
            fit_isentropic_exponent(config)
        )
        viscosity_fits.append(
            {
                "model_poly_degree": model_poly_degree,
                "df": df_viscosity,
                "Viscosity Relative Error Mean (%)": df_viscosity[
                    "Viscosity Relative Error (%)"
                ].mean(),
                "Viscosity Relative Error Max (%)": np.abs(
                    df_viscosity["Viscosity Relative Error (%)"]
                ).max(),
                "Viscosity Relative Error CI (%)": 1.96
                * df_viscosity["Viscosity Relative Error (%)"].std()
                / np.sqrt(len(df_viscosity)),
                "functional_form_text": viscosity_model_functional_form_text,
            }
        )
        density_fits.append(
            {
                "model_poly_degree": model_poly_degree,
                "df": df_density,
                "Density Relative Error Max (%)": df_density[
                    "Density Relative Error (%)"
                ].max(),
                "Density Relative Error CI (%)": 1.96
                * df_density["Density Relative Error (%)"].std()
                / np.sqrt(len(df_density)),
                "functional_form_text": density_model_functional_form_text,
            }
        )
        isentropic_exponent_fits.append(
            {
                "model_poly_degree": model_poly_degree,
                "df": df_isentropic_exponent,
                "Isentropic Exponent Relative Error Max (%)": df_isentropic_exponent[
                    "Isentropic Exponent Relative Error (%)"
                ].max(),
                "Isentropic Exponent Relative Error CI (%)": 1.96
                * df_isentropic_exponent["Isentropic Exponent Relative Error (%)"].std()
                / np.sqrt(len(df_isentropic_exponent)),
                "functional_form_text": isentropic_exponent_model_functional_form_text,
            }
        )
    return (
        pd.DataFrame(viscosity_fits),
        pd.DataFrame(density_fits),
        pd.DataFrame(isentropic_exponent_fits),
    )


def get_best_model(df: pd.DataFrame, error_column: str) -> pd.DataFrame:
    """
    Return the best model from the DataFrame based on the minimum error.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the model results.
    error_column: str
        The column containing the error to be minimized.

    Returns
    -------
    pd.DataFrame
        The best model based on the minimum error.
    """
    best_model_idx = df[error_column].idxmin()
    return df.loc[best_model_idx]


def save_best_model_to_disk(best_model: pd.DataFrame, model_type: str, filename: str):
    """
    Save the best model to disk as a .txt file and the model DataFrame as a .h5 file.

    Parameters
    ----------
    best_model: pd.DataFrame
        The best model to be saved.
    model_type: str
        The type of model being saved.
    filename: str
        The filename to save the model as.
    """
    with open(f"models/{model_type} Model {filename}.txt", "w", encoding="utf-8") as file:
        file.write(best_model["functional_form_text"])
        file.write(f"\n{best_model.df.describe()}")

    best_model.to_pickle(f"models/{model_type} Model {filename}.pkl")


if __name__ == "__main__":
    for config in CONFIG_LIST:
        viscosity_fits_df, density_fits_df, isentropic_exponent_fits_df = (
            fit_fluid_property_models(
                reference_data_directory=config["reference_data_directory"],
                reference_data_filename=config["reference_data_filename"],
                reference_fluid=config["reference_fluid"],
            )
        )

        viscosity_best_model = get_best_model(
            viscosity_fits_df, "Viscosity Relative Error Max (%)"
        )
        density_best_model = get_best_model(
            density_fits_df, "Density Relative Error Max (%)"
        )
        isentropic_exponent_best_model = get_best_model(
            isentropic_exponent_fits_df, "Isentropic Exponent Relative Error Max (%)"
        )

        save_best_model_to_disk(
            viscosity_best_model,
            "Viscosity",
            config["reference_data_filename"].replace(".xlsx", ""),
        )
        save_best_model_to_disk(
            density_best_model,
            "Density",
            config["reference_data_filename"].replace(".xlsx", ""),
        )
        save_best_model_to_disk(
            isentropic_exponent_best_model,
            "Isentropic Exponent",
            config["reference_data_filename"].replace(".xlsx", ""),
        )
