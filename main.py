"""Script to fit gas viscosity and density models to reference data and save the best models as .txt files to disk."""
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm

from fit_fluid_properties import fit_gas_density, fit_gas_viscosity


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

for config in CONFIG_LIST:
    viscosity_fits = []
    density_fits = []
    for model_poly_degree in tqdm(np.arange(1, 11, 1), desc="Model Poly Degree"):
        config["model_poly_degree"] = model_poly_degree
        df_viscosity, viscosity_model_functional_form_text = fit_gas_viscosity(config)
        df_density, density_model_functional_form_text = fit_gas_density(config)

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

    viscosity_fits_df = pd.DataFrame(viscosity_fits)
    density_fits_df = pd.DataFrame(density_fits)

    viscosity_best_model_idx = viscosity_fits_df[
        "Viscosity Relative Error Max (%)"
    ].idxmin()
    viscosity_best_model = viscosity_fits_df.loc[viscosity_best_model_idx]

    density_best_model_idx = density_fits_df["Density Relative Error Max (%)"].idxmin()
    density_best_model = density_fits_df.loc[density_best_model_idx]

    with open(
        f'models/Viscosity Model {config["reference_data_filename"].replace(".xlsx", ".txt")}',
        "w",
        encoding="utf-8",
    ) as file:
        file.write(viscosity_best_model["functional_form_text"])
        file.write(f"\n{viscosity_best_model.df.describe()}")

    with open(
        f'models/Density Model {config["reference_data_filename"].replace(".xlsx", ".txt")}',
        "w",
        encoding="utf-8",
    ) as file:
        file.write(density_best_model["functional_form_text"])
        file.write(f"\n{density_best_model.df.describe()}")
