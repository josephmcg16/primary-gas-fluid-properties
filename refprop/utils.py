import numpy as np
import pandas as pd
from enum import Enum


def generate_temperature_pressure_samples(
    n_grid_temperature: int,
    n_grid_pressure: int,
    temperature_range: tuple = (100, 1000),
    pressure_range: tuple = (100, 1000),
    method: str = "meshgrid",
):
    """
    Generate a Design of Experiments (DOE) with samples for temperature and pressure.

    Parameters
    ----------
    n_grid_temperature : int
        Number of samples for the temperature grid.
    n_grid_pressure : int
        Number of samples for the pressure grid.
    method : str
        Method to generate the samples. Options are "meshgrid" and "lhs".

    Returns
    -------
    pd.DataFrame
        DataFrame with the samples.
    """
    if method == "meshgrid":
        temperature = np.linspace(*temperature_range, n_grid_temperature)
        pressure = np.linspace(*pressure_range, n_grid_pressure)
        temperature, pressure = np.meshgrid(temperature, pressure)
        doe = pd.DataFrame(
            {
                "Temperature [K]": temperature.flatten(),
                "Pressure [Pa]": pressure.flatten(),
            }
        )
    elif method == "lhs":
        doe = lhs(2, n_grid_temperature * n_grid_temperature) * np.array(
            [
                temperature_range[1] - temperature_range[0],
                pressure_range[1] - pressure_range[0],
            ]
        ) + np.array([temperature_range[0], pressure_range[0]])
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
    return doe

