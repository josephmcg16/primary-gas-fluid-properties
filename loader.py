"""This module contains the Loader class to load the reference fluid properties data from the excel file."""
import numpy as np
import pandas as pd


class Loader:
    """
    Loader class to load reference fluid properties data from an excel file.

    Attributes
    ----------
        data_path: str
            The path to the excel file containing the reference fluid properties data.
        df: pd.DataFrame
            The dataframe containing the reference fluid properties data.

    Parameters
    -------
        data_path: str
            The path to the excel file containing the reference fluid properties data.
    
    Methods
    -------
        load_data(fluid_property_name: str, temperature_name: str = "Temperature (C)", pressure_name: str = "Pressure (bar)"):
            Load the reference data and return the input and output data for the fluid.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: pd.DataFrame = pd.read_excel(self.data_path)

    def load_data(
        self,
        fluid_property_name: str,
        temperature_name="Temperature (C)",
        pressure_name="Pressure (bar)",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the requested input temperature and pressure data and output fluid properties data.

        Parameters
        ----------
        fluid_property_name: str
            The name of the fluid property to be predicted.
        temperature_name: str
            The name of the temperature column in the excel file.
        pressure_name: str
            The name of the pressure column in the excel file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The input and output data for the fluid.
        """

        return (
            self.df[[temperature_name, pressure_name]].values,
            self.df[fluid_property_name].values,
        )

    def __call__(self, *args, **kwds) -> np.ndarray:
        """Load the reference data and return the input and output data for the fluid."""
        return self.load_data(*args, **kwds)
