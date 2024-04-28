import numpy as np
import pandas as pd

class Loader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: pd.DataFrame = None

    def load_data(
        self,
        fluid_property_name: str,
        temperature_name="Temperature (C)",
        pressure_name="Pressure (bar)",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load the reference data.

        Args:
            sheet_name (str): The sheet name to load.

        Returns:
            tuple[np.ndarray, np.ndarray]: X = [temperature, pressure], y = selected fluid property.
        """
        df = pd.read_excel(self.data_path)
        self.df = df
        return (
            df[[temperature_name, pressure_name]].values,
            df[fluid_property_name].values,
        )

    def __call__(self, *args, **kwds) -> np.ndarray:
        return self.load_data(*args, **kwds)