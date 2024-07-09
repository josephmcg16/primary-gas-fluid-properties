"""Module for scraping Thermophysical Properties of Fluid Systems data from the Nist Webbook."""

from urllib.error import HTTPError
from tqdm import tqdm
import pandas as pd
import numpy as np

from fluid_constants import SpecificGasConstants


def scrape_isotherm(
    fluid_id: str,
    isotherm: float,
    min_pressure: float,
    max_pressure: float,
    increment_pressure: float = 0.0,  # Default to scrape the maximum number of points.
) -> pd.DataFrame:
    """Scrape fluid properties data from the NIST Webbook for an isotherm.

    Args:
        fluid_id (str): The fluid ID to scrape.
        isotherm (float): The process temperature, degC.
        min_pressure (float): The minimum pressure to scrape, bar(a).
        max_pressure (float): The maximum pressure to scrape, bar(a).
        increment_pressure (float): The increments in the pressure range to scrape, bar(a).

    Returns:
        pd.DataFrame: The scraped fluid data.
    """
    scraper_string = (
        f"https://webbook.nist.gov/cgi/fluid.cgi?T={isotherm}&PLow={min_pressure}&"
        f"PHigh={max_pressure}&PInc={increment_pressure}&Digits=5&ID={fluid_id}&Action=Load&"
        "Type=IsoTherm&TUnit=C&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fkg&WUnit=m%2Fs&VisUnit=Pa*s&"
        "STUnit=N%2Fm&RefState=DEF"
    )
    try:
        return pd.read_html(scraper_string)[0]
    except HTTPError:
        print("Error: The data could not be scraped from the NIST Webbook.")
        return -1


def scrape_point(
    fluid_id: str,
    temperature: float,
    pressure: float,
) -> pd.DataFrame:
    """Scrape fluid properties data from the NIST Webbook for a pressure and temperature.

    Args:
        fluid_id (str): The fluid ID to scrape.
        temperature (float): The process temperature, degC.
        pressure (float): The process pressure, bar(a).

    Returns:
        pd.DataFrame: The scraped fluid data.
    """
    scraper_string = (
        f"https://webbook.nist.gov/cgi/fluid.cgi?T={temperature}&PLow={pressure}&"
        f"PHigh={pressure}&PInc=0&Digits=5&ID={fluid_id}&Action=Load&"
        "Type=IsoTherm&TUnit=C&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fkg&WUnit=m%2Fs&VisUnit=Pa*s&"
        "STUnit=N%2Fm&RefState=DEF"
    )
    try:
        return pd.read_html(scraper_string, index_col=False)[0]
    except HTTPError:
        print("Error: The data could not be scraped from the NIST Webbook.")
        return -1


def scrape_properties(
    fluid_id: str,
    temperature_range: np.ndarray,
    pressure_range: np.ndarray,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Scrape fluid properties data from the NIST Webbook for a range of temperatures and pressures.

    Args:
        fluid_id (str): The fluid ID to scrape.
        temperature_range (np.ndarray): The range of process temperatures, degC.
        pressure_range (np.ndarray): The range of process pressures, bar(a).
        show_progress (bool): Whether to show a progress bar.

    Returns:
        pd.DataFrame: The scraped fluid data.
    """
    if show_progress:
        temperature_range = tqdm(
            temperature_range, unit="isotherm", desc="Scraping NIST Webbook"
        )

    df = pd.DataFrame()
    for temperature in temperature_range:
        try:
            df = pd.concat(
                [
                    df,
                    scrape_isotherm(
                        fluid_id,
                        temperature,
                        pressure_range[0],
                        pressure_range[-1],
                        pressure_range[1] - pressure_range[0],
                    ),
                ]
            )
        except HTTPError:
            print("Error: The data could not be scraped from the NIST Webbook.")
            return -1
    return df


class Scraper:
    def __init__(self, fluid_id: str):
        self.fluid_id = fluid_id
        self.df = pd.DataFrame()

    def scrape(
        self,
        temp_range: np.ndarray,
        pressure_range: np.ndarray,
        show_progress: bool = False,
    ):
        """Scrape the fluid properties and display in the Treeview."""
        self.df = scrape_properties(
            self.fluid_id, temp_range, pressure_range, show_progress
        )
        self.df["Compressibility"] = (
            self.df["Pressure (bar)"]
            * 10**5
            / (
                self.df["Density (kg/m3)"]
                * (self.df["Temperature (C)"] + 273.15)
                * SpecificGasConstants.fluid_id_dict[self.fluid_id]
            )
        )
        return self.df

    def __call__(
        self,
        temp_range: np.ndarray,
        pressure_range: np.ndarray,
        show_progress: bool = False,
    ):
        self.scrape(temp_range, pressure_range, show_progress)
        return self.df


if __name__ == "__main__":
    from fluid_constants import NistWebbookFluidIDs

    FLUID_ID = NistWebbookFluidIDs.HYDROGEN
    TEMPERATURE_RANGE = np.linspace(-10, 50, 100)  # degC
    PRESSURE_RANGE = np.linspace(0.002, 2, 100)  # bar(a)
    PRESSURE_RANGE = np.linspace(2, 130, 300)  # bar(a)

    scraper = Scraper(fluid_id=FLUID_ID)
    fluidprop_data = scraper.scrape(
        TEMPERATURE_RANGE, PRESSURE_RANGE, show_progress=True
    )
    fluidprop_data.to_excel(
        "../data/H2 Nist Properties 0.002-130 bar.xlsx", index=False, sheet_name="CH4"
    )
