"""Generate fluid properties for each sample in a design of experiments (DOE)."""
import pandas as pd
from tqdm import tqdm
from refprop.utils import generate_temperature_pressure_samples
from refprop.refprop import RefpropInterface


def rename_columns(df:pd.DataFrame):
    """Rename columns to more descriptive names for scikit-learn modules."""
    df["Temperature (C)"] = df["T"] - 273.15  # K to C
    df["Pressure (bar)"] = df["P"] / 1e5  # Pa to bar
    df.drop(columns=["T", "P"], inplace=True)
    df.rename(columns={
        "Z": "Compressibility",
        "D": "Density (kg/m3)",
        "VIS": "Viscosity (Pa*s)",
        "ISENK": "Isentropic Exponent",
        "CSTAR": "Critical Flow Factor"
    }, inplace=True)
    return df


def get_fluid_property(doe:pd.DataFrame, hFld:str, hOut:str, z:list, show_progress:bool=False, rename:bool=False):
    """Get fluid properties for each sample in the doe.
    
    Args
    ----
    doe: pd.DataFrame
        DataFrame containing the samples to be evaluated.
    hFld: str
        Path to the fluid file(s). E.g., 'refprop/FLUIDS/PROPANE.FLD;refprop/FLUIDS/ETHANE.FLD'
    hOut: str
        Output properties to be calculated. E.g., 'Z,D,VIS,ISENK,CSTAR'
    z: list
        Composition array for each fluid in hFld.
    show_progress: bool
        Show tqdm progress bar.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the fluid properties for each sample in the
        doe.
    """
    if show_progress:
        progress_bar = tqdm
    else:
        progress_bar = lambda x, total: x
    refprop = RefpropInterface(r"T:\Joseph McGovern\Code\GitHub\refprop-dotnet\refprop", hFld)
    df = pd.DataFrame()
    for index, sample in progress_bar(doe.iterrows(), total=len(doe)):
        refprop_output = refprop.refprop2dll(
            hFld,
            "TP",  # Input string of properties (Temperature and Pressure)
            hOut,  # Output properties to be calculated
            21,  # mass base SI units
            0,
            sample["Temperature [K]"],  # Temperature in K
            sample["Pressure [Pa]"],  # Pressure in Pa
            z  # composition array
        )
        df = pd.concat([df, pd.DataFrame(refprop_output, index=[index])])

    df["T"] = doe["Temperature [K]"]
    df["P"] = doe["Pressure [Pa]"]

    if rename:
        df = rename_columns(df)
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df = df[cols]
    return df


if __name__ == "__main__":
    import yaml

    with open("config\\GENERATE_DATA.yaml", "r", encoding='utf-8') as file:
        df_config = pd.DataFrame(yaml.safe_load(file)).T

    df_dict = {}
    for fluid, config in df_config.iterrows():
        print(f"Generating data for {fluid}")
       
        df_dict[fluid] = get_fluid_property(
            doe=generate_temperature_pressure_samples(
                n_grid_temperature=int(config["N_GRID_TEMPERATURE"]),
                n_grid_pressure=int(config["N_GRID_PRESSURE"]),
                temperature_range=(config["TMIN"], config["TMAX"]),  # K
                pressure_range=(config["PMIN"], config["PMAX"]),  # Pa
            ),
            hFld=f"refprop/FLUIDS/{fluid}.FLD",
            hOut="Z,D,VIS,ISENK,CSTAR",
            z=[1.0] + [0.0] * 19,
            show_progress=True,
            rename=True
        )
        df_dict[fluid].to_excel(f"data\\{fluid} PROPERTIES.xlsx", index=False)
