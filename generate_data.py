import pandas as pd
from tqdm import tqdm
from refprop.utils import generate_temperature_pressure_samples
from refprop.refprop import RefpropInterface


def rename_columns(df:pd.DataFrame):
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


def get_fluid_property(doe:pd.DataFrame, hFld:str, hOut:str, z:list, show_progress:bool=False):
    if show_progress:
        progress_bar = tqdm
    else:
        progress_bar = lambda x: x
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
            show_progress=True
        )
        df_dict[fluid].to_excel(f"data\\{fluid} PROPERTIES.xlsx", index=False)
