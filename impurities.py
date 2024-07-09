import pandas as pd
import numpy as np

import yaml

from tqdm import tqdm

from refprop.refprop import RefpropInterface
from refprop.utils import generate_temperature_pressure_samples


DLL_PATH = r"T:\Joseph McGovern\Code\GitHub\refprop-dotnet\refprop"


def get_fluid_property(doe: pd.DataFrame, hFld: str, hOut: str, z: list, composition_basis: str = "mol", show_progress: bool = False):
    if show_progress:
        progress_bar = tqdm
    else:
        progress_bar = lambda x, total: x
    refprop = RefpropInterface(DLL_PATH, hFld)
    df = pd.DataFrame()
    for index, sample in progress_bar(doe.iterrows(), total=len(doe)):
        if composition_basis == "vol":
            pass
            # TODO: Convert vol/vol to mol/mol

        refprop_output = refprop.refprop2dll(
            hFld,
            "TP",  # Input string of properties (Temperature and Pressure)
            hOut,  # Output properties to be calculated
            21,  # mass base SI units
            0,
            sample["Temperature [K]"],  # Temperature in K
            sample["Pressure [Pa]"],  # Pressure in Pa
            z  # composition array, mol/mol or vol/vol
        )
        df = pd.concat([df, pd.DataFrame(refprop_output, index=[index])])
    df["T"] = doe["Temperature [K]"]
    df["P"] = doe["Pressure [Pa]"]
    return df


def get_impurities_properties(fluids: list, doe: pd.DataFrame, df_ref: pd.DataFrame, fluid_property:str, composition_basis, show_progress:bool=False):
    df_dict = {}
    for fluid in fluids:
        print(fluid) if show_progress else None
        df_dict[fluid] = get_fluid_property(doe, f"refprop/FLUIDS/{fluid}", fluid_property, [1.0] + [0.0] * 19, composition_basis, show_progress=show_progress)
        df_dict[fluid][f"{fluid_property}_Err"] = (df_dict[fluid][fluid_property] - df_ref[fluid_property])
    return df_dict


def get_impact_metrics(df_impurities: pd.DataFrame, fluid_property: str):
    integral_series = pd.Series({
        fluid: np.trapz(
            y=np.trapz(
                y=df_impurities[fluid].pivot(index="T", columns="P", values=f"{fluid_property}_Err"),
                x=df_impurities[fluid]["T"].unique()
            ),
            x=df_impurities[fluid]["P"].unique()
        ) for fluid in df_impurities
    }, name="Integral of Density Deviation over Temperature and Pressure Range")

    sorted_series = integral_series.reindex(integral_series.abs().sort_values(ascending=False).index)
    return sorted_series


def normalize_impurities(impurities, impurities_upper_limits, base_compositions_lower_limits, normalized=None, current_sum=0.0):
    if normalized is None:
        normalized = {}

    # Calculate target sum from the base compositions lower limits
    target = 1 - pd.Series(base_compositions_lower_limits).sum()
    
    if impurities.empty or current_sum >= target:
        normalized.update(base_compositions_lower_limits)
        return pd.Series(normalized, name="Normalized Worst-Case Compositions")

    impurity = impurities.index[0]
    limit = impurities_upper_limits[impurity]
    
    if current_sum + limit <= target:
        normalized[impurity] = limit
        return normalize_impurities(impurities[1:], impurities_upper_limits, base_compositions_lower_limits, normalized, current_sum + limit)
    else:
        remaining_limit = target - current_sum
        if remaining_limit > 0:
            normalized[impurity] = remaining_limit
        normalized.update(base_compositions_lower_limits)
        return pd.Series(normalized, name="Normalized Worst-Case Compositions")


if __name__ == "__main__":
    import plotly.graph_objects as go
    import plotly.express as px

    with open('config\\CO2_IMPURITIES.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    BASE_COMPOSITIONS_LOWER_LIMITS = pd.Series(config['BASE_COMPOSITIONS_LOWER_LIMITS'], name="Base Compositions [mol/mol]")
    IMPURITIES_UPPER_LIMITS = pd.Series(config['IMPURITIES_UPPER_LIMITS'], name="Impurities Upper Limits [mol/mol]")

    DOE = generate_temperature_pressure_samples(
        config["DOE"]["n_grid_temperature"],  # n_grid_temperature
        config["DOE"]["n_grid_pressure"],  # n_grid_pressure
        config["DOE"]["temperature_range"],  # temperature_range
        config["DOE"]["pressure_range"],  # pressure_range
    )

    df_ref = get_fluid_property(
        doe=DOE,
        hFld=";".join(
            BASE_COMPOSITIONS_LOWER_LIMITS.add_prefix("refprop/FLUIDS/").index
        ),  # input fluids
       hOut= config['FLUID_PROPERTY'],  # output fluid property
        z=list(BASE_COMPOSITIONS_LOWER_LIMITS.values) + \
            [0.0] * (20 - len(BASE_COMPOSITIONS_LOWER_LIMITS)),  # composition array,
        composition_basis=config['COMPOSITIONS_BASIS'],    
    )
    df_impurities = get_impurities_properties(
        fluids=list(IMPURITIES_UPPER_LIMITS.keys()),
        doe=DOE,
        df_ref=df_ref,
        fluid_property=config['FLUID_PROPERTY'],
        composition_basis=config['COMPOSITIONS_BASIS']
    )

    impact_metrics_density = get_impact_metrics(df_impurities, config['FLUID_PROPERTY'])
    compositions_worst_case = normalize_impurities(
        impurities=impact_metrics_density,
        impurities_upper_limits=IMPURITIES_UPPER_LIMITS,
        base_compositions_lower_limits=BASE_COMPOSITIONS_LOWER_LIMITS
    )

    df_worst_case = get_fluid_property(
        doe=DOE,
        hFld=";".join(compositions_worst_case.add_prefix("refprop/FLUIDS/").index), 
        hOut=config['FLUID_PROPERTY'], 
        z=list(compositions_worst_case.values) + [0.0] * (20 - compositions_worst_case.shape[0]),
        composition_basis=config['COMPOSITIONS_BASIS'],
    )
    df_worst_case["Density Deviation [%]"] = ((df_worst_case.D - df_ref.D) / df_ref.D * 100)
    
    print(impact_metrics_density)
    print(compositions_worst_case)
    print(compositions_worst_case.sum())
    print(df_worst_case["Density Deviation [%]"].describe())

    fig = px.imshow(
        df_worst_case.pivot(index="T", columns="P", values="Density Deviation [%]"),
        labels=dict(x="Pressure [Pa]", y="Temperature [K]", color="Density Deviation [%]"),
        title=f"METHANE Density Deviation [%] Worst Case Scenario",
        x=df_worst_case["P"].unique(),
        y=df_worst_case["T"].unique(),
        aspect="auto"
    )
    fig.show()
