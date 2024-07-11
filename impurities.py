"""Script to calculate the impact of impurities on fluid properties calculations."""
import pandas as pd
import numpy as np
import yaml

import plotly.express as px

from refprop.utils import generate_temperature_pressure_samples
from refprop.refprop import RefpropInterface
from generate_data import get_fluid_property


def get_impurities_properties(
        fluids: list, doe: pd.DataFrame, df_ref: pd.DataFrame, fluid_property:str, show_progress:bool=False
    ):
    """Get fluid properties df for each impurities in the fluids list."""
    df_dict = {}
    for fluid in fluids:
        if show_progress:
            print(fluid)
        df_dict[fluid] = get_fluid_property(
            doe=doe,
            hFld=f"refprop/FLUIDS/{fluid}",
            hOut=fluid_property,
            z=[1.0] + [0.0] * 19,
            show_progress=show_progress
        )
        df_dict[fluid][f"{fluid_property}_Err"] = (
            df_dict[fluid][fluid_property] - df_ref[fluid_property]
        )
    return df_dict


def get_impact_metrics(df_impurities: pd.DataFrame, fluid_property: str):
    """Integrate fluid properties error between impurities and pure fluid over operating range."""
    integral_series = pd.Series({
        fluid: np.trapz(
            y=np.trapz(
                y=df_impurities[fluid].pivot(
                    index="T", columns="P", values=f"{fluid_property}_Err"
                ),
                x=df_impurities[fluid]["T"].unique()
            ),
            x=df_impurities[fluid]["P"].unique()
        ) for fluid in df_impurities
    }, name=f"Integral of {fluid_property} Deviation over Temperature and Pressure Range")

    sorted_series = integral_series.reindex(integral_series.abs().sort_values(ascending=False).index)
    return sorted_series


def sort_metrics(impact_metrics):
    """Sorts the impact metrics in a worst-case order."""
    sorted_devs = impact_metrics.reindex(impact_metrics.abs().sort_values(ascending=False).index)
    
    positive_devs = sorted_devs[sorted_devs >= 0]
    negative_devs = sorted_devs[sorted_devs < 0]
    
    worst_case_order = []
    worst_case_indices = []
    
    while not positive_devs.empty or not negative_devs.empty:
        if not positive_devs.empty:
            worst_case_order.append(positive_devs.iloc[0])
            worst_case_indices.append(positive_devs.index[0])
            positive_devs = positive_devs.iloc[1:]
        if not negative_devs.empty:
            worst_case_order.append(negative_devs.iloc[0])
            worst_case_indices.append(negative_devs.index[0])
            negative_devs = negative_devs.iloc[1:]
    
    worst_case_series = pd.Series(worst_case_order, index=worst_case_indices)
    return worst_case_series


def normalize_impurities(
        impurities,
        impurities_upper_limits,
        base_compositions_lower_limits,
        normalized=None,
        current_sum=0.0
    ) -> pd.Series:
    """Sort impurities by impact metrics and normalize them to the worst-case scenario 
    (i.e., total composition of 100%)."""
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
        return normalize_impurities(
            impurities[1:],
            impurities_upper_limits,
            base_compositions_lower_limits,
            normalized,
            current_sum + limit
        )
    remaining_limit = target - current_sum
    if remaining_limit > 0:
        normalized[impurity] = remaining_limit
    normalized.update(base_compositions_lower_limits)
    return pd.Series(normalized, name="Normalized Worst-Case Compositions")


if __name__ == "__main__":
    def main():
        # Load configuration
        with open("config\\CO2_IMPURITIES.yaml", 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Define constants from configuration
        BASE_COMPOSITIONS_LOWER_LIMITS = pd.Series(
            config['BASE_COMPOSITIONS_LOWER_LIMITS'], 
            name="Base Compositions [vol/vol]"
        )
        IMPURITIES_UPPER_LIMITS = pd.Series(
            config['IMPURITIES_UPPER_LIMITS'],
            name="Impurities Upper Limits [vol/vol]"
        )

        # Generate temperature and pressure samples
        doe = generate_temperature_pressure_samples(
            config["DOE"]["n_grid_temperature"],
            config["DOE"]["n_grid_pressure"],
            config["DOE"]["temperature_range"],
            config["DOE"]["pressure_range"],
        )

        # Get fluid property for reference composition
        df_ref = get_fluid_property(
            doe=doe,
            hFld=";".join(
                BASE_COMPOSITIONS_LOWER_LIMITS.add_prefix("refprop/FLUIDS/").index
            ),
            hOut=config['FLUID_PROPERTY'],
            z=list(BASE_COMPOSITIONS_LOWER_LIMITS.values) + \
                [0.0] * (20 - len(BASE_COMPOSITIONS_LOWER_LIMITS)), 
        )

        # Get impurities properties
        df_impurities = get_impurities_properties(
            fluids=list(IMPURITIES_UPPER_LIMITS.keys()),
            doe=doe,
            df_ref=df_ref,
            fluid_property=config['FLUID_PROPERTY'],
        )

        # Calculate impact metrics
        impact_metrics = {
            config['FLUID_PROPERTY']: get_impact_metrics(
                df_impurities, config['FLUID_PROPERTY']
            )
        }

        # Sort impurities based on impact metrics
        impurities_sorted = sort_metrics(impact_metrics[config['FLUID_PROPERTY']]).index

        # Calculate normalized impurities
        total_impurities = 1 - BASE_COMPOSITIONS_LOWER_LIMITS.sum()
        sum_impurities = 0
        impurities_sorted_normalized = {}
        for impurity in impurities_sorted:
            sum_impurities += IMPURITIES_UPPER_LIMITS[impurity]
            if sum_impurities < total_impurities:
                impurities_sorted_normalized[impurity] = IMPURITIES_UPPER_LIMITS[impurity]
            else:
                impurities_sorted_normalized[impurity] = IMPURITIES_UPPER_LIMITS[impurity] - (sum_impurities - total_impurities)
                break

        compositions = pd.concat([BASE_COMPOSITIONS_LOWER_LIMITS, pd.Series(impurities_sorted_normalized)])

        # Function to convert compositions to moles per unit volume
        def convert_compositions_to_moles_unit_volume(compositions, temperature_filling, pressure_filling):
            def get_molar_mass(fluid):
                refprop = RefpropInterface(r"T:\Joseph McGovern\Code\GitHub\refprop-dotnet\refprop", fluid)
                refprop.setup_refprop()
                M = refprop.fluid_info['Molar Mass [g/mol]'] / 1000  # kg/mol
                return M

            def get_property(fluid, fluid_property, temperature, pressure):
                return get_fluid_property(
                    pd.DataFrame({"Temperature [K]": [temperature], "Pressure [Pa]": [pressure]}, index=[0]), 
                    f"refprop/FLUIDS/{fluid}",
                    fluid_property, 
                    [1.0] + [0] * 19
                )[fluid_property].values[0]  # kg/m3

            def get_moles_unit_volume(x_v, M, rho):
                rho_m = rho / M
                return x_v * rho_m

            df_filling = pd.DataFrame()
            for fluid in compositions.index:
                rho = get_property(
                    fluid, "D", temperature_filling, pressure_filling
                )  # kg/m3
                M = get_molar_mass(fluid)  # kg/mol
                n = get_moles_unit_volume(compositions[fluid], M, rho)
                df_filling = pd.concat([
                    df_filling,
                    pd.DataFrame({
                        "Filling Temperature [K]": [temperature_filling],
                        "Filling Pressure [Pa]": [pressure_filling],
                        "Filling Density [kg/m3]": [rho],
                        "Molar Mass [kg/mol]": [M],
                        "Moles Unit Volume [mol/L]": [n],
                    }, index=[fluid])
                ])
            df_filling["Composition [mol/mol]"] = df_filling["Moles Unit Volume [mol/L]"] / df_filling["Moles Unit Volume [mol/L]"].sum()
            df_filling["Composition [L/L]"] = compositions.values
            return df_filling

        # Determine worst-case compositions based on the basis
        if config['COMPOSITIONS_BASIS'] == "volume":
            compositions_worst_case = convert_compositions_to_moles_unit_volume(
                compositions=compositions, 
                temperature_filling=config['FILLING_CONDITIONS']['TEMPERATURE'], 
                pressure_filling=config['FILLING_CONDITIONS']['PRESSURE']
            )["Composition [mol/mol]"]
        elif config['COMPOSITIONS_BASIS'] == "mole":
            compositions_worst_case = compositions
        else:
            raise Exception("Invalid 'COMPOSITIONS_BASIS'")

        # Get fluid property for worst-case composition
        df_worst_case = get_fluid_property(
            doe=doe,
            hFld=";".join(compositions_worst_case.add_prefix("refprop/FLUIDS/").index),
            hOut=config['FLUID_PROPERTY'],
            z=list(compositions_worst_case.values) + \
                [0.0] * (20 - compositions_worst_case.shape[0]),
        )

        # Calculate deviation
        deviation_col = f"{config['FLUID_PROPERTY']} Deviation [%]"
        df_worst_case[deviation_col] = (df_worst_case.D - df_ref.D) / df_ref.D * 100.0
        print(df_worst_case.describe())

        # Plot results
        fig_im = px.imshow(
            df_worst_case.pivot(
                index="T", columns="P", values=deviation_col
            ),
            labels=dict(x="Pressure [Pa]", y="Temperature [K]", color=deviation_col),
            title=f"{deviation_col} vs. Temperature and Pressure",
            aspect="auto",
        )
        fig_im.update_xaxes(autorange=0)
        fig_im.update_yaxes(autorange=0)
        fig_im.show()

        fig_hist = px.histogram(
            df_worst_case, x=deviation_col,
            title=f"{deviation_col} Histogram",
        )
        fig_hist.show()
    main()
