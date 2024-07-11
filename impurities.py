"""Script to calculate the impact of impurities on fluid properties calculations."""
import pandas as pd
import numpy as np
import yaml

from refprop.utils import generate_temperature_pressure_samples
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
        """Main function to run the script."""
        with open('config\\CO2_IMPURITIES.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        base_compositions_lower_limits = pd.Series(
            config['BASE_COMPOSITIONS_LOWER_LIMITS'], 
            name="Base Compositions [mol/mol]"
        )
        impurities_upper_limits = pd.Series(
            config['IMPURITIES_UPPER_LIMITS'],
            name="Impurities Upper Limits [mol/mol]"
        )

        doe = generate_temperature_pressure_samples(
            config["DOE"]["n_grid_temperature"],
            config["DOE"]["n_grid_pressure"],
            config["DOE"]["temperature_range"],
            config["DOE"]["pressure_range"],
        )

        df_ref = get_fluid_property(
            doe=doe,
            hFld=";".join(
                base_compositions_lower_limits.add_prefix("refprop/FLUIDS/").index
                ),
            hOut=config['FLUID_PROPERTY'],
            z=list(base_compositions_lower_limits.values) + \
                [0.0] * (20 - len(base_compositions_lower_limits)), 
        )

        df_impurities = get_impurities_properties(
            fluids=list(impurities_upper_limits.keys()),
            doe=doe,
            df_ref=df_ref,
            fluid_property=config['FLUID_PROPERTY'],
        )

        impact_metrics_density = get_impact_metrics(df_impurities, config['FLUID_PROPERTY'])
        compositions_worst_case = normalize_impurities(
            impurities=impact_metrics_density,
            impurities_upper_limits=impurities_upper_limits,
            base_compositions_lower_limits=base_compositions_lower_limits
        )

        df_worst_case = get_fluid_property(
            doe=doe,
            hFld=";".join(compositions_worst_case.add_prefix("refprop/FLUIDS/").index),
            hOut=config['FLUID_PROPERTY'],
            z=list(compositions_worst_case.values) + \
                [0.0] * (20 - compositions_worst_case.shape[0]),
        )
        deviation_col = f"{config['FLUID_PROPERTY']} Deviation [%]"
        df_worst_case[deviation_col] = (df_worst_case.D - df_ref.D) / df_ref.D * 100.0

        print(f"Impact Metrics: \n{impact_metrics_density}\n{'-' * 80}")
        print(f"Worst-Case Compositions: \n{compositions_worst_case}\n{'-' * 80}")
        print(f"Worst-Case Sum: {compositions_worst_case.sum()}\n{'-' * 80}")
        print(f"Worst-Case Deviation Metrics: \n"
              f"{df_worst_case[deviation_col].describe()}")

    main()
