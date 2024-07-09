"""Class for handling of constants within the package."""
from dataclasses import dataclass


@dataclass
class NistWebbookFluidIDs:
    """Dataclass for the fluid IDs used in the NIST Webbook scraper."""
    WATER = "C7732185"
    NITROGEN = "C7727379"
    HYDROGEN = "C1333740"
    CARBON_DIOXIDE = "C124389"
    METHANE = "C74828"


@dataclass
class SpecificGasConstants:
    WATER = 461.5  # J/kgK
    HYDROGEN = 4124.478823  # J/kgK
    NITROGEN = 296.8  # J/kgK
    CARBON_DIOXIDE = 189  # J/kgK
    METHANE = 518.3  # J/kgK

    fluid_id_dict = {
        NistWebbookFluidIDs.WATER: WATER,
        NistWebbookFluidIDs.HYDROGEN: HYDROGEN,
        NistWebbookFluidIDs.NITROGEN: NITROGEN,
        NistWebbookFluidIDs.CARBON_DIOXIDE: CARBON_DIOXIDE,
        NistWebbookFluidIDs.METHANE: METHANE,
    }

    fluid_name_dict = {
        "WATER": WATER,
        "HYDROGEN": HYDROGEN,
        "NITROGEN": NITROGEN,
        "CO2": CARBON_DIOXIDE,
        "METHANE": METHANE,
    }


@dataclass
class SutherlandsConstants:
    """https://doc.comsol.com/5.5/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.27.html"""
    WATER = {
        "mu_0": 1.716e-5,  # viscosity at reference temperature, Pa.s
        "T_0": 291.15,  # reference temperature, K
        "S": 110.4,  # sutherland temperature, K
    }
    HYDROGEN = {
        "mu_0": 8.411e-6,
        "T_0": 273,
        "S": 97,
    }
    NITROGEN = {
        "mu_0": 1.663e-5,
        "T_0": 273,
        "S": 107,
    }
    CARBON_DIOXIDE = {
        "mu_0": 1.37e-5,
        "T_0": 273,
        "S": 222,
    }
    METHANE = {
        "mu_0": 1.49e-5,
        "T_0": 273.15,
        "S": 169,
    }

    fluid_id_dict = {
        NistWebbookFluidIDs.WATER: WATER,
        NistWebbookFluidIDs.HYDROGEN: HYDROGEN,
        NistWebbookFluidIDs.NITROGEN: NITROGEN,
        NistWebbookFluidIDs.CARBON_DIOXIDE: CARBON_DIOXIDE,
        NistWebbookFluidIDs.METHANE: METHANE,
    }

    fluid_name_dict = {
        "WATER": WATER,
        "HYDROGEN": HYDROGEN,
        "NITROGEN": NITROGEN,
        "CO2": CARBON_DIOXIDE,
        "METHANE": METHANE,
    }