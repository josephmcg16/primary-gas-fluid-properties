import ctypes as ct
import os
import pandas as pd
from tqdm import tqdm


class RefpropInterface:
    """
    Interface to REFPROP using the REFPRP64.DLL library.

    Parameters
    ----------
    install_path : str
        Path to the REFPROP installation directory.
    fluid : str
        Fluid name to use in REFPROP.

    Attributes
    ----------
    dll_path : str
        Path to the REFPROP DLL.
    fluid : str
        Fluid name to use in REFPROP.
    molar_mass : float
        Molar mass of the fluid in g/mol.
    refprop : ctypes.WinDLL
        REFPROP DLL object.
    MAX_STRING_LENGTH : int
        Maximum string length for the error messages.

    Methods
    -------
    _setup_refprop()
        Setup the REFPROP library and retrieve fluid constants.
    calculate_properties(temperature, pressure)
        Calculate the fluid properties of the fluid at a given temperature and pressure.
    calculate_thermal_properties(temperature, pressure)
        Calculate the dynamic viscosity and thermal conductivity of the fluid at a given temperature and pressure.
    calculate_critical_flow_factor(temperature, pressure, velocity=0.0)
        Calculate the critical flow factor of the fluid at a given temperature, pressure, and velocity.
    refprop2dll(hFld, hIn, hOut, iUnits, iFlag, a, b, z)
        Short version of subroutine REFPROP.
    """

    MAX_STRING_LENGTH = 255

    def __init__(self, install_path, fluid):
        """
        Initialize the REFPROP interface.

        Parameters
        ----------
        install_path : str
            Path to the REFPROP installation directory.
        fluid : str
            Fluid name to use in REFPROP.

        Raises
        ------
        Exception
            If the HMX.BNC file is not found.
        Exception
            If the fluid file is not found.
        Exception
            If the REFPROP SETUP function fails.
        """
        self.dll_path = os.path.join(install_path, "REFPRP64.DLL")
        self.fluid = fluid
        self.refprop = ct.WinDLL(self.dll_path)

    def setup_refprop(self):
        """
        Setup the REFPROP library and retrieve fluid constants.

        Raises
        ------
        Exception
            If the HMX.BNC file is not found.
        Exception
            If the fluid file is not found.
        Exception
            If the REFPROP SETUP function fails.
        """
        ierr = ct.c_int()
        herr = ct.create_string_buffer(self.MAX_STRING_LENGTH)
        hmx_bnc_path = os.path.join(os.path.dirname(self.dll_path), "HMX.BNC")

        if not os.path.isfile(hmx_bnc_path):
            raise Exception(f"HMX.BNC file not found: {hmx_bnc_path}")

        self.refprop.SETPATHdll.argtypes = [
            ct.c_char_p,
            ct.POINTER(ct.c_int),
            ct.c_char_p,
        ]
        self.refprop.SETPATHdll.restype = None
        self.refprop.SETPATHdll(
            os.path.dirname(self.dll_path).encode("utf-8"), ct.byref(ierr), herr
        )
        if ierr.value != 0:
            raise Exception(f"REFPROP SETPATH error: {herr.value.decode('utf-8')}")

        fluid_path = os.path.join(
            os.path.dirname(self.dll_path), "FLUIDS", f"{self.fluid}.FLD"
        )
        if not os.path.isfile(fluid_path):
            raise Exception(f"Fluid file not found: {fluid_path}")

        ncomp = ct.c_int(1)
        hFiles = ct.create_string_buffer(fluid_path.encode("utf-8"), 255)
        hFmix = ct.create_string_buffer("HMX.BNC".encode("utf-8"), 255)
        hrf = ct.create_string_buffer("DEF".encode("utf-8"), 3)

        self.refprop.SETUPdll.argtypes = [
            ct.POINTER(ct.c_int),  # ncomp
            ct.c_char_p,  # hFiles
            ct.c_char_p,  # hFmix
            ct.c_char_p,  # hrf
            ct.POINTER(ct.c_int),  # ierr
            ct.c_char_p,  # herr
            ct.c_int,  # hFiles_length
            ct.c_int,  # hFmix_length
            ct.c_int,  # hrf_length
            ct.c_int,  # herr_length
        ]
        self.refprop.SETUPdll.restype = None
        self.refprop.SETUPdll(
            ct.byref(ncomp),
            hFiles,
            hFmix,
            hrf,
            ct.byref(ierr),
            herr,
            255,  # hFiles_length
            255,  # hFmix_length
            3,  # hrf_length
            self.MAX_STRING_LENGTH,  # herr_length
        )
        if ierr.value != 0:
            raise Exception(f"REFPROP SETUP error: {herr.value.decode('utf-8')}")

        # Retrieve fluid constants using INFOdll
        icomp = ct.c_int(1)
        wmm = ct.c_double()
        Ttrp = ct.c_double()
        Tnbpt = ct.c_double()
        Tc = ct.c_double()
        Pc = ct.c_double()
        Dc = ct.c_double()
        Zc = ct.c_double()
        acf = ct.c_double()
        dip = ct.c_double()
        Rgas = ct.c_double()

        self.refprop.INFOdll.argtypes = [
            ct.POINTER(ct.c_int),  # icomp
            ct.POINTER(ct.c_double),  # wmm
            ct.POINTER(ct.c_double),  # Ttrp
            ct.POINTER(ct.c_double),  # Tnbpt
            ct.POINTER(ct.c_double),  # Tc
            ct.POINTER(ct.c_double),  # Pc
            ct.POINTER(ct.c_double),  # Dc
            ct.POINTER(ct.c_double),  # Zc
            ct.POINTER(ct.c_double),  # acf
            ct.POINTER(ct.c_double),  # dip
            ct.POINTER(ct.c_double),  # Rgas
        ]
        self.refprop.INFOdll.restype = None

        self.refprop.INFOdll(
            ct.byref(icomp),
            ct.byref(wmm),
            ct.byref(Ttrp),
            ct.byref(Tnbpt),
            ct.byref(Tc),
            ct.byref(Pc),
            ct.byref(Dc),
            ct.byref(Zc),
            ct.byref(acf),
            ct.byref(dip),
            ct.byref(Rgas),
        )

        self.fluid_info = {
            "Molar Mass [g/mol]": wmm.value,
            "Triple Point Temperature [K]": Ttrp.value,
            "Normal Boiling Point Temperature [K]": Tnbpt.value,
            "Critical Temperature [K]": Tc.value,
            "Critical Pressure [kPa]": Pc.value,
            "Critical Density [mol/L]": Dc.value,
            "Critical Compressibility Factor": Zc.value,
            "Accentric Factor": acf.value,
            "Dipole Moment [Debye]": dip.value,
            "Gas Constant [J/mol-K]": Rgas.value,}

    def refprop2dll(self, hFld, hIn, hOut, iUnits, iFlag, a, b, z):
        """
        Short version of subroutine REFPROP.

        Parameters
        ----------
        hFld : str
            Fluid string.
        hIn : str
            Input string of properties being sent to the routine.
        hOut : str
            Output string of properties to be calculated.
        iUnits : int
            The unit system to be used for the input and output properties (such as SI, English, etc.)
        iFlag : int
            Flag to specify if the routine SATSPLN should be called (where a value of 1 activates the call).
        a : float
            First input property as specified in the hIn variable.
        b : float
            Second input property as specified in the hIn variable.
        z : list of float
            Molar composition (array of size ncmax=20).

        Returns
        -------
        dict
            Dictionary with the results.

        Raises
        ------
        Exception
            If the REFPROP2dll function fails.
        """
        ierr = ct.c_int()
        herr = ct.create_string_buffer(self.MAX_STRING_LENGTH)
        hFld_c = ct.create_string_buffer(hFld.encode('ascii'), self.MAX_STRING_LENGTH)
        hIn_c = ct.create_string_buffer(hIn.encode('ascii'), self.MAX_STRING_LENGTH)
        hOut_c = ct.create_string_buffer(hOut.encode('ascii'), self.MAX_STRING_LENGTH)

        iUnits_c = ct.c_int(iUnits)
        iFlag_c = ct.c_int(iFlag)
        a_c = ct.c_double(a)
        b_c = ct.c_double(b)
        z_c = (ct.c_double * 20)(*z)
        Output = (ct.c_double * 200)()
        q = ct.c_double()

        self.refprop.REFPROP2dll.argtypes = [
            ct.c_char_p,  # hFld
            ct.c_char_p,  # hIn
            ct.c_char_p,  # hOut
            ct.POINTER(ct.c_int),  # iUnits
            ct.POINTER(ct.c_int),  # iFlag
            ct.POINTER(ct.c_double),  # a
            ct.POINTER(ct.c_double),  # b
            ct.POINTER(ct.c_double * 20),  # z
            ct.POINTER(ct.c_double * 200),  # Output
            ct.POINTER(ct.c_double),  # q
            ct.POINTER(ct.c_int),  # ierr
            ct.c_char_p,  # herr
            ct.c_int,  # hFld_length
            ct.c_int,  # hIn_length
            ct.c_int,  # hOut_length
            ct.c_int,  # herr_length
        ]
        self.refprop.REFPROP2dll.restype = None

        self.refprop.REFPROP2dll(
            hFld_c,
            hIn_c,
            hOut_c,
            ct.byref(iUnits_c),
            ct.byref(iFlag_c),
            ct.byref(a_c),
            ct.byref(b_c),
            ct.byref(z_c),
            ct.byref(Output),
            ct.byref(q),
            ct.byref(ierr),
            herr,
            self.MAX_STRING_LENGTH,
            self.MAX_STRING_LENGTH,
            self.MAX_STRING_LENGTH,
            self.MAX_STRING_LENGTH,
        )

        if ierr.value != 0 and ierr.value != -117:
            print(f"hFld: {hFld}, hIn: {hIn}, hOut: {hOut}")
            print(f"a: {a}, b: {b}")
            print(f"z: {z}")
            raise Exception(f"REFPROP2dll error: {herr.value.decode('ascii')}")
        return {hOut_property: Output[i] for i, hOut_property in enumerate(hOut.split(","))}


if __name__ == "__main__":
    from utils import generate_temperature_pressure_samples

    FLUID_NAME = "NITROGEN"

    doe = generate_temperature_pressure_samples(
        n_grid_temperature=1000,
        n_grid_pressure=1000,
        temperature_range=(278.15, 323.15),  # K
        pressure_range=(200, 20101325),  # Pa
    )

    refprop = RefpropInterface(r"T:\Joseph McGovern\Code\GitHub\refprop-dotnet\refprop", FLUID_NAME)

    df = pd.DataFrame()
    for index, sample in tqdm(doe.iterrows(), total=len(doe)):
        try:
            refprop_output = refprop.refprop2dll(
                FLUID_NAME,
                "TP",  # Input string of properties (Temperature and Pressure)
                "T,P,D,VIS,ISENK,CSTAR",  # Output properties to be calculated
                21,  # mass base SI units
                0,
                sample["Temperature [K]"],  # Temperature in K
                sample["Pressure [Pa]"],  # Pressure in Pa
                [1.0] + [0.0] * 19  # Assuming pure hydrogen
            )
            df = pd.concat([df, pd.DataFrame(refprop_output, index=[index])])
        except Exception as e:
            print(f"Error at index {index}: {e}")
            print(f"Sample: {sample}")
            raise e


    df.to_csv(f"data\\{FLUID_NAME} output.csv", index=False)
