import pandas as pd
import warnings
from importlib import resources

def get_disinfection_table(fluence, wavelength=222, room=None):
    """
    Retrieve and format inactivtion data for this room.

    Currently assumes all lamps are GUV222. in the future will need something
    cleverer than this
    """
    
    fname = "disinfection_table.csv"
    path = resources.files("guv_calcs.data").joinpath(fname)
    df = pd.read_csv(path)
    
    valid_wavelengths = df["wavelength [nm]"].unique()
    if wavelength not in valid_wavelengths:
        warnings.warn(f"No data is available for wavelength {wavelength} nm.")
    
    df = df[df["Medium"] == "Aerosol"]
    df = df[df["wavelength [nm]"] == wavelength]

    # calculate eACH before filling nans
    k1 = df["k1 [cm2/mJ]"].fillna(0).astype(float)
    k2 = df["k2 [cm2/mJ]"].fillna(0).astype(float)
    f = df["% resistant"].str.rstrip("%").astype("float").fillna(0) / 100
    eACH = (k1 * (1 - f) + k2 - k2 * (1 - f)) * fluence * 3.6
    df["eACH-UV"] = eACH.round(2)
    
    
    newkeys = ["eACH-UV"]
    
    if room is not None:
        volume = room.get_volume()
        # convert to cubic feet for cfm
        if room.units == "meters":
            volume = volume / (0.3048 ** 3)
        cadr_uv_cfm = eACH * volume / 60
        cadr_uv_lps = cadr_uv_cfm * 0.47195    
        df["CADR-UV [cfm]"] = cadr_uv_cfm.round(2)
        df["CADR-UV [lps]"] = cadr_uv_lps.round(2)
        newkeys += ["CADR-UV [cfm]","CADR-UV [lps]"]

    newkeys += [
        "Organism",
        "Species",
        "Strain",
        "Type (Viral)",
        "Enveloped (Viral)",
        "k1 [cm2/mJ]",
        "k2 [cm2/mJ]",
        "% resistant",
        "Medium (specific)",
        "Full Citation",
    ]
    df = df[newkeys].fillna(" ")
    df = df.rename(
        columns={"Medium (specific)": "Medium", "Full Citation": "Reference"}
    )
    df = df.sort_values("Species")

    return df