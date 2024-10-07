import pandas as pd
import warnings
from importlib import resources
import numpy as np
import csv
from ._helpers import load_csv

def get_disinfection_table(fluence, wavelength=222, room=None):
    """
    Retrieve and format inactivtion data for this room.

    Currently assumes all lamps are GUV222. in the future will need something
    cleverer than this
    """

    fname = "UVC Inactivation Constants.csv"
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
    df["eACH-UV"] = eACH.round(1)

    newkeys = [
        "eACH-UV"
    ]

    if room is not None:
        volume = room.get_volume()
        # convert to cubic feet for cfm
        if room.units == "meters":
            volume = volume / (0.3048 ** 3)
        cadr_uv_cfm = eACH * volume / 60
        cadr_uv_lps = cadr_uv_cfm * 0.47195
        df["CADR-UV [cfm]"] = cadr_uv_cfm.round(1)
        df["CADR-UV [lps]"] = cadr_uv_lps.round(1)
        newkeys += ["CADR-UV [cfm]", "CADR-UV [lps]"]

    newkeys += [
        "Organism",
        "Species",
        "Strain",
        "Type",
        "Enveloped (Viral)",
        "k1 [cm2/mJ]",
        "k2 [cm2/mJ]",
        "% resistant",
        "Medium (specific)",
        "Full Citation",
        "URL",
    ]
    df = df[newkeys].fillna(" ")
    df = df.rename(
        columns={"Medium (specific)": "Medium", "Full Citation": "Reference"}
    )
    df = df.sort_values("Species")

    return df


def get_spectral_weightings():
    """
    Return a dict of all the relevant spectral weightings by wavelength
    """

    fname = "UV Spectral Weighting Curves.csv"
    path = resources.files("guv_calcs.data").joinpath(fname)
    with path.open("rb") as file:
        weights = file.read()

    csv_data = load_csv(weights)
    reader = csv.reader(csv_data, delimiter=",")
    headers = next(reader, None)  # get headers

    data = {}
    for header in headers:
        data[header] = []
    for row in reader:
        for header, value in zip(headers, row):
            data[header].append(float(value))

    spectral_weightings = {}
    for i, (key, val) in enumerate(data.items()):
        spectral_weightings[key] = np.array(val)
    return spectral_weightings
    
def get_tlv(ref, standard):
    """
    return the value of the UV dose not to be exceeded over 8 hours, 
    assuming a monochromatic wavelength
    TODO: some way to verify the standard label is in the keys
    """
    
    weights = get_spectral_weightings()
    tlv_wavelengths = weights["Wavelength (nm)"]
    tlv_values = weights[standard]
    weighting = np.interp(ref, tlv_wavelengths, tlv_values)
    tlv = 3 / weighting  # value not to be exceeded in 8 hours
    return tlv