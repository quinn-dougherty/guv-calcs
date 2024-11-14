import os
import pandas as pd
import warnings
from importlib import resources
import numpy as np
import csv
from .spectrum import Spectrum
from ._helpers import load_csv


def get_full_disinfection_table():
    fname = "UVC Inactivation Constants.csv"
    path = resources.files("guv_calcs.data").joinpath(fname)
    return pd.read_csv(path)


def get_disinfection_table(fluence, wavelength=None, room=None):
    """
    Retrieve and format inactivtion data for this room.

    Currently assumes all lamps are GUV222. in the future will need something
    cleverer than this
    """

    df = get_full_disinfection_table()

    valid_wavelengths = df["wavelength [nm]"].unique()
    if wavelength not in valid_wavelengths:
        warnings.warn(f"No data is available for wavelength {wavelength} nm.")

    df = df[df["Medium"] == "Aerosol"]
    if wavelength is not None:
        df = df[df["wavelength [nm]"] == wavelength]

    # calculate eACH before filling nans
    k1 = df["k1 [cm2/mJ]"].fillna(0).astype(float)
    k2 = df["k2 [cm2/mJ]"].fillna(0).astype(float)
    f = df["% resistant"].str.rstrip("%").astype("float").fillna(0) / 100
    eACH = (k1 * (1 - f) + k2 - k2 * (1 - f)) * fluence * 3.6
    df["eACH-UV"] = eACH.round(1)

    newkeys = ["eACH-UV"]

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
        columns={
            "Medium (specific)": "Medium",
            "Full Citation": "Reference",
            "Organism": "Category",
        }
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


def get_standards():
    """
    load possible standards, as a list of strings
    """
    return list(get_spectral_weightings().keys())[1:]


def get_tlvs(ref, std=0):
    """
    return the value of the UV dose not to be exceeded over 8 hours,
    assuming a monochromatic wavelength. standard may be either an exact 
    string match or integer corresponding to this dictionary:
    0: `ANSI IES RP 27.1-22`
    1: `IEC 62471-6:2022`
    """
    
    standards = ["ANSI IES RP 27.1-22", "IEC 62471-6:2022"]
    # check user inputs
    msg =f"{std} is not a valid spectral weighting standard. Select one of {standards}"
    if isinstance(std,int):
        if std>len(standards)-1:
            raise KeyError(msg)
        else:
            key = standards[std]
    elif isinstance(std, str):
        if std not in standards:
            # check for a substring
            subkey = ''.join([os.path.commonprefix([std,val]) for val in standards])
            if subkey not in standards:
                raise KeyError(msg)
            else:
                key = subkey
        else:
            key = std
    else:
        raise TypeError(f"{type(std)} is not a valid type for argument std")
            
    if key == standards[0]:
        skinkey = key + " (Skin)"
        eyekey = key + " (Eye)"
    elif key == standards[1]:
        skinkey  = key + " (Eye/Skin)"
        eyekey = skinkey

    skin_tlv = get_tlv(ref, skinkey)
    eye_tlv = get_tlv(ref, eyekey)

    return skin_tlv, eye_tlv
    

def get_tlv(ref, standard=0):
    """
    return the value of the UV dose not to be exceeded over 8 hours,
    assuming a monochromatic wavelength. standard may be either an exact 
    string match or integer corresponding to this dictionary:
    0: `ANSI IES RP 27.1-22 (Eye)`
    1: `ANSI IES RP 27.1-22 (Skin)`
    2: `IEC 62471-6:2022 (Eye/Skin)`
    """

    weights = get_spectral_weightings()
    valid_keys = list(weights.keys())[1:]
    msg =f"{standard} is not a valid spectral weighting standard. Select one of {valid_keys}"
    if isinstance(standard,str):
        if standard not in valid_keys:
            raise KeyError(msg)
        else:
            key = standard
    elif isinstance(standard,int):
        if standard>len(valid_keys)-1:
            raise KeyError(msg)
        else:
            key = valid_keys[standard]
    else:
        raise TypeError(f"{type(standard)} is not a valid type for argument std")
        
    if isinstance(ref,(int,float)):
        tlv_wavelengths = weights["Wavelength (nm)"]
        tlv_values = weights[key]
        weighting = np.interp(ref, tlv_wavelengths, tlv_values)
        tlv = 3 / weighting  # value not to be exceeded in 8 hours
    elif isinstance(ref,Spectrum):
        tlv = ref.get_tlv(key)
    else:
        msg = f"Argument `ref` must be either float, int, or Spectrum object, not {type(ref)}"
        raise TypeError(msg)
    return tlv

def get_version(path) -> dict:

    version = {}
    with open(path) as f:
        exec(f.read(), version)
    return version["__version__"]
