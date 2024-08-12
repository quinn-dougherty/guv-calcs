import json
import pathlib
import numpy as np
from pathlib import Path
import io
import csv

def get_version(path) -> dict:

    version = {}
    with open(path) as f:
        exec(f.read(), version)
    return version["__version__"]


def parse_loadfile(filedata):
    """
    validate and parse a loadfile
    """

    try:
        dct = json.loads(filedata)
    except json.JSONDecodeError:
        path = Path(filedata)
        if path.is_file():
            if path.suffix.lower() != ".guv":
                raise ValueError("Please provide a valid .guv file")
            with open(filedata, "r") as json_file:
                dct = json.load(json_file)
        else:
            raise FileNotFoundError("Please provide a valid .guv file")

    return dct


def load_csv(datasource):
    """load csv data from either path or bytes"""
    if isinstance(datasource, (str, pathlib.PosixPath)):
        filepath = Path(datasource)
        filetype = filepath.suffix.lower()
        if filetype != ".csv":
            raise TypeError("Currently, only .csv files are supported")
        csv_data = open(datasource, mode="r", newline="")
    elif isinstance(datasource, bytes):
        # Convert bytes to a string using io.StringIO to simulate a file
        csv_data = io.StringIO(datasource.decode("utf-8"))
    else:
        raise TypeError(f"File type {type(datasource)} not valid")
    return csv_data


def check_savefile(filename, ext):
    """
    enforce that a savefile has the correct extension
    """

    if not ext.startswith("."):
        ext = "." + ext

    if isinstance(filename, str):
        if not filename.lower().endswith(ext):
            filename += ext
    elif isinstance(filename, pathlib.PosixPath):
        if not filename.suffix == ext:
            filename = filename.parent / (filename.name + ext)
    return filename


def validate_spectra(spectra, required_keys=None):
    """check that a spectra passed not-from-source"""

    # first, spectra must be a dict
    if not isinstance(spectra, dict):
        raise TypeError("Must be dict")

    # check any required keys
    if required_keys is not None:
        if isinstance(required_keys, list):
            for key in required_keys:
                validate_key(key, spectra)
        elif isinstance(required_keys, str):
            validate_key(required_keys, spectra)

    # check that all values within a dict are the same length
    if len(np.unique([len(val) for key, val in spectra.items()])) > 1:
        raise ValueError("All entries in the spectra dict must be of the same length")

    return spectra


def validate_key(key, dct):
    if key not in dct:
        raise KeyError("Required key {key} is absent.")


def rows_to_bytes(rows, encoding="cp1252"):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)

    # Get the CSV data from buffer, convert to bytes
    csv_data = buffer.getvalue()
    csv_bytes = csv_data.encode(encoding)  # encode to bytes
    return csv_bytes


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png"
    )  # You can change the format as needed (e.g., 'jpeg', 'pdf')
    buf.seek(0)  # Rewind the buffer
    return buf.getvalue()
