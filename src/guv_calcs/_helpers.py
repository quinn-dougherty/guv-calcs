import json
import pathlib
import numpy as np
from pathlib import Path
from io import StringIO


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy and bytes data types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode("utf8")
        return json.JSONEncoder.default(self, obj)


def get_version(path) -> dict:

    version = {}
    with open(path) as f:
        exec(f.read(), version)
    return version["__version__"]


def parse_json(jsondata):
    # parse input type
    FILE = False
    if isinstance(jsondata, str):
        if Path(jsondata).is_file():
            FILE = True
        if jsondata.lower().endswith(".json"):
            FILE = True
    elif isinstance(jsondata, pathlib.PosixPath):
        FILE = True
    else:
        raise ValueError("Could not parse jsondata")

    if FILE:
        with open(jsondata, "r") as json_file:
            dct = json.load(json_file)
    else:
        dct = json.loads(jsondata)
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
        # Convert bytes to a string using StringIO to simulate a file
        csv_data = StringIO(datasource.decode("utf-8"))
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
