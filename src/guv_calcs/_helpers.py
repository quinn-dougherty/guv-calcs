import json
import pathlib
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy and bytes data types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode("utf8")
        return json.JSONEncoder.default(self, obj)


def parse_json(jsondata):
    # parse input type
    FILE = False
    if isinstance(jsondata, str):
        if jsondata.endswith(".json"):
            FILE = True
        else:
            dct = json.loads(jsondata)
    elif isinstance(jsondata, pathlib.PosixPath):
        FILE = True
    else:
        raise ValueError("Could not parse jsondata")
    if FILE:
        with open(jsondata, "r") as json_file:
            dct = json.load(json_file)
    return dct
