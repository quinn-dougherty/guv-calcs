import json
import pathlib
from pathlib import Path
import numpy as np
import io
import csv
import plotly.io as pio
from plotly.graph_objs._figure import Figure as plotly_fig
from matplotlib.figure import Figure as mpl_fig


def parse_loadfile(filedata):
    """
    validate and parse a loadfile
    """

    if isinstance(filedata, (str, bytes or bytearray)):
        try:
            dct = json.loads(filedata)
        except json.JSONDecodeError:
            path = Path(filedata)
            dct = load_file(path)
    elif isinstance(filedata, pathlib.PosixPath):
        dct = load_file(filedata)

    return dct


def load_file(path):
    """load json from a"""
    if path.is_file():
        if path.suffix.lower() != ".guv":
            raise ValueError("Please provide a valid .guv file")
        with open(path, "r") as json_file:
            try:
                dct = json.load(json_file)
            except json.JSONDecodeError:
                raise ValueError(".guv file is malformed")
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
        csv_data = open(datasource, mode="r")
    elif isinstance(datasource, bytes):
        # Convert bytes to a string using io.StringIO to simulate a file
        csv_data = io.StringIO(datasource.decode("utf-8"), newline="")
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
    if isinstance(fig, mpl_fig):
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png"
        )  # You can change the format as needed (e.g., 'jpeg', 'pdf')
        buf.seek(0)  # Rewind the buffer
        byt = buf.getvalue()
    elif isinstance(fig, plotly_fig):
        byt = pio.to_image(fig, format="png", scale=1)
    else:
        raise TypeError("This figure type cannot be converted to bytes")
    return byt


def get_lamp_positions(num_lamps, x, y, num_divisions=100):
    """
    generate a list of (x,y) positions for a lamp given room dimensions and
    the number of lamps desired
    """
    lst = [new_lamp_position(i + 1, x, y) for i in range(num_lamps)]
    return np.array(lst).T


def new_lamp_position(lamp_idx, x, y, num_divisions=100):
    """
    get the default position for an additional new lamp
    x and y are the room dimensions
    first index is 1, not 0.
    """
    xp = np.linspace(0, x, num_divisions + 1)
    yp = np.linspace(0, y, num_divisions + 1)
    xidx, yidx = _get_idx(lamp_idx, num_divisions=num_divisions)
    return xp[xidx], yp[yidx]


def _get_idx(num_points, num_divisions=100):
    grid_size = (num_divisions, num_divisions)
    return _place_points(grid_size, num_points)[-1]


def _place_points(grid_size, num_points):
    M, N = grid_size
    grid = np.zeros(grid_size)
    points = []

    # Place the first point in the center
    center = (M // 2, N // 2)
    points.append(center)
    grid[center] = 1  # Marking the grid cell as occupied

    for _ in range(1, num_points):
        max_dist = -1
        best_point = None

        for x in range(M):
            for y in range(N):
                if grid[x, y] == 0:
                    # Calculate the minimum distance to all existing points
                    min_point_dist = min(
                        [np.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in points]
                    )
                    # Calculate the distance to the nearest boundary
                    min_boundary_dist = min(x, M - 1 - x, y, N - 1 - y)
                    # Find the point where the minimum of these distances is maximized
                    min_dist = min(min_point_dist, min_boundary_dist)

                    if min_dist > max_dist:
                        max_dist = min_dist
                        best_point = (x, y)

        if best_point:
            points.append(best_point)
            grid[best_point] = 1  # Marking the grid cell as occupied
    return points
