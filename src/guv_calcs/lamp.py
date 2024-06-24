from pathlib import Path
import csv
import inspect
import json
from io import StringIO
import pathlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from photompy import read_ies_data, plot_ies, total_optical_power
from .trigonometry import to_cartesian, to_polar, attitude
from ._helpers import NumpyEncoder, parse_json


class Lamp:
    """
    Represents a lamp with properties defined by a photometric data file.
    This class handles the loading of IES photometric data, orienting the lamp in 3D space,
    and provides methods for moving, rotating, and aiming the lamp

    Parameters:
    lamp_id: str
        A unique identifier for the lamp. Only required parameter.
    name: str
        Non-unique display name for the lamp
    filename: Path, str, or None
        If None or not pathlike, `filedata` must not be None
    filedata: Path or bytes or None
        Set by `filename` if filename is pathlike.
    x, y, z: floats
        Sets initial position of lamp in cartesian space
    angle: float
        Sets lamps initial rotation on its own axis.
    aimx, aimy, aimz: floats
        Sets initial aim point of lamp in cartesian space.
    spectra_source: Path or bytes or None
        Data source for spectra
    spectra_weight_source: Path or bytes or None
        Data source for spectral weighting
    spectra: dict
        Dictionary where keys are labels and values are arraylikes of shape (2,N) where N = the number of
        (wavelength, relative intensity) pairs that define the lamp's spectra. Set by `spectra_source` if provided,
        otherwise None, or may be passed directly. An unweighted spectra will have the key 'Unweighted'. If an unlabeled
        arraylike is passed, it will be assumed to be unweighted and weighted spectra will be calculated on that basis, if
        a `spectral_weight_source` was passed.
    spectral_weightings: dict
        Dictionary where keys are labels for a particular spectral weighting, and values are arraylikes of shape (2,N)
        where N = the number of (wavelength, relative intensity) pairs.
    intensity_units: str
        generally assumed to be `mW/Sr`. Future features will support other units, like uW/cm2
    radiation_type: str
        set from ies file keywords. Currently, only UVC222 is supported for GUV features.
    enabled: bool
        determines if lamp participates in calculations
    """
    
    def __init__(
        self,
        lamp_id,
        name=None,
        filename=None,
        filedata=None,
        x=None,
        y=None,
        z=None,
        angle=None,
        aimx=None,
        aimy=None,
        aimz=None,
        spectra_source=None,
        spectra=None,
        spectral_weight_source=None,
        spectral_weightings=None,
        intensity_units=None,
        radiation_type=None,
        max_irradiances=None,
        enabled=None,
    ):
        self.lamp_id = lamp_id
        self.name = lamp_id if name is None else name
        self.enabled = True if enabled is None else enabled
        # position
        self.x = 0.0 if x is None else x
        self.y = 0.0 if y is None else y
        self.z = 0.0 if z is None else z
        self.position = np.array([self.x, self.y, self.z])
        # orientation
        self.angle = 0.0 if angle is None else angle
        self.aimx = self.x if aimx is None else aimx
        self.aimy = self.y if aimy is None else aimy
        self.aimz = self.z - 1.0 if aimz is None else aimz
        self.aim(self.aimx, self.aimy, self.aimz)  # updates heading and bank

        # misc
        self.intensity_units = "mW/Sr" if intensity_units is None else intensity_units
        self.radiation_type = radiation_type
        # calc zone values will be stored here
        self.max_irradiances = {} if max_irradiances is None else max_irradiances

        # spectral weightings
        self.spectral_weight_source = spectral_weight_source
        self.spectral_weightings = (
            {} if spectral_weightings is None else spectral_weightings
        )
        if (
            self.spectral_weight_source is not None
            and len(self.spectral_weightings) == 0
        ):
            self._load_spectral_weightings()
        # spectra - unweighted and weighted
        self.spectra = {} if spectra is None else spectra
        self.spectra_source = spectra_source
        if self.spectra_source is not None and len(self.spectra) == 0:
            self._load_spectra()
            self._update_spectra()

        # load file and coordinates
        self.filename = filename
        self.filedata = filedata
        self._check_filename()

        # filename is just a label, filedata controls everything.
        if self.filedata is not None:
            self._load()
            self._orient()

    def _load_csv(self, datasource):
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

    def _load_spectra(self):
        """load spectral data from source"""
        csv_data = self._load_csv(self.spectra_source)
        reader = csv.reader(csv_data, delimiter=",")
        # read each line
        spectra = []
        for i, row in enumerate(reader):
            try:
                wavelength, intensity = map(float, row)
                spectra.append((wavelength, intensity))
            except ValueError:
                if i == 0:  # probably a header
                    continue
                else:
                    warnings.warn(f"Skipping invalid datarow: {row}")
        self.spectra["Unweighted"] = np.array(spectra).T

    def load_spectra(self, spectra_source):
        """
        external method to set self.spectra_source and invoke internal method
        _load_spectra to read the source into self.spectra.
        If weightings are present, update weighted spectra also
        If spectra_source is none, self.spectra will reset to empty
        """
        self.spectra_source = spectra_source
        if self.spectra_source is None:
            self.spectra = {}  # reset to empty
        else:
            self._load_spectra()
            if self.spectral_weight_source is not None:
                self._load_spectral_weightings()
                self._update_spectra()

    def _load_spectral_weightings(self):
        """load weightings"""
        csv_data = self._load_csv(self.spectral_weight_source)
        reader = csv.reader(csv_data, delimiter=",")
        headers = next(reader, None)  # get headers

        data = {}
        for header in headers:
            data[header] = []
        for row in reader:
            for header, value in zip(headers, row):
                data[header].append(float(value))

        # update self.spectra
        wavelengths = data[headers[0]]
        for i, (key, val) in enumerate(data.items()):
            if i == 0:
                continue
            else:
                values = np.array(val)
                self.spectral_weightings[key] = np.stack((wavelengths, values)).tolist()

    def _update_spectra(self):
        """
        weight the unweighted spectra by all potential spectral weightings and add to
        the self.spectra dict
        """

        if self.spectra is not None:
            wavelengths = self.spectra["Unweighted"][0]
            intensities = self.spectra["Unweighted"][1]
            maxval = max(intensities)
            for key, val in self.spectral_weightings.items():
                # update weights to match the spectral wavelengths we've got
                weights = np.interp(wavelengths, val[0], val[1])
                # self.spectral_weightings[key] = np.stack((wavelengths, weights))
                # weight spectra
                weighted_intensity = intensities * weights
                ratio = maxval / max(weighted_intensity)
                self.spectra[key] = np.stack(
                    (wavelengths, weighted_intensity * ratio)
                ).tolist()

    def load_weighted_spectra(self, spectral_weight_source):
        """
        external method to set self.spectral_weight_source and invoke internal method
        _load_weighted_spectra to read the weightings and update self.spectra
        """
        self.spectral_weight_source = spectral_weight_source
        self._load_spectral_weightings()
        self._update_spectra()

    def _check_filename(self):
        """
        determine datasource
        if filename is a path AND file exists AND filedata is None, it replaces filedata
        otherwise filedata stays the same
        """
        FILE_IS_PATH = False
        # if filename is string, check if it's a path
        if isinstance(self.filename, (str, pathlib.PosixPath)):
            if Path(self.filename).is_file():
                FILE_IS_PATH = True
        # if filename is a path and exists, it will replace filedata, but only if filedata wasn't specified to begin with
        if FILE_IS_PATH and self.filedata is None:
            self.filedata = self.filename

    def _load(self):
        """
        Loads lamp data from an IES file and initializes photometric properties.
        """
        self.lampdict = read_ies_data(self.filedata)
        self.valdict = self.lampdict["full_vals"]
        self.thetas = self.valdict["thetas"]
        self.phis = self.valdict["phis"]
        self.values = self.valdict["values"]
        self.interpdict = self.lampdict["interp_vals"]

        units_type = self.lampdict["units_type"]
        if units_type == 1:
            self.units = "feet"
        elif units_type == 2:
            self.units = "meters"
        else:
            msg = "Lamp dimension units could not be determined. Your ies file may be malformed. Units of meters are being assumed."
            warnings.warn(msg)
            self.units = "meters"

        self.dimensions = [
            self.lampdict["width"],
            self.lampdict["length"],
            self.lampdict["height"],
        ]
        self.input_watts = self.lampdict["input_watts"]
        self.keywords = self.lampdict["keywords"]
        if "_RADIATIONTYPE" in self.keywords.keys():
            self.radiation_type = self.keywords["_RADIATIONTYPE"]

    def _orient(self):
        """
        Initializes the orientation of the lamp based on its photometric data.
        """

        # true value coordinates
        tgrid, pgrid = np.meshgrid(self.thetas, self.phis)
        tflat, pflat = tgrid.flatten(), pgrid.flatten()
        tflat = 180 - tflat  # to account for reversed z direction
        x, y, z = to_cartesian(tflat, pflat, 1)
        self.coords = np.array([x, y, z]).T

        # photometric web coordinates
        xp, yp, zp = to_cartesian(tflat, pflat, self.values.flatten())
        self.photometric_coords = np.array([xp, yp, zp]).T

    def _recalculate_aim_point(self, dimensions=None, distance=None):
        """
        internal method to call if setting tilt/bank or orientation/heading
        if `dimensions` is passed, `distance` is not used
        """
        distance = 1 if distance is None else distance
        heading_rad = np.radians(self.heading)
        # Correcting bank angle for the pi shift
        bank_rad = np.radians(self.bank - 180)

        # Convert from spherical to Cartesian coordinates
        dx = np.sin(bank_rad) * np.cos(heading_rad)
        dy = np.sin(bank_rad) * np.sin(heading_rad)
        dz = np.cos(bank_rad)
        if dimensions is not None:
            distances = []
            dimx, dimy, dimz = dimensions
            if dx != 0:
                distances.append((dimx - self.x) / dx if dx > 0 else self.x / -dx)
            if dy != 0:
                distances.append((dimy - self.y) / dy if dy > 0 else self.y / -dy)
            if dz != 0:
                distances.append((dimz - self.z) / dz if dz > 0 else self.z / -dz)
            distance = min([d for d in distances])
        self.aim_point = self.position + np.array([dx, dy, dz]) * distance
        self.aimx, self.aimy, self.aimz = self.aim_point

    def get_total_power(self):
        """return the lamp's total optical power"""
        self.total_optical_power = total_optical_power(self.interpdict)
        return self.total_optical_power

    def reload(self, filename=None, filedata=None):
        """replace the ies file without erasing any position/rotation/eing information"""

        self.filename = filename
        self.filedata = filedata
        # if filename is a path, filedata is filename
        self._check_filename

        if self.filedata is not None:
            self._load()
            self._orient()
        else:
            self.lampdict = None
            self.valdict = None
            self.thetas = None
            self.phis = None
            self.values = None
            self.interpdict = None
            self.units = None
            self.dimensions = None
            self.input_watts = None
            self.keywords = None
            self.coords = None
            self.photometric_coords = None
            self.spectra = {}

    def transform(self, coords, scale=1):
        """
        Transforms the given coordinates based on the lamp's orientation and position.
        Applies rotation, then aiming, then scaling, then translation.
        Scale parameter should generally only be used for photometric_coords
        """
        # in case user has updated x y and z
        coords = np.array(attitude(coords.T, roll=0, pitch=0, yaw=self.angle)).T
        coords = np.array(
            attitude(coords.T, roll=0, pitch=self.bank, yaw=self.heading)
        ).T
        coords = (coords.T / scale).T + self.position
        return coords

    def get_cartesian(self, scale=1, sigfigs=9):
        """Return lamp's true position coordinates in cartesian space"""
        return self.transform(self.coords, scale=scale).round(sigfigs)

    def get_polar(self, sigfigs=9):
        """Return lamp's true position coordinates in polar space"""
        cartesian = self.transform(self.coords) - self.position
        return np.array(to_polar(*cartesian.T)).round(sigfigs)

    def move(self, x=None, y=None, z=None):
        """Designate lamp position in cartesian space"""
        # determine new position   selected_lamp.
        x = self.x if x is None else x
        y = self.y if y is None else y
        z = self.z if z is None else z
        position = np.array([x, y, z])
        # update aim point based on new position
        diff = position - self.position
        self.aim_point += diff
        self.aimx, self.aimy, self.aimz = self.aim_point
        # update position
        self.position = position
        self.x, self.y, self.z = self.position
        return self

    def rotate(self, angle):
        """designate lamp orientation with respect to its z axis"""
        self.angle = angle
        return self

    def set_orientation(self, orientation, dimensions=None, distance=None):
        """
        set orientation/heading.
        alternative to setting aim point with `aim`
        distinct from rotation; applies to a tilted lamp. to rotate a lamp along its axis,
        use the `rotate` method
        """
        # orientation = (orientation + 360) % 360
        self.heading = orientation
        self._recalculate_aim_point(dimensions=dimensions, distance=distance)

    def set_tilt(self, tilt, dimensions=None, distance=None):
        """
        set tilt/bank
        alternative to setting aim point with `aim`
        """
        # tilt = (tilt + 360) % 360
        self.bank = tilt
        self._recalculate_aim_point(dimensions=dimensions, distance=distance)

    def aim(self, x=None, y=None, z=None):
        """aim lamp at a point in cartesian space"""
        x = self.aimx if x is None else x
        y = self.aimy if y is None else y
        z = self.aimz if z is None else z
        self.aim_point = np.array([x, y, z])
        self.aimx, self.aimy, self.aimz = self.aim_point
        xr, yr, zr = self.aim_point - self.position
        self.heading = np.degrees(np.arctan2(yr, xr))
        self.bank = np.degrees(np.arctan2(np.sqrt(xr ** 2 + yr ** 2), zr) - np.pi)
        # self.heading = (heading+360)%360
        # self.bank = (bank+360)%360
        return self

    def plot_ies(self, title=""):
        """standard polar plot of an ies file"""
        fig, ax = plot_ies(fdata=self.valdict, title=title)
        return fig, ax

    def plot_spectra(self, title=None, fig=None, figsize=(6.4, 4.8), yscale="linear"):
        """
        plot the spectra of the lamp. at minimum, the unweighted spectra, possibly all
        weighted spectra as well.

        `yscale` is generally either "linear" or "log", but any matplotlib scale is permitted
        """

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]

        if len(self.spectra) > 0:
            for key, val in self.spectra.items():
                linestyle = "-" if key == "Unweighted" else "--"
                ax.plot(val[0], val[1], label=key, linestyle=linestyle)
            ax.legend()
            ax.grid(True, which="both", ls="--", c="gray", alpha=0.3)
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Relative intensity [%]")
            ax.set_yscale(yscale)

            title = self.name if title is None else title
            ax.set_title(title)
        return fig

    def plot_3d(
        self,
        elev=45,
        azim=-45,
        title="",
        figsize=(6, 4),
        show_cbar=False,
        alpha=0.7,
        cmap="rainbow",
        fig=None,
        ax=None,
    ):
        """plot in cartesian 3d space of the true positions of the irradiance values"""
        x, y, z = self.transform(self.coords).T
        intensity = self.values.flatten()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=intensity, cmap="rainbow", alpha=alpha)
        if self.aim_point is not None:
            ax.plot(
                *np.array((self.aim_point, self.position)).T,
                linestyle="--",
                color="black",
                alpha=0.7,
            )
        ax.set_title(title)
        ax.view_init(azim=azim, elev=elev)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return fig, ax

    def plot_web(
        self,
        elev=30,
        azim=-60,
        title="",
        figsize=(6, 4),
        color="#cc61ff",
        alpha=0.4,
        xlim=None,
        ylim=None,
        zlim=None,
    ):
        """plot photometric web, where distance r is set by the irradiance value"""
        scale = self.values.max()
        x, y, z = self.transform(self.photometric_coords, scale=scale).T
        Theta, Phi, R = to_polar(*self.photometric_coords.T)
        tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, color=color, alpha=alpha)
        if self.aim_point is not None:
            ax.plot(
                *np.array((self.aim_point, self.position)).T,
                linestyle="--",
                color="black",
                alpha=0.7,
            )
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        ax.view_init(azim=azim, elev=elev)
        return fig, ax

    @classmethod
    def from_json(cls, jsondata):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        data = parse_json(jsondata)
        # convert the contents of these dicts from lists to arrays
        special_keys = ["spectra", "spectral_weightings"]
        for key, val in data.items():
            if key in special_keys:
                data[key] = {k: np.array(v) for k, v in data[key].items()}
        return cls(**{k: v for k, v in data.items() if k in keys})

    def to_json(self):
        # Create a dictionary of all instance variables
        data = {attr: getattr(self, attr) for attr in vars(self)}
        return json.dumps(data, cls=NumpyEncoder)
