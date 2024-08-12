from pathlib import Path
import csv
import inspect
import json
from importlib import resources
import pathlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from photompy import read_ies_data, plot_ies, total_optical_power
from .trigonometry import to_cartesian, to_polar, attitude
from ._helpers import load_csv, validate_spectra, rows_to_bytes


class Lamp:
    """
    Represents a lamp with properties defined by a photometric data file.
    This class handles the loading of IES photometric data, orienting the lamp in 3D space,
    and provides methods for moving, rotating, and aiming the lamp.

    Arguments
    -------------------
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
        Optional. Data source for spectra. Alternatively, you
        may pass a 'spectra' dict directly. *If a spectra is passed,
        spectra_source will be ignored.*
    spectra: dict
        Opotional. Dictionary containing at least the keys "Wavelength" and
        "Unweighted Relative Intensity", where the values of each key are an array of the same
        size. The values of the "Unweighted Relative Intensity" key correspond to the raw
        relative intensity values of the lamp spectra. Any further keys
        are optional. If provided, will supercede spectra_source.
    enabled: bool
        Determines if lamp participates in calculations. A lamp may be created
        and added to a room, but disabled.

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
        enabled=None,
    ):

        """
        TODO:
            1. possibly worth removing the spectra and spectral_weightings arguments
        and have it only possible for users to set them from a source?
            2. probably __init__ needs to change initialization strategy
            3. possibly Spectra should be its own class? idk...
            4. in general much to think about

        """

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

        # calc zone values will be stored here
        self.max_irradiances = {}
        # spectral weightings
        self.spectral_weightings = {}
        self._load_spectral_weightings()

        # load spectra - unweighted and weighted
        self.spectra_source = spectra_source
        if spectra is None:
            self.spectra = {}
            if self.spectra_source is not None:
                self._load_spectra()
                self._update_spectra()
        else:
            self.spectra = validate_spectra(spectra)

        # load file and coordinates
        self.filename = filename
        self.filedata = filedata
        self._check_filename()

        # filename is just a label, filedata controls everything.
        if self.filedata is not None:
            self._load()
            self._orient()

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

    def get_total_power(self):
        """return the lamp's total optical power"""
        self.total_optical_power = total_optical_power(self.interpdict)
        return self.total_optical_power

    def get_cartesian(self, scale=1, sigfigs=9):
        """Return lamp's true position coordinates in cartesian space"""
        return self.transform(self.coords, scale=scale).round(sigfigs)

    def get_polar(self, sigfigs=9):
        """Return lamp's true position coordinates in polar space"""
        cartesian = self.transform(self.coords) - self.position
        return np.array(to_polar(*cartesian.T)).round(sigfigs)

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
                if key == "Wavelength":
                    continue
                linestyle = "-" if key == "Unweighted Relative Intensity" else "--"
                alpha = 1 if key == "Unweighted Relative Intensity" else 0.7
                ax.plot(
                    self.spectra["Wavelength"],
                    val,
                    label=key,
                    linestyle=linestyle,
                    alpha=alpha,
                )
            ax.legend()
            ax.grid(True, which="both", ls="--", c="gray", alpha=0.3)
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Relative intensity [%]")
            ax.set_yscale(yscale)

            title = self.name if title is None else title
            ax.set_title(title)
        return fig

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
        """
        plot in cartesian 3d space of the true positions of the irradiance values
        mostly a convenience visualization function. Generally irradiance values
        should use a polar plot.
        """
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

    def reload(self, filename=None, filedata=None):
        """
        replace the ies file without erasing any position/rotation/eing information
        can be used to load an ies file after initialization
        """

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

    def load_spectra(self, spectra_source):
        """
        external method to reload the spectra after initialization
        If weightings are present, update weighted spectra also
        If spectra_source is none, self.spectra will reset to empty
        """
        self.spectra_source = spectra_source
        if self.spectra_source is None:
            self.spectra = {}  # reset to empty
        else:
            self._load_spectra()
            self._load_spectral_weightings()
            self._update_spectra()

    def _load_spectra(self):
        
        """load spectral data from source"""
        csv_data = load_csv(self.spectra_source)
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
        self.spectra["Wavelength"] = np.array(spectra).T[0]
        self.spectra["Unweighted Relative Intensity"] = np.array(spectra).T[1]

    def _load_spectral_weightings(self):
        """load spectral weightings"""
        # load weights from within package
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

        for i, (key, val) in enumerate(data.items()):
            if i == 0:
                self.spectral_weightings["Wavelength"] = np.array(val)
                continue
            else:
                self.spectral_weightings[key] = np.array(val)

    def _update_spectra(self):
        """
        weight the unweighted spectra by all potential spectral weightings and add to
        the self.spectra dict
        """

        if self.spectra is not None:
            wavelengths = self.spectra["Wavelength"]
            intensities = self.spectra["Unweighted Relative Intensity"]
            maxval = max(intensities)
            weighted_wavelengths = self.spectral_weightings["Wavelength"]
            for key, val in self.spectral_weightings.items():
                if key == "Wavelength":
                    continue
                # update weights to match the spectral wavelengths we've got
                weights = np.interp(wavelengths, weighted_wavelengths, val)
                # weight spectra
                weighted_intensity = intensities * weights
                ratio = maxval / max(weighted_intensity)
                self.spectra[key] = weighted_intensity * ratio
        else:
            warnings.warn(
                "Spectra was not updated with weights as a spectra was not provided."
            )

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
            self.filedata = Path(self.filename).read_text()

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

    @classmethod
    def from_dict(cls, data):
        """initialize class from dict"""
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        for k, v in data["spectra"].items():
            if isinstance(v, str):
                lst = list(map(float, v.split(", ")))
            elif isinstance(v, list):
                lst = v
            data["spectra"][k] = np.array(lst)
        return cls(**{k: v for k, v in data.items() if k in keys})

    def save_spectra(self, fname=None):
        """"""
        rows = [list(self.spectra.keys())]
        rows += list(np.array(list(self.spectra.values())).T)
        csv_bytes = rows_to_bytes(rows)
        if fname is not None:
            with open(fname, "wb") as csvfile:
                csvfile.write(csv_bytes)
        else:
            return csv_bytes

    def save_ies(self, fname=None):
        if isinstance(self.filedata,str):
            iesbytes = self.filedata.encode("utf-8")
        elif isinstance(self.filedata,bytes):
            iesbytes = self.filedata
        if fname is not None:
            with open(fname, "wb") as file:
                file.write(iesbytes)
        else:
            return iesbytes

    def save_lamp(self, filename=None):
        """
        save just the minimum number of parameters required to re-instantiate the lamp
        Returns dict. If filename is not None, saves dict as json.
        Does not save calculation data like max_irradiances.
        """

        data = {}
        data["lamp_id"] = self.lamp_id
        data["name"] = self.name
        data["x"] = self.x
        data["y"] = self.y
        data["z"] = self.z
        data["angle"] = self.angle
        data["aimx"] = self.aimx
        data["aimy"] = self.aimy
        data["aimz"] = self.aimz

        if isinstance(self.filedata, bytes):
            filedata = self.filedata.decode("utf-8")
        elif isinstance(self.filedata, str) or self.filedata is None:
            filedata = self.filedata
        else:
            raise TypeError(f"Filedata must be str or bytes, not {type(self.filedata)}")
        data["filedata"] = filedata

        # this is just so that the file looks nicer in a text editor
        spectra_string = {}
        for key in ["Wavelength","Unweighted Relative Intensity"]:
            spectra_string[key] = ", ".join(map(str, self.spectra[key]))
        data["spectra"] = spectra_string

        if filename is not None:
            with open(filename, "w") as json_file:
                json.dump(data, json_file, indent=4)

        return data
