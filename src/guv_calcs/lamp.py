from pathlib import Path
from importlib import resources
import inspect
import json
import pathlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from photompy import read_ies_data, write_ies_data, plot_ies, total_optical_power
from .spectrum import Spectrum
from .trigonometry import to_cartesian, to_polar, attitude
from ._data import get_tlvs

VALID_LAMPS = [
    "beacon",
    "lumenizer_zone",
    "sterilray",
    "ushio_b1",
    "ushio_b1.5",
    "uvpro222_b1",
    "uvpro222_b2",
    "visium",
]

KRCL_KEYS = [
    "krypton chloride",
    "krypton-chloride",
    "krypton_chloride",
    "krcl",
    "kr-cl",
    "kr cl",
]

LPHG_KEYS = [
    "low pressure mercury",
    "low-pressure mercury",
    "mercury",
    "lphg",
    "lp-hg",
    "lp hg",
]


class Lamp:
    """
    Represents a lamp with properties defined by a photometric data file.
    This class handles the loading of IES photometric data, orienting the lamp in 3D space,
    and provides methods for moving, rotating, and aiming the lamp.

    Arguments
    -------------------
    lamp_id: str
        A unique identifier for the lamp object.
    name: str, default=None
        Non-unique display name for the lamp. If None set by lamp_id
    filename: Path, str
        If None or not pathlike, `filedata` must not be None
    filedata: Path or bytes, default=None
        Set by `filename` if filename is pathlike.
    x, y, z: floats, default=[0,0,0]
        Sets initial position of lamp in cartesian space
    angle: float, default=0
        Sets lamps initial rotation on its own axis.
    aimx, aimy, aimz: floats, default=[0,0,z-1]
        Sets initial aim point of lamp in cartesian space.
    guv_type: str
        Optional label for type of GUV source. Presently available:
        ["Krypton chloride (222 nm)", "Low-pressure mercury (254 nm)", "Other"]
    wavelength: float
        Optional label for principle GUV wavelength. Set from guv_type if guv_type
        is not "Other".
    spectra_source: Path or bytes, default=None
        Optional. Data source for spectra. May be a filepath, a binary stream,
        or a dict where the first value contains values of wavelengths, and
        the second value contains values of relative intensity.
    width, length, depth: floats, default=[None, None]
        x-axis and y-axis source extent, plus fixture depth in the units
        provided. If not provided, will be read from the .ies file. Note that
        the ies file interface labels `depth` as `height` instead.
    units: str or int in [1, 2] or None
        `feet` or `meters`. 1 corresponds to feet, 2 to `meters`. If not
        provided, will be read from .ies file, and lengt and width parameters
        will be ignored.
    source_density: int or float, default=1
        parameter that determines the fineness of the source discretization.
        Grid size follows fibonacci sequence. For an approximately square
        source, SD=1 => 1x1 grid, SD=2 => 3x3 grid, SD=3 => 5x5 grid. This is
        to ensure that a center point is always present while ensuring evenness
        of grid size.
    relative_map: arraylike
        A relative intensity map for non-uniform sources. Must be of the same
        size as the grid generated
    enabled: bool, defualt=True
        Determines if lamp participates in calculations. A lamp may be created
        and added to a room, but disabled.

    """

    def __init__(
        self,
        lamp_id=None,
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
        intensity_units=None,
        guv_type=None,
        wavelength=None,
        spectra_source=None,
        width=None,
        length=None,
        depth=None,
        units=None,
        source_density=None,
        relative_map=None,
        enabled=None,
    ):

        """
        TODO:
            probably __init__ needs to change initialization strategy. some kind of from_file?
            unfortunately this object is initialized with two files...maybe that is an issue

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

        # units of values, since not directly specified in .ies file and they can vary for GUV fixtures
        if intensity_units is not None:
            msg = f"Intensity unit {intensity_units} not recognized. Using default value `mW/Sr`"
            if isinstance(intensity_units, int):
                if intensity_units == 0:
                    self.intensity_units = "mW/sr"
                elif intensity_units == 1:
                    self.intensity_units = "uW/cm2"
                else:
                    warnings.warn(msg, stacklevel=3)
                    self.intensity_units = "mW/sr"
            elif isinstance(intensity_units, str):
                if intensity_units.lower() in ["mw/sr", "uw/cm2", "uw/cm²"]:
                    self.intensity_units = intensity_units
                else:
                    warnings.warn(msg, stacklevel=3)
                    self.intensity_units = "mW/sr"
            else:
                msg = f"Datatype {type(intensity_units)} for intensity units not recognized. Using default value `mW/Sr`"
                self.intensity_units = "mW/sr"
        else:
            self.intensity_units = "mW/sr"

        # source type & wavelength
        self.guv_type = guv_type
        if guv_type is not None:
            if any([key in guv_type.lower() for key in KRCL_KEYS]):
                self.wavelength = 222
            elif any([key in guv_type.lower() for key in LPHG_KEYS]):
                self.wavelength = 254
            else:
                self.wavelength = wavelength
        else:
            self.wavelength = wavelength
        if self.wavelength is not None:
            if not isinstance(self.wavelength, (int, float)):
                msg = f"Wavelength must be int or float, not {type(self.wavelength)}"
                raise TypeError(msg)

        # source values
        self.width = width
        self.length = length
        self.depth = depth
        self.units = units
        self.source_density = 1 if source_density is None else source_density
        self.relative_map = self._load_relmap(relative_map)

        self.grid_points = None  # populated from ies data
        self.photometric_distance = None  # ditto

        # aim
        self.aim(self.aimx, self.aimy, self.aimz)  # updates heading and bank

        self.spectra_source = spectra_source
        self.spectra = self._load_spectra(spectra_source)

        # load file and coordinates
        self.filename = filename
        self.filedata = filedata
        self._check_filename()

        # values that will be filled in if there is filedata available
        self.lampdict = None
        self.thetas = None
        self.phis = None
        self.values = None


        # filename is just a label, filedata controls everything.
        if self.filedata is not None:
            self._load()
            self._orient()

    @classmethod
    def from_dict(cls, data):
        """initialize class from dict"""
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        if data["spectra"] is not None:
            data["spectra_source"] = {}
            for k, v in data["spectra"].items():
                if isinstance(v, str):
                    lst = list(map(float, v.split(", ")))
                elif isinstance(v, list):
                    lst = v
                data["spectra_source"][k] = np.array(lst)
        return cls(**{k: v for k, v in data.items() if k in keys})

    @classmethod
    def from_keyword(
        cls,
        key,
        lamp_id=None,
        name=None,
        x=None,
        y=None,
        z=None,
        angle=None,
        aimx=None,
        aimy=None,
        aimz=None,
        intensity_units=None,
        guv_type=None,
        wavelength=None,
        width=None,
        length=None,
        depth=None,
        units=None,
        source_density=None,
        relative_map=None,
        enabled=None,
    ):

        if key.lower() in VALID_LAMPS:
            path = "guv_calcs.data.lamp_data"
            fname = key.lower() + ".ies"
            spec_fname = key.lower() + ".csv"
            filename = resources.files(path).joinpath(fname)
            spectra_source = resources.files(path).joinpath(spec_fname)
        else:
            msg = f"{key} is not a valid lamp key. Photometric and spectral data will not be loaded."
            warnings.warn(msg, stacklevel=3)
            filename = None
            spectra_source = None

        return cls(
            lamp_id=lamp_id,
            name=name,
            filename=filename,
            x=x,
            y=y,
            z=z,
            angle=angle,
            aimx=aimx,
            aimy=aimy,
            aimz=aimz,
            intensity_units=intensity_units,
            guv_type=guv_type,
            wavelength=wavelength,
            spectra_source=spectra_source,
            width=width,
            length=length,
            depth=depth,
            units=units,
            source_density=source_density,
            relative_map=relative_map,
            enabled=enabled,
        )

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
        self.grid_points = self._generate_source_points()
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
        # update grid points
        self.grid_points = self._generate_source_points()
        return self

    def transform(self, coords, scale=1):
        """
        Transforms the given coordinates based on the lamp's orientation and position.
        Applies rotation, then aiming, then scaling, then translation.
        Scale parameter should generally only be used for photometric_coords
        """
        coords = np.array(attitude(coords.T, roll=0, pitch=0, yaw=self.angle)).T
        coords = np.array(
            attitude(coords.T, roll=0, pitch=self.bank, yaw=self.heading)
        ).T
        coords = (coords.T / scale).T + self.position
        return coords

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
        return self

    def set_tilt(self, tilt, dimensions=None, distance=None):
        """
        set tilt/bank
        alternative to setting aim point with `aim`
        """
        # tilt = (tilt + 360) % 360
        self.bank = tilt
        self._recalculate_aim_point(dimensions=dimensions, distance=distance)
        return self

    def set_source_density(self, source_density):
        """change source discretization"""
        self.source_density = source_density
        self.grid_points = self._generate_source_points()

    def set_source_width(self, width):
        """change x-axis extent of lamp emissive surface"""
        self.width = width
        self.grid_points = self._generate_source_points()

    def set_source_length(self, length):
        """change y-axis extent of lamp emissive surface"""
        self.length = length
        self.grid_points = self._generate_source_points()

    def get_total_power(self):
        """return the lamp's total optical power"""
        return total_optical_power(self.lampdict["interp_vals"])

    def get_limits(self, standard=0):
        """
        get the threshold limit values for this lamp. Returns tuple
        (skin_limit, eye_limit) Will use the lamp spectrum if provided;
        if not provided will use wavelength; if neither is defined, returns
        (None, None). Standard may be a string in:
            [`ANSI IES RP 27.1-22`, `IEC 62471-6:2022`]
        Or an integer corresponding to the index of the desired standard.
        """
        if self.spectra is not None:
            skin_tlv, eye_tlv = get_tlvs(self.spectra, standard)
        elif self.wavelength is not None:
            skin_tlv, eye_tlv = get_tlvs(self.wavelength, standard)
        else:
            skin_tlv, eye_tlv = None, None
        return skin_tlv, eye_tlv

    def get_cartesian(self, scale=1, sigfigs=9):
        """Return lamp's true position coordinates in cartesian space"""
        return self.transform(self.coords, scale=scale).round(sigfigs)

    def get_polar(self, sigfigs=9):
        """Return lamp's true position coordinates in polar space"""
        cartesian = self.transform(self.coords) - self.position
        return np.array(to_polar(*cartesian.T)).round(sigfigs)

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
            self.thetas = None
            self.phis = None
            self.values = None
            self.units = None
            self.dimensions = None
            self.input_watts = None
            self.keywords = None
            self.coords = None
            self.photometric_coords = None
            self.spectra = None

    def load_spectra(self, spectra_source):
        """external method for loading spectra after lamp object has been instantiated"""
        self.spectra = self._load_spectra(spectra_source)

    def load_relmap(self, relative_map):
        """external method for loading relative intensity map after lamp object has been instantiated"""
        self.rel_map = self._load_relmap(relative_map)

    def save_ies(self, fname=None, original=False):
        """
        Save the current lamp paramters as an .ies file; alternatively, save the original ies file.
        """
        if self.lampdict is not None:
            if original:
                iesbytes = write_ies_data(self.lampdict, valkey="original_vals")
            else:
                updated_lampdict = self.get_lampdict()
                iesbytes = write_ies_data(updated_lampdict, valkey="original_vals")

            # write to file if provided, otherwise
            if fname is not None:
                with open(fname, "wb") as file:
                    file.write(iesbytes)
            else:
                return iesbytes
        else:
            return None

    def save_lamp(self, filename=None):
        """
        save just the minimum number of parameters required to re-instantiate the lamp
        Returns dict. If filename is not None, saves dict as json.
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
        data["intensity_units"] = self.intensity_units
        data["guv_type"] = self.guv_type
        data["wavelength"] = self.wavelength
        data["width"] = self.width
        data["length"] = self.length
        data["depth"] = self.depth
        data["units"] = self.units
        data["source_density"] = self.source_density
        data["relative_map"] = list(self.relative_map) if self.relative_map is not None else None

        data["filename"] = self.filename
        filedata = self.save_ies()
        data["filedata"] = filedata.decode() if filedata is not None else None
        

        if self.spectra is not None:
            spectra_dict = self.spectra.to_dict(as_string=True)
            keys = list(spectra_dict.keys())[0:2]  # keep the first two keys only
            data["spectra"] = {key: spectra_dict[key] for key in keys}
        else:
            data["spectra"] = None

        if filename is not None:
            with open(filename, "w") as json_file:
                json.dump(data, json_file, indent=4)

        return data

    def plot_ies(self, title=""):
        """standard polar plot of an ies file"""
        fig, ax = plot_ies(fdata=self.lampdict["full_vals"], title=title)
        return fig, ax

    def plot_grid_points(self, fig=None, ax=None, title="", figsize=(6, 4)):
        """plot the discretization of the emissive surface"""

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]

        ax.scatter(*self.grid_points.T[0:2])
        if self.width:
            ax.set_xlim(-self.width / 2, self.width / 2)
        if self.length:
            ax.set_ylim(-self.length / 2, self.length / 2)
        if title is None:
            title = "Source density = " + str(self.source_density)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        return fig, ax

    def plot_relative_map(self, fig=None, ax=None, title="", figsize=(6, 4)):
        """plot the relative intensity map of the emissive surface"""
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]

        if self.width and self.length:
            extent = [
                -self.width / 2,
                self.width / 2,
                -self.length / 2,
                self.length / 2,
            ]
            img = ax.imshow(self.relative_map, extent=extent)
        else:
            img = ax.imshow(self.relative_map)
        cbar = fig.colorbar(img, pad=0.03)
        cbar.set_label("Surface relative intensity", loc="center")

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

    def plot_3d(
        self,
        elev=45,
        azim=-45,
        title="",
        figsize=(6, 4),
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
            else:
                if self.filedata is None:
                    warnings.warn(
                        f"File {self.filename} not found. Provide a valid file or the filedata."
                    )
        # if filename is a path and exists, it will replace filedata, but only if filedata wasn't specified to begin with
        if FILE_IS_PATH and self.filedata is None:
            self.filedata = Path(self.filename).read_text()

    def _load(self):
        """
        Loads lamp data from an IES file and initializes photometric properties.
        """
        
        self.lampdict = read_ies_data(self.filedata)
        self.thetas = self.lampdict["full_vals"]["thetas"]
        self.phis = self.lampdict["full_vals"]["phis"]
        self.values = self.lampdict["full_vals"]["values"]

        if not all([self.width, self.length, self.units]):
            if any([self.width, self.length, self.units]):
                msg = "Width, length, and units arguments will be ignored and set from the .ies file instead."
                warnings.warn(msg, stacklevel=2)
            units_type = self.lampdict["units_type"]
            if units_type == 1:
                self.units = "feet"
            elif units_type == 2:
                self.units = "meters"
            else:
                msg = "Lamp dimension units could not be determined. Your ies file may be malformed. Units of meters are being assumed."
                warnings.warn(msg, stacklevel=2)
                self.units = "meters"

            self.width = self.lampdict["width"]
            self.length = self.lampdict["length"]
            self.depth = self.lampdict["height"]

        self.photometric_distance = max(self.width, self.length) * 10
        self.grid_points = self._generate_source_points()

        if self.relative_map is None:
            self.relative_map = np.ones(len(self.grid_points))
        if len(self.relative_map) != len(self.grid_points):
            self.relative_map = np.ones(len(self.grid_points))

        # self.input_watts = self.lampdict["input_watts"]
        # self.keywords = self.lampdict["keywords"]

    def get_lampdict(self):
        """
        update the lampdict object for writing a new ies file
        TODO: maybe include multiplier somehow?
        """

        lampdict = self.lampdict.copy()
        if self.units == "feet":
            lampdict["units_type"] = 1
        elif self.units == "meters":
            lampdict["units_type"] = 2

        lampdict["width"] = self.width
        lampdict["length"] = self.length
        lampdict["height"] = self.depth
        return lampdict

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

    def _load_spectra(self, spectra_source):
        """initialize a Spectrum object from the source"""
        if isinstance(spectra_source, dict):
            spectra = Spectrum.from_dict(spectra_source)
        elif isinstance(spectra_source, (str, pathlib.Path, bytes)):
            spectra = Spectrum.from_file(spectra_source)
        elif isinstance(spectra_source, tuple):
            spectra = Spectrum(spectra_source[0], spectra_source[1])
        elif spectra_source is None:
            spectra = None
        else:
            spectra = None
            msg = f"Datatype {type(spectra_source)} not recognized spectral data source"
            warnings.warn(msg, stacklevel=3)
        return spectra

    def _load_relmap(self, arg):
        """check filetype and return correct relative_map as array"""
        if arg is None:
            rel_map = None
        elif isinstance(arg, (str, pathlib.Path)):
            # check if this is a file
            if Path(arg).is_file():
                rel_map = np.genfromtxt(Path(arg), delimiter=",")
            else:
                msg = f"File {arg} not found. relative_map will not be used."
                warnings.warn(msg, stacklevel=3)
                rel_map = None
        elif isinstance(arg, bytes):
            rel_map = np.frombuffer(arg)
        elif isinstance(arg, (list, np.array)):
            rel_map = np.array(arg)
        else:
            msg = f"Argument type {type(arg)} for argument relative_map. relative_map will not be used."
            warnings.warn(msg, stacklevel=3)
            rel_map = None
        return rel_map

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
        self.grid_points = self._generate_source_points()

    def _generate_raw_points(self):
        """generate grid points, prior to transforming them"""
        num_points = self.source_density + self.source_density - 1
        num_points_v = max(
            num_points, num_points * int(round(self.width / self.length))
        )
        num_points_u = max(
            num_points, num_points * int(round(self.length / self.width))
        )
        if num_points_u % 2 == 0:
            num_points_u += 1
        if num_points_v % 2 == 0:
            num_points_v += 1

        # spacing = min(self.length, self.width) / num_points
        spacing_v = self.width / num_points_v
        spacing_u = self.length / num_points_u

        # If there's only one point, place it at the center
        if num_points_v == 1:
            v_points = np.array([0])  # Single point at the center of the width
        else:
            startv = -self.width / 2 + spacing_v / 2
            stopv = self.width / 2 - spacing_v / 2
            v_points = np.linspace(startv, stopv, num_points_v)

        if num_points_u == 1:
            u_points = np.array([0])  # Single point at the center of the length
        else:
            startu = -self.length / 2 + spacing_u / 2
            stopu = self.length / 2 - spacing_u / 2
            u_points = np.linspace(startu, stopu, num_points_u)

        return np.meshgrid(v_points, u_points)

    def _generate_source_points(self):
        """
        generate the points with which the calculations should be performed.
        If the source is approximately square and source_density is 1, only
        one point is generated. If source is more than twice as long as wide,
        (or vice versa), 2 or more points will be generated even if density is 1.
        Total number of points will increase quadratically with density.
        """

        # generate the points

        if all([self.width, self.length, self.source_density]):

            vv, uu = self._generate_raw_points()
            # get the normal plane to the aim point
            # Normalize the direction vector (normal vector)
            direction = self.position - self.aim_point
            normal = direction / np.linalg.norm(direction)

            # Generate two vectors orthogonal to the normal
            if np.allclose(normal, [1, 0, 0]):
                # if normal is close to x-axis, use y and z to define the plane
                u = np.array([0, 1, 0])
            else:
                u = np.cross(normal, [1, 0, 0])
            u = u / np.linalg.norm(u)  # ensure it's unit length

            # Second vector orthogonal to both the normal and u
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)  # ensure it's unit length
            # Calculate the 3D coordinates of the points, with an overall shift by the original point
            grid_points = (
                self.position + np.outer(uu.flatten(), u) + np.outer(vv.flatten(), v)
            )
            # reverse so that the 'upper left' point is first
            grid_points = grid_points[::-1]

            # self.corners = np.array([
            # self.position + self.width/2*v + self.length/2*u,
            # self.position + self.width/2*v - self.length/2*u,
            # self.position - self.width/2*v + self.length/2*u,
            # self.position - self.width/2*v - self.length/2*u
            # ])

        else:
            grid_points = self.position
            # self.corners = np.array([self.position]*4)

        return grid_points
