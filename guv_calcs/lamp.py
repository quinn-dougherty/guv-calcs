from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from ies_utils import read_ies_data, plot_ies, total_optical_power
from .trigonometry import to_cartesian, to_polar, attitude


class Lamp:
    """
    Represents a lamp with properties defined by a photometric data file.
    This class handles the loading of IES photometric data, orienting the lamp in 3D space,
    and provides methods for moving, rotating, and aiming the lamp
    """

    def __init__(
        self,
        lamp_id,
        name=None,
        filename=None,
        x=None,
        y=None,
        z=None,
        angle=None,
        aim_point=None,
        aimx=None,
        aimy=None,
        aimz=None,
        intensity_units=None,
    ):
        self.lamp_id = lamp_id
        self.name = name if name is not None else lamp_id
        # position
        self.x = 0.0 if x is None else x
        self.y = 0.0 if y is None else y
        self.z = 0.0 if z is None else z
        self.position = np.array([self.x, self.y, self.z])
        # orientation
        self.angle = 0.0 if angle is None else angle
        self.aimx = self.x if aimx is None else aimx
        self.aimy = self.y if aimy is None else aimy
        self.aimz = 0 if aimz is None else aimz
        self.aim(self.aimx, self.aimy, self.aimz)  # updates heading and bank

        # load file and coordinates
        self.filename = filename
        self.intensity_units = "mW/Sr" if intensity_units is None else intensity_units
        if self.filename is not None:
            self._load()
            self._orient()

    def _load(self):
        """
        Loads lamp data from an IES file and initializes photometric properties.
        """
        self.lampdict = read_ies_data(self.filename)
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

    def get_total_power(self):
        """return the lamp's total optical power"""
        self.total_optical_power = total_optical_power(self.interpdict)
        return self.total_optical_power

    def reload(self, filename):
        """replace the ies file without erasing any position/rotation/aiming information"""
        self.filename = filename
        self._load()
        self._orient()

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

    def get_cartesian(self, scale=1, sigfigs=9):
        """Return lamp's true position coordinates in cartesian space"""
        return self.transform(self.coords, scale=scale).round(sigfigs)

    def get_polar(self, sigfigs=9):
        """Return lamp's true position coordinates in polar space"""
        cartesian = self.transform(self.coords) - self.position
        return np.array(to_polar(*cartesian.T)).round(sigfigs)
        
    def set_name(self,name):
        self.name = name

    def move(self, x, y, z):
        """Designate lamp position in cartesian space"""
        self.x = x
        self.y = y
        self.z = z
        position = np.array([x, y, z])
        self.position = position
        diff = position - self.position
        self.aim_point += diff 
        return self

    def rotate(self, angle):
        """designate lamp orientation with respect to its z axis"""
        self.angle = angle
        return self

    def aim(self, x, y, z):
        """aim lamp at a point in cartesian space"""

        self.aim_point = np.array([x, y, z])
        xr, yr, zr = self.aim_point - self.position
        self.heading = np.degrees(np.arctan2(yr, xr))
        self.bank = np.degrees(np.arctan2(np.sqrt(xr ** 2 + yr ** 2), zr) - np.pi)
        return self

    def plot_ies(self, title="", figsize=(6.4, 4.8)):
        """standard polar plot of an ies file"""
        fig, ax = plot_ies(self.filename, title=title, figsize=figsize)
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
                alpha=0.7
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
                alpha=0.7
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
