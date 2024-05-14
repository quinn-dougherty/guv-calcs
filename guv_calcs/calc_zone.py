import numpy as np
from abc import ABC, abstractmethod
from ies_utils import get_intensity
from .trigonometry import attitude, to_polar


class CalcZone(ABC):
    """
    Abstract base class representing a calculation zone.

    This class provides a template for setting up zones within which various
    calculations related to lighting conditions are performed. Subclasses should
    provide specific implementations of the coordinate setting method.
    """

    def __init__(self, zone_id, dimensions, offset, fov80, vert, horiz):
        self.zone_id = zone_id
        self.dimensions = dimensions
        self.offset = True if offset is None else offset
        self.fov80 = False if fov80 is None else fov80
        self.vert = False if vert is None else vert
        self.horiz = False if horiz is None else horiz
        # these will all be calculated after spacing is set, which is set in the subclass
        self.spacing = None
        self.num_points = None
        self.xp = None
        self.yp = None
        self.zp = None
        self.coords = None
        self.values = None

    @abstractmethod
    def _set_coords(self):
        """
        Set up the coordinates in the calculation zone.

        This method should be implemented by all subclasses to define how the coordinates
        are structured based on the zone's dimensions and offset. The implementation
        will depend on whether the zone is a volume or a plane.
        """
        pass

    def _update(self):
        """
        Update the number of points based on the spacing, and then the points
        """
        self.num_points = [
            int(dim / space) for dim, space in zip(self.dimensions, self.spacing)
        ]
        if self.offset:
            self.points = [
                np.linspace(div / 2, val - div / 2, int(val / div))
                for val, div in zip(self.dimensions, self.spacing)
            ]
        else:
            self.points = [
                np.linspace(0, val, int(val / div))
                for val, div in zip(self.dimensions, self.spacing)
            ]
        self._set_coords()

    def set_spacing(self, spacing):
        """
        Set the spacing between points for calculations and update coordinates.
        """
        self.spacing = spacing
        self._update()

    def calculate_values(self, lamps: list):
        """
        Calculate and return irradiance values at all coordinate points within the zone.
        """
        total_values = np.zeros(self.coords.shape[0])
        for lamp in lamps:
            # determine lamp placement + calculate relative coordinates
            rel_coords = self.coords - lamp.position
            # store the theta and phi data based on this orientation
            Theta0, Phi0, R0 = to_polar(*rel_coords.T)
            # apply all transformations that have been applied to this lamp, but in reverse
            rel_coords = np.array(
                attitude(rel_coords.T, roll=0, pitch=0, yaw=-lamp.heading)
            ).T
            rel_coords = np.array(
                attitude(rel_coords.T, roll=0, pitch=-lamp.bank, yaw=0)
            ).T
            rel_coords = np.array(
                attitude(rel_coords.T, roll=0, pitch=0, yaw=-lamp.angle)
            ).T
            Theta, Phi, R = to_polar(*rel_coords.T)
            values = np.array(
                [
                    get_intensity(theta, phi, lamp.valdict) / r ** 2
                    for theta, phi, r in zip(Theta, Phi, R)
                ]
            )
            if self.fov80:
                values[Theta0 < 50] = 0
            if self.vert:
                values *= np.sin(np.radians(Theta0))
            if self.horiz:
                values *= np.cos(np.radians(Theta0))
            if lamp.intensity_units == "mW/Sr":
                total_values += values / 10  # convert from mW/Sr to uW/cm2
            else:
                raise KeyError("Units not recognized")
        total_values = total_values.reshape(
            self.num_points
        )  # reshape to correct dimensions
        total_values = np.ma.masked_invalid(
            total_values
        )  # mask any nans near light source
        self.values = total_values
        return self.values


class CalcVol(CalcZone):
    """
    Represents a volumetric calculation zone.
    A subclass of CalcZone designed for three-dimensional volumetric calculations.
    """

    def __init__(
        self,
        zone_id,
        dimensions,
        spacing=None,
        offset=None,
        fov80=None,
        vert=None,
        horiz=None,
    ):

        if len(dimensions) != 3:
            raise ValueError("CalcVol requires exactly three dimensions.")

        super().__init__(zone_id, dimensions, offset, fov80, vert, horiz)
        self.spacing = [0.25, 0.25, 0.1] if spacing is None else spacing

        if len(self.spacing) != len(dimensions):
            raise ValueError(
                "Dimensions of spacing must be equal to dimensions of calc zone"
            )

        self._update()

    def _set_coords(self):
        self.xp, self.yp, self.zp = self.points
        X, Y, Z = [grid.reshape(-1) for grid in np.meshgrid(*self.points)]
        self.coords = np.array((X, Y, Z)).T


class CalcPlane(CalcZone):
    """
    Represents a planar calculation zone.
    A subclass of CalcZone designed for two-dimensional planar calculations at a specific height.
    """

    def __init__(
        self,
        zone_id,
        height,
        dimensions,
        spacing=None,
        offset=None,
        fov80=None,
        vert=None,
        horiz=None,
    ):

        if len(dimensions) != 2:
            raise ValueError("CalcPlane requires exactly two dimensions.")

        super().__init__(zone_id, dimensions, offset, fov80, vert, horiz)
        self.height = height
        self.spacing = [0.1, 0.1] if spacing is None else spacing

        if len(self.spacing) != len(dimensions):
            raise ValueError(
                "Dimensions of spacing must be equal to dimensions of calc zone"
            )

        self._update()

    def _set_coords(self):
        """
        Setup the coordinate grid for volumetric calculations based on the provided dimensions and spacing.
        """
        self.xp, self.yp = self.points
        X, Y = [grid.reshape(-1) for grid in np.meshgrid(*self.points)]
        xy_coords = np.array([np.array((x0, y0)) for x0, y0 in zip(X, Y)])
        zs = np.ones(xy_coords.shape[0]) * self.height
        self.coords = np.stack([xy_coords.T[0], xy_coords.T[1], zs]).T
