import numpy as np
from .calc_zone import CalcPlane
from .trigonometry import to_polar


class ReflectanceManager:
    """
    Class for managing reflective surfaces and their interactions

    Attributes:
    room: guv_calcs.Room
        instance of the parent class
    reflectances: dict { str: float}
        dict with keys [`floor`, `ceiling`, `south`, `north`, `east`, `west`]
        and float values between 0 and 1. All values default 0.0
    x_spacings: dict { str: float}
        dict with same keys as `reflectances` and float values greater than 0.
        All values default 0.5. Determines spacing in the relative x direction
    u_spacings: dict { str: float}
        dict with same keys as `reflectances` and float values greater than 0.
        All values default 0.5. Determines spacing in the relative y direction

    """

    def __init__(
        self,
        room,
        reflectances=None,
        x_spacings=None,
        y_spacings=None,
    ):

        self.room = room
        self.zone_dict = {}

        keys = ["floor", "ceiling", "south", "north", "east", "west"]
        default_reflectances = {surface: 0.0 for surface in keys}
        default_spacings = {surface: 0.5 for surface in keys}

        self.reflectances = {**default_reflectances, **(reflectances or {})}
        self.x_spacings = {**default_spacings, **(x_spacings or {})}
        self.y_spacings = {**default_spacings, **(y_spacings or {})}

        self.surfaces = {}
        self._initialize_surfaces()

    def set_reflectance(self, R, wall_id=None):
        """set reflectance by wall_id or, if wall_if is None, to all walls"""
        keys = self.reflectances.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                # self.reflectances[wall] = R
                self.surfaces[wall].R = R
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                # self.reflectances[wall_id] = R
                self.surfaces[wall_id].R = R
        # update the dict
        for key, val in self.surfaces.items():
            self.reflectances[key] = val.R

    def set_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        """set x and y spacing by wall_id or, if wall_if is None, to all walls"""
        keys = self.x_spacings.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                self.surfaces[wall].plane.set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces[wall_id].plane.set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )
        for key, val in self.surfaces.items():
            self.x_spacings[key] = val.plane.x_spacing
            self.y_spacings[key] = val.plane.y_spacing

    def _initialize_surfaces(self):

        for wall, reflectance in self.reflectances.items():
            x1, x2, y1, y2, height, ref_surface = self._get_surface_dimensions(wall)
            plane = CalcPlane(
                zone_id=wall,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                height=height,
                ref_surface=ref_surface,
                x_spacing=self.x_spacings[wall],
                y_spacing=self.y_spacings[wall],
            )
            self.surfaces[wall] = ReflectiveSurface(R=reflectance, plane=plane)

    def _get_surface_dimensions(self, wall):
        """retrieve the dimensions of a particular wall based on its id"""
        if wall == "floor":
            x1, x2, y1, y2 = 0, self.room.x, 0, self.room.y
            height = 0
            ref_surface = "xy"
        elif wall == "ceiling":
            x1, x2, y1, y2 = 0, self.room.x, 0, self.room.y
            height = self.room.z
            ref_surface = "xy"
        elif wall == "south":
            x1, x2, y1, y2 = 0, self.room.x, 0, self.room.z
            height = 0
            ref_surface = "xz"
        elif wall == "north":
            x1, x2, y1, y2 = 0, self.room.x, 0, self.room.z
            height = self.room.y
            ref_surface = "xz"
        elif wall == "west":
            x1, x2, y1, y2 = 0, self.room.y, 0, self.room.z
            height = 0
            ref_surface = "yz"
        elif wall == "east":
            x1, x2, y1, y2 = 0, self.room.y, 0, self.room.z
            height = self.room.x
            ref_surface = "yz"
        return x1, x2, y1, y2, height, ref_surface

    def update_dimensions(self):
        """update the wall dimensions based on changes to the Room parent class"""
        for wall, surface in self.surfaces.items():
            x1, x2, y1, y2, height, _ = self._get_surface_dimensions(wall)
            surface.plane.height = height
            surface.plane.set_dimensions(x1, x2, y1, y2)

    def calculate_incidence(self, hard=False):
        """calculate the incident irradiances on all reflective walls"""
        for wall, R in self.reflectances.items():
            valid_lamps = self.room._get_valid_lamps()
            self.surfaces[wall].calculate_incidence(valid_lamps, hard=hard)

    def calculate_reflectance(self, zone, hard=False):
        """
        calculate the reflectance contribution to a calc zone from each surface
        """

        threshold = zone.base_values.mean() * 0.01  # 1% of total value

        total_values = {}
        for wall, surface in self.surfaces.items():
            if surface.R * surface.plane.values.mean() > threshold:

                surface.calculate_reflectance(zone, hard=hard)
                values = surface.zone_dict[zone.zone_id]["values"]
                total_values[wall] = values
            else:
                total_values[wall] = np.zeros(zone.num_points)

        self.zone_dict[zone.zone_id] = total_values

        return total_values

    def get_total_reflectance(self, zone):
        """sum over all surfaces to get the total reflected values for that calc zone"""
        dct = self.zone_dict[zone.zone_id]
        values = np.zeros(zone.num_points)
        for wall, surface_vals in dct.items():
            if surface_vals is not None:
                values += surface_vals * self.reflectances[wall]
        return values

    # def calculate_interreflectance(self):
    # """
    # calculate the contribution of each reflective surface to each
    # other reflective surface
    # """


class ReflectiveSurface:
    """
    Class that represents a single reflective surface defined by a calculation
    zone and a float value R between 0 and 1.
    """

    def __init__(self, R, plane):

        if not isinstance(R, float):
            raise TypeError("R must be a float in range [0, 1]")
        if R > 1 or R < 0:
            raise ValueError("R must be a float in range [0, 1]")

        if not isinstance(plane, CalcPlane):
            raise TypeError("plane must be a CalcPlane object")

        self.R = R
        self.plane = plane
        self.zone_dict = {}

    def calculate_incidence(self, lamps, hard=False):
        """calculate incoming radiation"""
        self.plane.calculate_values(lamps=lamps, hard=hard)

    def calculate_reflectance(self, zone, hard=False):
        """
        TODO: this can be sped up by storing each calculation for each lamp,
        and only recalculating the contribution from each lamp...maybe? idk
        maybe that doesn't work after all.

        Actually since calculating the incidence is fast

        calculate the reflective contribution of this reflective surface
        to a provided calculation zone

        Arguments:
            zone: a calculation zone onto which reflectance is calculated
            lamp: optional. if provided, and incidence not yet calculated, uses this
            lamp to calculate incidence. mostly this is just for
        """

        # first make sure incident irradiance is calculated
        if self.plane.values is None:
            raise ValueError("Incidence must be calculated before reflectance")

        if self.zone_dict.get(zone.zone_id) is None:
            self.zone_dict[zone.zone_id] = {}

        calc_state = zone.get_calc_state()
        update_state = zone.get_update_state()
        surface_state = self.plane.get_calc_state()
        NEW_ZONE = self.zone_dict[zone.zone_id].get("values") is None
        ZONE_RECALC = calc_state != self.zone_dict[zone.zone_id].get("calc_state")
        ZONE_UPDATE = update_state != self.zone_dict[zone.zone_id].get("update_state")
        SURFACE_UPDATE = surface_state != self.zone_dict[zone.zone_id].get(
            "surface_state"
        )

        CALCULATE = NEW_ZONE or SURFACE_UPDATE or hard

        if CALCULATE or ZONE_RECALC:
            distances, angles, theta = self._calculate_coordinates(zone)
        else:
            distances = self.zone_dict[zone.zone_id]["distances"]
            angles = self.zone_dict[zone.zone_id]["angles"]
            theta = self.zone_dict[zone.zone_id]["theta"]

        if CALCULATE or ZONE_UPDATE:
            I_r = self.plane.values[:, :, np.newaxis, np.newaxis, np.newaxis]
            element_size = self.plane.x_spacing * self.plane.y_spacing

            values = (I_r * abs(np.cos(angles)) * element_size) / (
                np.pi * distances ** 2
            )

            values = self._apply_filters(values, theta, zone)

            # Sum over all self.plane points to get total values at each volume point
            values = np.sum(values, axis=(0, 1))  # Collapse the dimensions
            values = values.reshape(*zone.num_points)
        else:
            values = self.zone_dict[zone.zone_id]["values"]

        # update the state
        self.zone_dict[zone.zone_id] = {
            "distances": distances,
            "angles": angles,
            "theta": theta,
            "values": values,
            "calc_state": calc_state,
            "update_state": update_state,
            "surface_state": surface_state,
        }

        # Ensure the final array has the correct shape and return multiplied by R
        return values * self.R

    def _apply_filters(self, values, theta, zone):

        theta = theta.reshape(values.shape)

        # clean nans
        if np.isnan(values).any():
            values = np.ma.masked_invalid(values)

        # apply vertical field of view
        values[theta < 90 - zone.fov_vert / 2] = 0
        if zone.vert:
            values *= np.sin(np.radians(theta))
        if zone.horiz:
            values *= abs(np.cos(np.radians(theta)))

        return values

    def _calculate_coordinates(self, zone):
        """
        return the angles and distances between the points of the reflective
        surface and the calculation zone

        this is the expensive step!
        """
        surface_points = self.plane.coords.reshape(*self.plane.num_points, 3)
        volume_points = zone.coords.reshape(*zone.num_points, 3)

        differences = (
            volume_points - surface_points[:, :, np.newaxis, np.newaxis, np.newaxis, :]
        )
        distances = np.linalg.norm(differences, axis=-1)

        cos_theta = differences[..., 2] / distances
        angles = np.arccos(cos_theta)

        # for angle-based differences
        Theta0, Phi0, R0 = to_polar(*differences.reshape(-1, 3).T)

        return distances, angles, Theta0
