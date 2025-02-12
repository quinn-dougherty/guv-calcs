import numpy as np
from .calc_zone import CalcPlane
from .trigonometry import to_polar


class ReflectanceManager:
    def __init__(
        self,
        room,
        reflectances=None,
        x_spacings=None,
        y_spacings=None,
    ):

        self.room = room

        keys = ["floor", "ceiling", "south", "north", "east", "west"]
        default_reflectances = {surface: 0.0 for surface in keys}
        default_spacings = {surface: 0.5 for surface in keys}

        self.reflectances = {**default_reflectances, **(reflectances or {})}
        self.x_spacings = {**default_spacings, **(x_spacings or {})}
        self.y_spacings = {**default_spacings, **(y_spacings or {})}

        self.surfaces = {}
        self._initialize_surfaces()

    def set_reflectance(self, R, wall_id=None):
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

    def set_y_spacing(self, y_spacing, wall_id=None):
        self._set_val(self.y_spacings, y_spacing, wall_id)

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
        for wall, surface in self.surfaces.items():
            x1, x2, y1, y2, height, _ = self._get_surface_dimensions(wall)
            surface.plane.height = height
            surface.plane.set_dimensions(x1, x2, y1, y2)

    def calculate_incidence(self):
        """calculate the incident irradiances"""
        for wall, R in self.reflectances.items():
            if R > 0:
                self.surfaces[wall].calculate_incidence(self.room.lamps)

    def calculate_reflectance(self, calc_zone):
        """calculate the reflectance contribution to a calc zone from each surface"""
        total_values = np.zeros(calc_zone.coords.shape[0])
        for key, surface in self.surfaces.items():
            if surface.R > 0:
                surface.calculate_reflectance(calc_zone)
                values = surface.zone_values[calc_zone.zone_id]
                values = values.reshape(total_values.shape)
                total_values += values * surface.R
        return total_values


class ReflectiveSurface:
    def __init__(self, R, plane):

        if not isinstance(R, float):
            raise TypeError("R must be a float in range [0, 1]")
        if R > 1 or R < 0:
            raise ValueError("R must be a float in range [0, 1]")

        if not isinstance(plane, CalcPlane):
            raise TypeError("plane must be a CalcPlane object")

        self.R = R
        self.plane = plane
        self.zone_values = {}

    def calculate_incidence(self, lamps):
        """calculate incoming radiation"""
        self.plane.calculate_values(lamps=lamps)

    def calculate_reflectance(self, calc_zone, lamp=None):
        """
        calculate the reflective contribution of this reflective surface
        to a provided calculation zone
        """

        # first make sure incident irradiance is calculated
        if self.plane.values is None:
            if lamp is not None:
                self.calculate_incidence({lamp.lamp_id: lamp})
            else:
                raise ValueError("Incidence must be calculated before reflectance")

        I_r = self.plane.values[:, :, np.newaxis, np.newaxis, np.newaxis]
        surface_points = self.plane.coords.reshape(*self.plane.num_points, 3)
        volume_points = calc_zone.coords.reshape(*calc_zone.num_points, 3)

        differences = (
            volume_points - surface_points[:, :, np.newaxis, np.newaxis, np.newaxis, :]
        )
        distances = np.linalg.norm(differences, axis=-1)

        cos_theta = differences[..., 2] / distances
        angles = np.arccos(cos_theta)

        grid_element_size = self.plane.x_spacing * self.plane.y_spacing
        # no I don't know why this factor of 10 is here but it makes the math work out
        nom = I_r * abs(np.cos(angles)) * grid_element_size * 10
        denom = np.pi * distances ** 2
        values = nom / denom
        # # clean nans
        # if np.isnan(values.any()):
        # values = np.ma.masked_invalid(values)

        # apply angle-based differences
        Theta0, Phi0, R0 = to_polar(*differences.reshape(-1, 3).T)
        Theta0 = Theta0.reshape(values.shape)

        # apply vertical field of view
        values[Theta0 < 90 - calc_zone.fov_vert / 2] = 0
        if calc_zone.vert:
            values *= np.sin(np.radians(Theta0))
        if calc_zone.horiz:
            values *= abs(np.cos(np.radians(Theta0)))

        # Sum over all self.plane points to get total values at each volume point
        values = np.sum(values, axis=(0, 1))  # Collapse the self.surface dimensions
        values = values.reshape(*calc_zone.num_points)

        self.zone_values[calc_zone.zone_id] = values
        # Ensure the final array has the correct shape and return multiplied by R
        return values * self.R
