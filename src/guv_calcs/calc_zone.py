import inspect
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
from photompy import get_intensity
from .trigonometry import attitude, to_polar
from ._helpers import rows_to_bytes


class CalcZone(object):
    """
    Base class representing a calculation zone.

    This class provides a template for setting up zones within which various
    calculations related to lighting conditions are performed. Subclasses should
    provide specific implementations of the coordinate setting method.

    NOTE: I changed this from an abstract base class to an object superclass
    to make it more convenient to work with the website, but this class doesn't really
    work on its own

    Parameters:
    --------
    zone_id: str
        identification tag for internal tracking
    name: str, default=None
        externally visible name for zone
    dimensions: array of floats, default=None
        array of len 2 if CalcPlane, of len 3 if CalcVol
    offset: bool, default=True
    fov80: bool, default=False
        apply 80 degree field of view filtering - used for calculating eye limits
    vert: bool, default=False
        calculate vertical irradiance only
    horiz: bool, default=False
        calculate horizontal irradiance only
    dose: bool, default=False
        whether to calculate a dose over N hours or just fluence
    hours: float, default = 8.0
        number of hours to calculate dose over. Only relevant if dose is True.
    enabled: bool, default = True
        whether or not the calc zone is enabled for calculations
    """

    def __init__(
        self,
        zone_id,
        name=None,
        offset=None,
        fov80=None,
        vert=None,
        horiz=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
        values=None,
    ):
        self.zone_id = zone_id
        self.name = zone_id if name is None else name
        self.offset = True if offset is None else offset
        self.fov80 = False if fov80 is None else fov80
        self.vert = False if vert is None else vert
        self.horiz = False if horiz is None else horiz
        self.dose = False if dose is None else dose
        if self.dose:
            self.units = "mJ/cm²"
        else:
            self.units = "uW/cm²"
        self.hours = 8.0 if hours is None else abs(hours)  # only used if dose is true
        self.enabled = True if enabled is None else enabled
        self.show_values = True if show_values is None else show_values

        # these will all be calculated after spacing is set, which is set in the subclass
        self.calctype = "Zone"
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.z1 = None
        self.z2 = None
        self.height = None
        self.x_spacing = None
        self.y_spacing = None
        self.z_spacing = None
        self.num_points = None
        self.xp = None
        self.yp = None
        self.zp = None
        self.coords = None
        self.values = values

    def save_zone(self, filename=None):

        data = {}
        data["zone_id"] = self.zone_id
        data["name"] = self.name
        data["offset"] = self.offset
        data["fov80"] = self.fov80
        data["vert"] = self.vert
        data["horiz"] = self.horiz
        data["dose"] = self.dose
        data["hours"] = self.hours
        data["enabled"] = self.enabled
        data["show_values"] = self.show_values
        data["x1"] = self.x1
        data["x2"] = self.x2
        data["x_spacing"] = self.x_spacing
        data["y1"] = self.y1
        data["y2"] = self.y2
        data["y_spacing"] = self.y_spacing
        if isinstance(self, CalcPlane):
            data["height"] = self.height
            data["calctype"] = "Plane"
        elif isinstance(self, CalcVol):
            data["z1"] = self.z1
            data["z2"] = self.z2
            data["z_spacing"] = self.z_spacing
            data["calctype"] = "Volume"

        if filename is not None:
            with open(filename, "w") as json_file:
                json_file.write(json.dumps(data))

        return data

    @classmethod
    def from_dict(cls, data):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        return cls(**{k: v for k, v in data.items() if k in keys})

    def set_dimensions(self, dimensions):
        raise NotImplementedError

    def set_spacing(self, spacing):
        raise NotImplementedError

    def _write_rows(self):
        raise NotImplementedError

    def set_offset(self, offset):
        if type(offset) is not bool:
            raise TypeError("Offset must be either True or False")
        self.offset = offset
        self._update

    def set_value_type(self, dose):
        """
        if true values will be in dose over time
        if false
        """
        if type(dose) is not bool:
            raise TypeError("Dose must be either True or False")

        # convert values if they need converting
        if self.values is not None:
            if dose and not self.dose:
                self.values = self.values * 3.6 * self.hours
            elif self.dose and not dose:
                self.values = self.values / (3.6 * self.hours)

        self.dose = dose
        if self.dose:
            self.units = "mJ/cm2"
        else:
            self.units = "uW/cm2"

    def set_dose_time(self, hours):
        if type(hours) not in [float, int]:
            raise TypeError("Hours must be numeric")
        self.hours = hours

    def _transform_lamp_coords(self, rel_coords, lamp):
        """
        transform zone coordinates to be consistent with any lamp
        transformations applied, and convert to polar coords for further
        operations
        """
        Theta0, Phi0, R0 = to_polar(*rel_coords.T)
        # apply all transformations that have been applied to this lamp, but in reverse
        rel_coords = np.array(
            attitude(rel_coords.T, roll=0, pitch=0, yaw=-lamp.heading)
        ).T
        rel_coords = np.array(attitude(rel_coords.T, roll=0, pitch=-lamp.bank, yaw=0)).T
        rel_coords = np.array(
            attitude(rel_coords.T, roll=0, pitch=0, yaw=-lamp.angle)
        ).T
        Theta, Phi, R = to_polar(*rel_coords.T)
        return Theta, Phi, R

    def _calculate_nearfield(self, lamp, R, values):
        """
        calculate the values within the photometric distance
        over a discretized source
        """
        near_idx = np.where(R < lamp.photometric_distance)
        # set current values to zero
        values[near_idx] = 0
        # redo calculation in a loop
        num_points = len(lamp.grid_points)

        for point, val in zip(lamp.grid_points, lamp.relative_map):
            rel_coords = self.coords - point
            Theta, Phi, R = self._transform_lamp_coords(rel_coords, lamp)
            Theta_n, Phi_n, R_n = Theta[near_idx], Phi[near_idx], R[near_idx]
            near_values = get_intensity(Theta_n, Phi_n, lamp.interpdict) / R_n ** 2
            near_values = near_values * val / num_points
            values[near_idx] += near_values
        return values

    def _calculate_single_lamp(self, lamp):
        """calculate the zone values for a single lamp"""
        rel_coords = self.coords - lamp.position
        Theta, Phi, R = self._transform_lamp_coords(rel_coords, lamp)
        values = get_intensity(Theta, Phi, lamp.interpdict) / R ** 2
        if lamp.source_density > 0:
            values = self._calculate_nearfield(lamp, R, values)

        Theta0, Phi0, R0 = to_polar(*rel_coords.T)
        if self.fov80:
            values[Theta0 < 50] = 0
        if self.vert:
            values *= np.sin(np.radians(Theta0))
        if self.horiz:
            values *= np.cos(np.radians(Theta0))

        return values

    def calculate_values(self, lamps: list):
        """
        Calculate and return irradiance values at all coordinate points within the zone.
        """
        total_values = np.zeros(self.coords.shape[0])
        self.lamp_values = {}
        for lamp_id, lamp in lamps.items():
            if lamp.filedata is not None and lamp.enabled:
                # determine lamp placement + calculate relative coordinates
                values = self._calculate_single_lamp(lamp)
                values = values / 10  # convert from mW/Sr to uW/cm2

                total_values += values
                # save the max value to the lamp object
                lamp.max_irradiances[self.zone_id] = total_values.max()

                # add lamp-specific values to dict
                if self.dose:
                    values = values * 3.6 * self.hours
                if np.isnan(values.any()):
                    values = np.ma.masked_invalid(values)
                self.lamp_values[lamp_id] = values.reshape(*self.num_points)

        total_values = total_values.reshape(*self.num_points)

        if np.isnan(total_values.any()):  # mask any nans near light source
            total_values = np.ma.masked_invalid(total_values)
        self.values = total_values
        # convert to dose
        if self.dose:
            self.values = self.values * 3.6 * self.hours

        return self.values

    def export(self, fname=None):

        try:
            rows = self._write_rows()  # implemented in subclass
            csv_bytes = rows_to_bytes(rows)

            if fname is not None:
                with open(fname, "wb") as csvfile:
                    csvfile.write(csv_bytes)
            else:
                return csv_bytes
        except NotImplementedError:
            pass


class CalcVol(CalcZone):
    """
    Represents a volumetric calculation zone.
    A subclass of CalcZone designed for three-dimensional volumetric calculations.
    """

    def __init__(
        self,
        zone_id,
        name=None,
        x1=None,
        x2=None,
        y1=None,
        y2=None,
        z1=None,
        z2=None,
        num_x=None,
        num_y=None,
        num_z=None,
        x_spacing=None,
        y_spacing=None,
        z_spacing=None,
        offset=None,
        fov80=None,
        vert=None,
        horiz=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
        values=None,
    ):

        super().__init__(
            zone_id=zone_id,
            name=name,
            offset=offset,
            fov80=fov80,
            vert=vert,
            horiz=horiz,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
            values=values,
        )
        self.calctype = "Volume"
        self.x1 = 0 if x1 is None else x1
        self.x2 = 6 if x2 is None else x2
        self.y1 = 0 if y1 is None else y1
        self.y2 = 4 if y2 is None else y2
        self.z1 = 0 if z1 is None else z1
        self.z2 = 2.7 if z2 is None else z2

        self.num_x = 30 if num_x is None else abs(num_x)
        self.num_y = 30 if num_y is None else abs(num_y)
        self.num_z = 10 if num_z is None else abs(num_z)

        default_x = abs(self.x2 - self.x1) / self.num_x
        default_y = abs(self.y2 - self.y1) / self.num_y
        default_z = abs(self.z2 - self.z1) / self.num_z

        # only allow setting by spacing if num_x / num_y / num_z are None
        if num_x is None:
            self.x_spacing = default_x if x_spacing is None else abs(x_spacing)
            self.num_x = int(abs((self.x2 - self.x1) / self.x_spacing))
        else:
            self.x_spacing = default_x
            if x_spacing is not None:
                msg = "Passed x_spacing value will be ignored, num_x used instead"
                warnings.warn(msg, stacklevel=3)

        if num_y is None:
            self.y_spacing = default_y if y_spacing is None else abs(y_spacing)
            self.num_y = int(abs((self.y2 - self.y1) / self.y_spacing))
        else:
            self.y_spacing = default_y
            if y_spacing is not None:
                msg = "Passed y_spacing value will be ignored, num_y used instead"
                warnings.warn(msg, stacklevel=3)

        if num_z is None:
            self.z_spacing = default_z if z_spacing is None else abs(z_spacing)
            self.num_z = int(abs((self.z2 - self.z1) / self.z_spacing))
        else:
            self.z_spacing = default_z
            if z_spacing is not None:
                msg = "Passed z_spacing value will be ignored, num_z used instead"
                warnings.warn(msg, stacklevel=3)

        self._update()

    def set_dimensions(self, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None):
        self.x1 = self.x1 if x1 is None else x1
        self.x2 = self.x2 if x2 is None else x2
        self.y1 = self.y1 if y1 is None else y1
        self.y2 = self.y2 if y2 is None else y2
        self.z1 = self.z1 if z1 is None else z1
        self.z2 = self.z2 if z2 is None else z2
        self._update()

    def set_spacing(self, x_spacing=None, y_spacing=None, z_spacing=None):
        self.x_spacing = self.x_spacing if x_spacing is None else abs(x_spacing)
        self.y_spacing = self.y_spacing if y_spacing is None else abs(y_spacing)
        self.z_spacing = self.z_spacing if z_spacing is None else abs(z_spacing)
        numx = int(abs(self.x2 - self.x1) / self.x_spacing)
        numy = int(abs(self.y2 - self.y1) / self.y_spacing)
        numz = int(abs(self.z2 - self.z1) / self.z_spacing)
        if numx == 0:
            msg = f"x_spacing too large. Minimum spacing:{self.x2-self.x1}"
            warnings.warn(msg, stacklevel=3)
            numx += 1
        if numy == 0:
            msg = f"y_spacing too large. Minimum spacing:{self.y2-self.y1}"
            warnings.warn(msg, stacklevel=3)
            numy += 1
        if numz == 0:
            msg = f"z_spacing too large. Minimum spacing:{self.z2-self.z1}"
            warnings.warn(msg, stacklevel=3)
            numz += 1
        self.num_x = numx
        self.num_y = numy
        self.num_z = numz
        self._update()

    def set_num_points(self, num_x=None, num_y=None, num_z=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        self.num_x = self.num_x if num_x is None else num_x
        self.num_y = self.num_y if num_y is None else num_y
        self.num_z = self.num_z if num_z is None else num_z
        if self.num_x == 0:
            warnings.warn("Number of x points must be at least 1")
            self.num_x += 1
        if self.num_y == 0:
            warnings.warn("Number of y points must be at least 1")
            self.num_y += 1
        if self.num_z == 0:
            warnings.warn("Number of z points must be at least 1")
            self.num_z += 1
        self._update()
        return self

    def _update(self):
        """
        Update the number of points based on the spacing, and then the points
        """
        if self.offset:
            xmult = -1 if self.x1 > self.x2 else 1
            ymult = -1 if self.y1 > self.y2 else 1
            zmult = -1 if self.z1 > self.z2 else 1
        else:
            xmult, ymult, zmult = 0, 0, 0

        if self.x1 == self.x2:
            self.num_x = 1
        if self.y1 == self.y2:
            self.num_y = 1
        if self.z1 == self.z2:
            self.num_z = 1

        x_offset = xmult * (self.x_spacing / 2)
        y_offset = ymult * (self.y_spacing / 2)
        z_offset = zmult * (self.z_spacing / 2)

        xpoints = np.linspace(self.x1 + x_offset, self.x2 - x_offset, self.num_x)
        ypoints = np.linspace(self.y1 + y_offset, self.y2 - y_offset, self.num_y)
        zpoints = np.linspace(self.z1 + z_offset, self.z2 - z_offset, self.num_z)

        self.points = [xpoints, ypoints, zpoints]
        self.xp, self.yp, self.zp = self.points

        X, Y, Z = [
            grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")
        ]
        self.coords = np.array((X, Y, Z)).T
        self.coords = np.unique(self.coords, axis=0)

        self.num_points = np.array([len(self.xp), len(self.yp), len(self.zp)])

    def _write_rows(self):
        """
        export solution to csv file
        designed to be in the same format as the Acuity Visual export
        """

        header = """Data format notes:

         Data consists of numZ horizontal grids of fluence rate values; each grid contains numX by numY points.

         numX; numY; numZ are given on the first line of data.
         The next line contains numX values; indicating the X-coordinate of each grid column.
         The next line contains numY values; indicating the Y-coordinate of each grid row.
         The next line contains numZ values; indicating the Z-coordinate of each horizontal grid.
         A blank line separates the position data from the first horizontal grid of fluence rate values.
         A blank line separates each subsequent horizontal grid of fluence rate values.

         fluence rate values are given in µW/cm²
         
         """
        lines = header.split("\n")
        rows = [[line] for line in lines]
        rows += [self.num_points]
        rows += self.points
        num_x = self.num_points[0]
        num_y = self.num_points[1]
        num_z = self.num_points[2]
        for i in range(num_z):
            rows += [""]
            if self.values is None:
                rows += [[""] * num_x] * num_y
            else:
                rows += self.values.T[i].tolist()
        return rows


class CalcPlane(CalcZone):
    """
    Represents a planar calculation zone.
    A subclass of CalcZone designed for two-dimensional planar calculations at a specific height.
    """

    def __init__(
        self,
        zone_id,
        name=None,
        x1=None,
        x2=None,
        y1=None,
        y2=None,
        height=None,
        num_x=None,
        num_y=None,
        x_spacing=None,
        y_spacing=None,
        offset=None,
        fov80=None,
        vert=None,
        horiz=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
        values=None,
    ):

        super().__init__(
            zone_id=zone_id,
            name=name,
            offset=offset,
            fov80=fov80,
            vert=vert,
            horiz=horiz,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
            values=values,
        )
        self.calctype = "Plane"
        self.height = 1.9 if height is None else height
        self.x1 = 0 if x1 is None else x1
        self.x2 = 6 if x2 is None else x2
        self.y1 = 0 if y1 is None else y1
        self.y2 = 4 if y2 is None else y2

        self.num_x = 30 if num_x is None else abs(num_x)
        self.num_y = 30 if num_y is None else abs(num_y)

        default_x = abs(self.x2 - self.x1) / self.num_x
        default_y = abs(self.y2 - self.y1) / self.num_y

        # only allow setting by spacing if num_x / num_y are None
        if num_x is None:
            self.x_spacing = default_x if x_spacing is None else abs(x_spacing)
            self.num_x = int(abs((self.x2 - self.x1) / self.x_spacing))
        else:
            self.x_spacing = default_x
            if x_spacing is not None:
                msg = "Passed x_spacing value will be ignored, num_x used instead"
                warnings.warn(msg, stacklevel=3)
        if num_y is None:
            self.y_spacing = default_y if y_spacing is None else abs(y_spacing)
            self.num_y = int(abs((self.y2 - self.y1) / self.y_spacing))
        else:
            self.y_spacing = default_y
            if y_spacing is not None:
                msg = "Passed y_spacing value will be ignored, num_y used instead"
                warnings.warn(msg, stacklevel=3)
        self._update()

    def set_height(self, height):
        """set height of calculation plane. currently we only support vertical planes"""
        if type(height) not in [float, int]:
            raise TypeError("Height must be numeric")
        self.height = height
        self._update()
        return self

    def set_dimensions(self, x1=None, x2=None, y1=None, y2=None):
        """set the dimensions and update the coordinate points"""
        self.x1 = self.x1 if x1 is None else x1
        self.x2 = self.x2 if x2 is None else x2
        self.y1 = self.y1 if y1 is None else y1
        self.y2 = self.y2 if y2 is None else y2
        self._update()
        return self

    def set_spacing(self, x_spacing=None, y_spacing=None):
        """set the fineness of the grid spacing and update the coordinate points"""
        self.x_spacing = self.x_spacing if x_spacing is None else abs(x_spacing)
        self.y_spacing = self.y_spacing if y_spacing is None else abs(y_spacing)
        numx = int(abs(self.x2 - self.x1) / self.x_spacing)
        numy = int(abs(self.y2 - self.y1) / self.y_spacing)
        if numx == 0:
            msg = f"x_spacing too large. Minimum spacing:{self.x2-self.x1}"
            warnings.warn(msg, stacklevel=3)
            numx += 1
        if numy == 0:
            msg = f"y_spacing too large. Minimum spacing:{self.y2-self.y1}"
            warnings.warn(msg, stacklevel=3)
            numy += 1
        self.num_x = numx
        self.num_y = numy
        self._update()
        return self

    def set_num_points(self, num_x=None, num_y=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        self.num_x = self.num_x if num_x is None else abs(num_x)
        self.num_y = self.num_y if num_y is None else abs(num_y)
        if self.num_x == 0:
            warnings.warn("Number of x points must be at least 1")
            self.num_x += 1
        if self.num_y == 0:
            warnings.warn("Number of y points must be at least 1")
            self.num_y += 1
        self.x_spacing = abs((self.x2 - self.x1) / round(self.num_y))
        self.y_spacing = abs((self.y2 - self.y1) / round(self.num_x))
        self._update()
        return self

    def _update(self):
        """
        Update the number of points based on the spacing, and then the points
        """
        if self.offset:
            xmult = -1 if self.x1 > self.x2 else 1
            ymult = -1 if self.y1 > self.y2 else 1
        else:
            xmult, ymult = 0, 0

        if self.x1 == self.x2:
            self.num_x = 1
        if self.y1 == self.y2:
            self.num_y = 1

        x_offset = xmult * (self.x_spacing / 2)
        y_offset = ymult * (self.y_spacing / 2)

        xpoints = np.linspace(self.x1 + x_offset, self.x2 - x_offset, self.num_x)
        ypoints = np.linspace(self.y1 + y_offset, self.y2 - y_offset, self.num_y)

        self.points = [xpoints, ypoints]
        self.xp, self.yp = self.points

        X, Y = [grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")]
        xy_coords = np.array([np.array((x0, y0)) for x0, y0 in zip(X, Y)])
        zs = np.ones(xy_coords.shape[0]) * self.height

        self.coords = np.stack([xy_coords.T[0], xy_coords.T[1], zs]).T
        self.coords = np.unique(self.coords, axis=0)

        self.num_points = np.array([len(self.xp), len(self.yp)])

    def plot_plane(self, fig=None, ax=None, vmin=None, vmax=None, title=None):
        """Plot the image of the radiation pattern"""
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]

        title = "" if title is None else title
        if self.values is not None:
            vmin = self.values.min() if vmin is None else vmin
            vmax = self.values.max() if vmax is None else vmax
            extent = [self.x1, self.x2, self.y1, self.y2]
            img = ax.imshow(self.values.T[::-1], extent=extent, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(
                img,
                pad=0.03,  # orientation=orientation, location=location
            )
            ax.set_title(title)
            cbar.set_label(self.units, loc="center")
        return fig, ax

    def _write_rows(self, fname=None):
        """
        export solution to csv
        """
        # if fname is None:
        # fname = self.name+".csv"
        num_x = self.num_points[0]
        num_y = self.num_points[1]

        rows = [[""] + self.points[0].tolist()]
        if self.values is None:
            vals = [[""] * num_y] * num_x
        else:
            vals = self.values
        rows += np.concatenate(([np.flip(self.points[1])], vals)).T.tolist()
        rows += [""]
        # zvals
        rows += [[""] + [self.height] * num_x] * num_y
        return rows
