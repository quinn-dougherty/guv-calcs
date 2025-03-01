import numpy as np

from .trigonometry import attitude, to_polar
from photompy import get_intensity


class LightingCalculator:
    """
    Performs all computations for a calculation zone
    """

    def __init__(self, zone):
        self.zone = zone

    def compute(self, lamps, hard=False):
        """
        Calculate and return irradiance values at all coordinate points within the zone.
        """

        # this will only recalculate if the lamp has changed--unless it is a 'hard' recalculation
        self.zone.lamp_values_base = {
            lamp_id: self._calculate_lamp(lamp, hard) for lamp_id, lamp in lamps.items()
        }

        # this step is always cheap
        self.zone.lamp_values = {
            lamp_id: self._apply_filters(lamps[lamp_id], values.copy())
            for lamp_id, values in self.zone.lamp_values_base.items()
        }

        # sum the base lamp values
        if self.zone.fov_horiz < 360 and len(lamps) > 1:
            values = self._calculate_horizontal_fov(lamps)
        else:
            values = sum(self.zone.lamp_values.values())

        # reshape
        values = values.reshape(*self.zone.num_points)

        return values

    def _calculate_lamp(self, lamp, hard=False):
        """
        Calculate the zone values for a single lamp
        """

        NEW_LAMP = self.zone.lamp_values_base.get(lamp.lamp_id) is None
        LAMP_UPDATE = lamp.calc_state != lamp.get_calc_state()
        ZONE_UPDATE = self.zone.calc_state != self.zone.get_calc_state()

        if hard or NEW_LAMP or LAMP_UPDATE or ZONE_UPDATE:
            # get coords
            rel_coords = self.zone.coords - lamp.position
            Theta, Phi, R = self._transform_lamp_coords(rel_coords, lamp)

            # fetch intensity values from photometric data
            interpdict = lamp.lampdict["interp_vals"]
            values = get_intensity(Theta, Phi, interpdict) / R ** 2

            # near field only if necessary
            if lamp.surface.source_density > 0 and lamp.surface.photometric_distance:
                values = self._calculate_nearfield(lamp, R, values)

            if np.isnan(values.any()):  # mask any nans near light source
                values = np.ma.masked_invalid(values)

        else:
            values = self.zone.lamp_values_base[lamp.lamp_id]

        return values

    def _apply_filters(self, lamp, values):
        """
        update the values of a single lamp based on the calc zone properties,
        but which don't require a full recalculation
        """

        rel_coords = self.zone.coords - lamp.position
        Theta0, Phi0, R0 = to_polar(*rel_coords.T)
        # apply vertical field of view
        values[Theta0 < 90 - self.zone.fov_vert / 2] = 0

        if self.zone.vert:
            values *= np.sin(np.radians(Theta0))
        if self.zone.horiz:
            values *= abs(np.cos(np.radians(Theta0)))

        if lamp.intensity_units.lower() == "mw/sr":
            values = values / 10  # convert from mW/Sr to uW/cm2

        # reshape
        values = values.reshape(*self.zone.num_points)

        return values

    def _calculate_nearfield(self, lamp, R, values):
        """
        calculate the values within the photometric distance
        over a discretized source
        """
        near_idx = np.where(R < lamp.surface.photometric_distance)
        # set current values to zero
        values[near_idx] = 0
        # redo calculation in a loop
        num_points = len(lamp.surface.surface_points)
        for point, val in zip(
            lamp.surface.surface_points, lamp.surface.intensity_map.reshape(-1)
        ):
            rel_coords = self.zone.coords - point
            Theta, Phi, R = self._transform_lamp_coords(rel_coords, lamp)
            Theta_n, Phi_n, R_n = Theta[near_idx], Phi[near_idx], R[near_idx]
            interpdict = lamp.lampdict["interp_vals"]
            near_values = get_intensity(Theta_n, Phi_n, interpdict) / R_n ** 2
            near_values = near_values * val / num_points
            values[near_idx] += near_values
        return values

    def _calculate_horizontal_fov(self, lamps):
        """
        Vectorized function to compute the largest possible value for all lamps
        within a horizontal view field.
        """

        # Compute relative coordinates: Shape (num_points, num_lamps, 3)
        lamp_positions = np.array(
            [lamp.position for lamp in lamps.values() if lamp.enabled]
        )
        rel_coords = self.zone.coords[:, None, :] - lamp_positions[None, :, :]

        # Calculate horizontal angles (in degrees)
        angles = np.degrees(np.arctan2(rel_coords[..., 1], rel_coords[..., 0]))
        angles = angles % 360  # Wrap; Shape (N, M)
        angles = angles[:, :, None]  # Expand; Shape (N, M, 1)

        # Compute pairwise angular differences for all rows
        diffs = np.abs(angles - angles.transpose(0, 2, 1))
        diffs = np.minimum(diffs, 360 - diffs)  # Wrap angular differences to [0, 180]

        # Create the adjacency mask for each pair within 180 degrees
        adjacency = diffs <= self.zone.fov_horiz / 2  # Shape (N, M, M)

        # current values to be transformed
        vals = self.zone.lamp_values.values()
        values = np.array([val.reshape(-1) for val in vals]).T
        # Sum the values for all connected components (using the adjacency mask)
        value_sums = adjacency @ values[:, :, None]  # Shape (N, M, 1)
        # Remove the last singleton dimension,
        value_sums = value_sums.squeeze(-1)  # Shape (N, M)

        return np.max(value_sums, axis=1)  # Shape (N,)

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
