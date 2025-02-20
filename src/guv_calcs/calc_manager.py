import numpy as np

from .trigonometry import attitude, to_polar
from photompy import get_intensity


class LightingCalculator:
    """
    Performs all calculations for a calculation zone
    """

    def __init__(self, zone):
        self.zone = zone

    def compute(self, lamps, ref_manager=None):
        """
        Calculate and return irradiance values at all coordinate points within the zone.
        Expensive! only to be run if necessary
        """

        self.zone.lamp_values_base = {
            lamp_id: self._calculate_lamp(lamp) for lamp_id, lamp in lamps.items()
        }

        if ref_manager is not None:
            # calculate reflectance -- may be expensive!
            ref_manager.calculate_reflectance(self.zone)

        self.zone.values = self.update(lamps, ref_manager)

        return self.zone.values

    def update(self, lamps, ref_manager=None):
        """
        (Relatively) cheaply update the values property
        Called from within the main compute function but may also be called externally

        run this if a property has changed that does not require a full recalculation
        includes: vert, horiz, fov_vert, reflectance, dose
        """

        self.zone.lamp_values = {
            lamp_id: self._apply_filters(lamps[lamp_id], values)
            for lamp_id, values in self.zone.lamp_values_base.items()
        }

        # this should be implemented inside the reflectance module separately
        # or possibly entirely restructured since both lamps and reflective surfaces
        # will contribute to the eye dose

        values = np.array(list(self.zone.lamp_values.values()))
        if self.zone.fov_horiz < 360 and len(lamps) > 1:
            self.zone.base_values = self._calculate_horizontal_fov(values.T, lamps)
        else:
            self.zone.base_values = values.sum(axis=0)

        if ref_manager is not None:
            self.zone.reflected_values = ref_manager.get_total_reflectance(self.zone)

        # reshape
        self.zone.lamp_values = {
            k: v.reshape(*self.zone.num_points)
            for k, v in self.zone.lamp_values.items()
        }
        self.zone.base_values = self.zone.base_values.reshape(*self.zone.num_points)

        # add in reflected values
        self.zone.values = self.zone.base_values + self.zone.reflected_values

        # convert to dose
        if self.zone.dose:
            mult = 3.6 * self.zone.hours
            self.zone.values = self.zone.values * mult
            self.zone.lamp_values = {
                k: v * mult for k, v in self.zone.lamp_values.items()
            }

        return self.zone.values

    def _calculate_lamp(self, lamp):
        """
        Calculate the zone values for a single lamp
        """
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

    def _calculate_horizontal_fov(self, values, lamps):
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
