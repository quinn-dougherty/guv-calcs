import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import plotly.graph_objs as go
from .trigonometry import to_polar


class Room:
    """
    Represents a room containing lamps and calculation zones.

    The room is defined by its dimensions and can contain multiple lamps and
    different calculation zones for evaluating lighting conditions and other
    characteristics.
    """

    def __init__(self, dimensions=None, units=None):
        self.units = None
        self.dimensions = None
        self.x = None
        self.y = None
        self.z = None
        self.volume = None

        default_units = "meters" if units is None else units.lower()
        self.set_units(default_units)

        default_dimensions = (
            [6.0, 4.0, 2.7] if self.units == "meters" else [20.0, 13.0, 9.0]
        )
        default_dimensions = default_dimensions if dimensions is None else dimensions
        self.set_dimensions(default_dimensions)

        self.lamps = {}
        self.calc_zones = {}

    def _check_position(self, dimensions):
        """
        Internal method to check if an object's dimensions exceed the room's boundaries.
        """
        for coord, roomcoord in zip(dimensions, self.dimensions):
            if coord > roomcoord:
                warnings.warn("Object exceeds room boundaries!", stacklevel=2)

    def set_units(self, units):
        """set room units"""
        if units not in ["meters", "feet"]:
            raise KeyError("Valid units are `meters` or `feet`")
        self.units = units

    def set_dimensions(self, dimensions):
        """set room dimensions"""
        if len(dimensions) != 3:
            raise ValueError("Room requires exactly three dimensions.")
        self.dimensions = np.array(dimensions)
        self.x, self.y, self.z = self.dimensions
        self.volume = self.x * self.y * self.z

    def get_units(self):
        """return room units"""
        return self.units

    def get_dimensions(self):
        """return room dimensions"""
        return self.dimensions

    def get_volume(self):
        """return room volume"""
        return self.volume

    def add_lamp(self, lamp):
        """
        Adds a lamp to the room if it fits within the room's boundaries.
        """
        self._check_position(lamp.position)
        self.lamps[lamp.lamp_id] = lamp

    def remove_lamp(self, lamp_id):
        """remove a lamp from the room"""
        del self.lamps[lamp_id]

    def add_calc_zone(self, calc_zone):
        """
        Adds a calculation zone to the room if it fits within the room's boundaries.
        """
        self._check_position(calc_zone.dimensions)
        self.calc_zones[calc_zone.zone_id] = calc_zone

    def remove_calc_zone(self, zone_id):
        """remove calculation zone from room"""
        del self.calc_zones[zone_id]

    def calculate(self):
        """
        Triggers the calculation of lighting values in each calculation zone based on the current lamps in the room.
        """
        for name, zone in self.calc_zones.items():
            zone.calculate_values(lamps=self.lamps)

    def plotly(self, fig=None, select_id=None, title="", color="#cc61ff", alpha=0.4):
        if fig is None:
            fig = go.Figure()
        traces = [trace.name for trace in fig.data]
        for lamp_id, lamp in self.lamps.items():
            if lamp.filename is not None and lamp.visible is True:
                x, y, z = lamp.transform(
                    lamp.photometric_coords, scale=lamp.values.max()
                ).T
                Theta, Phi, R = to_polar(*lamp.photometric_coords.T)
                tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))
                label = lamp.lamp_id
                if select_id is not None and select_id == lamp_id:
                    lampcolor = color  #'#ff6161'
                else:
                    lampcolor = "#5e8ff7"

                lamptrace = go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=tri.simplices[:, 0],
                    j=tri.simplices[:, 1],
                    k=tri.simplices[:, 2],
                    color=lampcolor,
                    opacity=alpha,
                    name=label,
                    showlegend=False,
                )
                aimtrace = go.Scatter3d(
                    x=[lamp.position[0], lamp.aim_point[0]],
                    y=[lamp.position[1], lamp.aim_point[1]],
                    z=[lamp.position[2], lamp.aim_point[2]],
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name=label + "_aim",
                    showlegend=False,
                )
                if lamptrace.name not in traces:
                    fig.add_trace(lamptrace)
                    fig.add_trace(aimtrace)
                else:
                    fig.update_traces(
                        x=x,
                        y=y,
                        z=z,
                        i=tri.simplices[:, 0],
                        j=tri.simplices[:, 1],
                        k=tri.simplices[:, 2],
                        color=lampcolor,
                        selector=({"name": lamptrace.name}),
                    )
                    fig.update_traces(
                        x=[lamp.position[0], lamp.aim_point[0]],
                        y=[lamp.position[1], lamp.aim_point[1]],
                        z=[lamp.position[2], lamp.aim_point[2]],
                        selector=({"name": aimtrace.name}),
                    )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, self.x]),
                yaxis=dict(range=[0, self.y]),
                zaxis=dict(range=[0, self.z]),
                aspectratio=dict(
                    x=self.x / self.z, y=self.y / self.z, z=self.z / self.z
                ),
            ),
            height=750,
        )
        fig.update_scenes(camera_projection_type="orthographic")
        return fig

    def plot(
        self,
        fig=None,
        ax=None,
        elev=30,
        azim=-45,
        title="",
        color="#cc61ff",
        alpha=0.4,
        select_id=None,
    ):
        """
        Generates a 3D plot of the room and the lamps in it
        """
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.view_init(azim=azim, elev=elev)
        ax.set_xlim(0, self.x)
        ax.set_ylim(0, self.y)
        ax.set_zlim(0, self.z)
        for lampid, lamp in self.lamps.items():
            if lamp.filename is not None:
                label = lamp.name
                if select_id is not None and select_id == lampid:
                    lampcolor = color  #'#ff6161'
                else:
                    lampcolor = "blue"
                x, y, z = lamp.transform(
                    lamp.photometric_coords, scale=lamp.values.max()
                ).T
                Theta, Phi, R = to_polar(*lamp.photometric_coords.T)
                tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))
                ax.plot_trisurf(
                    x,
                    y,
                    z,
                    triangles=tri.simplices,
                    label=label,
                    color=lampcolor,
                    alpha=alpha,
                )
                ax.plot(
                    *np.array((lamp.aim_point, lamp.position)).T,
                    linestyle="--",
                    linewidth=1,
                    color="black",
                    alpha=0.7,
                )

        return fig, ax
