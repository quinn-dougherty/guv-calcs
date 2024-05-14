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

    def __init__(self, dimensions, units: str = "meters"):
        self.dimensions = dimensions
        self.units = units
        self.x, self.y, self.z = self.dimensions
        self.volume = self.x * self.y * self.z
        self.lamps = []
        self.calc_zones = {}

    def _check_position(self, dimensions):
        """
        Internal method to check if an object's dimensions exceed the room's boundaries.
        """
        for coord, roomcoord in zip(dimensions, self.dimensions):
            if coord > roomcoord:
                warnings.warn("Object exceeds room boundaries!", stacklevel=2)

    def add_lamp(self, lamp):
        """
        Adds a lamp to the room if it fits within the room's boundaries.
        """
        self._check_position(lamp.position)
        self.lamps.append(lamp)

    def add_calc_zone(self, calc_zone):
        """
        Adds a calculation zone to the room if it fits within the room's boundaries.
        """
        self._check_position(calc_zone.dimensions)
        self.calc_zones[calc_zone.zone_id] = calc_zone

    def calculate(self):
        """
        Triggers the calculation of lighting values in each calculation zone based on the current lamps in the room.
        """
        for name, zone in self.calc_zones.items():
            zone.calculate_values(lamps=self.lamps)

    def plot(
        self,
        elev=45,
        azim=-45,
        title="",
        figsize=(6, 4),
        color="#cc61ff",
        alpha=0.4,
        use_plotly=False,
    ):
        """
        Generates a 3D plot of the room and the lamps in it
        """
        if use_plotly:
            fig = go.Figure()
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        for lamp in self.lamps:
            x, y, z = lamp.transform(lamp.photometric_coords, scale=lamp.values.max()).T
            Theta, Phi, R = to_polar(*lamp.photometric_coords.T)
            tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))
            if use_plotly:
                fig.add_trace(
                    go.Mesh3d(
                        x=x,
                        y=y,
                        z=z,
                        i=tri.simplices[:, 0],
                        j=tri.simplices[:, 1],
                        k=tri.simplices[:, 2],
                        color=color,
                        opacity=alpha,
                        name=lamp.filename.stem,
                    )
                )
                if lamp.aim_point is not None:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[lamp.position[0], lamp.aim_point[0]],
                            y=[lamp.position[1], lamp.aim_point[1]],
                            z=[lamp.position[2], lamp.aim_point[2]],
                            mode="lines",
                            line=dict(color="black", width=2, dash="dash"),
                            showlegend=False,
                        )
                    )
            else:
                ax.plot_trisurf(
                    x, y, z, triangles=tri.simplices, color=color, alpha=alpha
                )
                if lamp.aim_point is not None:
                    ax.plot(
                        *np.array((lamp.aim_point, lamp.position)).T,
                        linestyle="--",
                        color="black",
                        alpha=0.7
                    )
        if use_plotly:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[0, self.x]),
                    yaxis=dict(range=[0, self.y]),
                    zaxis=dict(range=[0, self.z]),
                    aspectratio=dict(x=self.x, y=self.y, z=self.z),
                )
            )
            fig.update_scenes(camera_projection_type="orthographic")
            fig.show()
        else:
            ax.set_title(title)
            ax.view_init(azim=azim, elev=elev)
            ax.set_xlim(0, self.x)
            ax.set_ylim(0, self.y)
            ax.set_zlim(0, self.z)
            # fig.subplots_adjust(top=10, bottom=-10, hspace=10)
            plt.show()
        return fig
