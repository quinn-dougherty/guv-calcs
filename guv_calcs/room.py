import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import plotly.graph_objs as go
from guv_calcs.calc_zone import CalcZone, CalcPlane, CalcVol
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
        if isinstance(calc_zone, CalcPlane):
            dimensions = [calc_zone.x2, calc_zone.y2]
        elif isinstance(calc_zone, CalcVol):
            dimensions = [calc_zone.x2, calc_zone.y2, calc_zone.z2]
        elif isinstance(calc_zone, CalcZone):
            dimensions = self.dimensions
        self._check_position(dimensions)
        self.calc_zones[calc_zone.zone_id] = calc_zone

    def remove_calc_zone(self, zone_id):
        """remove calculation zone from room"""
        del self.calc_zones[zone_id]

    def calculate(self):
        """
        Triggers the calculation of lighting values in each calculation zone based on the current lamps in the room.
        """
        for name, zone in self.calc_zones.items():
            if zone.enable:
                zone.calculate_values(lamps=self.lamps)

    # def _set_visibility(self, fig, trace_id, val):
        # """change visibility of lamp or calc zone trace"""
        # traces = [trace.name for trace in fig.data]
        # if trace_id in traces:
            # # fig.data[traces.index(trace_id)].visible = val
            # fig.update_traces(visible=val,selector=({"name": trace_id}))
            # print(val)
        # return fig

    # def _get_visibility(self, fig, trace_id):
        # """get visibility status of a trace"""
        # traces = [trace.name for trace in fig.data]
        # if trace_id in traces:
            # vis = fig.data[traces.index(trace_id)].visible
        # else: 
            # vis = None
        # return vis

    def _set_color(self, select_id, label, enable):
        if not enable:
            color = "#d1d1d1" # grey
        elif select_id is not None and select_id == label:
            color = "#cc61ff" # purple
        else:
            color = "#5e8ff7" # blue
        return color

    def _update_trace_by_id(self, fig, trace_id, **updates):
        # Iterate through all traces
        for trace in fig.data:
            # Check if trace customdata matches the trace_id
            if trace.customdata and trace.customdata[0] == trace_id:
                # Update trace properties based on additional keyword arguments
                for key, value in updates.items():
                    setattr(trace, key, value)

    def _remove_traces_by_ids(self, fig, active_ids):
        # Convert fig.data, which is a tuple, to a list to allow modifications
        traces = list(fig.data)
        
        # Iterate in reverse to avoid modifying the list while iterating
        for i in reversed(range(len(traces))):
            trace = traces[i]
            # Check if the trace's customdata is set and its ID is not in the list of active IDs
            if trace.customdata and trace.customdata[0] not in active_ids:
                del traces[i]  # Remove the trace from the list
        fig.data = traces

    def _plot_lamp(self, lamp, fig, select_id=None, color="#cc61ff"):
        """plot lamp as a photometric web"""

        x, y, z = lamp.transform(lamp.photometric_coords, scale=lamp.values.max()).T
        Theta, Phi, R = to_polar(*lamp.photometric_coords.T)
        tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))
        lampcolor = self._set_color(select_id,label=lamp.lamp_id,enable=lamp.enable)
        
        lamptrace = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            color=lampcolor,
            opacity=0.4,
            name=lamp.name,
            customdata=[lamp.lamp_id],
            legendgroup="lamps",
            legendgrouptitle_text="Lamps",
            showlegend=True
        )
        aimtrace = go.Scatter3d(
            x=[lamp.position[0], lamp.aim_point[0]],
            y=[lamp.position[1], lamp.aim_point[1]],
            z=[lamp.position[2], lamp.aim_point[2]],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name=lamp.name,
            customdata=[lamp.lamp_id + "_aim"],
            showlegend=False,
        )
        
        traces = [trace.customdata[0] for trace in fig.data]
        if lamptrace.customdata[0] not in traces:
            fig.add_trace(lamptrace)
            fig.add_trace(aimtrace)
        else:
            self._update_trace_by_id(fig,lamp.lamp_id, 
                x=x,
                y=y,
                z=z,
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                color=lampcolor,
                name=lamp.name) 

            self._update_trace_by_id(fig, lamp.lamp_id+'_aim', 
                x=[lamp.position[0], lamp.aim_point[0]],
                y=[lamp.position[1], lamp.aim_point[1]],
                z=[lamp.position[2], lamp.aim_point[2]],
            )
        return fig

    def _plot_plane(self, zone, fig, select_id=None):
        """plot a calculation plane"""
        zonecolor = self._set_color(select_id,zone.zone_id,zone.enable)
        zonetrace = go.Scatter3d(
            x=zone.coords.T[0],
            y=zone.coords.T[1],
            z=zone.coords.T[2],
            mode="markers",
            marker=dict(size=2, color=zonecolor),
            opacity=0.5,
            legendgroup="zones",
            legendgrouptitle_text="Calculation Zones",
            showlegend=True,
            name=zone.name,
            customdata=[zone.zone_id],
        )
        traces = [trace.name for trace in fig.data]
        if zonetrace.name not in traces:
            fig.add_trace(zonetrace)
        else:
            self._update_trace_by_id(fig, zone.zone_id,
                x=zone.coords.T[0],
                y=zone.coords.T[1],
                z=zone.coords.T[2],
                marker=dict(size=2, color=zonecolor),
            )

        return fig

    def _plot_vol(self, zone, fig, select_id=None):
        # Define the vertices of the rectangular prism
        vertices = [
            (zone.x1, zone.y1, zone.z1),  # 0
            (zone.x2, zone.y1, zone.z1),  # 1
            (zone.x2, zone.y2, zone.z1),  # 2
            (zone.x1, zone.y2, zone.z1),  # 3
            (zone.x1, zone.y1, zone.z2),  # 4
            (zone.x2, zone.y1, zone.z2),  # 5
            (zone.x2, zone.y2, zone.z2),  # 6
            (zone.x1, zone.y2, zone.z2),  # 7
        ]

        # Define edges by vertex indices
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Side edges
        ]

        # Create lists for x, y, z coordinates
        x_coords = []
        y_coords = []
        z_coords = []

        # Append coordinates for each edge, separated by None to create breaks
        for v1, v2 in edges:
            x_coords.extend([vertices[v1][0], vertices[v2][0], None])
            y_coords.extend([vertices[v1][1], vertices[v2][1], None])
            z_coords.extend([vertices[v1][2], vertices[v2][2], None])

        zonecolor = self._set_color(select_id,label=zone.zone_id,enable=zone.enable)
        # Create a single trace for all edges
        zonetrace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="lines",
            line=dict(color=zonecolor, width=5, dash="dot"),
            legendgroup="zones",
            legendgrouptitle_text="Calculation Zones",
            name=zone.name,
            customdata=[zone.zone_id],
        )
        traces = [trace.name for trace in fig.data]
        if zonetrace.name not in traces:
            fig.add_trace(zonetrace)
        else:
            self._update_trace_by_id(fig, zone.zone_id,
                x=x_coords,
                y=y_coords,
                z=z_coords,
                line=dict(color=zonecolor, width=5, dash="dot"),
                )
        return fig

    def plotly(self, fig=None, select_id=None, title="",):
        """plot all """
        if fig is None:
            fig = go.Figure()

        # first delete any extraneous traces
        lamp_ids = list(self.lamps.keys())
        aim_ids = [lampid+'_aim' for lampid in lamp_ids]
        zone_ids = list(self.calc_zones.keys())
        for active_ids in [lamp_ids,aim_ids,zone_ids]:
            self._remove_traces_by_ids(fig, active_ids)

        # plot lamps
        for lamp_id, lamp in self.lamps.items():
            if lamp.filedata is not None:
                fig = self._plot_lamp(lamp=lamp, fig=fig, select_id=select_id)
                traces = [trace.name for trace in fig.data]
                # vis = self._get_visibility(fig, lamp_id)
                # fig = self._set_visibility(fig, lamp_id+"_aim", vis)
        for zone_id, zone in self.calc_zones.items():
            if isinstance(zone, CalcPlane):
                fig = self._plot_plane(zone=zone, fig=fig, select_id=select_id)
            elif isinstance(zone, CalcVol):
                fig = self._plot_vol(zone=zone, fig=fig, select_id=select_id)
        # set views
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
            autosize=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
            legend=dict(
                x=0,
                y=1,
                yanchor='top',
                xanchor='left',
                # orientation='h',
                # font=dict(family="Courier", size=12, color="blue"),
                # bgcolor="LightSteelBlue"
            ),
        )
        fig.update_scenes(camera_projection_type="orthographic")
        # print(dir(fig.layout.scene.camera))
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
        DEPRECATED
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
            if lamp.filename is not None and lamp.visible:
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
