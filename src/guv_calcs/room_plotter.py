import plotly.graph_objs as go
import numpy as np
from scipy.spatial import Delaunay
from .trigonometry import to_polar
from .calc_zone import CalcPlane, CalcVol


class RoomPlotter:
    def __init__(self, room):
        self.room = room

    def plotly(
        self,
        fig=None,
        select_id=None,
        title="",
    ):
        """plot all"""
        if fig is None:
            fig = go.Figure()

        # first delete any extraneous traces
        lamp_ids = list(self.room.lamps.keys())
        aim_ids = [lampid + "_aim" for lampid in lamp_ids]
        zone_ids = list(self.room.calc_zones.keys())
        for active_ids in [lamp_ids, aim_ids, zone_ids]:
            self._remove_traces_by_ids(fig, active_ids)

        # plot lamps
        for lamp_id, lamp in self.room.lamps.items():
            if lamp.filedata is not None:
                fig = self._plot_lamp(lamp=lamp, fig=fig, select_id=select_id)
        for zone_id, zone in self.room.calc_zones.items():
            if isinstance(zone, CalcPlane):
                fig = self._plot_plane(zone=zone, fig=fig, select_id=select_id)
            elif isinstance(zone, CalcVol):
                fig = self._plot_vol(zone=zone, fig=fig, select_id=select_id)
        # set views
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[0, self.room.x]),
                yaxis=dict(range=[0, self.room.y]),
                zaxis=dict(range=[0, self.room.z]),
                aspectratio=dict(
                    x=self.room.x / self.room.z,
                    y=self.room.y / self.room.z,
                    z=self.room.z / self.room.z,
                ),
            ),
            height=750,
            autosize=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
            legend=dict(
                x=0,
                y=1,
                yanchor="top",
                xanchor="left",
            ),
        )
        fig.update_scenes(camera_projection_type="orthographic")
        return fig

    def _set_color(self, select_id, label, enabled):
        if not enabled:
            color = "#d1d1d1"  # grey
        elif select_id is not None and select_id == label:
            color = "#cc61ff"  # purple
        else:
            color = "#5e8ff7"  # blue
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
        lampcolor = self._set_color(select_id, label=lamp.lamp_id, enabled=lamp.enabled)

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
            showlegend=True,
        )
        aimtrace = go.Scatter3d(
            x=[lamp.x, lamp.aimx],
            y=[lamp.y, lamp.aimy],
            z=[lamp.z, lamp.aimz],
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
            self._update_trace_by_id(
                fig,
                lamp.lamp_id,
                x=x,
                y=y,
                z=z,
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                color=lampcolor,
                name=lamp.name,
            )

            self._update_trace_by_id(
                fig,
                lamp.lamp_id + "_aim",
                x=[lamp.x, lamp.aimx],
                y=[lamp.y, lamp.aimy],
                z=[lamp.z, lamp.aimz],
            )
        return fig

    def _plot_plane(self, zone, fig, select_id=None):
        """plot a calculation plane"""
        zonecolor = self._set_color(select_id, zone.zone_id, zone.enabled)
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
            self._update_trace_by_id(
                fig,
                zone.zone_id,
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

        zonecolor = self._set_color(select_id, label=zone.zone_id, enabled=zone.enabled)
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
            self._update_trace_by_id(
                fig,
                zone.zone_id,
                x=x_coords,
                y=y_coords,
                z=z_coords,
                line=dict(color=zonecolor, width=5, dash="dot"),
            )
        # fluence isosurface
        if zone.values is not None and zone.show_values:
            X, Y, Z = np.meshgrid(*zone.points, indexing="ij")
            x, y, z = X.flatten(), Y.flatten(), Z.flatten()
            values = zone.values.flatten()
            isomin = zone.values.mean() / 2
            if zone.name + " Values" not in traces:
                zone_value_trace = go.Isosurface(
                    x=x,
                    y=y,
                    z=z,
                    value=values,
                    surface_count=3,
                    isomin=isomin,
                    opacity=0.25,
                    showscale=False,
                    colorbar=None,
                    name=zone.name + " Values",
                    customdata=[zone.zone_id + "_values"],
                    legendgroup="zones",
                    legendgrouptitle_text="Calculation Zones",
                    showlegend=True,
                )
                fig.add_trace(zone_value_trace)
            else:
                self._update_trace_by_id(
                    fig,
                    zone.zone_id,
                    x=x,
                    y=y,
                    z=z,
                    values=values,
                    isomin=isomin,
                )

        return fig
