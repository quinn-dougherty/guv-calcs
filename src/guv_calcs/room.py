import warnings
from pathlib import Path
import datetime
import inspect
import json
import zipfile
import io
import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objs as go
from .lamp import Lamp
from .calc_zone import CalcZone, CalcPlane, CalcVol
from .trigonometry import to_polar
from ._data import get_version
from ._helpers import parse_loadfile, check_savefile, fig_to_bytes


class Room:
    """
    Represents a room containing lamps and calculation zones.

    The room is defined by its dimensions and can contain multiple lamps and
    different calculation zones for evaluating lighting conditions and other
    characteristics.
    """

    def __init__(
        self,
        x=None,
        y=None,
        z=None,
        units=None,
        standard=None,
        reflectance_ceiling=None,
        reflectance_north=None,
        reflectance_east=None,
        reflectance_south=None,
        reflectance_west=None,
        reflectance_floor=None,
        air_changes=None,
        ozone_decay_constant=None,
    ):
        self.units = "meters" if units is None else units.lower()
        self.set_units(self.units)  # just checks that unit entry is valid
        default_dimensions = (
            [6.0, 4.0, 2.7] if self.units == "meters" else [20.0, 13.0, 9.0]
        )
        self.x = default_dimensions[0] if x is None else x
        self.y = default_dimensions[1] if y is None else y
        self.z = default_dimensions[2] if z is None else z
        self.set_dimensions()

        self.standard = (
            "ANSI IES RP 27.1-22 (America)" if standard is None else standard
        )

        self.reflectance_ceiling = (
            0 if reflectance_ceiling is None else reflectance_ceiling
        )
        self.reflectance_north = 0 if reflectance_north is None else reflectance_north
        self.reflectance_east = 0 if reflectance_east is None else reflectance_east
        self.reflectance_south = 0 if reflectance_south is None else reflectance_south
        self.reflectance_west = 0 if reflectance_west is None else reflectance_west
        self.reflectance_floor = 0 if reflectance_floor is None else reflectance_floor
        self.air_changes = 1.0 if air_changes is None else air_changes
        self.ozone_decay_constant = (
            2.7 if ozone_decay_constant is None else ozone_decay_constant
        )

        self.lamps = {}
        self.calc_zones = {}

    def to_dict(self):
        data = {}
        data["x"] = self.x
        data["y"] = self.y
        data["z"] = self.z
        data["units"] = self.units
        data["reflectance_ceiling"] = self.reflectance_ceiling
        data["reflectance_north"] = self.reflectance_north
        data["reflectance_east"] = self.reflectance_east
        data["reflectance_south"] = self.reflectance_south
        data["reflectance_west"] = self.reflectance_west
        data["reflectance_floor"] = self.reflectance_floor
        data["standard"] = self.standard
        data["air_changes"] = self.air_changes
        data["ozone_decay_constant"] = self.ozone_decay_constant

        dct = self.__dict__.copy()
        data["lamps"] = {k: v.save_lamp() for k, v in dct["lamps"].items()}
        data["calc_zones"] = {k: v.save_zone() for k, v in dct["calc_zones"].items()}
        return data

    def save(self, fname=None):

        savedata = {}
        version = get_version(Path(__file__).parent / "_version.py")
        savedata["guv-calcs_version"] = version

        now = datetime.datetime.now()
        now_local = datetime.datetime.now(now.astimezone().tzinfo)
        timestamp = now_local.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        savedata["timestamp"] = timestamp

        savedata["data"] = self.to_dict()

        if fname is not None:
            filename = check_savefile(fname, ".guv")
            with open(filename, "w") as json_file:
                json.dump(savedata, json_file, indent=4)
        else:
            return json.dumps(savedata, indent=4)

    @classmethod
    def load(cls, filedata):
        """load from a file"""

        load_data = parse_loadfile(filedata)

        saved_version = load_data["guv-calcs_version"]
        current_version = get_version(Path(__file__).parent / "_version.py")
        if saved_version != current_version:
            warnings.warn(
                f"This file was saved with guv-calcs {saved_version}, while you have {current_version} installed."
            )
        room_dict = load_data["data"]

        roomkeys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        room = cls(**{k: v for k, v in room_dict.items() if k in roomkeys})

        for lampid, lamp in room_dict["lamps"].items():
            room.add_lamp(Lamp.from_dict(lamp))

        for zoneid, zone in room_dict["calc_zones"].items():
            if zone["calctype"] == "Plane":
                room.add_calc_zone(CalcPlane.from_dict(zone))
            elif zone["calctype"] == "Volume":
                room.add_calc_zone(CalcVol.from_dict(zone))
        return room

    def export_zip(
        self,
        fname=None,
        include_plots=False,
        include_lamp_files=False,
        include_lamp_plots=False,
    ):
        """
        write the project file and all results files to a zip file. Optionally include
        extra files like lamp ies files, spectra files, and plots.
        """

        # save project
        data_dict = {"room.guv": self.save()}

        # save all results
        for zone_id, zone in self.calc_zones.items():
            if zone.calctype != "Zone":
                data_dict[zone.name + ".csv"] = zone.export()
                if include_plots:
                    if zone.calctype == "Plane":
                        # Save the figure to a BytesIO object
                        if zone.dose:
                            title = f"{zone.hours} Hour Dose ({zone.height} m)"
                        else:
                            title = f"Irradiance ({zone.height} m)"
                        fig, ax = zone.plot_plane(title=title)
                        data_dict[zone.name + ".png"] = fig_to_bytes(fig)

        for lamp_id, lamp in self.lamps.items():
            if lamp.filedata is not None:
                if include_lamp_files:
                    data_dict[lamp.name + ".ies"] = lamp.save_ies()
                if include_lamp_plots:
                    ies_fig, ax = lamp.plot_ies(lamp.name)
                    data_dict[lamp.name + "_ies.png"] = fig_to_bytes(ies_fig)
            if lamp.spectra is not None:
                if include_lamp_plots:
                    linfig, _ = lamp.spectra.plot(
                        title=lamp.name, yscale="linear", weights=True, label=True
                    )
                    logfig, _ = lamp.spectra.plot(
                        title=lamp.name, yscale="log", weights=True, label=True
                    )
                    linkey = lamp.name + "_spectra_linear.png"
                    logkey = lamp.name + "_spectra_log.png"
                    data_dict[linkey] = fig_to_bytes(linfig)
                    data_dict[logkey] = fig_to_bytes(logfig)

        zip_buffer = io.BytesIO()
        # Create a zip file within this BytesIO object
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Loop through the dictionary, adding each string/byte stream to the zip
            for filename, content in data_dict.items():
                # Ensure the content is in bytes
                if isinstance(content, str):
                    content = content.encode("utf-8")
                # Add the file to the zip; writing the bytes to a BytesIO object for the file
                file_buffer = io.BytesIO(content)
                zip_file.writestr(filename, file_buffer.getvalue())
        zip_bytes = zip_buffer.getvalue()

        if fname is not None:
            with open(fname, "wb") as f:
                f.write(zip_bytes)
        else:
            return zip_bytes

    def set_units(self, units):
        """set room units"""
        if units not in ["meters", "feet"]:
            raise KeyError("Valid units are `meters` or `feet`")
        self.units = units
        return self

    def set_dimensions(self, x=None, y=None, z=None):
        """set room dimensions"""
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        self.z = self.z if z is None else z
        self.dimensions = (self.x, self.y, self.z)
        self.volume = self.x * self.y * self.z
        return self

    def get_units(self):
        """return room units"""
        return self.units

    def get_dimensions(self):
        """return room dimensions"""
        self.dimensions = np.array((self.x, self.y, self.z))
        return self.dimensions

    def get_volume(self):
        """return room volume"""
        self.volume = self.x * self.y * self.z
        return self.volume

    def add_standard_zones(self):
        """
        convenience function. Add skin and eye limit calculation planes,
        plus whole room fluence.
        """
        if "UL8802" in self.standard:
            height = 1.9
            skin_horiz = False
            eye_vert = False
            fov80 = False
        else:
            height = 1.8
            skin_horiz = True
            eye_vert = True
            fov80 = True

        self.add_calc_zone(
            CalcVol(
                zone_id="WholeRoomFluence",
                name="Whole Room Fluence",
                x1=0,
                x2=self.x,
                y1=0,
                y2=self.y,
                z1=0,
                z2=self.z,
                num_x=int(self.x * 20),
                num_y=int(self.y * 20),
                num_z=int(self.z * 20),
                show_values=False,
            )
        )

        self.add_calc_zone(
            CalcPlane(
                zone_id="SkinLimits",
                name="Skin Dose (8 Hours)",
                height=height,
                x1=0,
                x2=self.x,
                y1=0,
                y2=self.y,
                num_x=int(self.x * 20),
                num_y=int(self.y * 20),
                vert=False,
                horiz=skin_horiz,
                fov80=False,
                dose=True,
                hours=8,
            )
        )

        self.add_calc_zone(
            CalcPlane(
                zone_id="EyeLimits",
                name="Eye Dose (8 Hours)",
                height=height,
                x1=0,
                x2=self.x,
                y1=0,
                y2=self.y,
                num_x=int(self.x * 20),
                num_y=int(self.y * 20),
                vert=eye_vert,
                horiz=False,
                fov80=fov80,
                dose=True,
                hours=8,
            )
        )
        return self

    def _check_position(self, dimensions, obj_name):
        """
        Method to check if an object's dimensions exceed the room's boundaries.
        """
        msg = None
        for coord, roomcoord in zip(dimensions, self.dimensions):
            if coord > roomcoord:
                msg = f"{obj_name} exceeds room boundaries!"
                warnings.warn(msg, stacklevel=2)
        return msg

    def check_lamp_position(self, lamp):
        return self._check_position(lamp.position, lamp.name)

    def check_zone_position(self, calc_zone):
        if isinstance(calc_zone, CalcPlane):
            dimensions = [calc_zone.x2, calc_zone.y2]
        elif isinstance(calc_zone, CalcVol):
            dimensions = [calc_zone.x2, calc_zone.y2, calc_zone.z2]
        elif isinstance(calc_zone, CalcZone):
            # this is a hack; a generic CalcZone is just a placeholder
            dimensions = self.dimensions
        return self._check_position(dimensions, calc_zone.name)

    def check_positions(self):
        msgs = []
        for lamp_id, lamp in self.lamps.items():
            msgs.append(self.check_lamp_position(lamp))
        for zone_id, zone in self.calc_zones.items():
            msgs.append(self.check_zone_position(zone))
        return msgs

    def add_lamp(self, lamp):
        """
        Adds a lamp to the room if it fits within the room's boundaries.
        """
        self.check_lamp_position(lamp)
        self.lamps[lamp.lamp_id] = lamp
        return self

    def remove_lamp(self, lamp_id):
        """remove a lamp from the room"""
        del self.lamps[lamp_id]
        return self

    def add_calc_zone(self, calc_zone):
        """
        Adds a calculation zone to the room if it fits within the room's boundaries.
        """
        self.check_zone_position(calc_zone)
        self.calc_zones[calc_zone.zone_id] = calc_zone
        return self

    def remove_calc_zone(self, zone_id):
        """remove calculation zone from room"""
        del self.calc_zones[zone_id]
        return self

    def calculate(self):
        """
        Triggers the calculation of lighting values in each calculation zone based on the current lamps in the room.
        """
        for name, zone in self.calc_zones.items():
            if zone.enabled:
                zone.calculate_values(lamps=self.lamps)
        return self

    def calculate_by_id(self, zone_id):
        """calculate just the calc zone selected"""
        self.calc_zones[zone_id].calculate_values(lamps=self.lamps)
        return self

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
        lamp_ids = list(self.lamps.keys())
        aim_ids = [lampid + "_aim" for lampid in lamp_ids]
        zone_ids = list(self.calc_zones.keys())
        for active_ids in [lamp_ids, aim_ids, zone_ids]:
            self._remove_traces_by_ids(fig, active_ids)

        # plot lamps
        for lamp_id, lamp in self.lamps.items():
            if lamp.filedata is not None:
                fig = self._plot_lamp(lamp=lamp, fig=fig, select_id=select_id)
        for zone_id, zone in self.calc_zones.items():
            if isinstance(zone, CalcPlane):
                fig = self._plot_plane(zone=zone, fig=fig, select_id=select_id)
            elif isinstance(zone, CalcVol):
                fig = self._plot_vol(zone=zone, fig=fig, select_id=select_id)
        # set views
        fig.update_layout(
            title=title,
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
                yanchor="top",
                xanchor="left",
            ),
        )
        fig.update_scenes(camera_projection_type="orthographic")
        return fig
