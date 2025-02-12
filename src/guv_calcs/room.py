import warnings
from pathlib import Path
import datetime
import inspect
import json
import zipfile
import io
import numpy as np
from .lamp import Lamp
from .calc_zone import CalcZone, CalcPlane, CalcVol
from ._data import get_version
from ._helpers import parse_loadfile, check_savefile, fig_to_bytes
from .room_plotter import RoomPlotter
from .reflectance import ReflectanceManager


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
        reflectances=None,
        reflectance_x_spacings=None,
        reflectance_y_spacings=None,
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

        self.standard = (
            "ANSI IES RP 27.1-22 (ACGIH Limits)" if standard is None else standard
        )
        self.ozone_decay_constant = (
            2.7 if ozone_decay_constant is None else ozone_decay_constant
        )
        self.air_changes = 1.0 if air_changes is None else air_changes

        self.ref_manager = ReflectanceManager(
            self,
            reflectances,
            reflectance_x_spacings,
            reflectance_y_spacings,
        )
        self.plotter = RoomPlotter(self)

        self.set_dimensions()

        self.lamps = {}
        self.calc_zones = {}
        self.calc_state = {}

    def set_reflectance(self, R, wall_id=None):
        self.ref_manager.set_reflectance(R=R, wall_id=wall_id)
        return self

    def set_reflectance_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        self.ref_manager.set_spacing(
            x_spacing=x_spacing, y_spacing=y_spacing, wall_id=wall_id
        )
        return self

    def to_dict(self):
        data = {}
        data["x"] = self.x
        data["y"] = self.y
        data["z"] = self.z
        data["units"] = self.units
        data["reflectances"] = self.ref_manager.reflectances
        data["reflectance_x_spacing"] = self.ref_manager.x_spacings
        data["reflectance_y_spacing"] = self.ref_manager.y_spacings
        data["standard"] = self.standard
        data["air_changes"] = self.air_changes
        data["ozone_decay_constant"] = self.ozone_decay_constant

        dct = self.__dict__.copy()
        data["lamps"] = {k: v.save_lamp() for k, v in dct["lamps"].items()}
        data["calc_zones"] = {k: v.save_zone() for k, v in dct["calc_zones"].items()}
        return data

    def save(self, fname=None):
        """save all relevant parameters to a json file"""
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

    def get_calc_state(self):
        """
        Save all the features in the room that, if changed, will require re-calculation
        """
        room_state = [
            self.ref_manager.reflectances.copy(),
            self.ref_manager.x_spacings.copy(),
            self.ref_manager.y_spacings.copy(),
            self.units,
        ]

        lamp_state = {}
        # save only the features that affect the calculation
        # maybe think about optimizing it later
        for key, lamp in self.lamps.items():
            intensity_map_orig = (
                lamp.surface.intensity_map_orig.sum()
                if lamp.surface.intensity_map_orig is not None
                else None
            )
            intensity_map = (
                lamp.surface.intensity_map.sum()
                if lamp.surface.intensity_map is not None
                else None
            )
            lamp_state[key] = [
                lamp.filedata,
                lamp.x,
                lamp.y,
                lamp.z,
                lamp.angle,
                lamp.aimx,
                lamp.aimy,
                lamp.aimz,
                lamp.intensity_units,  # can be optimized
                lamp.spectra_source,
                lamp.surface.length,  # only for nearfield
                lamp.surface.width,  # ""
                lamp.surface.depth,
                lamp.surface.units,  # ""
                lamp.surface.source_density,  # ""
                intensity_map_orig,
                intensity_map,  # ""
                lamp.enabled,  # can be optimized
            ]

        zone_state = {}

        for key, zone in self.calc_zones.items():
            if zone.calctype != "Zone":
                zone_state[key] = [
                    zone.offset,
                    zone.fov_vert,
                    zone.fov_horiz,  # can be optimized
                    zone.vert,
                    zone.horiz,
                    zone.enabled,
                    zone.x1,
                    zone.x2,
                    zone.x_spacing,
                    zone.num_x,
                    zone.y1,
                    zone.y2,
                    zone.y_spacing,
                    zone.num_y,
                ]
                if zone.calctype == "Plane":
                    zone_state[key] += [zone.height]
                elif zone.calctype == "Volume":
                    zone_state[key] += [zone.z1, zone.z2, zone.z_spacing, zone.num_z]

        calc_state = {}
        calc_state["room"] = room_state
        calc_state["lamps"] = lamp_state
        calc_state["calc_zones"] = zone_state
        return calc_state

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

        self.ref_manager.update_dimensions()
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
            fov_vert = 180
        else:
            height = 1.8
            skin_horiz = True
            eye_vert = True
            fov_vert = 80

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
                num_x=min(int(self.x * 20), 50),
                num_y=min(int(self.y * 20), 50),
                num_z=min(int(self.z * 20), 50),
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
                num_x=min(int(self.x * 20), 300),
                num_y=min(int(self.y * 20), 300),
                horiz=skin_horiz,
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
                num_x=min(int(self.x * 20), 300),
                num_y=min(int(self.y * 20), 300),
                vert=eye_vert,
                horiz=False,
                fov_vert=fov_vert,
                fov_horiz=180,
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

        self.ref_manager.calculate_incidence()

        for name, zone in self.calc_zones.items():
            if zone.enabled:
                zone.calculate_values(lamps=self.lamps, ref_manager=self.ref_manager)

        self.calc_state = self.get_calc_state()
        return self

    def calculate_by_id(self, zone_id):
        """calculate just the calc zone selected"""
        self.calc_zones[zone_id].calculate_values(lamps=self.lamps)
        self.calc_state = self.get_calc_state()
        return self

    def plotly(self, fig=None, select_id=None, title=""):
        """return a plotly figure of all the room's components"""
        return self.plotter.plotly(fig=fig, select_id=select_id, title=title)
