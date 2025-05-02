import warnings
from pathlib import Path
import datetime
import inspect
import json
import zipfile
import io
import numpy as np
from collections.abc import Iterable
from .lamp import Lamp
from .calc_zone import CalcZone, CalcPlane, CalcVol
from ._data import get_version
from ._helpers import parse_loadfile, check_savefile, fig_to_bytes, new_lamp_position
from .room_plotter import RoomPlotter
from .disinfection_calculator import DisinfectionCalculator
from .reflectance import ReflectanceManager

VALID_UNITS = ["meters", "feet"]


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
        enable_reflectance=None,
        reflectances=None,
        reflectance_x_spacings=None,
        reflectance_y_spacings=None,
        reflectance_max_num_passes=None,
        reflectance_threshold=None,
        air_changes=None,
        ozone_decay_constant=None,
    ):
        self.units = "meters" if units is None else units.lower()
        if self.units not in VALID_UNITS:
            raise KeyError(f"Invalid unit {units}")
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

        self.enable_reflectance = (
            True if enable_reflectance is None else enable_reflectance
        )
        self.ref_manager = ReflectanceManager(
            x=self.x,
            y=self.y,
            z=self.z,
            reflectances=reflectances,
            x_spacings=reflectance_x_spacings,
            y_spacings=reflectance_y_spacings,
            max_num_passes=reflectance_max_num_passes,
            threshold=reflectance_threshold,
        )
        self.plotter = RoomPlotter(self)
        self.disinfection = DisinfectionCalculator(self)

        self.set_dimensions()

        self.lamps = {}
        self.calc_zones = {}
        self.calc_state = {}
        self.update_state = {}

    def set_max_num_passes(self, max_num_passes):
        self.ref_manager.max_num_passes = max_num_passes
        return self

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
        data["enable_reflectance"] = self.enable_reflectance
        data["reflectances"] = self.ref_manager.reflectances
        data["reflectance_x_spacings"] = self.ref_manager.x_spacings
        data["reflectance_y_spacings"] = self.ref_manager.y_spacings
        data["reflectance_max_num_passes"] = self.ref_manager.max_num_passes
        data["reflectance_threshold"] = self.ref_manager.threshold
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
                    if zone.dose:
                        title = f"{zone.hours} Hour Dose"
                    else:
                        title = f"Irradiance"
                    if zone.calctype == "Plane":
                        # Save the figure to a BytesIO object
                        title += f" ({zone.height} m)"
                        fig, ax = zone.plot_plane(title=title)
                        data_dict[zone.name + ".png"] = fig_to_bytes(fig)
                    elif zone.calctype == "Volume":
                        fig = zone.plot_volume()
                        data_dict[zone.name + ".png"] = fig_to_bytes(fig)

        for lamp_id, lamp in self.lamps.items():
            if lamp.filedata is not None:
                if include_lamp_files:
                    data_dict[lamp.name + ".ies"] = lamp.save_ies()
                if include_lamp_plots:
                    ies_fig, ax = lamp.plot_ies(title=lamp.name)
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

        room_state = [  # temp..this should be put with the ref_manager eventually
            self.enable_reflectance,
            self.ref_manager.x_spacings.copy(),
            self.ref_manager.y_spacings.copy(),
        ]

        lamp_state = {}
        for key, lamp in self.lamps.items():
            if lamp.enabled:
                lamp_state[key] = lamp.get_calc_state()

        zone_state = {}
        for key, zone in self.calc_zones.items():
            if zone.calctype != "Zone" and zone.enabled:
                zone_state[key] = zone.get_calc_state()

        calc_state = {}
        calc_state["room"] = room_state
        calc_state["lamps"] = lamp_state
        calc_state["calc_zones"] = zone_state

        return calc_state

    def get_update_state(self):
        """
        Save all the features in the room that should NOT trigger
        a recalculation, only an update
        """

        room_state = [
            self.ref_manager.reflectances.copy(),
            self.units,
        ]

        lamp_state = {}
        for key, lamp in self.lamps.items():
            lamp_state[key] = lamp.get_update_state()

        zone_state = {}
        for key, zone in self.calc_zones.items():
            if zone.calctype != "Zone":
                zone_state[key] = zone.get_update_state()

        update_state = {}
        update_state["room"] = room_state
        update_state["lamps"] = lamp_state
        update_state["calc_zones"] = zone_state

        return update_state

    def set_units(self, units):
        """set room units"""
        if units not in ["meters", "feet"]:
            raise KeyError("Valid units are `meters` or `feet`")
        self.units = units
        self._update_standard_zones()
        self.harmonize_units()
        return self

    def harmonize_units(self):
        """ensure that all lamps in the state have the correct units"""
        for lamp in self.lamps.values():
            if lamp.surface.units != self.units:
                lamp.set_units(self.units)

    def set_dimensions(self, x=None, y=None, z=None):
        """set room dimensions"""
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        self.z = self.z if z is None else z
        self.dimensions = (self.x, self.y, self.z)
        self.volume = self.x * self.y * self.z

        self.ref_manager.update_dimensions(self.x, self.y, self.z)
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

    def get_disinfection_data(self, zone_id="WholeRoomFluence"):
        """return the fluence_dict, dataframe, and violin plot"""
        return self.disinfection.get_disinfection_data(zone_id=zone_id)

    def _update_standard_zones(self):
        """update the standard safety calculation zones based on the current standard and units"""
        if "UL8802" in self.standard:
            height = 1.9 if self.units == "meters" else 6.25
            skin_horiz = False
            eye_vert = False
            fov_vert = 180
        else:
            height = 1.8 if self.units == "meters" else 5.9
            skin_horiz = True
            eye_vert = True
            fov_vert = 80

        if "SkinLimits" in self.calc_zones.keys():
            self.calc_zones["SkinLimits"].set_height(height)
            self.calc_zones["SkinLimits"].horiz = skin_horiz
        if "EyeLimits" in self.calc_zones.keys():
            self.calc_zones["EyeLimits"].set_height(height)
            self.calc_zones["EyeLimits"].fov_vert = fov_vert
            self.calc_zones["EyeLimits"].vert = eye_vert

    def set_standard(self, standard):
        """update the photobiological safety standard the Room is subject to"""
        self.standard = standard
        self._update_standard_zones()
        return self

    def add_standard_zones(self):
        """
        convenience function. Add skin and eye limit calculation planes,
        plus whole room fluence.
        """

        # max_vol_val = 20
        # max_plane_val = 50

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
                # num_x=min(int(self.x * 20), max_vol_val),
                # num_y=min(int(self.y * 20), max_vol_val),
                # num_z=min(int(self.z * 20), max_vol_val),
                show_values=False,
            )
        )

        self.add_calc_zone(
            CalcPlane(
                zone_id="EyeLimits",
                name="Eye Dose (8 Hours)",
                x1=0,
                x2=self.x,
                y1=0,
                y2=self.y,
                # num_x=min(int(self.x * 20), max_plane_val),
                # num_y=min(int(self.y * 20), max_plane_val),
                horiz=False,
                fov_horiz=180,
                dose=True,
                hours=8,
            )
        )

        self.add_calc_zone(
            CalcPlane(
                zone_id="SkinLimits",
                name="Skin Dose (8 Hours)",
                x1=0,
                x2=self.x,
                y1=0,
                y2=self.y,
                # num_x=min(int(self.x * 20), max_plane_val),
                # num_y=min(int(self.y * 20), max_plane_val),
                dose=True,
                hours=8,
            )
        )

        # sets the height and field of view parameters
        self._update_standard_zones()

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
        if not isinstance(lamp, Lamp):
            raise TypeError(f"Must be type Lamp, not {type(lamp)}")
        # check units
        if lamp.surface.units != self.units:
            lamp.set_units(self.units)
        # check position
        self.check_lamp_position(lamp)
        self.lamps[lamp.lamp_id] = lamp
        return self

    def place_lamp(self, lamp):
        """
        Position a lamp as far from other lamps and the walls as possible
        """
        idx = len(self.lamps) + 1
        x, y = new_lamp_position(idx, self.x, self.y)
        lamp.move(x, y, self.z)
        self.add_lamp(lamp)
        return self

    def place_lamps(self, *args):
        """place multiple lamps in the room, as far away from each other and the walls as possible"""
        for obj in args:
            if isinstance(obj, Lamp):
                self.place_lamp(obj)
            else:
                msg = f"Cannot add object of type {type(obj).__name__} to Room."
                warnings.warn(msg, stacklevel=3)
        return self

    def remove_lamp(self, lamp_id):
        """remove a lamp from the room"""
        del self.lamps[lamp_id]
        return self

    def add_calc_zone(self, calc_zone):
        """
        Adds a calculation zone to the room if it fits within the room's boundaries.
        """
        if not isinstance(calc_zone, (CalcZone, CalcPlane, CalcVol)):
            raise TypeError(
                f"Must be CalcZone, CalcPlane, or CalcVol not {type(calc_zone)}"
            )
        self.check_zone_position(calc_zone)
        self.calc_zones[calc_zone.zone_id] = calc_zone
        return self

    def add(self, *args):
        """
        Add objects to the Room.

        - If an object is a Lamp, it is added as a lamp.
        - If an object is a CalcZone, CalcPlane, or CalcVol, it is added as a calculation zone.
        - If an object is iterable, it is recursively processed.
        - Otherwise, a warning is printed.
        """

        for obj in args:
            if isinstance(obj, Lamp):
                self.add_lamp(obj)
            elif isinstance(obj, (CalcZone, CalcPlane, CalcVol)):
                self.add_calc_zone(obj)
            elif isinstance(obj, dict):
                self.add(*obj.values())
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                self.add(*obj)  # Recursively process other iterables
            else:
                msg = f"Cannot add object of type {type(obj).__name__} to Room."
                warnings.warn(msg, stacklevel=3)

        return self

    def remove_calc_zone(self, zone_id):
        """remove calculation zone from room"""
        del self.calc_zones[zone_id]
        return self

    def _get_valid_lamps(self):
        """return"""
        return {
            k: v for k, v in self.lamps.items() if v.enabled and v.filedata is not None
        }

    def calculate(self, hard=False):
        """
        Triggers the calculation of lighting values in each calculation zone
        based on the current lamps in the room.

        If no updates have been made since the last calculate call that would
        require a full recalculation, either only an update will be performed
        or no recalculation will occur.

        If `hard` is True, this behavior is overriden and the full
        recalculation will be performed
        """

        self.harmonize_units()

        valid_lamps = self._get_valid_lamps()

        new_calc_state = self.get_calc_state()
        new_update_state = self.get_update_state()

        LAMP_RECALC = self.calc_state.get("lamps") != new_calc_state.get("lamps")
        REF_RECALC = self.calc_state.get("room") != new_calc_state.get("room")
        REF_UPDATE = self.update_state.get("room") != new_update_state.get("room")

        # calculate incidence on the surfaces if the reflectances or lamps have changed
        if (
            LAMP_RECALC or REF_RECALC or REF_UPDATE or hard
        ) and self.enable_reflectance:
            self.ref_manager.calculate_incidence(valid_lamps, hard=hard)

        ref_manager = self.ref_manager if self.enable_reflectance else None
        for name, zone in self.calc_zones.items():
            if zone.enabled:
                zone.calculate_values(
                    lamps=valid_lamps, ref_manager=ref_manager, hard=hard
                )
        # update calc states.
        self.calc_state = new_calc_state
        self.update_state = new_update_state

        # possibly this should be per-calc zone? idk maybe it's fine
        for lamp_id in valid_lamps.keys():
            new_calc_state = self.lamps[lamp_id].get_calc_state()
            self.lamps[lamp_id].calc_state = new_calc_state

        if len(valid_lamps) == 0:
            msg = "No valid lamps are present in the room--either lamps have been disabled, or filedata has not been provided."
            if len(self.lamps) == 0:
                msg = "No lamps are present in the room."
            warnings.warn(msg, stacklevel=3)

        return self

    def calculate_by_id(self, zone_id, hard=False):
        """calculate just the calc zone selected"""
        valid_lamps = {
            k: v for k, v in self.lamps.items() if v.enabled and v.filedata is not None
        }
        if len(valid_lamps) > 0:
            new_calc_state = self.get_calc_state()
            new_update_state = self.get_update_state()

            LAMP_RECALC = self.calc_state.get("lamps") != new_calc_state.get("lamps")
            REF_RECALC = self.calc_state.get("room") != new_calc_state.get("room")
            REF_UPDATE = self.update_state.get("room") != new_update_state.get("room")

            # calculate incidence on the surfaces if the reflectances or lamps have changed
            if (
                LAMP_RECALC or REF_RECALC or REF_UPDATE or hard
            ) and self.enable_reflectance:
                self.ref_manager.calculate_incidence(valid_lamps, hard=hard)
            ref_manager = self.ref_manager if self.enable_reflectance else None
            self.calc_zones[zone_id].calculate_values(
                lamps=valid_lamps, ref_manager=ref_manager, hard=hard
            )
            self.calc_state = self.get_calc_state()
            self.update_state = self.get_update_state()
        return self

    def plotly(self, fig=None, select_id=None, title=""):
        """return a plotly figure of all the room's components"""
        self.harmonize_units()
        return self.plotter.plotly(fig=fig, select_id=select_id, title=title)
