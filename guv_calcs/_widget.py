import streamlit as st
from guv_calcs.calc_zone import CalcPlane, CalcVol


def clear_lamp_cache(room):
    """
    remove any lamps from the room and the widgets that don't have an
    associated filename, and deselect the lamp.
    """
    if st.session_state.selected_lamp_id:
        selected_lamp = room.lamps[st.session_state.selected_lamp_id]
        if selected_lamp.filename is None:
            remove_lamp(selected_lamp)
            room.remove_lamp(st.session_state.selected_lamp_id)
    st.session_state.selected_lamp_id = None


def clear_zone_cache(room):
    """
    remove any calc zones from the room and the widgets that don't have an
    associated zone type, and deselect the zone
    """
    if st.session_state.selected_zone_id:
        selected_zone = room.calc_zones[st.session_state.selected_zone_id]
        if not isinstance(selected_zone, (CalcPlane, CalcVol)):
            remove_zone(selected_zone)
            room.remove_calc_zone(st.session_state.selected_zone_id)
    st.session_state.selected_zone_id = None


def initialize_lamp(lamp):
    """initialize lamp editing widgets with their present values"""
    keys = [
        f"name_{lamp.lamp_id}",
        f"pos_x_{lamp.lamp_id}",
        f"pos_y_{lamp.lamp_id}",
        f"pos_z_{lamp.lamp_id}",
        f"aim_x_{lamp.lamp_id}",
        f"aim_y_{lamp.lamp_id}",
        f"aim_z_{lamp.lamp_id}",
        f"rotation_{lamp.lamp_id}",
        f"orientation_{lamp.lamp_id}",
        f"tilt_{lamp.lamp_id}",
        f"enabled_{lamp.lamp_id}",
    ]
    vals = [
        lamp.name,
        lamp.x,
        lamp.y,
        lamp.z,
        lamp.aimx,
        lamp.aimy,
        lamp.aimz,
        lamp.angle,
        lamp.heading,
        lamp.bank,
        lamp.enabled,
    ]
    print(lamp.enabled)
    add_keys(keys, vals)


def initialize_zone(zone):
    """initialize zone editing widgets with their present values"""
    keys = [
        f"name_{zone.zone_id}",
        f"x1_{zone.zone_id}",
        f"y1_{zone.zone_id}",
        f"x2_{zone.zone_id}",
        f"y2_{zone.zone_id}",
        f"x_spacing_{zone.zone_id}",
        f"y_spacing_{zone.zone_id}",
        f"offset_{zone.zone_id}",
        f"enabled_{zone.zone_id}",
    ]
    if isinstance(zone, CalcPlane):
        keys.append(f"height_{zone.zone_id}"),
        keys.append(f"fov80_{zone.zone_id}"),
    elif isinstance(zone, CalcVol):
        keys.append(f"z1_{zone.zone_id}")
        keys.append(f"z2_{zone.zone_id}")
        keys.append(f"z_spacing_{zone.zone_id}")

    vals = [
        zone.name,
        zone.x1,
        zone.y1,
        zone.x2,
        zone.y2,
        zone.x_spacing,
        zone.y_spacing,
        zone.offset,
        zone.enabled,
    ]
    if isinstance(zone, CalcPlane):
        vals.append(zone.height)
        vals.append(zone.fov80)
    elif isinstance(zone, CalcVol):
        vals.append(zone.z1)
        vals.append(zone.z2)
        vals.append(zone.z_spacing)

    add_keys(keys, vals)


def update_lamp_name(lamp):
    """update lamp name from widget"""
    lamp.name = st.session_state[f"name_{lamp.lamp_id}"]


def update_zone_name(zone):
    """update zone name from widget"""
    zone.name = st.session_state[f"name_{zone.zone_id}"]


def update_lamp_visibility(lamp):
    """update whether lamp shows in plot or not from widget"""
    lamp.enabled = st.session_state[f"enabled_{lamp.lamp_id}"]


def update_zone_visibility(zone):
    """update whether calculation zone shows up in plot or not from widget"""
    zone.enabled = st.session_state[f"enabled_{zone.zone_id}"]


def update_plane_dimensions(zone):
    """update dimensions and spacing of calculation volume from widgets"""
    zone.x1 = st.session_state[f"x1_{zone.zone_id}"]
    zone.x2 = st.session_state[f"x2_{zone.zone_id}"]
    zone.y1 = st.session_state[f"y1_{zone.zone_id}"]
    zone.y2 = st.session_state[f"y2_{zone.zone_id}"]
    zone.height = st.session_state[f"height_{zone.zone_id}"]

    zone.x_spacing = st.session_state[f"x_spacing_{zone.zone_id}"]
    zone.y_spacing = st.session_state[f"y_spacing_{zone.zone_id}"]

    zone.offset = st.session_state[f"offset_{zone.zone_id}"]

    zone._update()


def update_vol_dimensions(zone):
    """update dimensions and spacing of calculation volume from widgets"""
    zone.x1 = st.session_state[f"x1_{zone.zone_id}"]
    zone.x2 = st.session_state[f"x2_{zone.zone_id}"]
    zone.y1 = st.session_state[f"y1_{zone.zone_id}"]
    zone.y2 = st.session_state[f"y2_{zone.zone_id}"]
    zone.z1 = st.session_state[f"z1_{zone.zone_id}"]
    zone.z2 = st.session_state[f"z2_{zone.zone_id}"]

    zone.x_spacing = st.session_state[f"x_spacing_{zone.zone_id}"]
    zone.y_spacing = st.session_state[f"y_spacing_{zone.zone_id}"]
    zone.z_spacing = st.session_state[f"z_spacing_{zone.zone_id}"]

    zone.offset = st.session_state[f"offset_{zone.zone_id}"]

    zone._update


def update_lamp_position(lamp):
    """update lamp position and aim point based on widget input"""
    x = st.session_state[f"pos_x_{lamp.lamp_id}"]
    y = st.session_state[f"pos_y_{lamp.lamp_id}"]
    z = st.session_state[f"pos_z_{lamp.lamp_id}"]
    lamp.move(x, y, z)
    # update widgets
    update_lamp_aim_point(lamp)


def update_lamp_orientation(lamp):
    """update lamp object aim point, and tilt/orientation widgets"""
    aimx = st.session_state[f"aim_x_{lamp.lamp_id}"]
    aimy = st.session_state[f"aim_y_{lamp.lamp_id}"]
    aimz = st.session_state[f"aim_z_{lamp.lamp_id}"]
    lamp.aim(aimx, aimy, aimz)
    st.session_state[f"orientation_{lamp.lamp_id}"] = lamp.heading
    st.session_state[f"tilt_{lamp.lamp_id}"] = lamp.bank


def update_from_tilt(lamp, room):
    """update tilt+aim point in lamp, and aim point widget"""
    tilt = st.session_state[f"tilt_{lamp.lamp_id}"]
    lamp.set_tilt(tilt, dimensions=room.dimensions)
    update_lamp_aim_point(lamp)


def update_from_orientation(lamp, room):
    """update orientation+aim point in lamp, and aim point widget"""
    orientation = st.session_state[f"orientation_{lamp.lamp_id}"]
    lamp.set_orientation(orientation, room.dimensions)
    update_lamp_aim_point(lamp)


def update_lamp_aim_point(lamp):
    """reset aim point widget if any other parameter has been altered"""
    st.session_state[f"aim_x_{lamp.lamp_id}"] = lamp.aimx
    st.session_state[f"aim_y_{lamp.lamp_id}"] = lamp.aimy
    st.session_state[f"aim_z_{lamp.lamp_id}"] = lamp.aimz


def remove_lamp(lamp):
    """remove widget parameters if lamp has been deleted"""
    keys = [
        f"name_{lamp.lamp_id}",
        f"pos_x_{lamp.lamp_id}",
        f"pos_y_{lamp.lamp_id}",
        f"pos_z_{lamp.lamp_id}",
        f"aim_x_{lamp.lamp_id}",
        f"aim_y_{lamp.lamp_id}",
        f"aim_z_{lamp.lamp_id}",
        f"rotation_{lamp.lamp_id}",
        f"orientation_{lamp.lamp_id}",
        f"tilt_{lamp.lamp_id}",
    ]
    remove_keys(keys)


def remove_zone(zone):
    """remove widget parameters if calculation zone has been deleted"""

    keys = [
        f"name_{zone.zone_id}",
        f"x_{zone.zone_id}",
        f"y_{zone.zone_id}",
        f"x_spacing_{zone.zone_id}",
        f"y_spacing_{zone.zone_id}",
        f"offset_{zone.zone_id}",
        f"enabled_{zone.zone_id}",
    ]
    if isinstance(zone, CalcPlane):
        keys.append(f"height_{zone.zone_id}")
        remove_keys(keys)
    elif isinstance(zone, CalcVol):
        keys.append(f"zdim_{zone.zone_id}")
        keys.append(f"zspace_{zone.zone_id}")
        remove_keys(keys)


def remove_keys(keys):
    """remove parameters from widget"""
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def add_keys(keys, vals):
    """initialize widgets with parameters"""
    for key, val in zip(keys, vals):
        st.session_state[key] = val
