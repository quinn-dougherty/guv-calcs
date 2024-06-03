import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from guv_calcs.room import Room
from guv_calcs.calc_zone import CalcPlane, CalcVol
from guv_calcs._sidebar import (
    lamp_sidebar,
    zone_sidebar,
    room_sidebar,
    results_sidebar,
    default_sidebar,
)
from guv_calcs._website_helpers import (
    get_local_ies_files,
    get_ies_files,
    add_standard_zones,
    add_new_lamp,
    add_new_zone,
)
from guv_calcs._widget import (
    initialize_lamp,
    initialize_zone,
    initialize_room,
    clear_lamp_cache,
    clear_zone_cache,
)

# layout / page setup
st.set_page_config(
    page_title="Illuminate-GUV",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("deprecation.showPyplotGlobalUse", False)  # silence this warning
st.write(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)
ss = st.session_state

SELECT_LOCAL = "Select local file..."
SPECIAL_ZONES = ["WholeRoomFluence", "SkinLimits", "EyeLimits"]

if "lampfile_options" not in ss:
    ies_files = get_local_ies_files()  # local files for testing
    vendored_lamps = get_ies_files()  # files from assays.osluv.org
    ss.vendored_lamps = vendored_lamps
    options = [None] + list(vendored_lamps.keys()) + [SELECT_LOCAL]
    ss.lampfile_options = options

# Check and initialize session state variables
if "room" not in ss:
    ss.room = Room()
    ss.room = add_standard_zones(ss.room)

    preview_lamp = st.query_params.get("preview_lamp")
    if preview_lamp:
        lamp_id = add_new_lamp(ss.room, interactive=False)
        lamp = ss.room.lamps[f"Lamp{lamp_id}"]
        fdata = requests.get(ss.vendored_lamps[preview_lamp]).content
        lamp.reload(filename=preview_lamp, filedata=fdata)
        ss.room.calculate()
        ss.editing = "results"
        st.rerun()

room = ss.room

if "editing" not in ss:
    ss.editing = None  # determines what displays in the sidebar

if "selected_lamp_id" not in ss:
    ss.selected_lamp_id = None  # use None when no lamp is selected

if "selected_zone_id" not in ss:
    ss.selected_zone_id = None  # use None when no lamp is selected

if "uploaded_files" not in ss:
    ss.uploaded_files = {}

if "fig" not in ss:
    ss.fig = go.Figure()
    # Adding an empty scatter3d trace
    ss.fig.add_trace(
        go.Scatter3d(
            x=[0],  # No data points yet
            y=[0],
            z=[0],
            opacity=0,
            showlegend=False,
            customdata=["placeholder"],
        )
    )
    ss.eyefig = plt.figure()
    ss.skinfig = plt.figure()
fig = ss.fig


# Set up overall layout
left_pane, right_pane = st.columns([4, 1])

with st.sidebar:
    # Lamp editing sidebar
    if ss.editing == "lamps" and ss.selected_lamp_id is not None:
        lamp_sidebar(room)
    # calc zone editing sidebar
    elif ss.editing in ["zones", "planes", "volumes"] and ss.selected_zone_id:
        zone_sidebar(room)
    # room editing sidebar
    elif ss.editing == "room":
        room_sidebar(room)
    elif ss.editing == "results":
        results_sidebar(room)
    else:
        default_sidebar(room)


with right_pane:
    calculate = st.button("Calculate!", type="primary", use_container_width=True)
    edit_room = st.button("Edit Room", use_container_width=True)

    # Dropdown menus for luminaires; map display names to IDs
    lamp_names = {None: None}
    for lamp_id, lamp in room.lamps.items():
        lamp_names[lamp.name] = lamp_id
    lamp_sel_idx = list(lamp_names.values()).index(ss.selected_lamp_id)
    selected_lamp_name = st.selectbox(
        "Select luminaire", options=list(lamp_names), index=lamp_sel_idx
    )
    selected_lamp_id = lamp_names[selected_lamp_name]
    if ss.selected_lamp_id != selected_lamp_id:
        # if different, update and rerun
        ss.selected_lamp_id = selected_lamp_id
        if ss.selected_lamp_id is not None:
            # if lamp is selected, open editing pane
            ss.editing = "lamps"
            selected_lamp = room.lamps[ss.selected_lamp_id]
            # initialize widgets in editing pane
            initialize_lamp(selected_lamp)
            # clear widgets of anything to do with zone editing if it's currently loaded
            clear_zone_cache(room)
        st.rerun()

    add_lamp = st.button("Add Luminaire", use_container_width=True)

    # Drop down menu for calculation zones
    zone_names = {None: None}
    for zone_id, zone in room.calc_zones.items():
        zone_names[zone.name] = zone_id
    zone_sel_idx = list(zone_names.values()).index(ss.selected_zone_id)
    selected_zone_name = st.selectbox(
        "Select calculation zone", options=list(zone_names), index=zone_sel_idx
    )
    selected_zone_id = zone_names[selected_zone_name]
    if ss.selected_zone_id != selected_zone_id:
        ss.selected_zone_id = selected_zone_id
        if ss.selected_zone_id is not None:
            selected_zone = room.calc_zones[ss.selected_zone_id]
            if isinstance(selected_zone, CalcPlane):
                ss.editing = "planes"
                initialize_zone(selected_zone)
            elif isinstance(selected_zone, CalcVol):
                ss.editing = "volumes"
                initialize_zone(selected_zone)
            else:
                ss.editing = "zones"
            clear_lamp_cache(room)
        st.rerun()
    add_calc_zone = st.button("Add Calculation Zone", use_container_width=True)

    # st.write("")
    # show_results = st.button("Show results")

    if calculate:
        room.calculate()
        ss.editing = "results"
        # clear out any other selected objects and remove ones that haven't been fully initialized
        clear_lamp_cache(room)
        clear_zone_cache(room)
        st.rerun()

    if edit_room:
        ss.editing = "room"
        initialize_room(room)
        clear_lamp_cache(room)
        clear_zone_cache(room)
        st.rerun()

    if add_lamp:
        add_new_lamp(room)

    if add_calc_zone:
        add_new_zone(room)

    # if show_results:
    # ss.editing = "results"
    # clear_lamp_cache(room)
    # clear_zone_cache(room)
    # st.rerun()

# plot
with left_pane:
    # fig, ax = room.plot(select_id=ss.selected_lamp_id)
    # st.pyplot(fig,use_container_width=True)
    if ss.selected_lamp_id:
        select_id = ss.selected_lamp_id
    elif ss.selected_zone_id:
        select_id = ss.selected_zone_id
    else:
        select_id = None
    fig = room.plotly(fig=fig, select_id=select_id)
    st.plotly_chart(fig, use_container_width=True, height=750)
