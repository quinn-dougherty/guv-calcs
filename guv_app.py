import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from guv_calcs.room import Room
from guv_calcs.lamp import Lamp
from guv_calcs._website_helpers import get_lamp_position, get_ies_files

st.set_option('deprecation.showPyplotGlobalUse', False) # silence this warning

# set up overall layout
st.set_page_config(layout="wide") # dreaming of a wide christmas

# Check and initialize session state variables
if 'room' not in st.session_state:
    st.session_state.room = Room()
room = st.session_state.room

if 'lamps' not in st.session_state:
    st.session_state.lamps = []

if 'selected_lamp_id' not in st.session_state:
    st.session_state.selected_lamp_id = None  # use None when no lamp is selected

ies_files = [None] + get_ies_files() + ["Select local file..."]
# Set up overall layout
left_pane, right_pane = st.columns([4, 5])

with left_pane:
    # Button to add a new lamp
    if st.button("Add Lamp"):
        new_lamp_id = len(st.session_state.lamps) + 1
        xpos, ypos = get_lamp_position(lamp_idx=new_lamp_id, x=room.x, y=room.y)
        # lamp label
        new_lamp = Lamp(lamp_id=f"Lamp{new_lamp_id}", x=xpos, y=ypos, z=room.z)
        st.session_state.lamps.append(new_lamp) # add to session
        room.add_lamp(new_lamp) # add to room
        # Automatically select the new lamp for editing
        st.session_state.selected_lamp_id = new_lamp.lamp_id

    # Dropdown to select a lamp for editing
    lamp_options = {lamp.lamp_id: lamp for lamp in st.session_state.lamps}
    selected_lamp_id = st.selectbox("Select a lamp to edit", list(lamp_options.keys()),index=len(lamp_options)-1 if lamp_options else 0)
    st.session_state.selected_lamp_id = selected_lamp_id if selected_lamp_id else None

# This condition checks if there is a selected lamp and opens the sidebar accordingly

if st.session_state.selected_lamp_id:
    selected_lamp = lamp_options[st.session_state.selected_lamp_id]
    
    with st.sidebar:
        st.write("Edit Luminaire")
        selected_lamp.name = st.text_input("Name", value=selected_lamp.name, key=f"name_{selected_lamp.lamp_id}")

        # File input
        fname = st.selectbox(
            "Select file", [None] + get_ies_files() + ["Select local file..."], key=f"file_{selected_lamp.lamp_id}"
        )
        if fname == "Select local file...":
            uploaded_file = st.file_uploader("Upload a file", key=f"upload_{selected_lamp.lamp_id}")
            if uploaded_file is not None:
                fname = uploaded_file.read()
                
        if fname not in [None, "Select local file..."]:
            if fname != selected_lamp.filename:
                selected_lamp.reload(fname)
            iesfig, ax = selected_lamp.plot_ies()
            st.pyplot(iesfig,use_container_width=True)

        # Position inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.number_input(
                "Position X",
                min_value=0.0,
                max_value=room.x,
                value=selected_lamp.x,
                step=0.1,
                key=f"pos_x_{selected_lamp.lamp_id}",
            )
        with col2:
            y = st.number_input(
                "Position Y",
                min_value=0.0,
                max_value=room.y,
                value=selected_lamp.y,
                step=0.1,
                key=f"pos_y_{selected_lamp.lamp_id}",
            )
        with col3:
            z = st.number_input(
                "Position Z",
                min_value=0.0,
                max_value=room.z,
                value=selected_lamp.z,
                step=0.1,
                key=f"pos_z_{selected_lamp.lamp_id}",
            )
        selected_lamp.move(x, y, z)

        # Rotation input
        rotation = st.number_input(
            "Rotation",
            value=selected_lamp.angle,
            min_value=0.0,
            max_value=360.0,
            step=1.0,
            key=f"rotation_{selected_lamp.lamp_id}",
        )
        selected_lamp.rotate(rotation)
        # Aim point inputs
        col4, col5, col6 = st.columns(3)
        with col4:
            aimx = st.number_input(
                "Aim X",
                value=selected_lamp.aimx,
                key=f"aim_x_{selected_lamp.lamp_id}",
            )
        with col5:
            aimy = st.number_input(
                "Aim Y",
                value=selected_lamp.aimy,
                key=f"aim_y_{selected_lamp.lamp_id}",
            )
        with col6:
            aimz = st.number_input(
                "Aim Z",
                value=selected_lamp.aimz,
                key=f"aim_z_{selected_lamp.lamp_id}",
            )
        selected_lamp.aim(aimx, aimy, aimz)
        
        col_save, col_del = st.columns(2)
        save_button = col_save.button("Save Changes")
        del_button = col_del.button("Delete Lamp",type="primary")

        if save_button:
            room.add_lamp(selected_lamp)
            st.success("Changes Saved!")
            st.sidebar.open = False
            st.rerun()
        if del_button:
            st.session_state.lamps.remove(selected_lamp)
            room.remove_lamp(selected_lamp.lamp_id)
            st.sidebar.open = False
            st.rerun()
with left_pane:
    # room plot!
    fig = room.plot()
    st.pyplot(fig,use_container_width=True)
    # 
    
    # set room dimensions and units
    col_a, col_b, col_c, col_d = st.columns(4)
    units = col_d.selectbox("Room units", ["meters", "feet"], index=0)

    x = col_a.number_input(
        "Room length (x)", value=room.x, format="%.2f", min_value=0.01, step=0.1,
    )
    y = col_b.number_input(
        "Room width (y)", value=room.y, format="%.2f", min_value=0.01, step=0.1,
    )
    z = col_c.number_input(
        "Room height (z)", value=room.z, format="%.2f", min_value=0.01, step=0.1,
    )
    dimensions = np.array((x, y, z))
    if units != room.units:
        room.set_units(units)
    if (dimensions != room.dimensions).all():
        room.set_dimensions(dimensions)            

# Plot the room layout or other visuals in the right pane if needed
with right_pane:
    calculate_button = st.button(
        "Calculate!", type="primary", use_container_width=False
    )
    