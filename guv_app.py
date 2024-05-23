import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objs as go
from guv_calcs.room import Room
from guv_calcs.lamp import Lamp
from guv_calcs.calc_zone import CalcZone, CalcPlane, CalcVol
from guv_calcs._website_helpers import get_lamp_position, get_ies_files

# layout / page setup
st.set_page_config(
    page_title='GUV Calcs',
    layout="wide",
    initial_sidebar_state='expanded', 
) 
st.set_option('deprecation.showPyplotGlobalUse', False) # silence this warning
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# Check and initialize session state variables
if 'room' not in st.session_state:
    st.session_state.room = Room()
room = st.session_state.room

if 'editing' not in st.session_state:
    st.session_state.editing = None   # determines what displays in the sidebar
    
if 'selected_lamp_id' not in st.session_state:
    st.session_state.selected_lamp_id = None  # use None when no lamp is selected
    
if 'selected_zone_id' not in st.session_state:
    st.session_state.selected_zone_id = None  # use None when no lamp is selected

if 'fig' not in st.session_state:     
    st.session_state.fig = go.Figure()
    # Adding an empty scatter3d trace
    st.session_state.fig.add_trace(go.Scatter3d(
        x=[0],  # No data points yet
        y=[0],
        z=[0],
        opacity=0,
        # mode='markers'
    ))
fig = st.session_state.fig

ies_files = [None] + get_ies_files() + ["Select local file..."]
# Set up overall layout
left_pane, right_pane = st.columns([4,1])

with st.sidebar:
    # Lamp editing sidebar
    if st.session_state.editing == 'lamps' and st.session_state.selected_lamp_id is not None:

        selected_lamp = room.lamps[st.session_state.selected_lamp_id]
        # st.session_state.selected_lamp = selected_lamp 
    
        st.subheader("Edit Luminaire")
                
        cola, colb = st.columns([3,1])       
        selected_lamp.name = cola.text_input("Name", 
                             value=selected_lamp.name, 
                             key=f"name_{selected_lamp.lamp_id}",
                             # label_visibility="collapsed"
                             )
        colb.write("")
        colb.write("")
        save = colb.button("Enter",use_container_width=True)
        if save:
            st.rerun()
        
        # File input
        options = [None] + get_ies_files() + ["Select local file..."]     
        fname_idx = options.index(selected_lamp.filename)
        fname = st.selectbox("Select file", options, key=f"file_{selected_lamp.lamp_id}",index=fname_idx) 
        
        if fname == "Select local file...":
            uploaded_file = st.file_uploader("Upload a file", key=f"upload_{selected_lamp.lamp_id}")
            if uploaded_file is not None:
                fname = uploaded_file.read()
                
        if fname not in [None, "Select local file..."]:
            if fname != selected_lamp.filename:
                selected_lamp.reload(fname)
            iesfig, iesax = selected_lamp.plot_ies()
            st.pyplot(iesfig,use_container_width=True)

        # Position inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.number_input(
                "Position X",
                min_value=0.0,
                # max_value=room.x,
                value=selected_lamp.x,
                step=0.1,
                key=f"pos_x_{selected_lamp.lamp_id}",
            )
        with col2:
            y = st.number_input(
                "Position Y",
                min_value=0.0,
                # max_value=room.y,
                value=selected_lamp.y,
                step=0.1,
                key=f"pos_y_{selected_lamp.lamp_id}",
            )
        with col3:
            z = st.number_input(
                "Position Z",
                min_value=0.0,
                # max_value=room.z,
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
        
        col7, col8 = st.columns(2)
        del_button = col7.button("Delete Lamp",type="primary",use_container_width=True)
        close_button = col8.button("Close",use_container_width=True)

        if close_button: # maybe replace with an enable/disable button?
            st.session_state.editing = None
            st.session_state.selected_lamp_id = None     
            if selected_lamp.filename is None:
                room.remove_lamp(selected_lamp.lamp_id)
            st.rerun()
        if del_button:
            # st.session_state.lamps.remove(selected_lamp)
            room.remove_lamp(selected_lamp.lamp_id)
            st.session_state.editing = None
            st.session_state.selected_lamp_id = None
            st.rerun()          
    # calc zone editing sidebar
    elif st.session_state.editing in ['zones','planes','volumes'] and st.session_state.selected_zone_id:
        st.subheader("Edit Calculation Zone")
        cola,colb = st.columns([3,1])
        calc_types = ["Plane","Volume"]
        zone_type = cola.selectbox("Select calculation type",options=calc_types)
        colb.write("")
        colb.write("")
        if colb.button("Go"):
            idx = st.session_state.selected_zone_id[-1]
            if zone_type=="Plane":       
                new_zone = CalcPlane(zone_id=st.session_state.selected_zone_id,name='CalcPlane'+idx)
                st.session_state.editing = 'planes'
            elif zone_type=="Volume":
                new_zone = CalcVol(zone_id=st.session_state.selected_zone_id,name='CalcVol'+idx)
                st.session_state.editing = 'volumes'
            room.add_calc_zone(new_zone)
            st.rerun()
        selected_zone = room.calc_zones[st.session_state.selected_zone_id]
        if st.session_state.editing in ['planes','volumes']:
            
            
            cola, colb = st.columns([3,1])
            selected_zone.name = cola.text_input("Name", 
                                value=selected_zone.name, 
                                key=f"name_{selected_zone.zone_id}",
                                # label_visibility="collapsed"
                                )
            colb.write("")
            colb.write("")
            save = colb.button("Enter",use_container_width=True)
            if save:
                st.rerun()
                                
        if st.session_state.editing=="planes":    
        
            col1,col2 = st.columns([2,1])
            # xy dimensions and height
            height = col1.number_input(
                    "Height",
                    min_value=0.0,
                    max_value=room.z,
                    value=selected_zone.height,
                    step=None,
                    key=f"height_{selected_zone.zone_id}",
                )
            col2.write('')
            col2.write('')
            col2.write(room.units)
            col2, col3 = st.columns(2)   
            with col2:
                x = st.number_input(
                    "X dimension",
                    min_value=0.0,
                    # max_value=room.x,
                    value=selected_zone.x,
                    step=0.1,
                    key=f"xdim_{selected_zone.zone_id}",
                )
                x_spacing = st.number_input(
                    "X spacing",
                    min_value=0.01,
                    # max_value=room.x,
                    value=selected_zone.x_spacing,
                    step=0.01,
                    key=f"xspace_{selected_zone.zone_id}",
                )
            with col3:
                y = st.number_input(
                    "Y dimension",
                    min_value=0.0,
                    # max_value=room.y,
                    value=selected_zone.y,
                    step=0.1,
                    key=f"ydim_{selected_zone.zone_id}",
                )
                y_spacing = st.number_input(
                    "Y spacing",
                    min_value=0.01,
                    # max_value=room.y,
                    value=selected_zone.y_spacing,
                    step=0.01,
                    key=f"yspace_{selected_zone.zone_id}",
                )
                
            options = ['All angles','Horizontal irradiance','Vertical irradiance']
            st.selectbox('Calculation type',options,index=0)
            st.checkbox('Offset',value=True)
            st.checkbox('Field of View 80°')
            
            
        if st.session_state.editing=="volumes": 
            col1, col2, col3 = st.columns(3)
            with col1:
                x = st.number_input(
                    "X dimension",
                    min_value=0.0,
                    # max_value=room.x,
                    value=room.x,
                    step=0.1,
                    key=f"xdim_{selected_zone.zone_id}",
                )
                x_spacing = st.number_input(
                    "X spacing",
                    min_value=0.01,
                    # max_value=room.x,
                    value=selected_zone.spacing[0],
                    step=0.01,
                    key=f"xspace_{selected_zone.zone_id}",
                )
            with col2:
                y = st.number_input(
                    "Y dimension",
                    min_value=0.0,
                    # max_value=room.y,
                    value=room.y,
                    step=0.1,
                    key=f"ydim_{selected_zone.zone_id}",
                )
                y_spacing = st.number_input(
                    "Y spacing",
                    min_value=0.01,
                    # max_value=room.y,
                    value=selected_zone.spacing[1],
                    step=0.01,
                    key=f"yspace_{selected_zone.zone_id}",
                )
            with col3:
                z = st.number_input(
                    "Z dimension",
                    min_value=0.0,
                    # max_value=room.z,
                    value=room.z,
                    step=0.1,
                    key=f"zdim_{selected_zone.zone_id}",
                )
                z_spacing = st.number_input(
                    "Z spacing",
                    min_value=0.01,
                    # max_value=room.z,
                    value=selected_zone.z_spacing,
                    step=0.01,
                    key=f"zspace_{selected_zone.zone_id}",
                )              
                
        
        if st.session_state.editing == 'zones':
            del_button = st.button("Cancel",use_container_width=True)
            close_button = None
        elif st.session_state.editing in ['planes','volumes']:
            col7, col8 = st.columns(2)
            del_button = col7.button("Delete Calc Zone",type="primary",use_container_width=True)
            close_button = col8.button("Close",use_container_width=True)

        if close_button: # maybe replace with an enable/disable button?
            st.session_state.editing = None
            st.session_state.selected_zone_id = None            
            st.rerun()
        if del_button:
            room.remove_calc_zone(st.session_state.selected_zone_id)
            st.session_state.editing = None
            st.session_state.selected_zone_id = None
            st.rerun()          
    # room editing sidebar
    elif st.session_state.editing == 'room':
        st.subheader("Edit Room")
        # set room dimensions and units
        col_a, col_b, col_c = st.columns(3)
        units = st.selectbox("Room units", ["meters", "feet"], index=0)

        x = col_a.number_input(
            "Room length (x)", value=room.x#, format="%.2f", min_value=0.01,# step=0.1,
        )
        y = col_b.number_input(
            "Room width (y)", value=room.y#, format="%.2f", min_value=0.01,# step=0.1,
        )
        z = col_c.number_input(
            "Room height (z)", value=room.z#, format="%.2f", min_value=0.01,# step=0.1,
        )
        dimensions = np.array((x, y, z))
        if units != room.units:
            room.set_units(units)
            st.rerun()
        if (dimensions != room.dimensions).any():
            room.set_dimensions(dimensions)       
            st.rerun()
            
        close_button = st.button("Close", use_container_width=True)
        if close_button:
            st.session_state.editing=None
            st.rerun()
    elif st.session_state.editing == 'results':
        st.subheader("Results")
        close_button = st.button("Close", use_container_width=True)
        if close_button:
            st.session_state.editing=None
            st.rerun()
    else:
        st.write("")
                
with right_pane:
    calculate = st.button(
            "Calculate!", type="primary", use_container_width=True
        )
    # st.divider()
    edit_room = st.button("Edit Room",use_container_width=True)
    # st.divider()
            
    # Dropdown menus for luminaires
    lamp_names = {None : None}
    for lamp_id,lamp in room.lamps.items():
        lamp_names[lamp.name]=lamp_id 
   
    lamp_sel_idx = list(lamp_names.values()).index(st.session_state.selected_lamp_id)
    selected_lamp_name = st.selectbox("Select luminaire", options=list(lamp_names),index=lamp_sel_idx)    
    selected_lamp_id = lamp_names[selected_lamp_name]
    # if different, update and rerun
    if st.session_state.selected_lamp_id!= selected_lamp_id:
        st.session_state.selected_lamp_id = selected_lamp_id
        if st.session_state.selected_lamp_id is not None:
            st.session_state.editing = 'lamps'
        st.rerun()
    add_lamp = st.button("Add Luminaire",use_container_width=True)
    
    # st.divider()
    
    # Drop down menu for calculation zones
    zone_names = {None : None}
    for zone_id, zone in room.calc_zones.items():
        zone_names[zone.name]=zone_id
    
    zone_sel_idx = list(zone_names.values()).index(st.session_state.selected_zone_id)
    selected_zone_name = st.selectbox("Select calculation zone", options=list(zone_names), index=zone_sel_idx)
    selected_zone_id = zone_names[selected_zone_name]
    if st.session_state.selected_zone_id!= selected_zone_id:
        st.session_state.selected_zone_id = selected_zone_id
        if st.session_state.selected_zone_id is not None:
            st.session_state.editing = 'zones'
        st.rerun()
    add_calc_zone = st.button("Add Calculation Zone",use_container_width=True)
    
    if calculate:
        room.calculate()
        st.session_state.editing = 'results'
        st.session_state.selected_zone_id = None
        st.session_state.selected_lamp_id = None
        st.rerun()
    
    if edit_room:
        st.session_state.editing = 'room'
        st.session_state.selected_zone_id = None
        st.session_state.selected_lamp_id = None
        st.rerun()
    
    # Adding new lamps        
    if add_lamp:
        #initialize lamp 
        new_lamp_idx = len(room.lamps) + 1
        # set initial position
        xpos, ypos = get_lamp_position(lamp_idx=new_lamp_idx, x=room.x, y=room.y)
        new_lamp_id = f"Lamp{new_lamp_idx}"
        new_lamp = Lamp(lamp_id=new_lamp_id, x=xpos, y=ypos, z=room.z)        
        # add to session and to room 
        # st.session_state.lamps.append(new_lamp) 
        room.add_lamp(new_lamp) 
        # Automatically select for editing
        st.session_state.editing = 'lamps'
        st.session_state.selected_lamp_id = new_lamp.lamp_id
        st.rerun()

    # Adding new calculation zones
    if add_calc_zone:
        # initialize calculation zone
        new_zone_idx = len(room.calc_zones) + 1
        new_zone_id = f"CalcZone{new_zone_idx}"        
        # this zone object contains nothing but the name and ID and will be 
        # replaced by a CalcPlane or CalcVol object
        new_zone = CalcZone(zone_id=new_zone_id,dimensions=room.dimensions)
        # add to room        
        room.add_calc_zone(new_zone)
        # select for editing
        st.session_state.editing = 'zones'
        st.session_state.selected_zone_id = new_zone_id        
        st.rerun()
        
    
        
with left_pane:   
    # room plot!
    # fig, ax = room.plot(select_id=st.session_state.selected_lamp_id)
    # st.pyplot(fig,use_container_width=True)
    fig = room.plotly(fig=fig, select_id=st.session_state.selected_lamp_id)
    st.plotly_chart(fig,use_container_width=True, height=1600)#, use_container_width=True)
    
    # # set room dimensions and units
    # col_a, col_b, col_c, col_d = st.columns(4)
    # units = col_d.selectbox("Room units", ["meters", "feet"], index=0)

    # x = col_a.number_input(
        # "Room length (x)", value=room.x, format="%.2f", min_value=0.01,# step=0.1,
    # )
    # y = col_b.number_input(
        # "Room width (y)", value=room.y, format="%.2f", min_value=0.01,# step=0.1,
    # )
    # z = col_c.number_input(
        # "Room height (z)", value=room.z, format="%.2f", min_value=0.01,# step=0.1,
    # )
    # dimensions = np.array((x, y, z))
    # if units != room.units:
        # room.set_units(units)
        # st.rerun()
    # if (dimensions != room.dimensions).any():
        # room.set_dimensions(dimensions)       
        # st.rerun()
        
    