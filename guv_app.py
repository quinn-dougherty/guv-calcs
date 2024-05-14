import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ies_utils import *

# configure the page
st.set_page_config(layout="wide") # i'm dreaming of a wide christmas
st.title('light it up')
col1, col2 = st.columns(2) # left input, right output


with col1:
    lampcol1, lampcol2 = st.columns(2)
         
    # set lamps
    root = Path("./ies_files")
    p = root.glob('**/*')
    ies_names = [x.stem for x in p if x.is_file() and x.suffix == '.ies']
    ies_name = lampcol1.selectbox('Select lamp', ies_names)
    ies_file = list(root.glob('**/'+ies_name+'.ies'))[0]
    
    num_lamps = lampcol2.number_input('Number of lamps',value=1,step=1,min_value=1)
    
    fig, ax = plot_ies(ies_file)
    st.pyplot(fig)
    
    # make the lampdict
    lamps = []
    
    
    
    
with col2:
    # set room dimensions
    col_a, col_b, col_c, col_d = st.columns(4)
    units = col_d.selectbox('Room units',['meters','feet'],index=0)
    if units == 'meters':
        defaultvals = [6.0, 4.0, 2.7]
    elif units == 'feet':
        defaultvals = [20.0, 13.0, 9.0]
    
    x = col_a.number_input('Room length (x)', value=defaultvals[0], format="%.2f",min_value=0.01)
    y = col_b.number_input('Room width (y)', value=defaultvals[1], format="%.2f",min_value=0.01)
    z = col_c.number_input('Room height (z)', value=defaultvals[2], format="%.2f",min_value=0.01)
       
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_zlim(z)
    st.pyplot(fig)