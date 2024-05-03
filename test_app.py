import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configure the page to use wide mode
st.set_page_config(layout="wide")
st.markdown("""
<style>
    /* Adjust the margin below headers */
    .css-1h5na1f { margin-bottom: 0px !important; }
    /* Reduce the spacing between widget labels and their inputs */
    .stNumberInput label { margin-bottom: 0px; }
</style>
""", unsafe_allow_html=True)

# Set up the title of the app
st.title('Interactive Function Plotter')

# Create two columns: one for inputs and one for the plot
col1, col2 = st.columns([3, 2])  # Adjust the ratio based on your content needs

# Inputs will be in the first column
with col1:
    func_type = st.selectbox(
        'Select function type',
        ('Linear', 'Quadratic', 'Sinusoidal')
    )

    # Creating a row for the input parameters
    if func_type == 'Linear':
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Slope")  # Title for the first input
            a = st.number_input('', value=1.0, format="%.1f")
        with col_b:
            st.write("Intercept")  # Title for the second input
            b = st.number_input('', value=0.0, format="%.1f")
        func = lambda x: a * x + b
    elif func_type == 'Quadratic':
        col_a, col_b, col_c = st.columns(3)
        a = col_a.number_input('Coefficient a', value=1.0, format="%.1f")
        b = col_b.number_input('Coefficient b', value=0.0, format="%.1f")
        c = col_c.number_input('Coefficient c', value=0.0, format="%.1f")
        func = lambda x: a * x**2 + b * x + c
    else:  # Sinusoidal
        col_a, col_b, col_c = st.columns(3)
        a = col_a.number_input('Amplitude', value=1.0, format="%.1f")
        b = col_b.number_input('Frequency', value=1.0, format="%.1f")
        c = col_c.number_input('Phase', value=0.0, format="%.1f")
        func = lambda x: a * np.sin(b * x + c)

# Generate x values
x = np.linspace(-10, 10, 400)

# Generate y values based on the selected function
y = func(x)

# Plotting will be in the second column
with col2:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)
