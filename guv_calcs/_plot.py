import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.graph_objs as go
import numpy as np
from scipy.spatial import Delaunay

def plot_tlvs(skin_values, eye_values, room, height, figsize=(8,3.5),title=''):
    """
    Plot the eye and skin doses in the plane
    """

    # set figure sizes
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.07],figure=fig)  # 2 plots and 1 for colorbar
    
    vmin = eye_values.min()
    vmax = skin_values.max()
    extent = [0, room['dimensions'][0], 0, room['dimensions'][1]]
    
    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(skin_values, extent=extent, vmin=vmin, vmax=vmax )
    title1 = 'Skin dose at ' + str(height) + ' ' + room['units']
    title1 += '\nMax: '+str(round(skin_values.max(),2))+ ' mJ/cm²'
    ax1.set_title(title1)
    
    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(eye_values, extent=extent, vmin=vmin, vmax=vmax )
    title2 = 'Eye dose at ' + str(height) + ' ' + room['units']
    title2 += '\nMax: '+str(round(eye_values.max(),2))+ ' mJ/cm²'
    ax2.set_title(title2)
    
    # Colorbar in the third column of GridSpec
    cbar_ax = plt.subplot(gs[2])
    cbar = fig.colorbar(im2, label = "mJ/cm²/8 hours", cax=cbar_ax, use_gridspec=False, shrink=0.9,extendrect=False)
    fig.suptitle(title,y=1.0)
    plt.show()
    return fig
    
def photometric_web(lamps, room):
    """
    plot photometric webs of lamps in the room
    I don't know how to make the aspect ratio correct :( 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for lamp in lamps:
        
        lampfile = lamp['file']
        lampdict = read_ies_data(lampfile)
        valdict = lampdict["full_vals"]
        thetas = valdict['thetas']
        phis = valdict['phis']
        values = valdict['values']
 
        tgrid, pgrid = np.meshgrid(thetas, phis)
        tflat, pflat, rflat = tgrid.flatten(), pgrid.flatten(), values.flatten()
        tflat = 180 - tflat  # to account for reversed z direction

        x,y,z = to_cartesian(tflat, pflat, rflat)
        x0, y0, z0 = lamp['position']

        if room['dimensions']=='feet':
            scale = max(rflat) / 3.28084
        else:
            scale = max(rflat)

        x = x / scale + x0
        y = y / scale + y0
        z = z / scale + z0
        tri = Delaunay(np.column_stack((tflat,pflat)))
        img = ax.plot_trisurf(x, y, z, triangles=tri.simplices, color='blue', alpha=0.3)
    
    xlim, ylim, zlim = room['dimensions']
    ax.set_xlim(0,xlim)
    ax.set_ylim(0,ylim)
    ax.set_zlim(0,zlim)
    fig.subplots_adjust(top=10, bottom=-10,hspace=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return fig
    
def photometric_web_plotly(lamps, room):
    fig = go.Figure()
    for lamp in lamps:
        # load 
        lampfile = lamp['file']
        lampdict = read_ies_data(lampfile)
        valdict = lampdict["full_vals"]
        thetas = valdict['thetas']
        phis = valdict['phis']
        values = valdict['values']
        
        tgrid, pgrid = np.meshgrid(thetas, phis)
        tflat, pflat, rflat = tgrid.flatten(), pgrid.flatten(), values.flatten()
        tflat = 180 - tflat  # to account for reversed z direction

        x,y,z = to_cartesian(tflat,pflat,rflat)
        x0, y0, z0 = lamp['position']
        xlim, ylim, zlim = room['dimensions']

        scale = max(rflat) / 2
        
        x = x / (scale*xlim) + x0
        y = y / (scale*ylim) + y0
        z = z / (scale*zlim) + z0

        spherical_points = np.column_stack((tflat, pflat))
        tri = Delaunay(spherical_points)
        
        fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=tri.simplices[:, 0],
        j=tri.simplices[:, 1],
        k=tri.simplices[:, 2],
        color='purple',
        opacity=0.3,
        name=lampfile.stem
        ))

    # Set the range of axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, xlim]),
            yaxis=dict(range=[0, ylim]),
            zaxis=dict(range=[0, zlim]),
            aspectratio=dict(x=10, y=10, z=2.5)
        )
    )

    fig.show()