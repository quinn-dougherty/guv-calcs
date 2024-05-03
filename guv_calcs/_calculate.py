import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ies_utils import read_ies_data, get_intensity

def to_polar(x, y, z):
    """
    convert arrays of [x,y,z] points to arrays of [phi,theta,r] in degrees
    """
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.degrees(np.arccos(-z/r))
    theta = np.nan_to_num(theta,nan=0)
    phi = np.degrees(np.arctan2(x,y))
    phi[np.where(phi<0)] = phi[np.where(phi<0)] + 360
    
    return np.array((theta,phi,r))

def to_cartesian(theta, phi, r):
    """
    convert from degrees polar to cartesian coordinates
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    x = r * np.sin(theta_rad) * np.sin(phi_rad)
    y = r * np.sin(theta_rad) * np.cos(phi_rad) 
    z = r * np.cos(theta_rad)
    return x, y, z

def calculate_values(coords:np.array,lamps:list, room:dict, fov80=False, horiz=False,vert=False):
    room_dimensions = room['dimensions']
    total_values = np.zeros(coords.shape[0])
    for lamp in lamps:
        # verify lamp position
        lamp_position = lamp['position']
        for lampcoord, roomcoord in zip(lamp_position,room_dimensions):
            if lampcoord>roomcoord:
                msg = 'Luminaire is outside of room boundaries!'
                warnings.warn(msg, stacklevel=2)
                
        # determine lamp placement + calculate relative coordinates
        rel_coords = coords - lamp_position
        X, Y, Z = rel_coords.T
        Theta, Phi, R = to_polar(X, Y, Z)
        # load ies file
        lampfile = lamp['file']
        lampdict = read_ies_data(lampfile)
        valdict = lampdict["full_vals"]
        values = np.array([get_intensity(theta, phi, valdict)/r**2 for theta, phi, r in zip(Theta, Phi, R)]) 
        if fov80:
            values[Theta<50] = 0
        if horiz:
            values *= np.sin(np.radians(Theta))
        if vert:
            values *= np.cos(np.radians(Theta))
        total_values += values / 10 
    return total_values

def calculate_fluence(lamps:list, room:dict):
    """
    Calculate the fluence at every point
    """
    room_dimensions = room['dimensions']
    x,y,z = room_dimensions
    xdiv, ydiv, zdiv = room['divisions']    
    points = [np.linspace(div/2,val-div/2,int(val/div)) for val,div in zip((x,y,z),(xdiv,ydiv,zdiv))]
    grids = np.meshgrid(*points)
    X,Y,Z = [grid.reshape(-1) for grid in grids]
    coords = np.array([np.array((x0,y0,z0)) for x0,y0,z0 in zip(X,Y,Z)])
    values = calculate_values(coords, lamps, room)
    values = values.reshape(grids[0].shape)
    return values

def calculate_plane(lamps, 
                    room, 
                    height = 1.9, 
                    div=0.1, 
                    fov80=False, 
                    horiz=False, 
                    vert=True,
                    dose=True):
    """
    calculate the uW/cm2 at points along a plane of arbitrary height
    """
    x,y,z = room['dimensions']
    if height>z or height<0:
        msg = 'Plane of height {} is outside of room boundaries!'.format(z)
        warnings.warn(msg, stacklevel=2)
    
    points = [np.linspace(div/2,val-div/2,int(val/div)) for val in (x,y)]
    grids = np.meshgrid(*points)
    X,Y = [grid.reshape(-1) for grid in grids]
    xy_coords = np.array([np.array((x0,y0,)) for x0,y0 in zip(X,Y)])
    zs = np.ones(xy_coords.shape[0])*height
    coords = np.stack([xy_coords.T[0],xy_coords.T[1],zs]).T
    values = calculate_values(coords, lamps, room, fov80=fov80, horiz=horiz, vert=vert)
    values = values.reshape(grids[0].shape)
    # convert from uW/cm2 to mJ/cm2
    if dose:
        values /= 1000 # in mJ
        values *= 3600 * 8 # over 8 hours   
    return values