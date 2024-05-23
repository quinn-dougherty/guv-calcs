import numpy as np
from pathlib import Path

# def add_standard_zones(room):
    # height = 1.9 if room.units=='meters' else 6.5

    # eyezone = CalcPlane(
        # zone_id='EyeLimits',
        # name='Eye Limits',
        # height=height,
        # dimensions=[room.x,room.y],
        # vert=True,
        # horiz=False,
        # fov80=True,
    # )
    # skinzone = CalcPlane(
        # zone_id='EyeLimits',
        # name='Eye Limits',
        # height=height,
        # dimensions=[room.x,room.y],
        # vert=True,
        # horiz=False,
        # fov80=False,
    # )
        # CalcPlane("Skin", height=height, dimensions=[room.x,room.y], vert=False, horiz=True, fov80=False))
        # CalcPlane("Eye", height=height, dimensions=[room.x,room.y], vert=True, horiz=False, fov80=True))


def get_lamp_position(lamp_idx, x, y, num_divisions=100):
    xp = np.linspace(0,x,num_divisions+1)
    yp = np.linspace(0,y,num_divisions+1)
    xidx,yidx = _get_idx(lamp_idx,num_divisions=num_divisions)
    return xp[xidx],yp[yidx]
    
def _get_idx(num_points, num_divisions=100):
    grid_size = (num_divisions,num_divisions)
    return _place_points(grid_size, num_points)[-1]

def _place_points(grid_size, num_points):
    M, N = grid_size
    grid = np.zeros(grid_size)
    points = []

    # Place the first point in the center
    center = (M // 2, N // 2)
    points.append(center)
    grid[center] = 1  # Marking the grid cell as occupied

    for _ in range(1, num_points):
        max_dist = -1
        best_point = None
        
        for x in range(M):
            for y in range(N):
                if grid[x, y] == 0:
                    # Calculate the minimum distance to all existing points
                    min_point_dist = min([np.sqrt((x - px)**2 + (y - py)**2) for px, py in points])
                    # Calculate the distance to the nearest boundary
                    min_boundary_dist = min(x, M - 1 - x, y, N - 1 - y)
                    # Find the point where the minimum of these distances is maximized
                    min_dist = min(min_point_dist, min_boundary_dist)
                    
                    if min_dist > max_dist:
                        max_dist = min_dist
                        best_point = (x, y)

        if best_point:
            points.append(best_point)
            grid[best_point] = 1  # Marking the grid cell as occupied
    return points
    
## deprecated / never used
# def get_lamp_positions(num, dimensions):
    # x, y, z = dimensions
    # nrows = int(np.ceil(np.sqrt(num)))  # Initial guess for the number of rows
    # ncols = int(np.ceil(num / nrows))  # Calculate the number of columns needed
    # nrowdivs = nrows + 1
    # ncoldivs = ncols + 1
    # x_coords = [i * x / nrowdivs for i in range(1, nrowdivs)]
    # y_coords = [i * y / ncoldivs for i in range(1, ncoldivs)]
    # xg, yg = np.meshgrid(x_coords, y_coords)
    # coords = np.array((xg.flatten()[0:num], yg.flatten()[0:num], np.ones(num) * z))
    # return coords.T


def get_ies_files():
    # set lamps
    root = Path("./ies_files")
    p = root.glob("**/*")
    ies_files = [x for x in p if x.is_file() and x.suffix == ".ies"]
    return ies_files
