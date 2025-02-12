import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from photompy import plot_ies
from .trigonometry import to_polar


class LampPlotter:
    """
    Class for plotting
    """

    def __init__(self, lamp):
        self.lamp = lamp

    def plot_ies(self, title=""):
        """standard polar plot of an ies file"""
        fig, ax = plot_ies(fdata=self.lamp.lampdict["full_vals"], title=title)
        return fig, ax

    def plot_web(
        self,
        elev=30,
        azim=-60,
        title="",
        figsize=(6, 4),
        color="#cc61ff",
        alpha=0.4,
        xlim=None,
        ylim=None,
        zlim=None,
    ):
        """plot photometric web, where distance r is set by the irradiance value"""
        scale = self.lamp.values.max()
        x, y, z = self.lamp.transform(self.lamp.photometric_coords, scale=scale).T
        Theta, Phi, R = to_polar(*self.lamp.photometric_coords.T)
        tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, color=color, alpha=alpha)
        if self.lamp.aim_point is not None:
            ax.plot(
                *np.array((self.lamp.aim_point, self.lamp.position)).T,
                linestyle="--",
                color="black",
                alpha=0.7,
            )
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        ax.view_init(azim=azim, elev=elev)
        return fig, ax

    def plot_3d(
        self,
        elev=45,
        azim=-45,
        title="",
        figsize=(6, 4),
        alpha=0.7,
        cmap="rainbow",
        fig=None,
        ax=None,
    ):
        """
        plot in cartesian 3d space of the true positions of the irradiance values
        mostly a convenience visualization function. Generally irradiance values
        should use a polar plot.
        """
        x, y, z = self.lamp.transform(self.lamp.coords).T
        intensity = self.lamp.values.flatten()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=intensity, cmap="rainbow", alpha=alpha)
        if self.lamp.aim_point is not None:
            ax.plot(
                *np.array((self.lamp.aim_point, self.lamp.position)).T,
                linestyle="--",
                color="black",
                alpha=0.7,
            )
        ax.set_title(title)
        ax.view_init(azim=azim, elev=elev)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return fig, ax
