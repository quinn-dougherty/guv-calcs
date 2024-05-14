import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_tlvs(skin_values, eye_values, x, y, height, units, figsize=(8, 3.5), title=""):
    """
    Plot the eye and skin doses in the plane
    """
    # set figure sizes
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1, 1, 0.07], figure=fig
    )  # 2 plots and 1 for colorbar

    vmin = eye_values.min()
    vmax = skin_values.max()
    extent = [0, x, 0, y]

    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(skin_values, extent=extent, vmin=vmin, vmax=vmax)
    title1 = "Skin dose at " + str(height) + " " + units
    title1 += "\nMax: " + str(round(skin_values.max(), 2)) + " mJ/cm²"
    ax1.set_title(title1)

    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(eye_values, extent=extent, vmin=vmin, vmax=vmax)
    title2 = "Eye dose at " + str(height) + " " + units
    title2 += "\nMax: " + str(round(eye_values.max(), 2)) + " mJ/cm²"
    ax2.set_title(title2)

    # Colorbar in the third column of GridSpec
    cbar_ax = plt.subplot(gs[2])
    cbar = fig.colorbar(
        im2,
        label="mJ/cm²/8 hours",
        cax=cbar_ax,
        use_gridspec=False,
        shrink=0.9,
        extendrect=False,
    )
    fig.suptitle(title, y=1.0)
    plt.show()
