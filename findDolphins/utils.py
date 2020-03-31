import numpy as np
import matplotlib.pyplot as plt

__all__ = ["readFileListIn", "make_random_cmap", "supressAxs", "debug_fig"]


def readFileListIn(filename: str):
    '''Function that reads in given file and returns videofilename, and
       framenumber.

    Parameters
    ----------

    filename : str or Path object
        Filename of the file to read in

    Returns
    -------

    videoFrameList : List[]
    '''

    videoFrameList = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            lineSplit = line.split(",")
            videofile = lineSplit[0]
            framenumber = int(lineSplit[1].rstrip())
            videoFrameList.append([videofile, framenumber])

    return videoFrameList


def make_random_cmap(ncolors=256, random_state=None):
    """
    Make a matplotlib colormap consisting of (random) muted colors.

    A random colormap is very useful for plotting segmentation images.
    Taken from photutils: https://photutils.readthedocs.io/en/stable/index.html#

    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.

    random_state : int or `~numpy.random.mtrand.RandomState`, optional
        The pseudo-random number generator state used for random
        sampling.  Separate function calls with the same
        ``random_state`` will generate the same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
        The matplotlib colormap with random colors.
    """

    from matplotlib import colors

    prng = np.random.RandomState(random_state)
    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    cmap = colors.ListedColormap(rgb)
    cmap.colors[0] = colors.hex2color("#000000")

    return cmap


def supressAxs(ax):
    '''Function that removes all labels and ticks from a figure

    Parameters
    ----------

    ax: matplotlib axis object

    Returns
    -------

    ax : matplotlib axis object
        Now with no ticks or labels

    '''

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return ax


def debug_fig(im1, im2, im3, im4, labels, cmaps, pos=None):
    '''Function that plots 4 images in one figure


    Parameters
    ----------

    im1, im2, im3, im4 : np.ndarray, 2D
        images which are to be plotted

    labels : List[str]
        List of subplot titles

    Returns
    -------

    '''

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    if pos is None:
        fig.canvas.manager.window.move(0, 0)
    else:
        if pos == 0:
            fig.canvas.manager.window.move(0, 0)
        elif pos == 1:
            fig.canvas.manager.window.move(int(1920 / 3), 0)
        elif pos == 2:
            fig.canvas.manager.window.move(2*int(1920 / 3), 0)
        elif pos == 3:
            fig.canvas.manager.window.move(0, int(1080/2))
        elif pos == 4:
            fig.canvas.manager.window.move(int(1920 / 3), int(1080/2))
        elif pos == 5:
            fig.canvas.manager.window.move(2*int(1920 / 3), int(1080/2))

    ax = ax.ravel()

    plt.subplots_adjust(top=0.926, bottom=0.031, left=0.023, right=0.977, hspace=0.179, wspace=0.05)
    ax = [supressAxs(i) for i in ax]

    ax[0].imshow(im1, cmap=cmaps[0], aspect="auto")
    ax[0].set_title(labels[0])

    ax[1].imshow(im2, aspect="auto", cmap=cmaps[1])
    ax[1].set_title(labels[1])

    ax[2].imshow(im3, aspect="auto", cmap=cmaps[2])
    ax[2].set_title(labels[2])

    ax[3].imshow(im4, aspect="auto", cmap=cmaps[3])
    ax[3].set_title(labels[3])

    return fig, ax
