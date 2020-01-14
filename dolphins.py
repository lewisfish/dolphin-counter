import sys
from argparse import ArgumentParser
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText
from scipy import ndimage as ndi

from skimage import io
from skimage.color import rgb2ycbcr
from skimage.filters import threshold_yen, sato, rank

from skimage.measure import regionprops, label
from skimage.morphology import square, opening, remove_small_objects
from skimage.morphology import watershed, skeletonize, disk

from skimage.util import img_as_ubyte
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from skimage.draw import ellipse


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


def debug_fig(im1, im2, im3, im4, labels):
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

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    plt.subplots_adjust(top=0.926, bottom=0.031, left=0.023, right=0.977, hspace=0.179, wspace=0.05)
    ax = [supressAxs(i) for i in ax]

    ax[0].imshow(im1, cmap=plt.cm.gray, aspect="auto")
    ax[0].set_title(labels[0])

    ax[1].imshow(im2, aspect="auto")
    ax[1].set_title(labels[1])

    ax[2].imshow(im3, aspect="auto", cmap=make_random_cmap())
    ax[2].set_title(labels[2])

    ax[3].imshow(im4, aspect="auto")
    ax[3].set_title(labels[3])


def gradient_watershed(image, threshold, debug=False):
    '''Function calculates a segmentation map based upon gradient-esques maps
       and watershedding.

    Parameters
    ----------

    image : np.ndarray or 2D array
        Image of object for which to segment

    threshold : np.ndarray, 2D
        Binary thresholded image.

    debug : bool, optional
        If True then various graphs of the functions intermediate state
        is plotted. Default is False.

    Returns
    -------

    labels : np.ndarray, 2D
        Image which has been segmented and labeled.

    '''

    # create markers based upon Sato filtering
    # see https://scikit-image.org/docs/stable/api/skimage.filters.html?highlight=sato#skimage.filters.sato
    markers = sato(image, black_ridges=True)
    # remove noise
    markers = img_as_ubyte(opening(markers, square(1)))
    tmp = markers.copy()
    markers = (markers < 3) * threshold
    tmpt = markers
    # markers = ndi.morphology.binary_fill_holes(markers, square(3))
    markers = label(markers)
    # remove small lines
    # markers = remove_small_objects(markers, min_size=50)

    # create array for which the watershed algorithm will fill
    edges = rank.gradient(image, disk(2))

    segm = watershed(edges, markers, mask=threshold)
    labels = label(segm)

    if debug:
        debug_fig(image, edges, labels, markers,
                  ["Image", "Edges", "Labels", "markers"])
    return labels


parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to single image to be analysed.")

parser.add_argument("-d", "--debug", action="store_true",
                    help="Display debug info.")
parser.add_argument("-dd", action="store_true",
                    help="Display debug info.")
args = parser.parse_args()

if args.file is None:
    raise IOError("No file provided!!")

try:
    img = io.imread(args.file)
except FileNotFoundError:
    sys.exit()

img = img[130:1030, 0:1990]
hsv = rgb2ycbcr(img)

data = hsv[:, :, 0]

ny, nx = data.shape
print(data.shape)
y, x = np.mgrid[:ny, :nx]

sigma_clip = SigmaClip(sigma=2.5)
bkg_estimator = MedianBackground()
bkg = Background2D(data, box_size=(10, 10), filter_size=(5, 5),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

bkgsub = data - bkg.background
bkgsub = bkgsub / np.amax(bkgsub)
bkgsub = img_as_ubyte(bkgsub)


thresh = threshold_yen(bkgsub, nbins=256)
binary = bkgsub >= 150
thresh = opening(binary, square(1))

labels = label(thresh)
areas = []

if args.debug:
    figd, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    plt.subplots_adjust(top=0.926, bottom=0.031, left=0.023, right=0.977, hspace=0.179, wspace=0.05)
    axs = [supressAxs(i) for i in axs]
    axs[0].imshow(data, cmap=plt.cm.gray)
    axs[1].imshow(bkg.background, cmap=plt.cm.gray)
    axs[2].imshow(bkgsub, cmap=plt.cm.gray)
    axs[3].imshow(thresh, cmap=plt.cm.gray)
labs = gradient_watershed(bkgsub, thresh, debug=args.dd)

fig, ax = plt.subplots(1, 1)
ax.imshow(img)
for region in regionprops(labels):
    a = region.major_axis_length
    b = region.minor_axis_length
    areas.append(region.area)
    if a > 0 and b > 0 and region.area < 50:
        if region.eccentricity > 0.5 and a < 10:
            theta = region.orientation
            centre = region.centroid
            ellipse = mpatches.Ellipse(centre[::-1], a, b,
                                       angle=np.rad2deg(theta),
                                       fill=False, color="red")
            if args.debug:
                ellipsecopy = copy(ellipse)
                axs[0].add_patch(ellipsecopy)
            ax.add_patch(ellipse)

plt.show()
