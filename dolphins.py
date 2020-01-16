from argparse import ArgumentParser
from copy import copy
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches

from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from skimage import io
from skimage.color import rgb2ycbcr
from skimage.draw import ellipse
from skimage.filters import threshold_yen, sato, rank, meijering
from skimage.measure import regionprops, label
from skimage.morphology import square, opening, watershed, disk
from skimage.util import img_as_ubyte

from utils import debug_fig, supressAxs


def gradient_watershed(image, threshold, debug=False, altMarker=False):
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
    if not altMarker:
        markers = meijering(image, black_ridges=True)
    else:
        markers = sato(image, black_ridges=True)  # meijring maybe better?
    # remove noise
    markers = img_as_ubyte(markers)
    tmp = markers.copy()
    markers = (markers < 3) * threshold
    tmpt = markers
    markers = label(markers)
    # remove small lines

    # create array for which the watershed algorithm will fill
    edges = rank.gradient(image, disk(1))

    segm = watershed(edges, markers, mask=threshold)
    labels = label(segm)

    if debug >= 2:
        debug_fig(image, edges, labels, markers,
                  ["Image", "Edges", "Labels", "markers"])
    return labels


parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to single image to be analysed.")

parser.add_argument("-d", "--debug", action="count", default=0,
                    help="Display debug info.")

args = parser.parse_args()

if args.file is None:
    raise IOError("No file provided!!")

try:
    img = io.imread(args.file)
except FileNotFoundError:
    sys.exit()

img = img[130:1030, 0:1990]
# convert to ycbcr space and take yc values
# as this appears to work better than converting to grayscale directly...
data = rgb2ycbcr(img)[:, :, 0]

ny, nx = data.shape
y, x = np.mgrid[:ny, :nx]

# use some astronomy functions to estimate background
sigma_clip = SigmaClip(sigma=1.8)
bkg_estimator = MedianBackground()
bkg = Background2D(data, box_size=(10, 10),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

# subtract background
bkgsub = data - 1.5*bkg.background
bkgsub = bkgsub / np.amax(np.abs(bkgsub))
bkgsub = img_as_ubyte(bkgsub)

# threshold image
thresh = threshold_yen(bkgsub)
binary = bkgsub >= 60
# remove noise
thresh = opening(binary, square(1))

labels = label(thresh)

if args.debug > 0:
    figd, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    plt.subplots_adjust(top=0.926, bottom=0.031, left=0.023, right=0.977, hspace=0.179, wspace=0.05)
    axs = [supressAxs(i) for i in axs]
    axs[0].imshow(data, cmap=plt.cm.gray)
    axs[1].imshow(bkg.background, cmap=plt.cm.gray)
    axs[2].imshow(bkgsub, cmap=plt.cm.gray)
    axs[2].set_title("bkg sub")
    axs[3].imshow(thresh, cmap=plt.cm.gray)

# preform watershedding
labs = gradient_watershed(bkgsub, thresh, debug=args.debug)

fig, ax = plt.subplots(1, 1)
ax.imshow(img)
for region in regionprops(labels):
    a = region.major_axis_length
    b = region.minor_axis_length
    # remove false positives
    if a > 0 and b > 0 and region.eccentricity > 0.6:
        theta = region.orientation
        centre = region.centroid
        ellipse = mpatches.Ellipse(centre[::-1], 2.*a, 2.*b,
                                   angle=-(90+np.rad2deg(theta)),
                                   fill=False, color="red")
        if args.debug > 0:
            ellipsecopy = copy(ellipse)
            axs[0].add_patch(ellipsecopy)
        ax.add_patch(ellipse)

plt.show()
