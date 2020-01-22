from argparse import ArgumentParser
from copy import copy
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from scipy import ndimage as ndi

from skimage import io
from skimage.color import rgb2ycbcr
from skimage.filters import threshold_yen, sato, rank, meijering, threshold_local
from skimage.measure import regionprops, label
from skimage.morphology import opening, watershed, disk, remove_small_objects, dilation
from skimage.util import img_as_ubyte, invert, img_as_float

from utils import debug_fig, make_random_cmap


def createMask(image, factor=1.3):
    '''Function creates a mask based upon the red channel
       Idea from "Detection of Dugongs from Unmanned Aerial Vehicles" F.Marie et al.


    Parameters
    ----------

    image : np.ndarray, 2D
        Image to be masked.

    factor : float, optional
        Factor which is used as a threshold to determine the creation of the mask.
        Default value is 1.3

    Returns
    -------

    mask : np.ndarray, 2D
        Image mask. Returns the 3 channel image which has been masked.

    '''

    rmean = np.mean(image[:, :, 0])
    gmean = np.mean(image[:, :, 1])
    bmean = np.mean(image[:, :, 2])

    # select pixels that are greater than f*mean, where f is some factor.
    rtrue = np.where(image[:, :, 0] > factor*rmean, 1, 0)
    gtrue = np.where(image[:, :, 1] > factor*gmean, 1, 0)
    btrue = np.where(image[:, :, 2] > factor*bmean, 1, 0)

    # create mask
    mask = np.logical_and(rtrue, gtrue)
    mask = np.logical_and(mask, btrue)
    mask = mask.astype(bool)

    # apply mask
    tmp = np.where(mask == 0)
    mask = image.copy()
    mask[tmp] = 0

    return mask


def estimate_background(image, sigma=100., boxsize=(90, 199), simple=True):
    '''Function estimates the background of provided image.

    Parameters
    ----------

    image : np.ndarray, 2D
        Image from which the background will be estimated.

    sigma : float, optional
        Sigma for either Gaussian filter.

    boxsize : Tuple(int), optional
        Size of box used in mean filter

    simple : bool, optional
        If False then uses a simple Gaussian blur to estimate background.
        If True uses mean filter

    Returns
    -------

    bkg : np.ndarray, 2D
        Estimated background.

    '''

    # Use Gaussian blur to create background
    if simple:
        bkg = ndi.uniform_filter(data, boxsize)
    else:
        bkg = ndi.gaussian_filter(data, sigma=sigma)

    return bkg


def get_threshold(image, local=False, block_size=11, offset=0):
    '''Function calculates the best thresholding value to binarise the image

    Parameters
    ----------

    image : np.ndarray, 2D
        Image to be binarised

    local : bool, optional
        If True then uses local thresholding. If False use a global value.

    block_size : int, optional
        Size of the blocks to use in local thresholding

    offset : int, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value.

    Returns
    -------

    thresholded : np.ndarray, 2D
        The binarised image.
    '''

    if local:
        local_thresh = threshold_local(image, block_size, offset=offset)
        thresholded = image > local_thresh
    else:
        global_thresh = threshold_yen(image)
        thresholded = image > global_thresh

    return thresholded


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

    markers = img_as_ubyte(markers)
    tmp = markers.copy()
    # threshold and mask
    markers = (markers < 3) * threshold
    tmpt = markers
    markers = label(markers)
    markers[markers == 1] = 0

    # create array for which the watershed algorithm will fill
    # based upon the gradient of the image
    edges = rank.gradient(image, disk(1))

    segm = watershed(edges, markers)
    labels = label(segm, connectivity=2)
    labels = remove_small_objects(labels, min_size=20)

    if debug >= 2:
        debug_fig(image, edges, labels, markers,
                  ["Image", "Edges", "Labels", "markers"],
                  [plt.cm.gray, None, make_random_cmap(), plt.cm.nipy_spectral],
                  pos=2)
    return labels


parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to single image to be analysed.")

parser.add_argument("-d", "--debug", action="count", default=0,
                    help="Display debug info.")

parser.add_argument("-np", "--noplot", action="store_true",
                    help="Suppress default plot output.")

args = parser.parse_args()

if args.file is None:
    raise IOError("No file provided!!")

start = time.time()

try:
    img = io.imread(args.file)
except FileNotFoundError:
    sys.exit()

img = img[130:1030, 0:1990]
# convert to ycbcr space and take yc values
# as this appears to work better than converting to grayscale directly...
data = rgb2ycbcr(img)[:, :, 0]

# create mask based upon red values
# Theory is that dolphins should have more red than the sea...
imgmask = createMask(img)

# estimate background then threshold it to get a mask
bkg = estimate_background(data, sigma=100.)
bkgMask = invert(get_threshold(bkg)).astype(int)
# combine masks
imgmask = imgmask[:, :, 0] * bkgMask
# convert to binary
imgmask = np.where(imgmask > 0, 1, 0)
# remove noise and then enlarge areas
# imgmask = opening(imgmask, disk(2))
# imgmask = dilation(imgmask, disk(5))

# subtract background, apply mask and renormalise
bkgsub = data - 1.5*bkg
bkgsub *= bkgMask
bkgsub = bkgsub / np.amax(np.abs(bkgsub))
bkgsub = img_as_ubyte(bkgsub)


# get location of probable dolphin
# thresh = get_threshold(imgmask, local=True, block_size=51)
# remove noise
# thresh = invert(opening(thresh, disk(6)))


if args.debug > 0:
    labels = ["Image", "Background est.", "Image - background", "Threshold"]
    cmaps = [plt.cm.gray for i in range(0, 4)]
    figd, axs = debug_fig(data, bkg, bkgsub, imgmask, labels, cmaps, pos=1)

# preform watershedding
labs = gradient_watershed(bkgsub, imgmask, debug=args.debug)

fig, ax = plt.subplots(1, 1)
fig.canvas.manager.window.move(0, 0)
ax.imshow(img)

dcount = 0
for region in regionprops(labs):
    a = region.major_axis_length
    b = region.minor_axis_length
    area = np.pi * a * b
    # remove false positives
    if a > 0 and b > 0 and region.eccentricity > 0.7 and area < 400:

        dcount += 1
        theta = region.orientation
        centre = region.centroid[::-1]
        ellipse = mpatches.Ellipse(centre, 2.*b, 2.*a,
                                   angle=-np.rad2deg(theta),
                                   fill=False, color="red")

        if args.debug > 0:
            # need to use copy() as cant add same artist to different figs for whatever reason...
            ellipsecopy = copy(ellipse)
            axs[0].add_patch(ellipsecopy)
        ax.add_patch(ellipse)

finish = time.time()
text = f"Total dolphins:{dcount}\n"
text += f"Total time:{finish-start:.03f}"
textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
ax.add_artist(textbox)
print(dcount)
if not args.noplot:
    plt.show()
