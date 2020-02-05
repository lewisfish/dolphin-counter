from copy import copy
import sys
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from scipy import ndimage as ndi
from scipy import stats

from skimage.color import rgb2ycbcr
from skimage.filters import threshold_yen, rank, meijering, threshold_local
from skimage.measure import regionprops, label
from skimage.morphology import watershed, disk, remove_small_objects
from skimage.util import img_as_ubyte, invert

from ocr import getMagnification
from utils import debug_fig, make_random_cmap, supressAxs


class Engine(object):
    '''Class existences so that Pool method can be used on main.
       Basically a way to pass the function arguments that are the same with
       one variable argument, i.e the file name'''

    def __init__(self, parameters):
        '''This sets the arguments for the function passed to pool via
           engine'''
        self.parameters = parameters

    def __call__(self, filename):
        '''This calls the function when engine is called on pool'''
        return main(filename, *self.parameters)


def redRatio(image: np.ndarray) -> np.ndarray:
    '''

    Parameters
    ----------

    Returns
    -------

    '''

    reds = image[:, :, 0]
    blues = image[:, :, 1]
    greens = image[:, :, 2]

    ratio = reds / (blues + greens)

    return ratio


def createMask(image: np.ndarray, factor=1.3) -> np.ndarray:
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


def estimate_background(image: np.ndarray, sigma=100., boxsize=(90, 199),
                        simple=False) -> np.ndarray:
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
        bkg = ndi.uniform_filter(image, boxsize)
    else:
        bkg = ndi.gaussian_filter(image, sigma=sigma)

    return bkg


def get_threshold(image: np.ndarray, local=False, block_size=11, offset=0) -> np.ndarray:
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


def gradient_watershed(image: np.ndarray, threshold: np.ndarray, magn: float,
                       debug=False, altMarker=False) -> np.ndarray:
    '''Function calculates a segmentation map based upon gradient-esques maps
       and watershedding.

    Parameters
    ----------

    image : np.ndarray or 2D array
        Image of object for which to segment

    threshold : np.ndarray, 2D
        Binary thresholded image.

    magn: float
        Magnification of the image as determined by getMagnification.

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

    # create array for which the watershed algorithm will fill
    # based upon the gradient of the image
    edges = rank.gradient(image, disk(1))
    if magn > 1.0:
        markers = remove_small_objects(markers, min_size=50)
        edges = -edges
    segm = watershed(edges, markers)
    labels = label(segm, connectivity=2)

    if debug >= 2:
        debug_fig(image, edges, labels, markers,
                  ["Image", "Edges", "Labels", "markers"],
                  [plt.cm.gray, None, make_random_cmap(), make_random_cmap()],
                  pos=2)
    return labels


def method1(image, gray, debug=0):

    # create mask based upon red values
    # Theory is that dolphins should have more red than the sea...
    imgmask = createMask(image)

    # probably should do something better with imgmask other than just take red channel...
    # setup code such that can drop in different methods to compare effectiveness...

    # estimate background then threshold it to get a mask
    background = estimate_background(gray, sigma=100.)
    bkgMask = invert(get_threshold(background)).astype(int)
    maskArea = 1. - ((np.sum(bkgMask)) / (bkgMask.shape[0]*bkgMask.shape[1]))
    if maskArea > .45:
        bkgFactor = 0.0
        imgmask = imgmask[:, :, 0]
    else:
        bkgFactor = 1.1
        # combine masks
        imgmask = imgmask[:, :, 0] * bkgMask

    # convert to binary
    imgmask = np.where(imgmask > 0, 1, 0)

    # subtract background, apply mask and renormalise
    backgroundSubtracted = gray - (bkgFactor*background)
    if maskArea < 0.45:
        backgroundSubtracted *= bkgMask
    backgroundSubtracted = backgroundSubtracted / np.amax(np.abs(backgroundSubtracted))
    backgroundSubtracted = img_as_ubyte(backgroundSubtracted)

    if debug > 2:
        labels = ["gray", "background", "backgroundSubtracted", "imgmask"]
        cmaps = [plt.cm.gray for i in range(0, 4)]
        figd, axs = debug_fig(gray, background, backgroundSubtracted, imgmask, labels, cmaps, pos=1)

        return backgroundSubtracted, imgmask, axs

    return backgroundSubtracted, imgmask, None


def remove_large_objects(segments, max_size):

    out = np.copy(segments)
    component_sizes = np.bincount(segments.ravel())

    too_large = component_sizes > max_size
    too_large_mask = too_large[segments]
    out[too_large_mask] = 0

    return out


def method2(image, gray, dolpLength, debug=0):
    from scipy.stats import sigmaclip

    red_ratio = image[:, :, 0] > .56
    red_ratio = dilation(red_ratio, disk(2))
    red_ratio = remove_small_objects(red_ratio, min_size=(dolpLength**2)/50.)
    red_ratio = remove_large_objects(label(red_ratio), max_size=dolpLength**2)

    if debug > 2:
        labels = ["image - bkg", "red_ratio", "mask", "None"]
        cmaps = [None, plt.cm.gray, plt.cm.gray, plt.cm.gray]
        figd, axs = debug_fig(image[:,:,0], first, second, third, labels, cmaps, pos=1)

        return red_ratio, axs

    return red_ratio, None


def main(filename, debug: int, noplot: bool, saveplot: bool):
    '''

    Parameters
    ----------

    filename :

    debug : int

    noplot : bool

    saveplot : bool

    Returns
    -------

    None

    '''

    if filename is None:
        raise IOError("No file provided!!")

    start = time.time()
    # use magnification given by image to remove false positives
    magn = getMagnification(str(filename))
    dolpLength = 22.38*magn + 22.38

    try:
        img = cv2.imread(str(filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    except FileNotFoundError:
        sys.exit()

    img = img[130:1030, 0:1990]
    # convert to ycbcr space and take yc values
    # as this appears to work better than converting to grayscale directly...
    data = rgb2ycbcr(img)[:, :, 0]

    if magn >= 4.0:
        labs, dax = method2(img, data, dolpLength, debug=debug)
    else:
        # preform watershedding
        bkgsub, imgmask, dax = method1(img, data, debug=debug)
        labs = gradient_watershed(bkgsub, imgmask, magn, debug=debug)

    if not noplot:
        fig, ax = plt.subplots(1, 1)
        fig.canvas.manager.window.move(0, 0)
        ax = supressAxs(ax)
        ax.imshow(img, aspect="auto")

    dcount = 0

    areas = []
    for r in regionprops(labs):
        a = r.major_axis_length
        b = r.minor_axis_length
        ecc = r.eccentricity
        # remove false positives
        if a > .25*dolpLength and b > 0 and ecc > 0.7 and ecc < 0.99 and a < 2.*dolpLength:
            areas.append(r.area)

    areas, *_ = stats.sigmaclip(areas, low=2.5, high=2.5)

    for region in regionprops(labs):
        a = region.major_axis_length
        b = region.minor_axis_length
        ecc = region.eccentricity
        # remove false positives
        if a > .25*dolpLength and b > 0 and ecc > 0.7 and ecc < 0.99 and a < 2.*dolpLength:
            if region.area in areas:
                dcount += 1
                theta = region.orientation
                centre = region.centroid[::-1]
                ellipse = mpatches.Ellipse(centre, 2.*b, 2.*a,
                                           angle=-np.rad2deg(theta),
                                           fill=False, color="red", linewidth=2.)

                if debug > 1:
                    # need to use copy() as cant add same artist to different figs for whatever reason...
                    ellipsecopy = copy(ellipse)
                    # dax[0].add_patch(ellipsecopy)
                if not noplot:
                    ax.add_patch(ellipse)

    finish = time.time()
    if not noplot:
        text = f"Total dolphins:{dcount}\n"
        text += f"Total time:{finish-start:.03f}\n"
        text += f"Magnification:{magn}"
        textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
        ax.add_artist(textbox)

    print(filename, dcount)

    if not noplot:
        if saveplot:
            fig.set_figheight(11.25)
            fig.set_figwidth(20)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.savefig(f"output/{str(filename.name)[:-4]}_output_003.png", dpi=96)
        else:
            plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    from multiprocessing import Pool
    from pathlib import Path
    import sys

    parser = ArgumentParser(description="Counts objects in a picture")

    parser.add_argument("-f", "--file", type=str,
                        help="Path to single image to be analysed.")
    parser.add_argument("-fo", "--folder", type=str,
                        help="Path to folder of images to be analysed.")

    parser.add_argument("-d", "--debug", action="count", default=0,
                        help="Display debug info.")

    parser.add_argument("-np", "--noplot", action="store_true",
                        help="Suppress default plot output.")
    parser.add_argument("-sp", "--saveplot", action="store_true",
                        help="Save output plot.")
    parser.add_argument("-n", "--ncores", type=int, default=1,
                        help="Specify the number of cores to use. Default is 1.")

    args = parser.parse_args()

    if args.file and args.folder:
        print("Can not have both folder and file arguments!!")
        sys.exit()

    if args.file is None and args.folder is None:
        print("Need image input!!")
        sys.exit()

    if args.folder:
        # Get all relevant files in folder
        files = Path(args.folder).glob("*.png")
    else:
        # just single file so place in a generator manually
        files = (Path(args.file) for i in range(1))

    if args.ncores != 1:
        pool = Pool(args.ncores)
        engine = Engine([args.debug, args.noplot, args.saveplot])

        results = pool.map(engine, files)
        pool.close()
        pool.join()
    else:
        for file in files:
            main(file, args.debug, args.noplot, args.saveplot)
