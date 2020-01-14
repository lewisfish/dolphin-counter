from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

from skimage import io
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, rank, sato, threshold_local
from skimage.measure import regionprops, label
from skimage.morphology import watershed, square, opening, disk
from skimage.morphology import remove_small_objects, skeletonize
from skimage.segmentation import relabel_sequential
from skimage.util import invert, img_as_ubyte


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


def get_large_small_regons(segmap, threshold, iterative=False):
    '''Function returns seeds, and large regions that contains many seeds

    Parameters
    ----------

    segmap : np.ndarray, int, 2D
        Binary image of original image.

    threshold : float
        Size of area to cut off individual seeds at

    iterative : bool, optional
        no use as of yet...

    Returns
    -------

    largeRegions : List[sliceObjects]
        List of bounding boxes which demarcate regions where many seeds are.

    smallRegions : List[sliceObjects]
        List of bounding boxes which demarcate regions where a seed is.
    '''

    largeRegions = []
    smallRegions = []

    for r in regionprops(segmap):
        if r.area > threshold:
            # keep area for reprocessing
            largeRegions.append(r)
        else:
            smallRegions.append(r)
            # area is a seed
    return largeRegions, smallRegions


def join_segs(seg1, seg2, min_size=50):
    '''Joins two segmentations with seg1 taking precedence

    Parameters
    ----------

    seg1, seg2 : np.ndarray, 2D
        2D labeled segmentation maps. Seg1 is used as a mask to join the
        two together.

    min_size : int, optional
        Size of small object to remove.

    Returns
    -------

    joined : np.ndarray, 2D
        2D array of labeled segmentation map which is the result of joining
        seg1 and seg2.
    '''

    joined = np.where(seg1 > 0, seg1, seg2)
    joined = label(joined)
    joined = remove_small_objects(joined, min_size=min_size)

    return joined


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

    text = f""
    for i, reg in enumerate(regionprops(im3)):
        text += f"area {i+1}: {reg.area:.0f}\n"

    textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
    ax[2].add_artist(textbox)

    ax[3].imshow(im4, aspect="auto")
    ax[3].set_title(labels[3])


def outerdist(image: np.ndarray):
    ''' Function calculates the outer distance transform from
        Segmentation of clustered nuclei with shape markers and marking
        function, Cheng et al.

    Parameters
    ----------

    image : np.ndarray or 2D array
        Binary image of object for which to compute the outer
        distance transform

    Returns
    -------
    dists : np.ndarray, 2D
        Image in which each pixel value is the distance to the nearest marker.

    markers : np.ndarray, 2D
        2D array where markers are individually labeled.
    '''

    edt = ndi.distance_transform_edt(image)
    coordinates = peak_local_max(edt, labels=image, min_distance=7, num_peaks=10)

    dists = np.zeros_like(image, dtype=np.float64)
    # loop over all pixels and if the image pixel is 1 then calculate distance
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] > 0:
                ldist = []
                # loop over all coordinates that are markers and calculate
                # euclidean distance
                for m in coordinates:
                    dist = np.sqrt((i - m[0])**2 + (j - m[1])**2)
                    ldist.append(dist)
                # take minimum of all calculated distance and assign the
                # current pixel this value
                mina = min(ldist)
                dists[i, j] = mina

    # create markers from coordinates
    markers = np.zeros_like(image, dtype=np.bool)
    nd_indices = tuple(coordinates.T)
    markers[nd_indices] = True

    return dists, markers


def distance_watershed(threshold: np.ndarray, debug=False, outer=False):
    '''Function uses a distance transform to create markers and preform
       watershed segmentation

    Parameters
    ----------

    threshold: np.ndarray, 2D
        Binary thresholded image. The segmentation map will be
        computed from this

    debug : bool, optional
        If True then various graphs of the functions intermediate state is
        plotted. Default is False.

    outer : bool, optional
        If True use the outer distance transform. If False use the
        inner distance transform. DEfaul is False.

    Returns
    -------

    labels : np.ndarray, 2D
        Image which has been segmented and labeled.
    '''

    if outer:
        dist, markers = outerdist(threshold)
        segm = watershed(dist, mask=threshold)
    else:
        dist = ndi.distance_transform_edt(threshold)
        peaks = peak_local_max(dist, indices=False, labels=threshold, min_distance=10)
        markers = label(peaks)
        segm = watershed(-dist, markers, mask=threshold)

    labels = label(segm)

    if debug:
        debug_fig(curImg, -dist, labels, markers,
                  ["Image", "Inner distance", "labels", "markers"])

    return labels


def gradient_watershed(image, threshold, debug=False):
    '''Function calculates a segmentation map based upon gradient-esques maps
       and watershedding.

    Parameters
    ----------

    image : np.ndarray or 2D array
        Image of object for which to segment. Grayscale

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
    markers = ndi.morphology.binary_fill_holes(markers, square(3))
    # transform markers into 1 pixel wide lines and label
    markers = skeletonize(markers)
    markers = label(markers)
    # remove small lines
    markers = remove_small_objects(markers, min_size=50)

    # create array for which the watershed algorithm will fill
    edges = rank.gradient(image, disk(2))

    segm = watershed(edges, markers, mask=threshold)
    labels = label(segm)

    if debug:
        debug_fig(tmpt, tmp, labels, markers,
                  ["Image", "Edges", "Labels", "markers"])

    return labels


MAXSIZE = 5000
MINSIZE = 1200

parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to single image to be analysed.")
parser.add_argument("-d", "--debug", action="store_true",
                    help="Display debug info.")
parser.add_argument("-dd", action="store_true")
parser.add_argument("-s", "--save", action="store_true",
                    help="Save images of individual objects.")
parser.add_argument("-g", "--gradient", action="store_true",
                    help="Use gradient_watershed in place of\
                    distance_watershed.")
parser.add_argument("-fo", "--folder", type=str, help="Location to save images.")
parser.add_argument("-t", "--size_threshold", type=float, help="blah", default=1.0)

args = parser.parse_args()

if args.file is None:
    raise IOError("No file provided!!")

try:
    img = io.imread(args.file)
except FileNotFoundError:
    sys.exit()

if args.save:
    if args.folder:
        name = Path(args.folder)
        if not name.exists():
            name.mkdir()
    else:
        name = Path()

gray = img_as_ubyte(rgb2gray(img))

# threshold image using Otsu's method
thresh = threshold_otsu(gray)
thresh = gray < thresh
# remove noise
thresh = opening(thresh, square(3))

labels = label(thresh)
labels = remove_small_objects(labels, min_size=200)


large, small = get_large_small_regons(labels, args.size_threshold*MAXSIZE)

fig, axs = plt.subplots(1, 1)
fig.set_figheight(11.25)
fig.set_figwidth(20)
plt.subplots_adjust(top=0.963, bottom=0.04, left=0.008, right=0.992, hspace=0.2, wspace=0.2)

axs.imshow(img, cmap=plt.cm.gray)
seedcount = 0

segmap = np.zeros_like(gray)

# isolate small seeds, and plot bounding box
for s in small:
    minr, minc, maxr, maxc = s.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, color="red", linewidth=2)
    axs.add_patch(rect)
    axs.scatter(*s.centroid[::-1], color="green", marker="x")
    seedcount += 1
    # add seed to segmentation map
    segmap[s.slice] = thresh[s.slice]

    seed = img[s.slice]

    if args.save:
        io.imsave(name / f"{seedcount:03}.png", seed)
origSegmap = segmap
segmap = label(segmap)

# isolate collections of seeds, and try to separate into individual seeds
for region in large:

    y1 = region.bbox[0]
    y2 = region.bbox[2]
    x1 = region.bbox[1]
    x2 = region.bbox[3]

    # make sure we are not outside image
    y1 = max(y1, 0)
    y2 = min(y2, thresh.shape[0])
    x1 = max(x1, 0)
    x2 = min(x2, thresh.shape[1])

    # get small image & threshold image
    curImg = invert(gray[y1:y2, x1:x2])
    curThresh = thresh[y1:y2, x1:x2]

    if args.gradient:
        labs = gradient_watershed(curImg, curThresh, debug=args.dd)
    else:
        labs = distance_watershed(curThresh, debug=args.dd, outer=True)

    # Loop over labeled regions and plot bounding box and save image
    # of individual seed
    for r in regionprops(labs):
        if r.area <= region.area and r.area > 1.1*MINSIZE:
            minr, minc, maxr, maxc = r.bbox

            y1 = region.bbox[0] + minr
            x1 = region.bbox[1] + minc
            rect = mpatches.Rectangle((x1, y1), maxc - minc, maxr - minr,
                                      fill=False, color="green", linewidth=2)

            x, y = r.centroid[::-1]
            x += region.bbox[1]
            y += region.bbox[0]

            if origSegmap[int(y), int(x)] == 0:

                axs.add_patch(rect)
                axs.scatter(x, y, color="green", marker="x")

                seedcount += 1

                y2 = y1 + (maxr - minr)
                x2 = x1 + (maxc - minc)

                segmap[y1:y2, x1:x2] = labs[r.slice]

                seed = img[y1:y2, x1:x2]
                if args.save:
                    io.imsave(name / f"{seedcount:03}.png", seed)

text = f"Total seeds:{seedcount}"
if args.debug or args.dd:
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Final segmentation")
    segmap = label(segmap)
    ax.imshow(segmap, cmap=make_random_cmap())

textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
axs.add_artist(textbox)
axs.set_title(f"Result for: {args.file}")
plt.show()
