import cv2
import numpy as np
from skimage import io
from skimage.morphology import remove_small_objects, label, area_closing
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

__all__ = ["getMagnification"]


def _initModel():
    '''Initialise the ML model
       https://towardsdatascience.com/an-actual-application-for-the-mnist-digits-classifier-bbd76548bf2f


        Parameters
        ----------

        None

        Returns
        -------

        model : tensorflow.python.keras.engine.sequential.Sequential
            The ML trained model
    '''

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(254, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    # convert from 2D input to 1D vectors
    model.add(tf.keras.layers.Flatten())
    # finish our model with densely connected layers
    model.add(tf.keras.layers.Dense(140, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # output layer with 10 units (one per each class 0-9)
    model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
    model.load_weights('Ml-test/image_to_number_model.hdf5')

    return model


def _processLabels(image, stats, label):
    '''Function returns image of a single digit cropped from labeled image and
       resized for ML model

        Parameters
        ----------

        image : np.ndarray, 2D
            Labeled image to be cropped and resized.

        stats : List
            List of various stats (eg. centroid, bbox etc.) for each labeled
            component in image.

        label : int
            Integer label for connected components.

        Returns
        -------

        digit : np.ndarray, 2D
            Array holding the digit to be classified.
    '''

    # discard parts of image that are not of current label
    digit = np.where(image != label, 0, 255)
    # get bbox coords
    x1 = max(0, stats[label, cv2.CC_STAT_LEFT] - 5)
    x2 = x1 + stats[label, cv2.CC_STAT_WIDTH] + 10
    y1 = stats[label, cv2.CC_STAT_TOP] - 5
    y2 = y1 + stats[label, cv2.CC_STAT_HEIGHT] + 10

    # crop and resize for ML classification.
    digit = digit[y1:y2, x1:x2]
    digit = np.array(digit, dtype="uint8")
    digit = cv2.resize(digit,
                       dsize=(28, 28),
                       interpolation=cv2.INTER_CUBIC)

    return digit


def getMagnification(filename, model, debug=False):
    '''Function uses ML OCR to determine the magnification of the frame from
       the drone video.

        Parameters
        ----------

        filename : str
            filename of the frame to determine the magnification from.

        debug : bool, optional
            If True then returns list of images and their classifications
            for debug purposes.


        Returns
        -------

        magnification : float
            The determined magnification level of the drone video.

    '''

    # Open image and convert to grayscale
    img = io.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    array = img
    array = array[80:140, 25:127]

    # Threshold, dilate and then crop to ROI.
    ret2, thresh = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    if np.mean(thresh) > 100.:
        ret, thresh = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contour, hier = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(thresh, [cnt], 0, 255, 1)
        array = area_closing(thresh, area_threshold=156, connectivity=1)
    else:
        array = cv2.dilate(thresh, kernel, iterations=1)

    # Label and remove noise and decimal point
    array = label(array)
    array = remove_small_objects(array, 100)

    # convert image back to binary and uint8 type
    array = np.where(array > 0, 255, 0)
    array = np.array(array, "uint8")

    # label again, this time with stats calculated
    output = cv2.connectedComponentsWithStats(array, 4, cv2.CV_32S)
    nlabels = output[0]  # number of labels
    array = output[1]    # Labeled image
    stats = output[2]    # List of stats for each label

    labels = []
    digits = []

    # Sort labels so that they are process in left to right order
    s = stats[1:, cv2.CC_STAT_LEFT]
    labelOrder = sorted(range(len(s)), key=lambda k: s[k])

    # classify digits
    for i in labelOrder:
        digits.append(_processLabels(array, stats, i+1))
        lab = model.predict_classes(digits[-1].reshape(1, 28, 28, 1).astype('float32')/255)
        labels.append(lab)

    # format and return magnification level
    first = labels[0]
    second = labels[1]
    if labels[2] != 3:
        if labels[2] != 8 and labels[1] != 0:
            third = labels[2]
        else:
            third = None
    else:
        third = None

    if third:
        magnification = float(f"{first[0]}{second[0]}.{third[0]}")
    else:
        magnification = float(f"{first[0]}.{second[0]}")

    if debug:
        return magnification, digits, labels
    else:
        return magnification


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    files = ["2019_11_23_16_18928.png", "2019_11_23_16_55322.png",
             "2019_11_23_16_55379.png", "2019_11_23_16_19134.png",
             "2019_11_23_16_55348.png", "2019_11_23_16_55380.png",
             "2019_11_23_16_19185.png", "2019_11_23_16_55373.png",
             "2019_11_23_16_55381.png", "2019_11_23_16_19468.png",
             "2019_11_23_16_55374.png", "2019_11_23_16_55382.png",
             "2019_11_23_16_22062.png", "2019_11_23_16_55375.png",
             "2019_11_23_16_55399.png", "2019_11_23_16_55245.png",
             "2019_11_23_16_55376.png", "2019_11_23_16_55425.png",
             "2019_11_23_16_55271.png", "2019_11_23_16_55377.png",
             "2019_11_23_16_55450.png", "2019_11_23_16_55296.png",
             "2019_11_23_16_55378.png", "2019_11_23_16_55476.png"]

    fig = plt.figure()
    outer_grid = fig.add_gridspec(5, 5, wspace=.5, hspace=0.)
    # Init ML model
    model = _initModel()
    for i, file in enumerate(files):

        magnification, digits, labels = getMagnification(file, debug=True)
        inner_grid = outer_grid[i].subgridspec(1, 3, wspace=0.0, hspace=0.0)

        ax0 = fig.add_subplot(inner_grid[0])
        ax1 = fig.add_subplot(inner_grid[1])
        ax2 = fig.add_subplot(inner_grid[2])

        ax0.imshow(digits[0])
        ax0.set_title(f"{labels[0]}")
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1.imshow(digits[1])
        ax1.set_title(f"{labels[1]}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        if labels[2] != 3:
            ax2.imshow(digits[2])
            ax2.set_title(f"{labels[2]}")
        else:
            ax2.imshow(np.zeros((28, 28)))
            ax2.set_title("[-]")
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig.add_subplot(ax0)
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)

    plt.show()
