import cv2
import numpy as np
from typing import List
import warnings
from skimage.morphology import remove_small_objects, label
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
            The ML trained model.
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


def _processLabels(image: np.ndarray, stats: List, label: int) -> np.ndarray:
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
    y1 = max(0, stats[label, cv2.CC_STAT_TOP] - 5)
    y2 = y1 + stats[label, cv2.CC_STAT_HEIGHT] + 10

    # crop and resize for ML classification.
    digit = digit[y1:y2, x1:x2]
    digit = np.array(digit, dtype="uint8")
    digit = cv2.resize(digit,
                       dsize=(28, 28),
                       interpolation=cv2.INTER_CUBIC)

    return digit


def getMagnification(frame: np.ndarray, debug=False) -> float:
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
    from setting import model

    if debug:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3)

    # Open image and convert to grayscale
    array = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # array = img
    array = array[92:130, 24:140]

    # Threshold, dilate and then crop to ROI.
    ret2, thresh = cv2.threshold(array, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thresh = label(thresh)
    thresh = remove_small_objects(thresh, 100)
    # convert image back to binary and uint8 type
    thresh = np.where(thresh > 0, 255, 0)
    thresh = np.array(thresh, "uint8")

    if debug:
        axs[0].imshow(array)

    if np.mean(thresh) > 100.:
        ret, thresh = cv2.threshold(array, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = label(thresh)
        thresh = remove_small_objects(thresh, 100)

        thresh = np.where(thresh > 0, 255, 0)
        array = np.array(thresh, "uint8")

    else:
        fraction = np.sum(thresh/255)/(thresh.shape[0]*thresh.shape[1])
        if fraction > 0.2:
            array = np.where(array < 50, 255, array)
            ret2, thresh = cv2.threshold(array, 210, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            thresh = label(thresh)
            thresh = remove_small_objects(thresh, 100)
            # convert image back to binary and uint8 type
            thresh = np.where(thresh > 0, 255, 0)
            thresh = np.array(thresh, "uint8")

            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            if debug:
                axs[1].imshow(thresh)
            array = thresh
        else:
            kernel = np.ones((2, 2), np.uint8)
            array = thresh
            array = cv2.dilate(thresh, kernel, iterations=1)

    # label again, this time with stats calculated
    output = cv2.connectedComponentsWithStats(array, 8, cv2.CV_32S)
    nlabels = output[0]  # number of labels
    array = output[1]    # Labeled image
    stats = output[2]    # List of stats for each label

    labels = []
    digits = []
    if debug:
        axs[2].imshow(array)

    # Sort labels so that they are processed in left to right order
    s = stats[1:, cv2.CC_STAT_LEFT]
    labelOrder = sorted(range(len(s)), key=lambda k: s[k])

    # classify digits
    for i in labelOrder:
        digits.append(_processLabels(array, stats, i+1))
        lab = model.predict_classes(digits[-1].reshape(1, 28, 28, 1).astype('float32')/255)
        labels.append(lab)

    if debug:
        print(labels)
        plt.show()

    # if method fails just return 1.0 magnification
    if len(labels) == 1:
        return 1.0

    # format and return magnification level
    first = labels[0]
    second = labels[1]

    if len(labels) > 2:
        if int(str(first[0])+str(second[0])) < 20.:
            if len(labels) == 3:
                third = None
            else:
                third = labels[2]
        else:
            third = None
    else:
        third = None

    if third:
        magnification = float(f"{first[0]}{second[0]}.{third[0]}")
    else:
        magnification = float(f"{first[0]}.{second[0]}")

    return magnification


if __name__ == '__main__':
    import glob as gb
    import time

    files = gb.glob("large/*.png")

    files.sort()

    # run tests on 1.0x magnification
    for i, file in enumerate(files[:631]):
        print(f"{i+1}/{len(files)}")
        start = time.time()
        magnification = getMagnification(file, debug=True)

        assert magnification - 1.0 < 0.2, print(file, magnification)
        finish = time.time()

    files = gb.glob("small/2019_*.png")
    files.sort()

    # run tests on different magnifications
    magns = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.1, 2.0, 3.3, 4.0, 4.0, 4.0, 4.2,
             4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 8.0, 9.4, 12.6, 10.1]

    for i, file in enumerate(files):
        print(f"{i+1}/{len(files)}")
        start = time.time()
        magnification = getMagnification(file, debug=False)
        assert magnification - magns[i] < 0.001, f"{file}, {magnification}"

        finish = time.time()
