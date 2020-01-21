import pandas as pd
import cv2
from PIL import Image
import pytesseract
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def initModel():
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


def processLabels(image, stats, label):

    digit = np.where(image != label, 0, 255)
    x1 = max(0, stats[label, cv2.CC_STAT_LEFT] - 5)
    x2 = x1 + stats[label, cv2.CC_STAT_WIDTH] + 10
    y1 = stats[label, cv2.CC_STAT_TOP] - 5
    y2 = y1 + stats[label, cv2.CC_STAT_HEIGHT] + 10

    digit = digit[y1:y2, x1:x2]
    digit = np.array(digit, dtype="uint8")
    digit = cv2.resize(digit,
                       dsize=(28, 28),
                       interpolation=cv2.INTER_CUBIC)

    return digit


model = initModel()

files = ["2019_11_23_16_18928.png", "2019_11_23_16_19185.png",
         "2019_11_23_16_22062.png", "2019_11_23_16_55374.png",
         "2019_11_23_16_55376.png", "2019_11_23_16_55378.png",
         "2019_11_23_16_55380.png", "2019_11_23_16_55382.png",
         "2019_11_23_16_19134.png", "2019_11_23_16_19468.png",
         "2019_11_23_16_55373.png", "2019_11_23_16_55375.png",
         "2019_11_23_16_55377.png", "2019_11_23_16_55379.png",
         "2019_11_23_16_55381.png", "2019_11_23_16_55245.png",
         "2019_11_23_16_55271.png", "2019_11_23_16_55296.png",
         "2019_11_23_16_55322.png", "2019_11_23_16_55348.png",
         "2019_11_23_16_55373.png", "2019_11_23_16_55399.png",
         "2019_11_23_16_55425.png", "2019_11_23_16_55450.png",
         "2019_11_23_16_55476.png"]

fig = plt.figure()
outer_grid = fig.add_gridspec(5, 5, wspace=.5, hspace=0.)

for i, file in enumerate(files):
    print(file)
    inner_grid = outer_grid[i].subgridspec(1, 3, wspace=0.0, hspace=0.0)
    img = Image.open(file).convert('LA')
    array = np.array(img)[:, :, 0]

    # array = 255-array
    ret2, array = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    array = cv2.dilate(array, kernel, iterations=2)
    array = array[80:140, 25:127]

    output = cv2.connectedComponentsWithStats(array, 4, cv2.CV_32S)
    nlabels = output[0]
    array = output[1]
    stats = output[2]

    # plt.imshow(array)
    # plt.show()

    labels = []
    digits = []
    for i in range(1, 4):
        digits.append(processLabels(array, stats, i))
        lab = model.predict_classes(digits[-1].reshape(1, 28, 28, 1).astype('float32')/255)
        labels.append(lab)

    # for dig in digits:
    #     plt.imshow(dig)
    #     plt.show()

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

    if labels[2] != 8 and labels[1] != 0:
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


# files = ["2019_11_23_16_18928.png", "2019_11_23_16_19185.png",
#          "2019_11_23_16_22062.png", "2019_11_23_16_55374.png",
#          "2019_11_23_16_55376.png", "2019_11_23_16_55378.png",
#          "2019_11_23_16_55380.png", "2019_11_23_16_55382.png",
#          "2019_11_23_16_19134.png", "2019_11_23_16_19468.png",
#          "2019_11_23_16_55373.png", "2019_11_23_16_55375.png",
#          "2019_11_23_16_55377.png", "2019_11_23_16_55379.png",
#          "2019_11_23_16_55381.png", "2019_11_23_16_55245.png",
#          "2019_11_23_16_55271.png", "2019_11_23_16_55296.png",
#          "2019_11_23_16_55322.png", "2019_11_23_16_55348.png",
#          "2019_11_23_16_55373.png", "2019_11_23_16_55399.png",
#          "2019_11_23_16_55425.png", "2019_11_23_16_55450.png",
#          "2019_11_23_16_55476.png"]