# split dataset into train, test, and validation sets.
import json

import pandas as pd


def tojson(dataframe, filename):

    array = dataframe.to_numpy()
    mydict = {}
    for row in array:
        name, frame, x0, y0, x1, y1, label = row
        if name not in mydict:
            mydict[name] = {}
        if frame not in mydict[name]:
            mydict[name][frame] = {"boxes": [], "labels": []}
        mydict[name][frame]["boxes"].append([x0, y0, x1, y1])
        mydict[name][frame]["labels"].append(label)

    with open(f"{filename}.json", "w") as fout:
        json.dump(mydict, fout)


file = "updated_labels.csv"

trainVideos = ["錄製_2019_11_23_16_16_02_506.mp4", "錄製_2019_11_29_16_14_59_189.mp4",
               "錄製_2019_11_24_16_10_26_600.mp4", "錄製_2019_11_20_10_13_54_900.mp4",
               "錄製_2019_11_28_09_42_38_725.mp4", "錄製_2019_11_28_07_43_03_380.mp4"]

validationVideos = ["錄製_2019_11_28_12_05_07_124.mp4", "錄製_2019_11_29_08_23_34_390.mp4",
                    "錄製_2019_11_28_16_11_37_783.mp4", "錄製_2019_11_21_07_25_55_214.mp4",
                    "錄製_2019_11_20_11_12_52_295.mp4", "錄製_2019_11_25_12_24_03_891.mp4"]

testVideos = ["錄製_2019_11_28_12_59_28_589.mp4", "錄製_2019_11_28_11_02_40_268.mp4",
              "錄製_2019_11_29_13_07_47_434.mp4"]

df = pd.read_csv(file, names=["filename", "framenumber", "x0", "y0", "x1", "y1",
                              "label", "comment1", "comment2", "comment3",
                              "comment4"], engine="python")
# remove all comments in dataset
df = df.drop(["comment1", "comment2", "comment3", "comment4"], axis=1)
# remove absoulte filepath
df.filename.replace(r"E\:\\.{40,}\\", "", regex=True, inplace=True)

# train
trainSet = df[df["filename"].isin(trainVideos)]
lengthTrain = len(trainSet.index)
assert lengthTrain == 13290

# validation
validSet = df[df["filename"].isin(validationVideos)]
lengthValid = len(validSet.index)
assert lengthValid == 3451

# test
testSet = df[df["filename"].isin(testVideos)]
lengthTest = len(testSet.index)
assert lengthTest == 3454

assert (lengthTest + lengthValid + lengthTrain) == 20195

# write out without header or index column
tojson(testSet, "test")
tojson(validSet, "valid")
tojson(trainSet, "train")

# trainSet.to_csv("train.csv", index=False, header=False)
# validSet.to_csv("valid.csv", index=False, header=False)
# testSet.to_csv("test.csv", index=False, header=False)
