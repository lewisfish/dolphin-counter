from pathlib import Path
import sys

import cv2


def _iter_dict(videodict):
    '''make dict into a generator yield the pertinent values and keys.
    '''

    for key in videodict:
        try:
            frames = videodict[key]["time"]
        except KeyError:
            print("No more objects to be analysed!!!")
            sys.exit()
        bboxs = videodict[key]["bbox"]
        for i, j in zip(frames, bboxs):
            yield key, i, j


def createDict(filename):

    # if user has already labeled some objects check file and only get
    # unlabeled objects from file.
    labelPath = Path("labels.csv")
    if labelPath.exists():
        labelFile = open(labelPath, "r")
        labelLines = labelFile.readlines()
        if len(labelLines) > 0:
            lastLine = labelLines[-1]
            lineSplit = lastLine.split(",")
            lastVideo = Path(lineSplit[0]).name
            lastFrameNum = int(lineSplit[1])

            x0 = int(lineSplit[2])
            y0 = int(lineSplit[3])
            x1 = int(lineSplit[4])
            y1 = int(lineSplit[5])
            lastCoords = [[x0, y0], [x1, y1]]
        else:
            lastFrameNum, lastCoords = None, None
    else:
        lastFrameNum, lastCoords = None, None

    f = open(filename, "r")
    lines = f.readlines()
    mydict = {}

    if lastFrameNum is None and lastCoords is None:
        boolFlag = True
    else:
        boolFlag = False

    for line in lines:
        # get video filename
        # store framenumber and bbox in a dict
        lineSplit = line.split(",")
        videoFile = lineSplit[0]
        if videoFile not in mydict:
            mydict[videoFile] = {}

        frameNum = int(lineSplit[1])

        x0 = int(lineSplit[2][2:])
        y0 = int(lineSplit[3])
        x1 = int(lineSplit[4])
        y1 = int(lineSplit[5][:-2])
        coords = [[x0, y0], [x1, y1]]
        if boolFlag:
            if "time" not in mydict[videoFile]:
                mydict[videoFile]["time"] = []
                mydict[videoFile]["bbox"] = []

            mydict[videoFile]["time"].append(frameNum)
            mydict[videoFile]["bbox"].append(coords)

        if coords == lastCoords and frameNum == lastFrameNum and str(lastVideo) == videoFile:
            boolFlag = True

    return _iter_dict(mydict)


if __name__ == '__main__':

    file = "test.dat"
    dictFrames = createDict(file)

    iterDict = iter_dict(dictFrames)
    for i in iterDict:
        print(i)
