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

    # if user has already labelled some objects check file and only get
    # unlablled objects from file.
    labelPath = Path("labels.dat")
    if labelPath.exists():
        labelFile = open(labelPath, "r")
        labelLines = labelFile.readlines()
        if len(labelLines) > 0:
            lastLine = labelLines[-1]
            lineSplit = lastLine.split(",")
            lastFrameNum = int(lineSplit[0])

            x0 = int(lineSplit[1][3:])
            y0 = int(lineSplit[2][:-1])
            x1 = int(lineSplit[3][2:])
            y1 = int(lineSplit[4][:-2])
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
        if line[0] == "#":
            videoFile = line[1:].strip()
            mydict[videoFile] = {}
            cap = cv2.VideoCapture(videoFile)  # converts to BGR by default
            fps = cap.get(cv2.CAP_PROP_FPS)  # get fps
            cap.release()
        else:
            # store framenumber and bbox in a dict
            lineSplit = line.split(",")
            frameNum = int(lineSplit[0])

            x0 = int(lineSplit[1][2:])
            y0 = int(lineSplit[2])
            x1 = int(lineSplit[3])
            y1 = int(lineSplit[4][:-2])
            coords = [[x0, y0], [x1, y1]]
            if boolFlag:
                if "time" not in mydict[videoFile]:
                    mydict[videoFile]["time"] = []
                    mydict[videoFile]["bbox"] = []

                mydict[videoFile]["time"].append(frameNum)
                mydict[videoFile]["bbox"].append(coords)

            if coords == lastCoords and frameNum == lastFrameNum:
                boolFlag = True

    return _iter_dict(mydict)


if __name__ == '__main__':

    file = "test.dat"
    dictFrames = createDict(file)

    iterDict = iter_dict(dictFrames)
    for i in iterDict:
        print(i)
