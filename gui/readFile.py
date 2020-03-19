from pathlib import Path
import sys


def _iter_dict(videodict):
    '''Make dict into a generator yield the pertinent values and keys.


    Parameters
    ----------

    videodict: dictionary
        Dictionary of information.

    Returns
    -------

    Generator

    '''

    for key in videodict:
        try:
            frames = videodict[key]["time"]
        except KeyError:
            print("No more objects to be analysed!!!")
            sys.exit()
        bboxs = videodict[key]["bbox"]
        lengths = videodict[key]["length"]
        for i, j, k in zip(frames, bboxs, lengths):
            yield key, i, j, k


def createDict(filename: str):
    '''Function creates dictionary that contains all the frames, and
       bounding boxes.

    Parameters
    ----------

    filename : str or Path object

    Returns
    -------

    generator that gives the videofilename, frame number, and bounding box

    '''

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
        # store framenumber, bbox, dolphin length in a dict
        lineSplit = line.split(",")
        videoFile = lineSplit[0]
        if videoFile not in mydict:
            mydict[videoFile] = {}
        if len(lineSplit) == 1:
            continue
        frameNum = int(lineSplit[1].lstrip())

        x0 = int(lineSplit[2][2:].lstrip())
        y0 = int(lineSplit[3].lstrip())
        x1 = int(lineSplit[4].lstrip())
        y1 = int(lineSplit[5].lstrip().rstrip()[:-1])
        coords = [[x0, y0], [x1, y1]]
        dolphinLength = float(lineSplit[-1])
        if boolFlag:
            if "time" not in mydict[videoFile]:
                mydict[videoFile]["time"] = []
                mydict[videoFile]["bbox"] = []
                mydict[videoFile]["length"] = []

            mydict[videoFile]["time"].append(frameNum)
            mydict[videoFile]["bbox"].append(coords)
            mydict[videoFile]["length"].append(dolphinLength)

        if coords == lastCoords and frameNum == lastFrameNum and str(lastVideo) == videoFile:
            boolFlag = True

    return _iter_dict(mydict)


if __name__ == '__main__':

    file = "test.dat"
    dictFrames = createDict(file)

    iterDict = iter_dict(dictFrames)
    for i in iterDict:
        print(i)
