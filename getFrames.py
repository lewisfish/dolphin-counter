import cv2
from argparse import ArgumentParser


def getTimes(file, fps):
    '''Function takes a filename and returns times in seconds from an expected
       format of hr:min:sec #frames
       Where #frames is the number of frames to return after this timestamp.
       If #frames is not provided then we assume just 1 frame is needed.

    Parameters
    ----------

    file : str
        Name of file to open and process.

    Returns
    -------

    times : List[float]
        List of times to capture frames.
    '''

    times = []

    with open(file, "r")as f:
        lines = f.readlines()
        for line in lines:
            numExtraFrames = line.split(" ")
            hour, minute, sec = numExtraFrames[0].split(":")
            time = (int(hour) * 60*60) + (int(minute) * 60) + int(sec)
            times.append(time)

    if len(numExtraFrames) == 1:
        return times, 1, 1
    elif len(numExtraFrames) == 2:
        return times, int(numExtraFrames[1]), 1
    else:
        return times, int(numExtraFrames[1]), int(numExtraFrames[2])


parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to single image to be analysed.")
parser.add_argument("-t", "--times", type=str,
                    help="Path to file which contains timestamps in the format\
                    hr:min:sec fro creating stills.")

args = parser.parse_args()

# open video file
cap = cv2.VideoCapture(args.file)  # converts to RGB by default
fps = cap.get(cv2.CAP_PROP_FPS)  # get fps

times, rangeFrames, step = getTimes(args.times, fps)

# loop over times to create frames
for i, time in enumerate(times):
    # get frame number from fps and timestamp in seconds
    frameNum = int(time * fps)
    # set position in video as frameNum
    for i in range(1, rangeFrames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        _, frame = cap.read()
        # save frame as png
        name = args.file[0:13] + f"_{frameNum}"
        print(name + ".png")
        cv2.imwrite("Ml-test/" + name + ".png", frame)
        frameNum += step

cap.release()
