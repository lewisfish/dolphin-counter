import cv2
from argparse import ArgumentParser


def getTimes(file):
    '''Function takes a filename and returns times in seconds from an expected
       format of hr:min:sec

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
            hour, minute, sec = line.split(":")
            time = (int(hour) * 60*60) + (int(minute) * 60) + int(sec)
            times.append(time)

    return times


parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to single image to be analysed.")
parser.add_argument("-t", "--times", type=str,
                    help="Path to file which contains timestamps in the format\
                    hr:min:sec fro creating stills.")

args = parser.parse_args()

# open video file
cap = cv2.VideoCapture(args.file)
fps = cap.get(cv2.CAP_PROP_FPS)  # get fps

times = getTimes(args.times)

# loop over times to create frames
for i, time in enumerate(times):
    # get frame number from fps and timestamp in seconds
    frameNum = int(time * fps)
    # set position in video as frameNum
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    _, frame = cap.read()
    # save frame as png
    name = args.file[0:13] + str(frameNum)
    print(name + ".png")
    cv2.imwrite(name + ".png", frame)

cap.release()
