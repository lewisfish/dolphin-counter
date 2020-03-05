from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import cv2


def getTimes(file: str) -> Tuple[List[int], int, int]:
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
    videos = []
    steps = []
    numbframes = []

    with open(file, "r")as f:
        lines = f.readlines()
        for line in lines:
            if ".mp4" in line:
                videoFlag = True
                parts = line.split(" ")
                video = parts[0]
                videos.append(video)

                hour, minute, sec = parts[1].split(":")
                time = (int(hour) * 60*60) + (int(minute) * 60) + int(sec)
                times.append(time)

                if len(parts) >= 3:
                    numbframes.append(int(parts[2]))
                    if len(parts) >= 4:
                        steps.append(int(parts[3]))
                else:
                    numbframes.append(1)
                    steps.append(1)

            else:
                numExtraFrames = line.split(" ")
                hour, minute, sec = numExtraFrames[0].split(":")
                time = (int(hour) * 60*60) + (int(minute) * 60) + int(sec)
                times.append(time)
                videos.append(None)

                if len(parts) >= 3:
                    numbframes.append(int(parts[1]))
                    steps.append(int(parts[2]))
                else:
                    numbframes.append(1)
                    steps.append(1)

    return times, numbframes, steps, videos


parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to video to be turned into frames")
parser.add_argument("-t", "--times", type=str,
                    help="Path to file which contains timestamps in the format\
                    hr:min:sec for creating stills.")
parser.add_argument("-fo", "--folder", type=str,
                    help="Folder to save frames in.")

args = parser.parse_args()

if args.folder is None:
    args.folder = "./"

times, rangeFrames, steps, videos = getTimes(args.times)

f = open("filelist.txt", "w")

listVideos = list(Path(args.folder).glob("**/*.mp4"))
# loop over times to create frames
for i, time in enumerate(times):
    # open video file
    if videos[i] is not None:
        filename = Path(videos[i]).name[:-4]
        for realVideo in listVideos:
            if filename in str(realVideo):
                filename = str(realVideo)
                break
        cap = cv2.VideoCapture(filename)  # converts to BGR by default
    else:
        filename = Path(args.file).name[:-4]
        cap = cv2.VideoCapture(args.file)

    fps = cap.get(cv2.CAP_PROP_FPS)  # get fps
    if steps[i] == -99:
        step = fps
    else:
        step = steps[i]

    # get frame number from fps and timestamp in seconds
    frameNum = int(time * fps)
    # set position in video as frameNum
    for j in range(0, rangeFrames[i]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        _, frame = cap.read()
        # save frame as png
        finalfilename = filename + f"_{frameNum}.png"
        f.write(args.folder + videos[i]+","+str(frameNum)+"\n")
        print(args.folder + videos[i]+","+str(frameNum))
        # print(f"{args.folder}" + finalfilename)
        # cv2.imwrite(f"{args.folder}/" + finalfilename, frame)
        frameNum += int(step)
    cap.release()
f.close()
