import cv2
from collections import OrderedDict
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from argparse import ArgumentParser


def createDict(filename: str):

    f = open(filename, "r")
    lines = f.readlines()
    mydict = {}
    for line in lines:
        if line[0] == "#":
            videoFile = line[1:].strip()
            mydict[videoFile] = {}
            cap = cv2.VideoCapture(videoFile)  # converts to BGR by default
            fps = cap.get(cv2.CAP_PROP_FPS)  # get fps
            cap.release()
        else:
            lineSplit = line.split(",")
            frameNum = int(lineSplit[0])

            x0 = int(lineSplit[1][2:])
            y0 = int(lineSplit[2])
            x1 = int(lineSplit[3])
            y1 = int(lineSplit[4][:-2])
            coords = [[x0, y0], [x1, y1]]

            if frameNum not in mydict[videoFile]:
                mydict[videoFile][frameNum] = []

            mydict[videoFile][frameNum].append(coords)

    return mydict


if __name__ == '__main__':

    parser = ArgumentParser(description="Render video from output of dolphin detection.")

    parser.add_argument("-f", "--file", type=str,
                        help="Path to output file to be analysed.")

    parser.add_argument("-v", "--video", type=str,
                        help="Path to video file to be cut up.")

    # parser.add_argument("-d", "--debug", action="count", default=0,
    #                     help="Display debug info.")
    parser.add_argument("-pt", "--plot", action="store_true",
                        help="Display plot of dolphin count over all frames.")
    parser.add_argument("-nv", "--novideo", action="store_true",
                        help="If provided do not render video.")

    args = parser.parse_args()

    file = args.file
    genny = createDict(file)

    videoFile = args.video
    genny = {k: OrderedDict(sorted(v.items())) for k, v in genny.items()}

    if not args.novideo:
        cap = cv2.VideoCapture(videoFile)  # converts to BGR by default
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()
        h, w, layers = frame.shape
        writer = cv2.VideoWriter("output-new-1.avi", cv2.VideoWriter_fourcc(*"XVID"), 1, (w, h))

    i = 0
    dolphinCount = []
    for video in genny:
        for time in genny[video]:

            if not args.novideo:
                cap.set(cv2.CAP_PROP_POS_FRAMES, time)
                _, frame = cap.read()

            numDolphins = 0
            for bbox in genny[video][time]:

                x1 = int(bbox[0][1])
                x2 = int(bbox[1][1])
                y1 = int(bbox[0][0]) + 130
                y2 = int(bbox[1][0]) + 130
                if not args.novideo:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                numDolphins += 1
            dolphinCount.append(numDolphins)
            # plt.imshow(frame)
            # plt.show()
            if not args.novideo:
                org = (50, 1040-100)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, f"{numDolphins}", org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                writer.write(frame.astype("uint8"))
            print(i)
            i += 1

    if args.plot:
        plt.plot(dolphinCount)
        plt.show()

    if not args.novideo:
        cv2.destroyAllWindows()
        cap.release()
        writer.release()
