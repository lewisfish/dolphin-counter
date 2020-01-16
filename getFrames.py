import cv2
import subprocess
from argparse import ArgumentParser


def getTimes(file):

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


out = subprocess.Popen(["ffmpeg", "-i", args.file], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)

stdout, stderr = out.communicate()

cap = cv2.VideoCapture(args.file)
fps = cap.get(cv2.CAP_PROP_FPS)
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

times = getTimes(args.times)

for i, time in enumerate(times):
    frameNum = int(time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    _, frame = cap.read()
    name = args.file[0:13] + str(frameNum)
    print(name + ".png")
    cv2.imwrite(name + ".png", frame)

cap.release()
