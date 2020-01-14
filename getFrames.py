import cv2
import numpy as np
import subprocess
import re


def parse_metadata(metadata, error):

    if error is not None:
        raise AttributeError(error)

    durationREGEX = r"\d{2}:\d{2}:\d{2}\.\d{2}"
    fpsREGEX = r"\d{2,3}\.\d{1,2}\ fps"

    metadata = metadata.decode("utf-8")
    matches = re.findall(durationREGEX, metadata, re.MULTILINE)
    time = matches[0]

    duration = int(time[:2]) * 60 * 60  # hours
    duration += (int(time[3:5]) * 60)  # minutes
    duration += float(time[6:])  # seconds

    fps = re.findall(fpsREGEX, metadata, re.MULTILINE)[0]
    fps = float(fps[:-3])

    return fps, duration


video = "2019_11_24_16_10_26_600.mp4"

out = subprocess.Popen(["ffmpeg", "-i", video], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)

stdout, stderr = out.communicate()

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

times = [12*60 + 26, 12*60 + 17, 12*60 + 28, 12*60 + 38, 14*60 + 19]
for i, time in enumerate(times):
    frameNum = time * fps
    # print(fps, totalFrameNumber, width, height, frameNum)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    _, frame = cap.read()
    name = video[0:13] + "_4"
    print(name + f"_{i}.png")
    cv2.imwrite(name + f"_{i}.png", frame)

cap.release()
