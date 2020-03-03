import bisect
import datetime
from pathlib import Path
from typing import Generator, Tuple, List

import cv2
import numpy as np
import pandas as pd

__all__ = ["getAltitude"]


def _getTimeDate(filename: Path) -> datetime.datetime:
    '''Function gets the date and video start time from video files name

    Parameters
    ----------

    filename : Path object
        Path to the video file

    Returns
    -------

    dt : datetime.datetime object
        The date and time of the drones flight start

    '''
    # remove folders and codec from filename
    name = filename.stem
    date = name[:10].split("_")
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])

    time = name[11:].split("_")
    hour = int(time[0])
    minute = int(time[1])
    second = int(time[2])
    millsecond = int(time[3])
    dt = datetime.datetime(year, month, day, hour, minute, second, millsecond)

    return dt


def _findStart(files: Generator[Path, None, None], targetTime: datetime.datetime) -> Tuple[Path, List, List, int]:
    '''Function matches the videos to a drone flight number. Also matches start
       time to the gps data which starts recording before the video records,
       and has a different tic from that of the video.

    Parameters
    ----------

    files : Generator object
        Generator object that yields the file names of the possible GPS data
        files.

    targetTime : datetime.datetime object
        The time to search for in the GPS data.

    Returns
    -------

    outfile, outalts, outtimes, curPosition : Tuple(Path, List, List, int)
        Outfile is the GPS-data file filename : Path or str
        outalts is a list of altitudes for the given video.
        outtimes is a list of times for the given video where GPS was recorded.
        curPosition is the location in the list of times/GPS data when the
        video feed begins.

    '''

    curPosition = 1e99
    for file in files:
        times = []
        df = pd.read_csv(file)
        time = df["Time"]

        # format first and last datetime and check if target falls between them.
        # as strptime is reallllllyyyy slow...
        t = time.iloc[0]
        fmttime = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
        times.append(fmttime)

        t = time.iloc[-1]
        fmttime = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
        times.append(fmttime)

        # find position of target time in times array
        position = bisect.bisect_left(times, targetTime)

        # if position not in list then it returns the length of the list as the position
        if position == 1 and position < curPosition:
            outfile = file
            curPosition = position
            outalts = df["Altitude/mm"]

            outtimes = []
            for t in time:
                fmttime = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
                outtimes.append(fmttime)

    return outfile, outalts, outtimes, curPosition


def getAltitude(videoFile: str, framenumber: int, gpsdataPath="videos+data/gps-data/") -> float:
    '''Function calculates the altitude of the drone given a video and a frame
       number from that video to match the GPS data to.

    Parameters
    ----------

    videoFile : Path object or str
        Path to the video file for which an altitude is required.

    framenumber : int
        Number of the frame for which the altitude is required.

    gpsdataPath : str, optional
        Path to GPS data folder

    Returns
    -------

    currentAltitude : float
        The calculated altitude.

    '''

    # open and set video to frame of interest, whilst collecting
    # metadata from video

    videoPath = Path(videoFile)
    cap = cv2.VideoCapture(str(videoFile))  # converts to BGR by default
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, framenumber)

    # get current position of video in msec from start
    # need to read frame for this to work...
    _, frame = cap.read()
    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    msec /= 1e3

    cap.release()

    dt = _getTimeDate(videoPath) + datetime.timedelta(seconds=3)  # adjust for video recording delay
    # form gpsdata filename
    date = str(dt.year) + str(dt.month) + str(dt.day)
    path = Path(gpsdataPath)
    files = path.glob(date + "*.csv")
    file, alts, times,  pos = _findStart(files, dt)

    # get the altitude
    frameTime = dt + datetime.timedelta(seconds=msec)

    position = bisect.bisect_left(times, frameTime)
    currentTime = times[position-1]
    currentAltitude = alts[position-1]

    return currentAltitude / 1e3  # convert to meters


if __name__ == '__main__':

    # given a frame number what is the altitude?
    altsa = []
    cnt = 0
    for i in range(0, 80000, 5000):
        altitude = getAltitude(Path("videos+data/2019_11_23_16_16_02_506.mp4"), i)
        altsa.append(altitude / 1e3)
        cnt += 1
