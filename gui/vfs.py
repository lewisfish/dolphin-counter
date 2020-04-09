# Adapted from imutils https://github.com/jrosebr1/imutils
from queue import Queue
import sys
from threading import Thread
import time

from cv2 import VideoCapture, CAP_PROP_POS_FRAMES


class FileVideoStream:
    def __init__(self, path, start, length, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not

        self.stream = VideoCapture(str(path))
        self.startFrame = start
        self.videoLength = length
        self.stream.set(CAP_PROP_POS_FRAMES, self.startFrame)

        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        self.currentNumber = 0

        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # add the frame to the queue
                self.Q.put(frame)
                self.currentNumber += 1

                if self.currentNumber >= self.videoLength:
                    self.stream.set(CAP_PROP_POS_FRAMES, self.startFrame)
                    self.currentNumber = 0

            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        frame = self.Q.get()
        return frame

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def clear(self):

        with self.Q.mutex:
            self.Q.queue.clear()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
