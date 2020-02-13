import cv2
import numpy as np


class Camera:
    def __init__(self):
        self.cam_name = None
        self.cap = None
        self.last_frame = np.zeros((1, 1, 3), np.uint8)

    def initialize(self, camera_name):
        self.cam_name = camera_name
        if self.cam_name == "":
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(self.cam_name)

    def get_frame(self, frameNumber):
        if self.cam_name == "":
            self.last_frame = np.zeros((761, 1500, 3), np.uint8)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
            ret, self.last_frame = self.cap.read()
            self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        return self.last_frame

    def close_camera(self):
        self.cap.release()

    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cam_name)


if __name__ == '__main__':
    cam = Camera(0)
    cam.initialize()
    print(cam)
    frame = cam.get_frame()
    print(frame)
    cam.close_camera()
