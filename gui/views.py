import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic

from models import Camera


class StartWindow(QMainWindow):
    def __init__(self, size, generatorFile):
        super().__init__()

        self.genny = generatorFile
        self.filename, self.currentFrameNumber, self.bbox = next(self.genny)
        # output is framenumber, bbox, class
        self.outFile = "labels.dat"
        self.dialogs = list()

        self.camera = Camera()
        self.camera.initialize(self.filename)

        mainWindow = uic.loadUi("gui/mainwindow.ui", self)
        self.update_image()

        # get button presses and send appropriate class to function
        self.dolphinAction.clicked.connect(lambda ch, i=0: self.saveLabelgetNextImage(i))
        self.whaleAction.clicked.connect(lambda ch, i=1: self.saveLabelgetNextImage(i))
        self.fishAction.clicked.connect(lambda ch, i=2: self.saveLabelgetNextImage(i))
        self.turtleAction.clicked.connect(lambda ch, i=3: self.saveLabelgetNextImage(i))
        self.sbirdAction.clicked.connect(lambda ch, i=4: self.saveLabelgetNextImage(i))
        self.fbirdAction.clicked.connect(lambda ch, i=5: self.saveLabelgetNextImage(i))
        self.logAction.clicked.connect(lambda ch, i=6: self.saveLabelgetNextImage(i))
        self.trashAction.clicked.connect(lambda ch, i=7: self.saveLabelgetNextImage(i))
        self.waveAction.clicked.connect(lambda ch, i=8: self.saveLabelgetNextImage(i))
        self.wcrestAction.clicked.connect(lambda ch, i=9: self.saveLabelgetNextImage(i))
        self.boatAction.clicked.connect(lambda ch, i=10: self.saveLabelgetNextImage(i))
        self.glareAction.clicked.connect(lambda ch, i=11: self.saveLabelgetNextImage(i))

    def writeToFile(self, filename, content):
        with open(filename, "a") as myfile:
            myfile.write("\n" + content)

    def show_video_inset(self, filename, currentFrame, bbox):
        '''Function initiates a popout video player of object of interest
        '''

        dialog = VideoPlayer(self.insetVideo, filename, currentFrame, bbox)
        self.dialogs.append(dialog)

    def intersection(self, a, b):

        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0:
            return None
        return (x, y, w, h)

    def saveLabelgetNextImage(self, item):
        '''If dolphin button clicked records object as a dolphin'''

        self.dialogs[-1].close()
        self.writeToFile(self.outFile, f"{self.currentFrameNumber}, {self.bbox}, {item}")
        self.get_next_image_data()
        self.update_image()

    def get_next_image_data(self):
        '''Gets next frame number and bounding box to show.
           Checks if source has changed'''

        try:
            newFile, self.currentFrameNumber, self.bbox = next(self.genny)
        except StopIteration:
            newFile = ""

        if newFile != self.filename:
            self.camera.close_camera()
            self.filename = newFile
            self.camera.initialize(self.filename)

    def update_image(self):
        '''Updates displayed image and shows ROI as an inset.'''

        frame = self.camera.get_frame(self.currentFrameNumber)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        if self.filename != "":
            x1 = self.bbox[0][1]
            x2 = self.bbox[1][1]
            y1 = self.bbox[0][0] + 130  # due to cropping in anaylsis
            y2 = self.bbox[1][0] + 130

            hdiff = int((y2 - y1) / 2)
            wdiff = int((x2 - x1) / 2)

            inset = frame[y1-hdiff:y2+hdiff, x1-wdiff:x2+wdiff].copy()
            insetHeight, insetWidth, channel = inset.shape
            bytesPerLine = 3 * insetWidth

            insetQimg = QImage(inset.data, insetWidth, insetHeight, bytesPerLine, QImage.Format_RGB888)
            insetPixmap = QPixmap(insetQimg)
            insetPixmap = insetPixmap.scaled(300, 300, Qt.KeepAspectRatio)
            self.resize(insetPixmap.width(), insetPixmap.height())

            self.insetImage.setPixmap(insetPixmap)

            # check if green rect is in inset
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Show "done!!" if no images left
            frame = cv2.putText(frame, 'Done!!', (750, 380), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)

        # update canvas image
        bytesPerLine = 3 * width
        qimg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap(qimg)
        self.pixmap = self.pixmap.scaled(1550, 787)

        self.resize(self.pixmap.width(), self.pixmap.height())
        self.imageAction.setPixmap(self.pixmap)

        self.show_video_inset(self.filename, self.currentFrameNumber, self.bbox)


class VideoPlayer(QMainWindow):
    """docstring for Second"""
    def __init__(self, insetVideo, filename, currentFrame, bbox):
        super(VideoPlayer, self).__init__()

        self.fileName = filename
        self.originalFrame = currentFrame
        self.frameNumber = currentFrame
        self.bbox = bbox
        self.videoLength = 20  # frames to loop over including orginal frame

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.getNextFrame)

        self.camera = Camera()
        self.camera.initialize(self.fileName)

        # set up UI
        self.image_view = insetVideo

        self.update_image()
        self.timer.start()

    def getNextFrame(self):
        if self.frameNumber <= self.originalFrame + self.videoLength:
            self.frameNumber += 1
        else:
            self.frameNumber = self.originalFrame - self.videoLength

        self.update_image()

    def get_extent_ROI(self, x1, x2, y1, y2):
        # get ROI
        hdiff = int((y2 - y1))
        wdiff = int((x2 - x1))

        return y1-hdiff, y2+hdiff, x1-wdiff, x2+wdiff

    def update_image(self):
        '''Updates displayed image and shows ROI as an inset.'''

        frame = self.camera.get_frame(self.frameNumber)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width

        # add 50 pixels to each side to increase video size
        x1 = self.bbox[0][1] - 50
        x2 = self.bbox[1][1] + 50
        y1 = self.bbox[0][0] + 130 - 50  # due to cropping in analysis
        y2 = self.bbox[1][0] + 130 + 50

        # constrain ROI to within orignal frame
        x1 = max(0, x1)
        x2 = min(width, x2)
        y1 = max(0, y1)
        y2 = min(height, y2)

        y1, y2, x1, x2 = self.get_extent_ROI(x1, x2, y1, y2)

        # update canvas image
        if self.fileName == "":
            newimage = cv2.putText(frame, 'Done!!', (750, 380), cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            newimage = frame[y1:y2, x1:x2].astype("uint8")

        h, w, _ = newimage.shape

        bytesPerLine = 3 * w

        pimg = QImage(newimage, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(pimg)
        pixmap = pixmap.scaled(325, 325)

        self.resize(pixmap.width(), pixmap.height())

        self.image_view.setPixmap(pixmap)

    def closeEvent(self, event):
        '''Closes the opened outfile on closing of QtApplication'''
        self.timer.stop()
        self.camera.close_camera()


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
