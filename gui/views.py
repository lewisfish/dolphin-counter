import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic

from models import Camera
from vfs import FileVideoStream


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

        dialog = VideoPlayer(self.filename, self.currentFrameNumber)
        self.dialogs.append(dialog)
        self.dialogs[-1].show()

    def writeToFile(self, filename, content):
        with open(filename, "a") as myfile:
            myfile.write("\n" + content)

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

        self.writeToFile(self.outFile, f"{self.currentFrameNumber}, {self.bbox}, {item}")
        self.get_next_image_data()

        self.dialogs[-1].update(self.filename, self.currentFrameNumber)

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

    def closeEvent(self, event):
        '''Closes the opened outfile on closing of QtApplication'''
        for dialog in self.dialogs:
            dialog.close()


class VideoPlayer(QMainWindow):
    """docstring for Second"""
    def __init__(self, filename, currentFrame):
        super(VideoPlayer, self).__init__()

        self.fileName = filename
        self.originalFrame = currentFrame
        self.frameNumber = 0
        self.videoLength = 50  # frames to loop over including orginal frame

        self.timer = QTimer(self)

        startFrame = self.originalFrame - int(self.videoLength/2)

        self.fvs = self.initVideo(startFrame, self.videoLength)

        self.timer.timeout.connect(self.update_image)

        # set up UI
        self.central_widget = QWidget()
        self.image_view = QLabel(self)
        # allows video to automatically rescale
        self.image_view.setScaledContents(True)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.image_view)
        self.setCentralWidget(self.central_widget)
        self.timer.start(40)

    def update(self, name, frame):
        self.fileName = name
        self.originalFrame = frame
        startFrame = max(self.originalFrame - int(self.videoLength/2), 1)
        self.fvs.stop()
        self.fvs.stream.release()

        self.fvs = None
        self.fvs = self.initVideo(startFrame, self.videoLength)

    def initVideo(self, start, length):
        return FileVideoStream(self.fileName, start, length).start()

    def resize(self, image, width):
        '''Function that resize an image and keeps image ratio.
        '''

        h, w, channel = image.shape
        ratio = width / float(w)
        dim = (width, int(h * ratio))

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def update_image(self):

        if not self.fvs.stopped:
            frame = self.fvs.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = self.resize(frame, 1500)

            height, width, channel = frame.shape
            bytesPerLine = 3 * width

            pimg = QImage(frame, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap(pimg)

            self.image_view.setPixmap(pixmap)

    def closeEvent(self, event):
        '''Closes the opened outfile on closing of QtApplication'''
        self.timer.stop()


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
