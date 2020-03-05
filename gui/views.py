from pathlib import Path

import cv2
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QDialog, QFileDialog

from models import Camera
from vfs import FileVideoStream


class StartWindow(QMainWindow):
    def __init__(self, size, generatorFile):
        super().__init__()

        self.videoDir = Path(QFileDialog.getExistingDirectory(self, "Select Video Directory"))

        self.inputGenerator = generatorFile
        self.filename, self.currentFrameNumber, self.bbox = next(self.inputGenerator)
        self.filename = self.videoDir / Path(self.filename)

        # output is framenumber, bbox, class
        self.outFile = "labels.csv"
        self.dialogs = list()

        # init "camera" which grabs frames from video to display
        self.camera = Camera()
        self.camera.initialize(self.filename)

        # LoadUi designed with QtCreator
        mainWindow = uic.loadUi("gui/mainwindow.ui", self)

        # Auto scale image when window resized
        self.imageAction.setScaledContents(True)
        # Show image
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
        self.multiDolphinAction.clicked.connect(lambda ch, i=12: self.saveLabelgetNextImage(i))
        self.otherAction.clicked.connect(lambda ch, i=13: self.saveLabelgetNextImage(i))

        # show video stream dialog
        dialog = VideoPlayer(self.filename, self.currentFrameNumber, self)
        self.dialogs.append(dialog)
        self.dialogs[-1].show()

    def writeToFile(self, filename, content):
        '''Function write data to file

        Parameters
        ----------
        filename : str

        content : str

        Returns
        -------
        None

        '''

        with open(filename, "a") as myfile:
            myfile.write("\n" + content)

    def saveLabelgetNextImage(self, item):
        '''If dolphin button clicked records object as a dolphin'''

        # write out label
        self.writeToFile(self.outFile, f"{self.filename}, {self.currentFrameNumber}, {self.bbox[0][0]}, {self.bbox[0][1]}, {self.bbox[1][0]}, {self.bbox[1][1]}, {item}")

        self.get_next_image_data()

        # update video stream
        self.dialogs[-1].update(self.filename, self.currentFrameNumber)

        self.update_image()

    def get_next_image_data(self):
        '''Gets next frame number and bounding box to show.
           Checks if source has changed'''

        try:
            newFile, self.currentFrameNumber, self.bbox = next(self.inputGenerator)
        except StopIteration:
            newFile = ""

        if newFile != self.filename:
            self.camera.close_camera()
            self.filename = newFile
            self.filename = self.videoDir / Path(self.filename)
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


class VideoPlayer(QDialog):
    """docstring for Second"""
    def __init__(self, filename, currentFrame, parent=None):
        super(VideoPlayer, self).__init__()

        self.fileName = filename
        self.originalFrame = currentFrame
        self.frameNumber = 0
        self.videoLength = 50  # frames to loop over including orginal frame
        self.setWindowTitle("Video Feed")
        self.timer = QTimer(self)
        startFrame = self.originalFrame - int(self.videoLength/2)

        self.fvs = self.initVideo(startFrame, self.videoLength)

        self.timer.timeout.connect(self.update_image)

        # set up UI
        self.image_view = QLabel(self)
        # allows video to automatically rescale
        self.image_view.setScaledContents(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_view)
        self.setLayout(self.layout)
        self.timer.start(40)

    def update(self, name, frame):
        '''Function updates file video stream with new file, init frame etc.
        '''

        self.fileName = name
        self.originalFrame = frame
        startFrame = max(self.originalFrame - int(self.videoLength/2), 1)
        self.fvs.stop()
        self.fvs.stream.release()

        self.fvs = None
        self.fvs = self.initVideo(startFrame, self.videoLength)

    def initVideo(self, start, length):
        ''' Get FileVideoStream object
        '''

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
