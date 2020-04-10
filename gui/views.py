# -*- coding: utf-8 -*-
import os
from pathlib import Path

from cv2 import rectangle, line, resize, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, CAP_PROP_FPS, INTER_AREA, cvtColor, COLOR_BGR2RGB
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QDialog, QFileDialog, QActionGroup

from models import Camera
from vfs import FileVideoStream


class StartWindow(QMainWindow):
    def __init__(self, size, generatorFile):
        super().__init__()

        self.videoDir = Path(QFileDialog.getExistingDirectory(self, "Select Video Directory"))
        self.videoFiles = list(self.videoDir.glob("**/*.mp4"))

        self.inputGenerator = generatorFile
        self.list_idx = 0

        self.filename, self.currentFrameNumber, self.bbox, self.dLength, self.readInComment = self.inputGenerator[self.list_idx]
        self.list_idx += 1
        self.filename = self.getFullFileName(self.filename)
        self.tick = 40  # interval between frames (ms)

        self.prevFilename = self.filename
        self.prevFrameNumber = self.currentFrameNumber
        self.prevBbox = self.bbox
        self.prevDlength = self.dLength
        self.prevComment = ""
        self.prevReadInComment = ""

        # output is framenumber, bbox, class
        self.outFile = "labels.csv"
        self.dialogs = list()

        # init "camera" which grabs frames from video to display
        self.camera = Camera()
        self.camera.initialize(self.filename)

        # LoadUi designed with QtCreator
        mainWindow = uic.loadUi("gui/mainwindow.ui", self)
        uniqueID = str(self.filename.name) + " " + str(self.currentFrameNumber)
        uniqueID += " " + str(self.bbox)
        self.label.setText(uniqueID)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Auto scale image when window resized
        self.imageAction.setScaledContents(True)
        # Show image
        self.update_image()

        # get button presses and send appropriate class to function
        self.dolphinAction.clicked.connect(lambda ch, i=0: self.saveLabelgetNextImage(i))
        self.birdAction.clicked.connect(lambda ch, i=1: self.saveLabelgetNextImage(i))
        self.multiDolphinAction.clicked.connect(lambda ch, i=2: self.saveLabelgetNextImage(i))
        self.whaleAction.clicked.connect(lambda ch, i=3: self.saveLabelgetNextImage(i))
        self.turtleAction.clicked.connect(lambda ch, i=4: self.saveLabelgetNextImage(i))
        self.unknownAction.clicked.connect(lambda ch, i=5: self.saveLabelgetNextImage(i))
        self.unknownnotcAction.clicked.connect(lambda ch, i=6: self.saveLabelgetNextImage(i))
        self.boatAction.clicked.connect(lambda ch, i=7: self.saveLabelgetNextImage(i))
        self.fishAction.clicked.connect(lambda ch, i=8: self.saveLabelgetNextImage(i))
        self.trashAction.clicked.connect(lambda ch, i=9: self.saveLabelgetNextImage(i))
        self.waterAction.clicked.connect(lambda ch, i=10: self.saveLabelgetNextImage(i))

        self.textEdit.setPlaceholderText("Comments")
        if self.readInComment is not None:
            self.textEdit.setPlaceholderText(self.readInComment)

        # Menu actions
        self.speedGroup = QActionGroup(self)
        self.speedGroup.addAction(self.MenuSpeed1_0)
        self.speedGroup.addAction(self.MenuSpeed0_5)
        self.speedGroup.addAction(self.MenuSpeed2_0)
        self.speedGroup.setExclusive(True)

        self.lengthGroup = QActionGroup(self)
        self.lengthGroup.addAction(self.MenuLength50)
        self.lengthGroup.addAction(self.MenuLength100)
        self.lengthGroup.addAction(self.MenuLength150)
        self.lengthGroup.addAction(self.MenuLength200)
        self.lengthGroup.setExclusive(True)

        self.speedGroup.triggered.connect(self.buttonSpeedState)
        self.lengthGroup.triggered.connect(self.buttonLengthState)

        # other button
        self.backAction.clicked.connect(self.getPreviousObject)

        # show video stream dialog
        dialog = VideoPlayer(self.filename, self.currentFrameNumber, self)
        self.dialogs.append(dialog)
        self.dialogs[-1].show()

    def getPreviousObject(self):
        '''Get previous frame, display it and remove last label from file'''

        self.backAction.setEnabled(False)
        self.removeLastLine(self.outFile)
        self.list_idx -= 1

        if self.prevFilename != self.filename:
            self.camera.close_camera()
            self.filename = self.prevFilename
            self.camera.initialize(self.filename)
        else:
            self.filename = self.prevFilename

        self.currentFrameNumber = self.prevFrameNumber
        self.bbox = self.prevBbox
        self.dLength = self.prevDlength
        self.comment = self.prevComment
        self.readInComment = self.prevReadInComment

        uniqueID = str(self.filename.name) + " " + str(self.currentFrameNumber)
        uniqueID += " " + str(self.bbox)
        self.label.setText(uniqueID)

        self.dialogs[-1].update(self.filename, self.currentFrameNumber)

        self.update_image()
        if self.readInComment is None:
            self.textEdit.insertPlainText(self.prevComment)
        else:
            self.textEdit.setPlaceholderText(self.readInComment)

    def removeLastLine(self, filename):
        '''Remove last line from a given file.'''

        lines = None
        with open(filename, "r") as fin:
            lines = fin.readlines()

        with open(filename, "w") as fout:
            fout.writelines([item for item in lines[:-1]])

    def buttonSpeedState(self, button):
        '''If a speed menu action is taken, then change the
           timer interval in the video player'''

        if button.text() == "1.0x":
            self.dialogs[-1].timer.setInterval(self.tick)
        elif button.text() == "2.0x":
            self.dialogs[-1].timer.setInterval(int(self.tick / 2))
        elif button.text() == "0.5x":
            self.dialogs[-1].timer.setInterval(self.tick * 2)

    def buttonLengthState(self, button):
        '''If a video length menu action is taken, then change the
           video length in the video player'''

        if button.text() == "2 secs":
            self.dialogs[-1].videoLength = 25
            self.dialogs[-1].fvs.videoLength = 25
            self.dialogs[-1].update(self.filename, self.currentFrameNumber)

        elif button.text() == "4 secs":
            self.dialogs[-1].videoLength = 50
            self.dialogs[-1].fvs.videoLength = 50
            self.dialogs[-1].update(self.filename, self.currentFrameNumber)

        elif button.text() == "6 secs":
            self.dialogs[-1].videoLength = 75
            self.dialogs[-1].fvs.videoLength = 75
            self.dialogs[-1].update(self.filename, self.currentFrameNumber)

        elif button.text() == "8 secs":
            self.dialogs[-1].videoLength = 100
            self.dialogs[-1].fvs.videoLength = 100
            self.dialogs[-1].update(self.filename, self.currentFrameNumber)

    def getFullFileName(self, target):
        '''Get the full filename path'''

        for file in self.videoFiles:
            if target in str(file):
                return file

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

        try:
            with open(filename, "r", encoding="utf-8") as myfile:
                text = myfile.read()
        except FileNotFoundError:
            text = "\n"

        with open(filename, "a", encoding="utf-8") as myfile:
            if not text.endswith("\n"):
                myfile.write("\n")
            # remove any newline characters so that the csv file is properly formatted.
            myfile.write(content.replace("\n", " ") + "\n")

    def saveLabelgetNextImage(self, item):
        '''If dolphin button clicked records object as a dolphin'''

        # write out label
        self.backAction.setEnabled(True)

        self.comment = self.textEdit.toPlainText()
        self.textEdit.clear()
        self.writeToFile(self.outFile, f"{self.filename}, {self.currentFrameNumber}, {self.bbox[0][0]}, {self.bbox[0][1]}, {self.bbox[1][0]}, {self.bbox[1][1]}, {item}, {self.comment}")

        self.get_next_image_data()

        # update video stream
        self.dialogs[-1].update(self.filename, self.currentFrameNumber)

        self.update_image()

    def get_next_image_data(self):
        '''Gets next frame number and bounding box to show.
           Checks if source has changed'''

        self.prevFilename = self.filename
        self.prevFrameNumber = self.currentFrameNumber
        self.prevBbox = self.bbox
        self.prevDlength = self.dLength
        self.prevComment = self.comment
        self.prevReadInComment = self.readInComment

        try:
            newFile, self.currentFrameNumber, self.bbox, self.dLength, self.readInComment = self.inputGenerator[self.list_idx]
            self.textEdit.setPlaceholderText("Comments")
            if self.readInComment is not None:
                self.textEdit.setPlaceholderText(self.readInComment)

            self.list_idx += 1
        except StopIteration:
            newFile = ""

        if newFile != self.filename:
            self.camera.close_camera()
            self.filename = newFile
            self.filename = self.getFullFileName(self.filename)  # self.videoDir / Path(self.filename)
            self.camera.initialize(self.filename)

        uniqueID = str(self.filename.name) + " " + str(self.currentFrameNumber)
        uniqueID += " " + str(self.bbox)
        self.label.setText(uniqueID)

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

            inset = frame[y1:y2, x1:x2].copy()
            insetHeight, insetWidth, channel = inset.shape
            bytesPerLine = 3 * insetWidth

            insetQimg = QImage(inset.data, insetWidth, insetHeight, bytesPerLine, QImage.Format_RGB888)
            insetPixmap = QPixmap(insetQimg)
            insetPixmap = insetPixmap.scaled(300, 300, Qt.KeepAspectRatio)
            self.resize(insetPixmap.width(), insetPixmap.height())

            self.insetImage.setPixmap(insetPixmap)

            # check if green rect is in inset
            rectangle(frame, (x1, y1), (x2, y2), (254, 97, 0), 2)
            # draw scale bar
            pt1 = (50, height - 50)
            pt2 = (50 + int(self.dLength), height - 50)
            line(frame, pt1, pt2, (0, 0, 0), 2)
        else:
            # Show "done!!" if no images left
            frame = putText(frame, 'Done!!', (750, 380), FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, LINE_AA)

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

        self.parent = parent
        self.fileName = filename
        self.originalFrame = currentFrame
        self.count = 0

        # frames to loop over including original frame, really half video length
        self.videoLength = int(100 / 2)
        self.setWindowTitle("Video Feed")
        self.timer = QTimer(self)
        startFrame = max(self.originalFrame - self.videoLength, 1)

        self.fvs = self.initVideo(startFrame, self.videoLength)

        self.timer.timeout.connect(self.update_video)

        # set up UI
        self.image_view = QLabel(self)
        # allows video to automatically rescale
        self.image_view.setScaledContents(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_view)
        self.setLayout(self.layout)
        # video speed ~ fps
        self.parent.tick = 1000./self.fvs.stream.get(CAP_PROP_FPS)

        self.timer.start(self.parent.tick)  # real time (ms)

    def update(self, name, frame):
        '''Function updates file video stream with new file, init frame etc.
        '''
        self.count = 0
        self.fileName = name
        self.originalFrame = frame
        startFrame = max(self.originalFrame - self.videoLength, 1)

        self.fvs.clear()
        self.fvs.stop()
        self.fvs.stream.release()
        self.fvs = None
        self.fvs = self.initVideo(startFrame, self.videoLength)

    def initVideo(self, start, length):
        ''' Get FileVideoStream object
        '''

        return FileVideoStream(self.fileName, start, length*2, 256).start()

    def resize(self, image, width):
        '''Function that resize an image and keeps image ratio.
        '''

        h, w, channel = image.shape
        ratio = width / float(w)
        dim = (width, int(h * ratio))

        return resize(image, dim, interpolation=INTER_AREA)

    def update_video(self):

        if not self.fvs.stopped:
            frame = self.fvs.read()
            self.count += 1
            frame = cvtColor(frame, COLOR_BGR2RGB)
            # draw scale bar
            height, _, _1 = frame.shape

            pt1 = (50, height - 50)
            pt2 = (50 + int(self.parent.dLength), height - 50)
            line(frame, pt1, pt2, (0, 0, 0), 2)

            minFrameRectShow = self.videoLength - 10
            maxFrameRectShow = self.videoLength + 10

            if self.count >= minFrameRectShow and self.count <= maxFrameRectShow:
                x1 = self.parent.bbox[0][1] - 20
                x2 = self.parent.bbox[1][1] + 20
                y1 = self.parent.bbox[0][0] + 110  # due to cropping in anaylsis
                y2 = self.parent.bbox[1][0] + 150

                rectangle(frame, (x1, y1), (x2, y2), (254, 97, 0), 2)

            frame = self.resize(frame, 1500)

            height, width, channel = frame.shape
            bytesPerLine = 3 * width

            pimg = QImage(frame, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap(pimg)

            self.image_view.setPixmap(pixmap)

        if self.count >= self.videoLength*2:
            self.count = 0

    def closeEvent(self, event):
        '''Closes the opened outfile on closing of QtApplication'''
        self.timer.stop()


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
