import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication
from PyQt5.QtGui import QPixmap, QImage

from models import Camera


class StartWindow(QMainWindow):
    def __init__(self, size, generatorFile):
        super().__init__()

        self.genny = generatorFile
        self.filename, self.currentFrameNumber, self.bbox = next(self.genny)

        self.camera = Camera()
        self.camera.initialize(self.filename)

        self.init_UI(size)
        self.update_image()

        self.dolphinAction.clicked.connect(self.update_image_dolph)
        self.otherAction.clicked.connect(self.update_image_other)

    def init_UI(self, size):
        self.size = size
        self.central_widget = QWidget()
        self.dolphinAction = QPushButton('Dolphin', self.central_widget)
        self.otherAction = QPushButton('Other', self.central_widget)
        self.image_view = QLabel(self)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.dolphinAction)
        self.layout.addWidget(self.otherAction)
        self.layout.addWidget(self.image_view)
        self.setCentralWidget(self.central_widget)

    def update_image_dolph(self):
        self.get_next_image_data()
        self.update_image()

    def update_image_other(self):
        self.get_next_image_data()
        self.update_image()

    def get_next_image_data(self):
        try:
            newFile, self.currentFrameNumber, self.bbox = next(self.genny)
        except StopIteration:
            newFile = ""

        if newFile != self.filename:
            self.camera.close_camera()
            self.filename = newFile
            self.camera.initialize(self.filename)

    def update_image(self):
        frame = self.camera.get_frame(self.currentFrameNumber)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        if self.filename != "":
            x1 = self.bbox[0][0]
            x2 = self.bbox[1][0]
            y1 = self.bbox[0][1]
            y2 = self.bbox[1][1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            frame = cv2.putText(frame, 'Done!!', (750, 380), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)
        qimg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qimg)
        pixmap = pixmap.scaled(1500, 1500, Qt.KeepAspectRatio)
        self.resize(pixmap.width(), pixmap.height())
        self.image_view.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
