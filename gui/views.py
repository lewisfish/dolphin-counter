import numpy as np

from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QLabel, QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider

from PyQt5.QtGui import QPixmap, QImage


class StartWindow(QMainWindow):
    def __init__(self, size, camera=None):
        super().__init__()
        self.camera = camera
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

        self.dolphinAction.clicked.connect(self.update_image_dolph)
        self.otherAction.clicked.connect(self.update_image_other)

    def update_image_dolph(self):
        self.update_image()

    def update_image_other(self):
        self.update_image()

    def update_image(self):
        frame = self.camera.get_frame()
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
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
