import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication
from PyQt5.QtGui import QPixmap, QImage

from models import Camera


class StartWindow(QMainWindow):
    def __init__(self, size, generatorFile):
        super().__init__()

        self.genny = generatorFile
        self.filename, self.currentFrameNumber, self.bbox = next(self.genny)
        self.outFile = open("labels.dat", "w")

        self.camera = Camera()
        self.camera.initialize(self.filename)

        self.init_UI(size)
        self.update_image()

        self.dolphinAction.clicked.connect(self.update_image_dolph)
        self.otherAction.clicked.connect(self.update_image_other)

    def intersection(self, a, b):

        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0:
            return None
        return (x, y, w, h)

    def init_UI(self, size):
        '''Sets up UI'''

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
        '''If dolphin button clicked records object as a dolphin'''

        self.outFile.write(f"{self.currentFrameNumber}, {self.bbox}, dolphin" + "\n")
        self.get_next_image_data()
        self.update_image()

    def update_image_other(self):
        '''If other button clicked records object as other'''

        self.get_next_image_data()
        self.outFile.write(f"{self.currentFrameNumber}, {self.bbox}, other" + "\n")
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

    def show_ROI(self, x1, x2, y1, y2, frame):
        # get ROI and resize it and show as an inset
        hdiff = int((y2 - y1) / 2)
        wdiff = int((x2 - x1) / 2)
        ROI = frame[y1-hdiff:y2+hdiff, x1-wdiff:x2+wdiff]
        ROI = cv2.resize(ROI, interpolation=cv2.INTER_NEAREST, dsize=(0, 0), fx=20, fy=20)
        heightROI, widthROI, _ = ROI.shape
        if heightROI >= frame.shape[0] or widthROI >= frame.shape[1]:
            ROI = frame[y1-hdiff:y2+hdiff, x1-wdiff:x2+wdiff]
            ROI = cv2.resize(ROI, interpolation=cv2.INTER_NEAREST, dsize=(0, 0), fx=10, fy=10)
            heightROI, widthROI, _ = ROI.shape

        # check if inset collides with ROI box. If so move it to opposite side.
        inter = self.intersection([0, 0, widthROI, heightROI], [x1, y1, (x2 - x1), (y2 - y1)])
        if inter:
            if inter[0] < width/2:
                frame[0:heightROI, width-widthROI:] = ROI
                cv2.rectangle(frame, (width-widthROI, 0), (width, heightROI), (0, 0, 0), 2)

            elif inter[0] > width/2:
                frame[0:heightROI, 0:widthROI] = ROI
                cv2.rectangle(frame, (0, 0), (widthROI, heightROI), (0, 0, 0), 2)

            else:
                print("Error!!! in show_ROI")
                sys.exit()
        else:
            frame[0:heightROI, 0:widthROI] = ROI
            cv2.rectangle(frame, (0, 0), (widthROI, heightROI), (0, 0, 0), 2)

        return frame

    def update_image(self):
        '''Updates displayed image and shows ROI as an inset.'''
        import matplotlib.pyplot as plt
        frame = self.camera.get_frame(self.currentFrameNumber)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        if self.filename != "":
            x1 = self.bbox[0][1]
            x2 = self.bbox[1][1]
            y1 = self.bbox[0][0] + 130  # due to cropping in anaylsis
            y2 = self.bbox[1][0] + 130

            frame = self.show_ROI(x1, x2, y1, y2, frame)
            # check if green rect is in inset
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Show "done!!" if no images left
            frame = cv2.putText(frame, 'Done!!', (750, 380), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)

        # update canvas image
        qimg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qimg)
        pixmap = pixmap.scaled(1500, 1500, Qt.KeepAspectRatio)
        self.resize(pixmap.width(), pixmap.height())
        self.image_view.setPixmap(pixmap)

    def close_event(self, event):
        '''Closes the opened outfile on closing of QtApplication'''
        self.outFile.close()


if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
