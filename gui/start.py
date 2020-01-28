from argparse import ArgumentParser

from PyQt5.QtWidgets import QApplication

from models import Camera
from views import StartWindow
from readFile import iter_dict, createDict

parser = ArgumentParser(description="Counts objects in a picture")

parser.add_argument("-f", "--file", type=str,
                    help="Path to single image to be analysed.")


args = parser.parse_args()

genVals = iter_dict(createDict(args.file))

camera = Camera(None)
# camera.initialize()

app = QApplication([])
screen = app.primaryScreen()
size = screen.size()

start_window = StartWindow(size, camera)
start_window.show()
app.exit(app.exec_())
