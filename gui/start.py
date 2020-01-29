from argparse import ArgumentParser

from PyQt5.QtWidgets import QApplication

from views import StartWindow
from readFile import createDict

parser = ArgumentParser(description="Helps label video data for ML.")

parser.add_argument("-f", "--file", type=str,
                    help="Path to file that contains frame and bbox data.")


args = parser.parse_args()

genVals = createDict(args.file)

app = QApplication([])
screen = app.primaryScreen()
size = screen.size()

start_window = StartWindow(size, genVals)
start_window.show()
app.exit(app.exec_())
