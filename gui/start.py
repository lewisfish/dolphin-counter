from PyQt5.QtWidgets import QApplication

from views import StartWindow
from readFile import createDict

file = "final-output-dlength.dat"
genVals = createDict(file)

app = QApplication(["Object labeler"])
screen = app.primaryScreen()
size = screen.size()

start_window = StartWindow(size, genVals)
start_window.show()
app.exit(app.exec_())
