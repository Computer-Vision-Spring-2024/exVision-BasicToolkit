import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets


class CanvasWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_canvas()

    def setup_canvas(self):
        # Main Viewport Layout
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Main Viewport Frame
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)

        # Main Viewport Canvas Layout
        self.canvas_layout = QtWidgets.QVBoxLayout(self.frame)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Create a figure and a canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.figure.patch.set_facecolor("none")

        # Add the canvas to the layout
        self.canvas_layout.addWidget(self.canvas)

        # Add the frame to the main layout
        self.layout.addWidget(self.frame)
