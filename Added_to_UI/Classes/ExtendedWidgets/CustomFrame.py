import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFrame


class CustomFrame(QFrame):
    imgDropped = pyqtSignal(np.ndarray, int, str)

    def __init__(self, title, frame_name, flag, parent=None):
        super(CustomFrame, self).__init__(parent)
        self.setAcceptDrops(True)

        self.dropped_path = None
        self.subplot = None
        self.figure = None
        self.canvas = None
        self.title = title
        self.frame_name = frame_name
        self.flag = flag
        self.setup_canvas()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self.dropped_path = url.toLocalFile()
            try:
                with open(self.dropped_path, "rb") as file:
                    # Check if it's an image file
                    if self.dropped_path.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".pgm")
                    ):
                        img = cv2.imread(self.dropped_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = self.to_grayscale(img)
                        if not self.flag == 3:
                            self.imgDropped.emit(img, self.flag, self.dropped_path)
                        else:
                            return
                    else:
                        print("Not an image file:", self.dropped_path)
            except Exception as e:
                print("Error accessing file:", e)

    def setup_canvas(self):
        """
        Description:
            - Makes a Layout to which a FigureCanvas object is added, to display the
            images in the hybrid tab.
        """
        self.setObjectName(self.frame_name)
        self.setFrameShape(self.StyledPanel)
        self.setFrameShadow(self.Raised)
        self.setStyleSheet("border: 1px solid white;")
        # Create figure and subplot
        self.figure = plt.figure()
        self.subplot = self.figure.add_subplot(111)
        self.subplot.set_title(self.title, color="white")
        self.subplot.axis("off")
        self.figure.patch.set_facecolor("none")
        # Create canvas
        self.canvas = FigureCanvas(self.figure)
        # Create layouts for frame
        layout_obj = QtWidgets.QVBoxLayout(self)
        # Add canvas to layout
        layout_obj.addWidget(self.canvas)

    def Display_image(self, img):
        """
        Description:
            - Display the given image in the canvas.

        Args:
            - img: The image to be displayed.
        """
        self.subplot.imshow(img, cmap="gray")
        self.canvas.draw()

    def to_grayscale(self, image):
        """
        Convert an image to grayscale by averaging the red, green, and blue channels for each pixel.

        Parameters:
        - image: numpy.ndarray
            The input image.

        Returns:
        - numpy.ndarray
            The grayscale image.
        """
        # Get the dimensions of the image
        height, width, _ = image.shape

        # Create an empty array to store the grayscale image
        grayscale_image = np.zeros((height, width), dtype=np.uint8)

        # Iterate over each pixel and use the linear approximation of gamma correction.
        for y in range(height):
            for x in range(width):
                r, g, b = image[y, x]
                grayscale_image[y, x] = 0.299 * r + 0.587 * g + 0.114 * b

        return grayscale_image
