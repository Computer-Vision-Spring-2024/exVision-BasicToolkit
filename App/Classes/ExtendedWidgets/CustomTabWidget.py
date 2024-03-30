from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QTabWidget


class CustomTabWidget(QTabWidget):
    imgDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super(CustomTabWidget, self).__init__(parent)
        self.setTabsClosable(True)
        self.setAcceptDrops(True)

        self.dropped_path = None

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
                        self.imgDropped.emit(self.dropped_path)
                    else:
                        print("Not an image file:", self.dropped_path)
            except Exception as e:
                print("Error accessing file:", e)
