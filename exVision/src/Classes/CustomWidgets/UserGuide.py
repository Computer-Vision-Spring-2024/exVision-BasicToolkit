from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QMovie, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTreeWidgetItem,
    QVBoxLayout,
)


class UserGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.setWindowTitle("User Guide")
        self.resize(900, 600)

    def setupUi(self):
        # Add an icon
        icon = QIcon()
        icon.addPixmap(
            QPixmap("Resources/Icons/App_Icon.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        self.setWindowIcon(icon)

        # GIFs
        self.gifs = [
            "README_resources/Import__gif.gif",
            "README_resources/Navigate.gif",
            "README_resources/noise.gif",
        ]  # Paths to your GIFs

        # Documentation text
        self.doc_texts = {
            "Additive Noise": "Documentation for Additive Noise",
            "Uniform": "Documentation for Uniform Noise",
            "Gaussian": "Documentation for Gaussian Noise",
            "Salt & Pepper": "Documentation for Salt & Pepper Noise",
            # Add the rest of your documentation texts here...
        }

        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(10)
        self.setFont(font)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("splitter")

        self.documentation_tree = QtWidgets.QTreeWidget(self.splitter)
        self.documentation_tree.setObjectName("documentation_tree")
        self.documentation_tree.headerItem().setText(0, "Documentation")

        self.vertical_frame = QFrame(self.splitter)
        self.vertical_frame.setFrameShape(QFrame.StyledPanel)
        self.vertical_frame.setFrameShadow(QFrame.Raised)

        self.vertical_layout = QVBoxLayout(self.vertical_frame)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)

        self.gif_frame = QFrame(self.vertical_frame)
        self.gif_frame.setFrameShape(QFrame.StyledPanel)
        self.gif_frame.setFrameShadow(QFrame.Raised)
        self.gif_frame.setObjectName("gif_frame")
        self.gif_viewer = QLabel(self.gif_frame)
        self.gif_viewer.setGeometry(QtCore.QRect(20, 20, 400, 300))
        self.gif_viewer.setAlignment(Qt.AlignCenter)
        self.gif_viewer.setObjectName("gif_label")
        self.vertical_layout.addWidget(self.gif_frame)

        self.frame_documentation = QFrame(self.vertical_frame)
        self.frame_documentation.setFrameShape(QFrame.StyledPanel)
        self.frame_documentation.setFrameShadow(QFrame.Raised)
        self.frame_documentation.setObjectName("frame_documentation")
        self.documentation_text = QLabel(self.frame_documentation)
        self.documentation_text.setGeometry(QtCore.QRect(20, 20, 400, 200))
        self.documentation_text.setWordWrap(True)
        self.documentation_text.setObjectName("documentation_text")
        self.vertical_layout.addWidget(self.frame_documentation)

        self.mainLayout = QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addWidget(self.splitter)

        self.populate_tree()
        self.documentation_tree.itemClicked.connect(self.display_content)

        self.retranslateUi()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.gif_viewer.setText(_translate("UserGuideDialog", ""))
        self.documentation_text.setText(_translate("UserGuideDialog", ""))

    def populate_tree(self):
        tasks = {
            "Task 01:": [
                ("Additive Noise", ["Uniform", "Gaussian", "Salt & Pepper"]),
                ("Low-Pass Filters", ["Average", "Gaussian", "Median"]),
                ("Edge Detection", ["Sobel", "Roberts", "Prewitt", "Canny"]),
                "Draw Histogram and Distribution Curve",
                "Equalizer Histogram",
                "Image Equalization",
                "Image Normalization",
                "Local and Global Thresholding",
                "Convert to Grayscale, plot R, G, and B histograms with its distribution function",
                "Frequency domain filters (high-pass & low-pass)",
                "Hybrid Images",
            ],
            "Task 02:": [
                "Active Contour (SNAKE)",
                ("Hough Transform:", ["Line", "Circle", "Ellipse"]),
            ],
            "Task 03:": [
                ("Feature Extraction", ["Harris Operator", "Lambda-Minus"]),
                "SIFT",
                (
                    "Image Matching using",
                    [
                        "Sum of Squared Differences (SSD)",
                        "Normalized Cross Corrleations",
                    ],
                ),
            ],
            "Task 04:": [
                (
                    "For gray images, and both global & local ways:",
                    [
                        "Optimal Thresholding",
                        "OTSU Thresholding",
                        "Spectral Thresholding",
                    ],
                ),
                "Map RGB to LUV",
                (
                    "For color images:",
                    [
                        "K-means",
                        "Region Growing",
                        "Agglomerative Clustering",
                        "Mean Shift",
                    ],
                ),
            ],
            "Task 05:": [
                "Detect Faces (color or grayscale)",
                "Recognize faces based on PCA/Eigen analysis",
                "Report performance and plot ROC curve",
            ],
        }

        for task, items in tasks.items():
            task_item = QTreeWidgetItem(self.documentation_tree)
            task_item.setText(0, task)
            for item in items:
                if isinstance(item, tuple):
                    parent_item = QTreeWidgetItem(task_item)
                    parent_item.setText(0, item[0])
                    for subitem in item[1]:
                        child_item = QTreeWidgetItem(parent_item)
                        child_item.setText(0, subitem)
                else:
                    child_item = QTreeWidgetItem(task_item)
                    child_item.setText(0, item)

    def display_content(self, item):
        item_text = item.text(0)
        if item_text in self.doc_texts:
            self.documentation_text.setText(self.doc_texts[item_text])
            gif_path = self.gifs[
                0
            ]  # Update this to select the appropriate GIF for each item
            movie = QMovie(gif_path)
            self.gif_viewer.setMovie(movie)
            movie.start()
        else:
            self.documentation_text.setText("No documentation available.")
            self.gif_viewer.clear()


"""
import sys

import fitz  # PyMuPDF
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QMovie, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QTreeWidgetItem,
    QVBoxLayout,
)


class UserGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.setWindowTitle("User Guide")
        self.resize(900, 600)

    def setupUi(self):
        # Add an icon
        icon = QIcon()
        icon.addPixmap(
            QPixmap("Resources/Icons/App_Icon.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        self.setWindowIcon(icon)

        # Paths to your GIFs
        self.gifs = [
            "README_resources/Import__gif.gif",
            "README_resources/Navigate.gif",
            "README_resources/noise.gif",
        ]

        # Documentation PDFs
        self.pdf_docs = {
            "Additive Noise": "path/to/AdditiveNoise.pdf",
            "Uniform": "path/to/UniformNoise.pdf",
            "Gaussian": "path/to/GaussianNoise.pdf",
            "Salt & Pepper": "Resources/Tasks_Documentation/Task01/Salt_and_Pepper.pdf",
            # Add the rest of your documentation PDFs here...
        }

        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(10)
        self.setFont(font)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("splitter")

        self.documentation_tree = QtWidgets.QTreeWidget(self.splitter)
        self.documentation_tree.setObjectName("documentation_tree")
        self.documentation_tree.headerItem().setText(0, "Documentation")

        self.vertical_frame = QFrame(self.splitter)
        self.vertical_frame.setFrameShape(QFrame.StyledPanel)
        self.vertical_frame.setFrameShadow(QFrame.Raised)

        self.vertical_layout = QVBoxLayout(self.vertical_frame)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)

        self.gif_frame = QFrame(self.vertical_frame)
        self.gif_frame.setFrameShape(QFrame.StyledPanel)
        self.gif_frame.setFrameShadow(QFrame.Raised)
        self.gif_frame.setObjectName("gif_frame")
        self.gif_viewer = QLabel(self.gif_frame)
        self.gif_viewer.setGeometry(QtCore.QRect(20, 20, 400, 300))
        self.gif_viewer.setAlignment(Qt.AlignCenter)
        self.gif_viewer.setObjectName("gif_label")
        self.vertical_layout.addWidget(self.gif_frame)

        self.scroll_area = QScrollArea(self.vertical_frame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QtWidgets.QWidget(self.scroll_area)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.pdf_viewer = QLabel(self.scroll_content)
        self.scroll_layout.addWidget(self.pdf_viewer)
        self.scroll_area.setWidget(self.scroll_content)
        self.vertical_layout.addWidget(self.scroll_area)

        self.mainLayout = QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addWidget(self.splitter)

        self.populate_tree()
        self.documentation_tree.itemClicked.connect(self.display_content)

        self.retranslateUi()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.gif_viewer.setText(_translate("UserGuideDialog", ""))
        self.pdf_viewer.setText(_translate("UserGuideDialog", ""))

    def populate_tree(self):
        tasks = {
            "Task 01:": [
                ("Additive Noise", ["Uniform", "Gaussian", "Salt & Pepper"]),
                ("Low-Pass Filters", ["Average", "Gaussian", "Median"]),
                ("Edge Detection", ["Sobel", "Roberts", "Prewitt", "Canny"]),
                "Draw Histogram and Distribution Curve",
                "Equalizer Histogram",
                "Image Equalization",
                "Image Normalization",
                "Local and Global Thresholding",
                "Convert to Grayscale, plot R, G, and B histograms with its distribution function",
                "Frequency domain filters (high-pass & low-pass)",
                "Hybrid Images",
            ],
            "Task 02:": [
                "Active Contour (SNAKE)",
                ("Hough Transform:", ["Line", "Circle", "Ellipse"]),
            ],
            "Task 03:": [
                ("Feature Extraction", ["Harris Operator", "Lambda-Minus"]),
                "SIFT",
                (
                    "Image Matching using",
                    [
                        "Sum of Squared Differences (SSD)",
                        "Normalized Cross Corrleations",
                    ],
                ),
            ],
            "Task 04:": [
                (
                    "For gray images, and both global & local ways:",
                    [
                        "Optimal Thresholding",
                        "OTSU Thresholding",
                        "Spectral Thresholding",
                    ],
                ),
                "Map RGB to LUV",
                (
                    "For color images:",
                    [
                        "K-means",
                        "Region Growing",
                        "Agglomerative Clustering",
                        "Mean Shift",
                    ],
                ),
            ],
            "Task 05": [
                "Detect Faces (color or grayscale)",
                "Recognize faces based on PCA/Eigen analysis",
                "Report performance and plot ROC curve",
            ],
        }

        for task, items in tasks.items():
            task_item = QTreeWidgetItem(self.documentation_tree)
            task_item.setText(0, task)
            for item in items:
                if isinstance(item, tuple):
                    parent_item = QTreeWidgetItem(task_item)
                    parent_item.setText(0, item[0])
                    for subitem in item[1]:
                        child_item = QTreeWidgetItem(parent_item)
                        child_item.setText(0, subitem)
                else:
                    child_item = QTreeWidgetItem(task_item)
                    child_item.setText(0, item)

    def display_content(self, item):
        item_text = item.text(0)
        if item_text in self.pdf_docs:
            self.display_pdf(self.pdf_docs[item_text])
        else:
            self.pdf_viewer.clear()
            self.gif_viewer.clear()

    def display_pdf(self, pdf_path):
        # Open the PDF
        document = fitz.open(pdf_path)
        # Render the first page to an image
        page = document.load_page(0)
        pix = page.get_pixmap()
        # Convert the image to QPixmap
        image = QtGui.QImage(
            pix.samples, pix.width, pix.height, pix.stride, QtGui.QImage.Format_RGB888
        )
        qpixmap = QPixmap.fromImage(image)
        self.pdf_viewer.setPixmap(qpixmap)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    user_guide = UserGuideDialog()
    user_guide.show()
    sys.exit(app.exec_())

"""
