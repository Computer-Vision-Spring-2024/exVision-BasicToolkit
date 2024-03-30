import sys

from PyQt5.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout


class HoughTransformGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.hough_transform = None  # Reference to the associated Noise effect

        self.initUI()

    def initUI(self):
        # Vertical layout for the main content
        self.main_layout = QVBoxLayout()

        # Horizontal layout for label and combo box
        self.hough_type_hbox = QHBoxLayout()
        self.hough_type_label = QLabel("HT Type")
        self.hough_type_combo_box = QComboBox()
        self.hough_type_combo_box.addItems(["Line", "Circle", "Ellipse"])
        self.hough_type_hbox.addWidget(self.hough_type_label)
        self.hough_type_hbox.addWidget(self.hough_type_combo_box)

        # Add the horizontal layout to the vertical layout
        self.main_layout.addLayout(self.hough_type_hbox)

        # Set the main layout of the group box
        self.setLayout(self.main_layout)
