from PyQt5.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel


class EdgeDetectorGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.edge_effect = None

        self.label = QLabel("Edge Detector")

        self.edge_widget_combo_box = QComboBox()
        self.lookup = [
            "sobel_3x3",
            "sobel_5x5",
            "roberts",
            "prewitt",
            "laplacian",
            "canny",
        ]
        self.edge_widget_combo_box.addItems(self.lookup)

        hbox = QHBoxLayout()
        hbox.addWidget(self.label)
        hbox.addWidget(self.edge_widget_combo_box)
        self.setLayout(hbox)
