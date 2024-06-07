from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


class EdgeDetectorGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.edge_effect = None
        self.main_layout = QVBoxLayout()
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
        self.edge_widget_combo_box.setCurrentIndex(5)
        hbox = QHBoxLayout()
        hbox.addWidget(self.label)
        hbox.addWidget(self.edge_widget_combo_box)
        self.low_threshold_label = QLabel("Low Threshold Ratio")
        self.low_threshold_spinbox = QDoubleSpinBox()
        self.low_threshold_spinbox.setSingleStep(0.01)
        # Intialize a value for BETA
        self.low_threshold_spinbox.setValue(0.09)
        lowthreshold_HLayout = QHBoxLayout()
        lowthreshold_HLayout.addWidget(self.low_threshold_label)
        lowthreshold_HLayout.addWidget(self.low_threshold_spinbox)

        self.high_threshold_label = QLabel("High Threshold Ratio")
        self.high_threshold_spinbox = QDoubleSpinBox()
        self.high_threshold_spinbox.setSingleStep(0.01)
        # Intialize a value for BETA
        self.high_threshold_spinbox.setValue(0.3)
        highthreshold_HLayout = QHBoxLayout()
        highthreshold_HLayout.addWidget(self.high_threshold_label)
        highthreshold_HLayout.addWidget(self.high_threshold_spinbox)

        self.main_layout.addLayout(hbox)
        self.main_layout.addLayout(lowthreshold_HLayout)
        self.main_layout.addLayout(highthreshold_HLayout)
        self.setLayout(self.main_layout)
