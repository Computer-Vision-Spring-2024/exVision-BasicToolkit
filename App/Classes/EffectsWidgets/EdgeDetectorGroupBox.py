from PyQt5.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QDoubleSpinBox


class EdgeDetectorGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.edge_effect = None
        main_layout = QVBoxLayout()
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
        self.low_threshold_label = QLabel("Low Threshold Ratio")
        self.low_threshold_spinbox = QDoubleSpinBox()
        self.low_threshold_spinbox.setSingleStep(0.01)
        # Intialize a value for BETA
        self.low_threshold_spinbox.setValue(0.05)
        lowthreshold_HLayout = QHBoxLayout()
        lowthreshold_HLayout.addWidget(self.low_threshold_label)
        lowthreshold_HLayout.addWidget(self.low_threshold_spinbox)
        
        self.high_threshold_label = QLabel("High Threshold Ratio")
        self.high_threshold_spinbox = QDoubleSpinBox()
        self.high_threshold_spinbox.setSingleStep(0.01)
        # Intialize a value for BETA
        self.high_threshold_spinbox.setValue(0.2)
        highthreshold_HLayout = QHBoxLayout()
        highthreshold_HLayout.addWidget(self.high_threshold_label)
        highthreshold_HLayout.addWidget(self.high_threshold_spinbox)

        
        main_layout.addLayout(hbox)
        main_layout.addLayout(lowthreshold_HLayout)
        main_layout.addLayout(highthreshold_HLayout)
        self.setLayout(main_layout)
