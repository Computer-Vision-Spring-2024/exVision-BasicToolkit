from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)


class ThresholdingGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.thresholding_effect = None

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Threshold Type
        threshold_type_layout = QHBoxLayout()
        self.threshold_type_label = QLabel("Threshold Type")
        self.threshold_type_comb = QComboBox()
        self.threshold_type_comb.addItems(["Local", "Global"])
        self.threshold_type_comb.setCurrentIndex(0)
        self.threshold_type_comb.currentIndexChanged.connect(
            self.update_threshold_options
        )
        threshold_type_layout.addWidget(self.threshold_type_label)
        threshold_type_layout.addWidget(self.threshold_type_comb)

        # Block Size (Local), Global Threshold (Global)
        self.block_size_label = QLabel("Block Size")
        self.block_size_slider = QSlider(Qt.Horizontal)
        self.block_size_spinbox = QSpinBox()
        self.block_size_spinbox.setValue(9)
        self.block_size_slider.valueChanged.connect(self.block_size_spinbox.setValue)
        self.block_size_spinbox.valueChanged.connect(self.block_size_slider.setValue)
        self.block_size_HLayout = QHBoxLayout()
        self.block_size_HLayout.addWidget(self.block_size_slider)
        self.block_size_HLayout.addWidget(self.block_size_spinbox)

        main_layout.addLayout(threshold_type_layout)
        main_layout.addWidget(self.block_size_label)
        main_layout.addLayout(self.block_size_HLayout)

        self.setLayout(main_layout)

    def update_threshold_options(self, index):
        if index == 0:  # Local
            self.block_size_label.setText("Block Size")
            self.block_size_slider.setRange(0, 255)
            self.block_size_spinbox.setRange(0, 255)
        elif index == 1:  # Global
            self.block_size_label.setText("Global Threshold Value")
            self.block_size_slider.setRange(0, 255)
            self.block_size_spinbox.setRange(0, 255)
