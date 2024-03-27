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


class NoiseGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.noise_effect = None  # Reference to the associated Noise effect

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Noise Type
        noise_type_layout = QHBoxLayout()
        self.noise_type_label = QLabel("Noise Type")
        self.noise_type_comb = QComboBox()
        self.noise_type_comb.addItems(["Uniform", "Gaussian", "Salt & Pepper"])
        self.noise_type_comb.currentIndexChanged.connect(self.update_noise_options)
        noise_type_layout.addWidget(self.noise_type_label)
        noise_type_layout.addWidget(self.noise_type_comb)

        # Lower Limit / Mean / Salt
        self.lower_label = QLabel("Lower Limit")
        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_spinbox = QSpinBox()
        self.lower_slider.valueChanged.connect(self.lower_spinbox.setValue)
        self.lower_spinbox.valueChanged.connect(self.lower_slider.setValue)
        self.lower_HLayout = QHBoxLayout()
        self.lower_HLayout.addWidget(self.lower_slider)
        self.lower_HLayout.addWidget(self.lower_spinbox)

        # Upper Limit / Standard Deviation / Pepper
        self.upper_label = QLabel("Upper Limit")
        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_spinbox = QSpinBox()
        self.upper_slider.valueChanged.connect(self.upper_spinbox.setValue)
        self.upper_spinbox.valueChanged.connect(self.upper_slider.setValue)
        self.upper_HLayout = QHBoxLayout()
        self.upper_HLayout.addWidget(self.upper_slider)
        self.upper_HLayout.addWidget(self.upper_spinbox)

        self.update_noise_options(0)  # Set default options for Uniform noise

        main_layout.addLayout(noise_type_layout)
        main_layout.addWidget(self.lower_label)
        main_layout.addLayout(self.lower_HLayout)
        main_layout.addWidget(self.upper_label)
        main_layout.addLayout(self.upper_HLayout)

        self.setLayout(main_layout)

    def update_noise_options(self, index):
        if index == 0:  # Uniform
            self.lower_label.setText("Lower Limit")
            self.upper_label.setText("Upper Limit")
            self.lower_slider.setRange(0, 100)
            self.upper_slider.setRange(0, 100)
            self.lower_spinbox.setRange(0, 100)
            self.upper_spinbox.setRange(0, 100)
        elif index == 1:  # Gaussian
            self.lower_label.setText("Mean")
            self.upper_label.setText("Standard Deviation")
            self.lower_slider.setRange(0, 100)
            self.upper_slider.setRange(0, 100)
            self.lower_spinbox.setRange(0, 100)
            self.upper_spinbox.setRange(0, 100)
        elif index == 2:  # Salt & Pepper
            self.lower_label.setText("Pepper")
            self.upper_label.setText("Salt")
            self.lower_slider.setRange(0, 100)
            self.upper_slider.setRange(0, 100)
            self.lower_spinbox.setRange(0, 100)
            self.upper_spinbox.setRange(0, 100)
