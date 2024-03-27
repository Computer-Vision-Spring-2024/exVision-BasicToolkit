from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class HybridGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.hybrid_images_effect = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        image1_layout = QHBoxLayout()
        image1_path_label = QLabel("Img1 Path:")
        self.line_edit1 = QLineEdit()
        self.line_edit1.setReadOnly(True)
        image1_layout.addWidget(image1_path_label)
        image1_layout.addWidget(self.line_edit1)

        combobox_layout = QHBoxLayout()
        combobox_label = QLabel("Our Images:")
        self.combobox = QComboBox()
        self.checkbox_img1 = QtWidgets.QCheckBox("Image1")
        combobox_layout.addWidget(self.combobox)
        combobox_layout.addWidget(self.checkbox_img1)

        image2_layout = QHBoxLayout()
        image2_path_label = QLabel("Img2 Path:")
        self.line_edit2 = QLineEdit()
        self.line_edit2.setReadOnly(True)
        image2_layout.addWidget(image2_path_label)
        image2_layout.addWidget(self.line_edit2)

        self.radio_low_pass = QtWidgets.QRadioButton("Low-pass on Image One")
        self.radio_high_pass = QtWidgets.QRadioButton("High-pass on Image One")
        # Set the first radio button to be initially selected
        self.radio_low_pass.setChecked(True)

        # Create spin boxes for cutoff frequencies
        low_pass_layout = QHBoxLayout()
        self.spin_low_pass = QtWidgets.QSpinBox()
        self.spin_low_pass.setRange(0, 1000)
        label_low_pass = QtWidgets.QLabel("Low-pass Cutoff:")
        low_pass_layout.addWidget(label_low_pass)
        low_pass_layout.addWidget(self.spin_low_pass)

        high_pass_layout = QHBoxLayout()
        self.spin_high_pass = QtWidgets.QSpinBox()
        self.spin_high_pass.setRange(0, 1000)
        label_high_pass = QtWidgets.QLabel("High-pass Cutoff:")
        high_pass_layout.addWidget(label_high_pass)
        high_pass_layout.addWidget(self.spin_high_pass)
        # Set initial values for the Cutoff frequencies
        self.spin_low_pass.setValue(20)
        self.spin_high_pass.setValue(10)

        main_layout.addLayout(image1_layout)

        main_layout.addLayout(image2_layout)
        main_layout.addWidget(combobox_label)
        main_layout.addLayout(combobox_layout)
        # Add radio buttons to layout
        main_layout.addWidget(self.radio_low_pass)
        main_layout.addWidget(self.radio_high_pass)

        # Add spin boxes to layout
        main_layout.addLayout(low_pass_layout)
        main_layout.addLayout(high_pass_layout)
        self.setLayout(main_layout)
